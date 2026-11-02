"""Step 7 evaluator checks; no 4B model loading or GPU generation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from meta_data_extraction.data_models import ExtractionResult, Provenance
from meta_data_extraction.evaluation import build_evaluator_metadata
from meta_data_extraction.finetune.evaluation import (
    EvaluationConfig,
    _summary,
    _strict_json_object,
    compare_conditions,
    prepare_condition_output,
    schema_errors,
    score_condition,
)
from meta_data_extraction.finetune.tokenization import load_tokenizer
from meta_data_extraction.finetune.tracking import checksum
from meta_data_extraction.finetune.validation import write_json


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ["generic.json", "photon_and_neutron.json"]


def config(output_dir: str = "outputs/finetune/evaluation") -> EvaluationConfig:
    return EvaluationConfig(
        version="1.0.0", prepare_config="configs/finetune_prepare.yaml",
        json_only_config="configs/finetune_json_only.yaml",
        structure_aware_config="configs/finetune_structure_aware.yaml",
        metrics_config="configs/evaluation_metrics.yaml",
        alignment_config="configs/alignment_helper.yaml", output_dir=output_dir,
        base_run_dir="outputs/finetune/base_evaluation", domain_name="photon_and_neutron",
        seed=42, max_seq_length=65536, max_new_tokens=16384,
        enable_thinking=False, decoding="greedy",
    )


def test_strict_json_rejects_nonobjects_duplicates_and_nonfinite_values():
    assert _strict_json_object('{"a": 1}') == {"a": 1}
    for text in ('[1]', '{"a": 1, "a": 2}', '{"a": NaN}'):
        with pytest.raises(ValueError):
            _strict_json_object(text)


def test_schema_contract_accepts_gold_and_reports_structural_and_reference_errors():
    gold = json.loads((ROOT / "data/gold/2010.02965.json").read_text(encoding="utf-8"))
    assert schema_errors(gold, SCHEMAS) == []
    broken = json.loads(json.dumps(gold))
    del broken["bibliographic"]["title"]
    broken["unknown"] = 1
    broken["datasets"][0]["sample_refs"] = ["sample_missing"]
    errors = schema_errors(broken, SCHEMAS)
    assert any("title: missing" in error for error in errors)
    assert any("unknown field" in error for error in errors)
    assert any("dangling" in error for error in errors)


def test_all_frozen_test_gold_records_satisfy_schema_contract():
    test_ids = json.loads((ROOT / "data/finetune/splits.json").read_text(encoding="utf-8"))["test"]
    failures = {}
    for paper_id in test_ids:
        gold = json.loads((ROOT / "data/gold" / f"{paper_id}.json").read_text(encoding="utf-8"))
        errors = schema_errors(gold, SCHEMAS)
        if errors:
            failures[paper_id] = errors
    assert failures == {}


def test_test_prompt_audit_uses_pinned_cached_tokenizer_without_gold(monkeypatch):
    from meta_data_extraction.finetune.evaluation import prepare_test_rows

    prepare = __import__("yaml").safe_load((ROOT / "configs/finetune_prepare.yaml").read_text(encoding="utf-8"))
    try:
        tokenizer, identity = load_tokenizer(prepare, local_files_only=True)
    except OSError:
        pytest.skip("Pinned tokenizer is not cached")
    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        assert not (path.parent.name == "gold" and path.suffix == ".json"), "Prompt preparation opened test gold"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    rows, data = prepare_test_rows(ROOT, config(), tokenizer, identity)
    assert len(rows) == 30 and [row["paper_id"] for row in rows] == data["test_ids"]
    assert all(row["prompt_tokens"] + 16384 <= 65536 for row in rows)


def test_condition_output_is_immutable_and_resumable(tmp_path):
    contract = {"version": "fixture", "condition": "m0"}
    prepare_condition_output(tmp_path / "run", contract)
    prepare_condition_output(tmp_path / "run", contract)
    with pytest.raises(ValueError, match="different contract"):
        prepare_condition_output(tmp_path / "run", {**contract, "condition": "m1"})


def test_summary_preserves_undefined_conditional_reference_scores():
    generation = [{"json_valid": False, "schema_valid": False, "status": "invalid_json",
                   "hit_output_limit": False, "latency_ms": 1.0, "generated_tokens": 0,
                   "peak_allocated_bytes": 0, "peak_reserved_bytes": 0} for _ in range(30)]
    fields = {name: {"f1": 0.0} for name in ("facilities", "samples", "datasets")}
    fields.update({"datasets.attrs.sample_refs": {"n_papers": 30},
                   "datasets.attrs.facility_refs": {"n_papers": 30}})
    summary = _summary(generation, fields, 30)
    assert summary["sample_refs_f1"] is None
    assert summary["facility_refs_f1"] is None


def _write_generation_fixture(output: Path, cfg: EvaluationConfig, *, one_invalid=False):
    split = json.loads((ROOT / "data/finetune/splits.json").read_text(encoding="utf-8"))
    ids = split["test"]
    manifest = {"evaluator": build_evaluator_metadata(ROOT / cfg.metrics_config),
                "data": {"paper_ids": ids}}
    write_json(output / "manifest.json", manifest)
    records = []
    for index, paper_id in enumerate(ids):
        gold = json.loads((ROOT / "data/gold" / f"{paper_id}.json").read_text(encoding="utf-8"))
        valid = not (one_invalid and index == 0)
        result = ExtractionResult(document_id=paper_id, domain_name=cfg.domain_name,
                                  output=gold if valid else {}, raw_output=json.dumps(gold) if valid else "broken",
                                  provenance=Provenance(config_name="fixture"),
                                  warnings=[] if valid else ["invalid fixture"])
        prediction = output / "predictions" / f"{paper_id}.json"
        write_json(prediction, result.model_dump(mode="json"))
        records.append({"paper_id": paper_id, "status": "generated" if valid else "invalid_json",
                        "json_valid": valid, "schema_valid": valid, "schema_errors": [],
                        "prompt_tokens": 1, "generated_tokens": 1 if valid else 0,
                        "ended_with_eos": valid, "hit_output_limit": False, "latency_ms": 1.0,
                        "peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
                        "raw_file": f"raw/{paper_id}.txt", "prediction_file": f"predictions/{paper_id}.json",
                        "raw_sha256": "unused", "prediction_sha256": checksum(prediction), "warnings": []})
    (output / "generation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def test_scoring_keeps_all_30_papers_and_penalizes_invalid_generation(tmp_path):
    cfg = config()
    output = tmp_path / "m0"
    _write_generation_fixture(output, cfg, one_invalid=True)
    metrics = score_condition(ROOT, cfg, "ablation_fixture_seed42", output)
    assert metrics["method"] == "ablation_fixed_endpoint"
    assert metrics["all_papers"]["n_papers"] == 30
    assert metrics["valid_predictions_only"]["n_papers"] == 29
    assert metrics["summary"]["papers_json_valid"] == 29
    assert metrics["summary"]["scoring_fallbacks"] == 1
    assert metrics["summary"]["facilities_f1"] < 1.0
    assert len(metrics["papers"]) == 30


def test_comparison_requires_identical_paper_accounting(tmp_path):
    cfg = config(str(tmp_path / "evaluation"))
    output = Path(cfg.output_dir)
    ids = json.loads((ROOT / "data/finetune/splits.json").read_text(encoding="utf-8"))["test"]
    evaluator = build_evaluator_metadata(ROOT / cfg.metrics_config)
    summary = {"json_valid_rate": 1.0, "schema_valid_rate": 1.0,
               "facilities_f1": 1.0, "samples_f1": 1.0, "datasets_f1": 1.0,
               "sample_refs_f1": 1.0, "facility_refs_f1": 1.0,
               "generic_metadata_macro_score": 1.0}
    for condition in ("m0", "m1", "m2"):
        write_json(output / condition / "metrics.json",
                   {"evaluator": evaluator, "aggregation_policy": {"primary": "all papers"},
                    "summary": summary, "papers": [{"paper_id": paper_id} for paper_id in ids]})
    comparison = compare_conditions(ROOT, cfg)
    assert list(comparison["conditions"]) == ["m0", "m1", "m2"]
    assert (output / "comparison.md").exists()
    changed = json.loads((output / "m2/metrics.json").read_text(encoding="utf-8"))
    changed["papers"].reverse()
    write_json(output / "m2/metrics.json", changed)
    with pytest.raises(ValueError, match="identical ordered papers"):
        compare_conditions(ROOT, cfg)
