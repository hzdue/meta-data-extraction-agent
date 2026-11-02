"""Fixed-endpoint exploratory development-evaluation contract checks."""
from __future__ import annotations

from pathlib import Path
import json

from meta_data_extraction.finetune.ablation import read_ablation_config
from meta_data_extraction.finetune.ablation_evaluation import (
    compare, endpoint_adapter, plan, read_ablation_evaluation_config,
)
from meta_data_extraction.finetune.validation import write_json


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return read_ablation_evaluation_config(ROOT / "configs/finetune_ablation_evaluation.yaml")


def test_plan_freezes_twelve_fixed_endpoints_outside_step7_output():
    config = configuration()
    rows = plan(ROOT, config)
    assert len(rows) == 12
    assert len({row["label"] for row in rows}) == 12
    assert {row["condition"] for row in rows} == set(config.conditions)
    assert {row["seed"] for row in rows} == {42, 123, 456}
    assert all(row["output"].startswith("outputs/finetune/ablation-development-evaluation/")
               for row in rows)
    assert all(row["adapter_sha256"] and row["checkpoint"].endswith(
        f"checkpoint-{row['endpoint_step']}") for row in rows)


def test_endpoint_evaluation_does_not_substitute_selected_base_checkpoint():
    matrix = read_ablation_config(ROOT / configuration().ablation_config)
    training, checkpoint, identity = endpoint_adapter(ROOT, matrix, "d_only", 123)
    # Two-paper selection chose step zero for this run, but development evaluation
    # must use the predeclared fixed-exposure step-six endpoint.
    assert identity["endpoint_step"] == 6
    assert checkpoint.name == "checkpoint-6"
    assert training.max_steps == 6 and training.seed == 123
    assert identity["planned_exposure"] == {"D": 24}


def test_comparison_requires_and_aggregates_all_twelve_runs(tmp_path):
    output = tmp_path / "development"
    config = configuration().model_copy(update={"output_dir": str(output)})
    paper_ids = [f"paper_{index}" for index in range(30)]
    summary = {
        "json_valid_rate": 1.0, "schema_valid_rate": .9,
        "facilities_f1": .7, "samples_f1": .8, "datasets_f1": .6,
        "sample_refs_f1": .75, "facility_refs_f1": .65,
        "generic_metadata_macro_score": .9, "mean_latency_ms": 1000.0,
        "mean_generated_tokens": 100.0, "papers_json_valid": 30,
        "papers_schema_valid": 27, "generation_failures": 0,
        "output_limit_failures": 0,
    }
    for condition in config.conditions:
        for seed in config.seeds:
            directory = output / f"{condition}_seed{seed}"
            write_json(directory / "metrics.json", {
                "summary": summary, "papers": [{"paper_id": paper_id} for paper_id in paper_ids],
            })
            write_json(directory / "manifest.json", {
                "model": {"adapter_sha256": f"fixture-{condition}-{seed}"},
            })
            rows = [{"paper_id": paper_id, "status": "generated", "schema_errors": []}
                    for paper_id in paper_ids]
            (directory / "generation.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = compare(ROOT, config)
    assert len(result["fixed_endpoint_conditions"]) == 4
    assert result["fixed_endpoint_conditions"]["matched_d_dr"]["aggregate"][
        "papers_reference_complete"]["mean"] == 30
    assert (output / "comparison.json").is_file()
    assert (output / "comparison.md").is_file()
