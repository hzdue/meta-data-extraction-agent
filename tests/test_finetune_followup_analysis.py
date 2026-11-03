"""Checks for the frozen Step 9 Phase A1 paired analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from meta_data_extraction.evaluation import evaluate
from meta_data_extraction.finetune.data import project_entities, project_references, reconstruct
from meta_data_extraction.finetune.followup_analysis import (
    FIELD_SPECS, paired_cluster_bootstrap, prediction_error_profile, prf,
    read_config, read_phase_a2_config, report_field_counts, report_field_f1,
)
from meta_data_extraction.finetune.followup_evaluation import (
    empty_references, end_to_end_reference_counts, inventory_from_output,
    strict_json_object, validate_reference_output,
)


ROOT = Path(__file__).resolve().parents[1]


def _report(tp=1, fp=0, fn=0):
    return {"fields": {spec["report_key"]: {"tp": tp, "fp": fp, "fn": fn}
                       for spec in FIELD_SPECS.values()}}


def test_followup_configuration_freezes_a1_comparison():
    config = read_config(ROOT / "configs/finetune_followup.yaml")
    assert config.phase_a1.control == "d_only"
    assert config.phase_a1.treatment == "matched_d_dr"
    assert config.phase_a1.seeds == [42, 123, 456]
    assert config.phase_a1.expected_papers == 30
    assert config.phase_a1.bootstrap_samples == 10_000
    a2 = read_phase_a2_config(ROOT / "configs/finetune_followup_a2.yaml").phase_a2
    assert a2.adapter_condition == "matched_d_dr"
    assert a2.inventories == ["oracle", "self_predicted", "d24_predicted"]
    assert a2.max_new_tokens == 4096


def test_prf_uses_evaluator_counts():
    assert prf({"tp": 3, "fp": 1, "fn": 2}) == {
        "tp": 3, "fp": 1, "fn": 2,
        "precision": .75, "recall": .6, "f1": pytest.approx(2 / 3),
    }


def test_reference_field_without_aligned_dataset_pair_is_undefined_not_perfect():
    report = _report()
    report["fields"][FIELD_SPECS["sample_refs"]["report_key"]] = {"n_pairs": 0}
    assert report_field_counts(report, "sample_refs") == {"tp": 0, "fp": 0, "fn": 0}
    assert report_field_f1(report, "sample_refs") is None


def test_prediction_profile_separates_structural_and_evaluator_errors():
    prediction = {"output": {
        "facilities": [{"id": "f1", "name": "Source"},
                       {"id": "f1", "name": " source "}],
        "samples": [{"id": "s1", "sample_name": "Sample"}],
        "datasets": [{
            "id": "d1", "dataset_name": "Data", "measurement_technique": "SANS",
            "sample_refs": ["s1", "missing", "s1"], "facility_refs": ["f1"],
        }],
    }}
    profile = prediction_error_profile(prediction, _report(tp=1, fp=2, fn=3))
    assert profile == {
        "missing_entities": 9,
        "unmatched_predictions": 6,
        "duplicate_id_excess": 1,
        "duplicate_identity_excess": 1,
        "missing_references": 6,
        "extra_or_mislinked_references": 4,
        "dangling_references": 1,
        "repeated_reference_excess": 1,
    }


def test_paired_bootstrap_resamples_papers_and_keeps_all_seeds():
    paper_ids = ["one", "two"]
    runs = {condition: {} for condition in ("d_only", "matched_d_dr")}
    for condition in runs:
        for seed in (42, 123, 456):
            reports = {}
            for paper_id in paper_ids:
                reports[paper_id] = _report(
                    tp=1 if condition == "d_only" else 2,
                    fp=1 if condition == "d_only" else 0,
                    fn=1 if condition == "d_only" else 0,
                )
            runs[condition][seed] = {"reports": reports}
    result = paired_cluster_bootstrap(
        runs, paper_ids, "facilities", samples=100, seed=42)
    assert result["mean_seed_delta_f1"] == pytest.approx(.5)
    assert result["paired_bootstrap_95_ci"] == pytest.approx([.5, .5])
    assert result["probability_delta_above_zero"] == 1.0


def test_phase_a2_inventory_removes_answers_and_requires_unique_ids():
    output = {
        "facilities": [{"id": "f1", "name": "Facility"}],
        "samples": [{"id": "s1", "sample_name": "Sample"}],
        "datasets": [{"id": "d1", "dataset_name": "Data",
                      "sample_refs": ["s1"], "facility_refs": ["f1"]}],
    }
    inventory = inventory_from_output(output)
    assert inventory["datasets"] == [{"id": "d1", "dataset_name": "Data"}]
    assert validate_reference_output({"references": [
        {"dataset_id": "d1", "field": "sample_refs", "target_ids": ["s1"]},
        {"dataset_id": "d1", "field": "facility_refs", "target_ids": ["f1"]},
    ]}, inventory)["references"][0]["target_ids"] == ["s1"]
    assert empty_references(inventory)["references"][0]["target_ids"] == []
    output["samples"].append({"id": "s1", "sample_name": "Duplicate"})
    with pytest.raises(ValueError, match="unique"):
        inventory_from_output(output)


def test_phase_a2_strict_json_and_reference_contract_reject_malformed_outputs():
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        strict_json_object('{"references": [], "references": []}')
    inventory = {"facilities": [], "samples": [{"id": "s1"}],
                 "datasets": [{"id": "d1"}]}
    with pytest.raises(ValueError, match="cover each dataset"):
        validate_reference_output({"references": []}, inventory)
    with pytest.raises(ValueError, match="Duplicate, invalid, or dangling"):
        validate_reference_output({"references": [
            {"dataset_id": "d1", "field": "sample_refs", "target_ids": ["missing"]},
            {"dataset_id": "d1", "field": "facility_refs", "target_ids": []},
        ]}, inventory)


def test_end_to_end_reference_score_penalizes_unaligned_entities():
    gold = {"datasets": [{"id": "gd", "sample_refs": ["gs"], "facility_refs": []}]}
    reconstructed = {"datasets": [{"id": "pd", "sample_refs": ["ps", "extra"],
                                    "facility_refs": []}]}
    report = {"fields": {
        "datasets": {"match_stages": [{"pred": "pd", "gold": "gd"}]},
        "samples": {"match_stages": [{"pred": "ps", "gold": "gs"}]},
        "facilities": {"match_stages": []},
    }}
    assert end_to_end_reference_counts(gold, reconstructed, report, "sample_refs") == {
        "tp": 1, "fp": 1, "fn": 0,
    }


def test_phase_a2_oracle_reconstruction_is_perfect_under_shared_evaluator(tmp_path):
    paper_id = "1901.08819"
    gold_path = ROOT / "data/gold" / f"{paper_id}.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    reconstructed = reconstruct(project_entities(gold), project_references(gold))
    prediction_path = tmp_path / "prediction.json"
    prediction_path.write_text(json.dumps({
        "document_id": paper_id, "output": reconstructed,
    }), encoding="utf-8")
    report = evaluate(gold_path, prediction_path, ROOT / "configs/evaluation_metrics.yaml")
    for field in ("facilities", "samples", "datasets"):
        assert report["fields"][field]["f1"] == 1.0
    for field in ("sample_refs", "facility_refs"):
        assert report["fields"][f"datasets.attrs.{field}"]["f1"] == 1.0
        counts = end_to_end_reference_counts(gold, reconstructed, report, field)
        assert counts["fp"] == 0 and counts["fn"] == 0
