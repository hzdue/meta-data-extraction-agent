"""Lightweight mechanistic diagnostics over frozen fine-tuning outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from meta_data_extraction.finetune.mechanistic_diagnostics import (
    _summarize_decomposition, paired_oracle_bootstrap, read_config,
    reference_error_decomposition, relabel_oracle_predictions, verify_frozen_configs,
)
from meta_data_extraction.finetune.validation import write_json


ROOT = Path(__file__).resolve().parents[1]


def test_mechanistic_configuration_freezes_equal_update_primary_contrast():
    config = read_config(ROOT / "configs/finetune_mechanistic_diagnostics.yaml")
    verify_frozen_configs(ROOT, config)
    assert config.error_decomposition.conditions == ["d_only", "matched_d_dr", "d32"]
    assert config.oracle_comparison.reused_condition == "matched_d_dr"
    assert config.oracle_comparison.generated_conditions == ["d32", "d_only"]
    assert config.oracle_comparison.primary_contrast == ["matched_d_dr", "d32"]
    assert config.oracle_comparison.model_selection_allowed is False


def test_reference_error_decomposition_partitions_inventory_and_linking_failures():
    gold = {
        "samples": [{"id": value} for value in ("s1", "s2", "s3", "s4", "s5")],
        "datasets": [
            {"id": "g1", "sample_refs": ["s1", "s2", "s5"]},
            {"id": "g2", "sample_refs": ["s3"]},
            {"id": "g3", "sample_refs": ["s4"]},
        ],
    }
    prediction = {"output": {
        "samples": [{"id": value} for value in ("ps1", "ps3", "ps5")],
        "datasets": [
            {"id": "pd1", "sample_refs": ["ps1", "unknown", "ps3"]},
            {"id": "unmatched", "sample_refs": ["ps1"]},
        ],
    }}
    report = {"fields": {
        "datasets": {"match_stages": [{"pred": "pd1", "gold": "g1"}]},
        "samples": {"match_stages": [
            {"pred": "ps1", "gold": "s1"},
            {"pred": "ps3", "gold": "s3"},
            {"pred": "ps5", "gold": "s5"},
        ]},
    }}
    result = reference_error_decomposition(gold, prediction, report, "sample_refs")
    assert result == {
        "gold_links": 5, "predicted_unique_links": 4,
        "true_positive_links": 1, "false_positive_links": 3,
        "false_negative_links": 4,
        "fp_unaligned_dataset": 1, "fp_unaligned_target": 1, "fp_wrong_link": 1,
        "fn_dataset_and_target_missing": 1, "fn_dataset_missing_only": 1,
        "fn_target_missing_only": 1, "fn_link_error_with_entities_present": 1,
        "fn_inventory_limited": 3, "recoverable_gold_links": 2,
    }
    summary = _summarize_decomposition([result])
    assert summary["observed_recall"] == pytest.approx(.2)
    assert summary["inventory_recall_ceiling"] == pytest.approx(.4)
    assert summary["inventory_limited_fraction_of_false_negatives"] == pytest.approx(.75)
    assert summary["link_recall_when_entities_available"] == pytest.approx(.5)


def _oracle_metrics(value):
    return {"papers": [
        {"paper_id": "one", "end_to_end_reference": {
            "sample_refs": {"tp": value, "fp": 2 - value, "fn": 2 - value},
            "facility_refs": {"tp": value, "fp": 2 - value, "fn": 2 - value},
        }},
        {"paper_id": "two", "end_to_end_reference": {
            "sample_refs": {"tp": value, "fp": 2 - value, "fn": 2 - value},
            "facility_refs": {"tp": value, "fp": 2 - value, "fn": 2 - value},
        }},
    ]}


def test_oracle_bootstrap_pairs_papers_and_retains_seeds():
    metrics = {
        "matched_d_dr": {seed: _oracle_metrics(2) for seed in (42, 123, 456)},
        "d32": {seed: _oracle_metrics(1) for seed in (42, 123, 456)},
    }
    result = paired_oracle_bootstrap(
        metrics, ["matched_d_dr", "d32"], "sample_refs", samples=100, seed=42)
    assert result["mean_seed_delta_f1"] == pytest.approx(.5)
    assert result["paired_paper_cluster_bootstrap_95_interval"] == pytest.approx([.5, .5])
    assert result["probability_delta_above_zero"] == 1.0


def test_oracle_relabel_updates_prediction_and_committed_checksum(tmp_path):
    output = tmp_path / "oracle"
    prediction = output / "reconstructed_predictions" / "paper.json"
    write_json(prediction, {"provenance": {"config_name": "oracle_seed42"}, "output": {}})
    generation = {"paper_id": "paper", "reconstructed_file":
                  "reconstructed_predictions/paper.json", "reconstructed_sha256": "old"}
    (output / "generation.jsonl").write_text(json.dumps(generation) + "\n", encoding="utf-8")
    relabel_oracle_predictions(output, "d32", 42)
    assert json.loads(prediction.read_text(encoding="utf-8"))["provenance"]["config_name"] == \
        "oracle_d32_seed42"
    committed = json.loads((output / "generation.jsonl").read_text(encoding="utf-8"))
    assert committed["reconstructed_sha256"] != "old"
