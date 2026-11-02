"""Controlled ablation matrix, schedule, reference patch, and selection checks."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")

from meta_data_extraction.finetune.ablation import (
    exposure_schedule, plan_matrix, read_ablation_config, reference_patch_record,
)
from meta_data_extraction.finetune.validation import (
    ValidationCallback, best_schema_entry, diagnostic_quality, paper_mean,
)


ROOT = Path(__file__).resolve().parents[1]


def matrix():
    return read_ablation_config(ROOT / "configs/finetune_ablation.yaml")


def synthetic_records():
    return [{"example_id": f"paper_{index}:{task}", "task": task,
             "input_ids": [index], "labels": [index]}
            for task in ("D", "E", "R", "P") for index in range(8)]


def test_frozen_matrix_has_matched_d_compute_and_reference_redesign_controls():
    plan = plan_matrix(matrix())
    assert plan["d_only"]["exposures"] == {"D": 24}
    assert plan["matched_d_der"]["exposures"] == {"D": 24, "E": 8, "R": 8}
    assert plan["matched_compute_der"]["exposures"] == {"D": 8, "E": 8, "R": 8}
    assert plan["matched_d_drp"]["exposures"] == {"D": 24, "P": 8}
    assert all(item["seeds"] == [42, 123, 456] for item in plan.values())


@pytest.mark.parametrize("condition", [
    "d_only", "matched_d_de", "matched_d_dr", "matched_d_der",
    "matched_compute_de", "matched_compute_dr", "matched_compute_der", "matched_d_drp",
])
def test_schedules_have_exact_deterministic_exposure_and_complete_windows(condition):
    expected = matrix().conditions[condition].exposures
    first = exposure_schedule(synthetic_records(), expected, 42)
    second = exposure_schedule(synthetic_records(), expected, 42)
    assert first == second
    assert Counter(row["task"] for row in first) == Counter(expected)
    assert len(first) % 4 == 0
    assert set(row["example_id"].split(":")[0] for row in first) == {f"paper_{i}" for i in range(8)}


def test_reference_patch_uses_final_dataset_nesting_and_exact_frozen_answers():
    records = [json.loads(line) for line in
               (ROOT / "data/finetune/prepared/train_structure_aware.jsonl").read_text(encoding="utf-8").splitlines()]
    source = next(row for row in records if row["task"] == "R")
    patch = reference_patch_record(source)
    references = json.loads(source["completion"][0]["content"])["references"]
    target = json.loads(patch["completion"][0]["content"])
    assert patch["task"] == "P" and patch["example_id"].endswith(":P")
    assert set(target) == {"datasets"}
    assert all(list(row) == ["id", "sample_refs", "facility_refs"] for row in target["datasets"])
    flattened = [(row["id"], field, row[field]) for row in target["datasets"]
                 for field in ("sample_refs", "facility_refs")]
    assert flattened == [(row["dataset_id"], row["field"], row["target_ids"]) for row in references]
    prompt = patch["prompt"][0]["content"]
    assert '"datasets": [{"id": "dataset_1", "sample_refs": []' in prompt
    assert "## Entity inventory (reference fields removed)" in prompt


def diagnostic(paper_id, *, schema_valid, sample_f1, facility_f1):
    return {"paper_id": paper_id, "status": "scored", "schema_valid": schema_valid,
            "conditional_reference_f1": {
                "sample_refs": {"f1": sample_f1, "n_pairs": 1},
                "facility_refs": {"f1": facility_f1, "n_pairs": 1},
            }}


def entry(step, loss, diagnostics):
    papers = [{"paper_id": "v1", "loss": loss, "supervised_tokens": 2},
              {"paper_id": "v2", "loss": loss, "supervised_tokens": 3}]
    return {"step": step, "loss": paper_mean(papers), "papers": papers,
            "diagnostics": diagnostics, "exposure": {}}


def test_schema_selection_prioritizes_completeness_then_references_then_loss():
    invalid_low_loss = entry(0, 0.01, [diagnostic("v1", schema_valid=False, sample_f1=1, facility_f1=1),
                                      diagnostic("v2", schema_valid=True, sample_f1=1, facility_f1=1)])
    valid_lower_refs = entry(6, 0.5, [diagnostic("v1", schema_valid=True, sample_f1=.4, facility_f1=.4),
                                     diagnostic("v2", schema_valid=True, sample_f1=.4, facility_f1=.4)])
    valid_higher_refs = entry(8, 0.9, [diagnostic("v1", schema_valid=True, sample_f1=.8, facility_f1=.8),
                                      diagnostic("v2", schema_valid=True, sample_f1=.8, facility_f1=.8)])
    assert diagnostic_quality(valid_higher_refs) == {"schema_valid_papers": 2, "mean_reference_f1": .8}
    assert best_schema_entry([invalid_low_loss, valid_lower_refs, valid_higher_refs])["step"] == 8


def test_schema_selection_excludes_intermediate_recovery_checkpoint():
    callback = object.__new__(ValidationCallback)
    callback.selection_policy = "schema_reference_loss"
    callback.config = type("Config", (), {"max_steps": 8})()
    callback.history = [
        entry(0, .7, [diagnostic("v1", schema_valid=True, sample_f1=.5, facility_f1=.5),
                      diagnostic("v2", schema_valid=True, sample_f1=.5, facility_f1=.5)]),
        entry(6, .4, [diagnostic("v1", schema_valid=True, sample_f1=1, facility_f1=1),
                      diagnostic("v2", schema_valid=True, sample_f1=1, facility_f1=1)]),
        entry(8, .6, [diagnostic("v1", schema_valid=True, sample_f1=.7, facility_f1=.7),
                      diagnostic("v2", schema_valid=True, sample_f1=.7, facility_f1=.7)]),
    ]
    assert callback.winner()["step"] == 8

