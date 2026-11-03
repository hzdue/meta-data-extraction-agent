"""Phase B D32 evaluation and frozen Gate B checks."""
from __future__ import annotations

from pathlib import Path

from meta_data_extraction.finetune.followup_phase_b_evaluation import (
    apply_gate_b, d32_endpoint, plan, read_phase_b_evaluation_config,
)
from meta_data_extraction.finetune.followup_training import read_phase_b_config


ROOT = Path(__file__).resolve().parents[1]


def configs():
    evaluation = read_phase_b_evaluation_config(
        ROOT / "configs/finetune_followup_b_evaluation.yaml")
    phase_b = read_phase_b_config(ROOT / evaluation.training_config)
    return evaluation, phase_b


def contrast(mean, *, positive=3, improved=5, worsened=1, share=.1):
    return {
        "pooled_f1_delta_across_seeds": {"mean": mean},
        "positive_seeds": positive,
        "paper_direction": {"improved": improved, "tied": 20,
                            "worsened": worsened, "not_comparable": 4},
        "largest_absolute_paper_share": share,
    }


def test_phase_b_evaluation_freezes_three_d32_endpoints_and_reuses_controls():
    evaluation, phase_b = configs()
    rows = plan(ROOT, evaluation, phase_b)
    assert len(rows) == 3
    assert {row["seed"] for row in rows} == {42, 123, 456}
    assert all(row["planned_exposure"] == {"D": 32} for row in rows)
    assert all(row["checkpoint"].endswith("checkpoint-8") for row in rows)
    assert evaluation.controls == ["d_only", "matched_d_dr"]
    training, checkpoint, identity = d32_endpoint(ROOT, phase_b, evaluation, 42)
    assert training.max_steps == identity["endpoint_step"] == 8
    assert checkpoint.name == "checkpoint-8"


def test_gate_b_prefers_d32_only_when_facilities_are_equivalent_and_samples_recover():
    _, phase_b = configs()
    contrasts = {
        "facilities": contrast(.02), "facility_refs": contrast(.01),
        "samples": contrast(-.04, positive=0),
        "sample_refs": contrast(-.05, positive=0), "datasets": contrast(0),
    }
    result = apply_gate_b(contrasts, phase_b.gate_b)
    assert result["outcome"] == "prefer_d32"
    assert all(result["prefer_d32_checks"].values())


def test_gate_b_proceeds_only_for_distributed_r_specific_facility_advantage():
    _, phase_b = configs()
    contrasts = {
        "facilities": contrast(.05), "facility_refs": contrast(.04),
        "samples": contrast(-.01), "sample_refs": contrast(-.01),
        "datasets": contrast(0),
    }
    result = apply_gate_b(contrasts, phase_b.gate_b)
    assert result["outcome"] == "proceed_phase_c"
    assert all(result["proceed_phase_c_checks"].values())
    contrasts["facility_refs"] = contrast(.04, share=.4)
    assert apply_gate_b(contrasts, phase_b.gate_b)["outcome"] == "inconclusive"
