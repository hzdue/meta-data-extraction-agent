"""Frozen Phase C evaluation and Gate C checks."""
from pathlib import Path

from meta_data_extraction.finetune.followup_phase_c_evaluation import (
    apply_gate_c, plan, read_phase_c_evaluation_config, verify_inputs,
)
from meta_data_extraction.finetune.followup_training import read_phase_c_config


ROOT = Path(__file__).resolve().parents[1]


def configs():
    evaluation = read_phase_c_evaluation_config(
        ROOT / "configs/finetune_followup_c_evaluation.yaml")
    verify_inputs(ROOT, evaluation)
    return evaluation, read_phase_c_config(ROOT / evaluation.training_config)


def contrast(mean, *, positive=3, negative=0, improved=8, worsened=2, share=.1):
    return {"pooled_f1_delta_across_seeds": {"mean": mean},
            "positive_seeds": positive, "negative_seeds": negative,
            "paper_direction": {"improved": improved, "tied": 20,
                                "worsened": worsened, "not_comparable": 0},
            "largest_absolute_paper_share": share}


def set_fields(default=0):
    return {field: contrast(default) for field in
            ("facilities", "samples", "datasets", "sample_refs", "facility_refs")}


def test_phase_c_evaluation_freezes_six_runs_and_reuses_two_controls():
    evaluation, phase = configs()
    rows = plan(evaluation, phase)
    assert len(rows) == 6
    assert {row["condition"] for row in rows} == {"r_then_d", "d_then_r"}
    assert {row["seed"] for row in rows} == {42, 123, 456}
    assert evaluation.controls == ["matched_d_dr", "d32"]
    assert evaluation.model_selection_allowed is False


def test_gate_c_shortlists_only_a_distributed_balanced_recovery():
    _, phase = configs()
    mixed, d32, order = set_fields(), set_fields(), set_fields()
    mixed["samples"] = contrast(.025)
    mixed["sample_refs"] = contrast(.03)
    result = apply_gate_c({"r_then_d_minus_matched_d_dr": mixed,
                           "r_then_d_minus_d32": d32,
                           "d_then_r_minus_r_then_d": order}, phase.gate_c)
    assert result["outcome"] == "shortlist_r_then_d"
    assert result["shortlist_r_then_d"] is True
    mixed["sample_refs"] = contrast(.03, share=.4)
    result = apply_gate_c({"r_then_d_minus_matched_d_dr": mixed,
                           "r_then_d_minus_d32": d32,
                           "d_then_r_minus_r_then_d": order}, phase.gate_c)
    assert result["shortlist_r_then_d"] is False


def test_gate_c_late_interference_requires_margin_seed_and_paper_support():
    _, phase = configs()
    mixed, d32, order = set_fields(), set_fields(), set_fields()
    order["sample_refs"] = contrast(-.04, positive=0, negative=3, improved=2, worsened=8)
    result = apply_gate_c({"r_then_d_minus_matched_d_dr": mixed,
                           "r_then_d_minus_d32": d32,
                           "d_then_r_minus_r_then_d": order}, phase.gate_c)
    assert result["late_interference"] is True
    assert result["outcome"] == "late_interference_only"
