"""Frozen Step 9 Phase B configuration and D32 schedule checks."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from meta_data_extraction.finetune.followup_training import (
    d32_schedule, phase_c_schedule, plan_phase_b, plan_phase_c,
    read_phase_b_config, read_phase_c_config,
)
from meta_data_extraction.finetune.training import SaveAtSteps


ROOT = Path(__file__).resolve().parents[1]


def config():
    return read_phase_b_config(ROOT / "configs/finetune_followup_b.yaml")


def records():
    return [
        {"example_id": f"paper_{index}:{task}", "task": task,
         "input_ids": [index], "labels": [index]}
        for task in ("D", "E", "R") for index in range(8)
    ]


def test_phase_b_freezes_d32_three_seeds_and_numeric_gate():
    phase = config()
    plan = plan_phase_b(phase)
    assert plan["planned_exposure"] == {"D": 32}
    assert plan["endpoint_updates"] == 8
    assert plan["seeds"] == [42, 123, 456]
    assert phase.condition.gradient_accumulation_steps == 4
    assert phase.gate_b.equivalence_margin_f1 == .03
    assert phase.gate_b.r_specific_facilities_margin_f1 == .04
    assert phase.gate_b.r_specific_facility_refs_margin_f1 == .03
    assert phase.gate_b.minimum_positive_seeds == 2
    assert phase.gate_b.maximum_largest_paper_absolute_share == .25


def test_d32_schedule_is_deterministic_and_covers_every_paper_once_per_cycle():
    first = d32_schedule(records(), 42)
    second = d32_schedule(records(), 42)
    other_seed = d32_schedule(records(), 123)
    assert first == second and first != other_seed
    assert Counter(row["task"] for row in first) == {"D": 32}
    assert Counter(row["example_id"] for row in first) == {
        f"paper_{index}:D": 4 for index in range(8)
    }
    for start in range(0, 32, 8):
        assert {row["example_id"] for row in first[start:start + 8]} == {
            f"paper_{index}:D" for index in range(8)
        }


def test_phase_c_freezes_two_orders_and_numeric_gate_after_gradient_prerequisite():
    phase = read_phase_c_config(ROOT / "configs/finetune_followup_c.yaml")
    plan = plan_phase_c(phase)
    assert list(plan["conditions"]) == ["r_then_d", "d_then_r"]
    assert all(value["planned_exposure"] == {"D": 24, "R": 8}
               for value in plan["conditions"].values())
    assert plan["conditions"]["r_then_d"]["boundary_update"] == 2
    assert plan["conditions"]["d_then_r"]["boundary_update"] == 6
    assert all(condition.train_sampling_strategy == "sequential"
               for condition in phase.conditions.values())
    assert phase.gate_c.sample_recovery_margin_f1 == .02
    assert phase.gate_c.equivalence_margin_f1 == .03
    assert phase.gate_c.late_interference_margin_f1 == .03
    assert phase.gate_c.maximum_largest_paper_absolute_share == .25


def test_phase_c_schedules_change_only_order_and_keep_task_pure_windows():
    phase = read_phase_c_config(ROOT / "configs/finetune_followup_c.yaml")
    first = phase_c_schedule(records(), phase.conditions["r_then_d"], 42)
    second = phase_c_schedule(records(), phase.conditions["r_then_d"], 42)
    reverse = phase_c_schedule(records(), phase.conditions["d_then_r"], 42)
    assert first == second and first != reverse
    assert Counter(row["task"] for row in first) == {"D": 24, "R": 8}
    assert [row["task"] for row in first[:8]] == ["R"] * 8
    assert [row["task"] for row in reverse[:24]] == ["D"] * 24
    for schedule in (first, reverse):
        assert all(len({row["task"] for row in schedule[start:start + 4]}) == 1
                   for start in range(0, 32, 4))
        assert Counter(row["example_id"].split(":")[0] for row in schedule) == {
            f"paper_{index}": 4 for index in range(8)
        }


def test_phase_c_boundary_checkpoint_fails_closed_on_sampler_drift():
    trainer = type("Trainer", (), {"exposure": {"examples": {"R": 2, "D": 6}}})()
    callback = SaveAtSteps((2,), trainer=trainer, expected_exposures={2: {"R": 8}})
    state, control = type("State", (), {"global_step": 2})(), type("Control", (), {})()
    with pytest.raises(ValueError, match="differs from the task boundary"):
        callback.on_step_end(None, state, control)
    trainer.exposure["examples"] = {"R": 8}
    result = callback.on_step_end(None, state, control)
    assert result.should_save is True
