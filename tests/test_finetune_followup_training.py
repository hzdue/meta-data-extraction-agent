"""Frozen Step 9 Phase B configuration and D32 schedule checks."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from meta_data_extraction.finetune.followup_training import (
    d32_schedule, plan_phase_b, read_phase_b_config,
)


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
