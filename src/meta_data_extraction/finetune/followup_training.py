"""Frozen Step 9 Phase B D32 schedule and training preparation."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from .parsing import text_hash
from .training_data import TrainingConfig, file_hash, prepare_training, read_config


class PhaseBCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: Literal["d32"]
    family: Literal["additional_training_control"]
    description: str = Field(min_length=1)
    exposures: dict[Literal["D"], int]
    cycle_size: Literal[8]
    cycles: Literal[4]
    gradient_accumulation_steps: Literal[4]
    endpoint_updates: Literal[8]
    checkpoint_interval: Literal[6]

    @model_validator(mode="after")
    def exact_d32(self):
        if self.exposures != {"D": 32}:
            raise ValueError("Phase B must contain exactly 32 complete-JSON examples")
        if self.cycle_size * self.cycles != 32:
            raise ValueError("Four complete eight-paper D cycles are required")
        if sum(self.exposures.values()) // self.gradient_accumulation_steps != self.endpoint_updates:
            raise ValueError("D32 exposure must produce exactly eight optimizer updates")
        return self


class ExistingControls(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ablation_config: str
    ablation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_dir: str
    shorter_control: Literal["d_only"]
    mixed_control: Literal["matched_d_dr"]
    endpoint_policy: Literal["fixed_exposure"]


class GateB(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_analysis: str
    source_analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison: Literal["matched_d_dr_minus_d32"]
    equivalence_margin_f1: float = Field(gt=0, lt=1)
    sample_recovery_margin_f1: float = Field(gt=0, lt=1)
    r_specific_facilities_margin_f1: float = Field(gt=0, lt=1)
    r_specific_facility_refs_margin_f1: float = Field(gt=0, lt=1)
    minimum_positive_seeds: Literal[2]
    minimum_net_improved_papers: int = Field(ge=1)
    maximum_largest_paper_absolute_share: float = Field(gt=0, lt=1)
    decision_rules: dict[Literal["prefer_d32", "proceed_phase_c", "otherwise"], str]

    @model_validator(mode="after")
    def frozen_thresholds(self):
        expected = {
            "equivalence_margin_f1": 0.03,
            "sample_recovery_margin_f1": 0.03,
            "r_specific_facilities_margin_f1": 0.04,
            "r_specific_facility_refs_margin_f1": 0.03,
            "maximum_largest_paper_absolute_share": 0.25,
        }
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise ValueError("Gate B numeric thresholds are frozen before D32")
        if set(self.decision_rules) != {"prefer_d32", "proceed_phase_c", "otherwise"}:
            raise ValueError("Gate B requires all three mutually exclusive outcomes")
        return self


class PhaseBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    phase: Literal["phase_b"]
    study_type: Literal["exploratory"]
    base_training_config: str
    base_training_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_root: str
    seeds: list[int]
    selection_policy: Literal["schema_reference_loss"]
    independent_test_split: None
    condition: PhaseBCondition
    existing_controls: ExistingControls
    gate_b: GateB

    @model_validator(mode="after")
    def complete_design(self):
        if self.seeds != [42, 123, 456]:
            raise ValueError("Phase B seeds are frozen to 42, 123 and 456")
        return self


def read_phase_b_config(path: Path) -> PhaseBConfig:
    return PhaseBConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _verify_hash(root: Path, relative: str, expected: str) -> None:
    path = root / relative
    if not path.is_file() or file_hash(path) != expected:
        raise ValueError(f"Frozen Phase B input changed: {relative}")


def _cycle_seed(seed: int, cycle: int) -> int:
    digest = hashlib.sha256(f"phase-b:d32:{seed}:{cycle}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def d32_schedule(records: list[dict], seed: int) -> list[dict]:
    """Four deterministic cycles, each covering every D paper exactly once."""
    source = sorted((row for row in records if row["task"] == "D"),
                    key=lambda row: row["example_id"])
    if len(source) != 8 or len({row["example_id"] for row in source}) != 8:
        raise ValueError("D32 requires exactly one frozen D record for each of eight papers")
    schedule: list[dict] = []
    for cycle in range(4):
        ordered = list(source)
        random.Random(_cycle_seed(seed, cycle)).shuffle(ordered)
        schedule.extend(deepcopy(ordered))
    for start in range(0, 32, 8):
        cycle = schedule[start:start + 8]
        if len({row["example_id"] for row in cycle}) != 8:
            raise ValueError("Every D32 cycle must cover all eight papers once")
    return schedule


def plan_phase_b(config: PhaseBConfig) -> dict:
    return {
        "phase": config.phase,
        "study_type": config.study_type,
        "condition": config.condition.name,
        "planned_exposure": config.condition.exposures,
        "endpoint_updates": config.condition.endpoint_updates,
        "seeds": config.seeds,
        "output_pattern": f"{config.output_root}/d32_seed<seed>",
        "gate_b": config.gate_b.model_dump(),
        "independent_test_split": config.independent_test_split,
    }


def prepare_phase_b(root: Path, config_path: Path, config: PhaseBConfig, seed: int,
                    *, offline: bool = False):
    if seed not in config.seeds:
        raise ValueError("Seed is not in the frozen Phase B design")
    _verify_hash(root, config.base_training_config, config.base_training_config_sha256)
    _verify_hash(root, config.existing_controls.ablation_config,
                 config.existing_controls.ablation_config_sha256)
    _verify_hash(root, config.gate_b.source_analysis, config.gate_b.source_analysis_sha256)

    base = read_config(root / config.base_training_config)
    if base.gradient_accumulation_steps != config.condition.gradient_accumulation_steps:
        raise ValueError("Base gradient accumulation differs from the D32 contract")
    tokenizer, records, validation, base_summary = prepare_training(
        root, base, offline=offline)
    schedule = d32_schedule(records, seed)
    values = base.model_dump()
    values.update(
        method="structure_aware",
        seed=seed,
        max_steps=config.condition.endpoint_updates,
        save_steps=config.condition.checkpoint_interval,
        validation_steps=config.condition.checkpoint_interval,
        output_dir=f"{config.output_root}/d32_seed{seed}",
    )
    training_config = TrainingConfig.model_validate(values)
    schedule_identity = [
        {"position": index, "example_id": row["example_id"], "task": row["task"],
         "tokens": len(row["input_ids"])}
        for index, row in enumerate(schedule)
    ]
    papers = Counter(row["example_id"].split(":")[0] for row in schedule)
    followup = {
        "version": config.version,
        "phase": "followup_phase_b",
        "study_type": config.study_type,
        "condition": config.condition.name,
        "family": config.condition.family,
        "description": config.condition.description,
        "seed": seed,
        "planned_exposure": config.condition.exposures,
        "total_examples": len(schedule),
        "endpoint_updates": config.condition.endpoint_updates,
        "per_paper_exposure": dict(sorted(papers.items())),
        "cycle_size": config.condition.cycle_size,
        "cycles": config.condition.cycles,
        "schedule_sha256": text_hash(json.dumps(schedule_identity, sort_keys=True)),
        "selection_policy": config.selection_policy,
        "endpoint_policy": config.existing_controls.endpoint_policy,
        "independent_test_split": config.independent_test_split,
        "gate_b_sha256": text_hash(json.dumps(config.gate_b.model_dump(), sort_keys=True)),
        "phase_config": str(config_path.relative_to(root)),
        "phase_config_sha256": file_hash(config_path),
    }
    summary = {
        **base_summary,
        "method": "followup_phase_b",
        "training_examples": len(schedule),
        "training_papers": len(papers),
        "task_counts": {"D": len(schedule)},
        "longest_training_sequence": max(len(row["input_ids"]) for row in schedule),
        "supervised_tokens_per_pass": sum(
            sum(token != -100 for token in row["labels"]) for row in schedule),
        "total_tokens_per_pass": sum(len(row["input_ids"]) for row in schedule),
        "followup": followup,
    }
    return training_config, tokenizer, schedule, validation, summary


class PhaseCStage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: Literal["D", "R"]
    examples: Literal[8, 24]


class PhaseCCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: Literal["r_then_d", "d_then_r"]
    family: Literal["training_order_control"]
    description: str = Field(min_length=1)
    stages: list[PhaseCStage]
    exposures: dict[Literal["D", "R"], int]
    gradient_accumulation_steps: Literal[4]
    boundary_update: Literal[2, 6]
    endpoint_updates: Literal[8]
    checkpoint_interval: Literal[6]
    train_sampling_strategy: Literal["sequential"]

    @model_validator(mode="after")
    def exact_order_control(self):
        expected = ([PhaseCStage(task="R", examples=8), PhaseCStage(task="D", examples=24)]
                    if self.name == "r_then_d" else
                    [PhaseCStage(task="D", examples=24), PhaseCStage(task="R", examples=8)])
        boundary = 2 if self.name == "r_then_d" else 6
        if self.stages != expected or self.exposures != {"D": 24, "R": 8}:
            raise ValueError("Phase C conditions must differ only in R8/D24 stage order")
        if self.boundary_update != boundary:
            raise ValueError("Phase C task boundary is not at the declared optimizer update")
        return self


class GateC(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    comparison: Literal["r_then_d_vs_mixed_and_d32"]
    sample_recovery_margin_f1: float
    equivalence_margin_f1: float
    late_interference_margin_f1: float
    minimum_positive_seeds: Literal[2]
    minimum_net_improved_papers: Literal[1]
    maximum_largest_paper_absolute_share: float
    decision_rules: dict[Literal["shortlist_r_then_d", "late_interference", "otherwise"], str]

    @model_validator(mode="after")
    def frozen_thresholds(self):
        expected = (self.sample_recovery_margin_f1, self.equivalence_margin_f1,
                    self.late_interference_margin_f1,
                    self.maximum_largest_paper_absolute_share)
        if expected != (0.02, 0.03, 0.03, 0.25):
            raise ValueError("Gate C numeric thresholds are frozen before order training")
        if set(self.decision_rules) != {"shortlist_r_then_d", "late_interference", "otherwise"}:
            raise ValueError("Gate C requires all declared interpretation outcomes")
        return self


class PhaseCPrerequisites(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ablation_config: str
    ablation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_b_evaluation_config: str
    phase_b_evaluation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gradient_result: str
    gradient_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gradient_status: Literal["complete"]
    user_authorization_date: Literal["2026-11-09"]


class PhaseCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    phase: Literal["phase_c"]
    study_type: Literal["post_hoc_exploratory"]
    base_training_config: str
    base_training_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_root: str
    seeds: list[int]
    selection_policy: Literal["schema_reference_loss"]
    independent_test_split: None
    conditions: dict[str, PhaseCCondition]
    prerequisites: PhaseCPrerequisites
    gate_c: GateC

    @model_validator(mode="after")
    def complete_design(self):
        if self.seeds != [42, 123, 456] or list(self.conditions) != ["r_then_d", "d_then_r"]:
            raise ValueError("Phase C seeds or condition order changed")
        if any(name != condition.name for name, condition in self.conditions.items()):
            raise ValueError("Phase C condition keys and names differ")
        return self


def read_phase_c_config(path: Path) -> PhaseCConfig:
    return PhaseCConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _stage_cycle_seed(seed: int, condition: str, task: str, cycle: int) -> int:
    digest = hashlib.sha256(f"phase-c:{condition}:{seed}:{task}:{cycle}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def phase_c_schedule(records: list[dict], condition: PhaseCCondition, seed: int) -> list[dict]:
    """Build exact task-pure accumulation windows with deterministic paper cycles."""
    by_task = {task: sorted((row for row in records if row["task"] == task),
                            key=lambda row: row["example_id"]) for task in ("D", "R")}
    if any(len(rows) != 8 or len({row["example_id"] for row in rows}) != 8
           for rows in by_task.values()):
        raise ValueError("Phase C requires one frozen D and R record per training paper")
    schedule = []
    for stage in condition.stages:
        cycles = stage.examples // 8
        for cycle in range(cycles):
            ordered = list(by_task[stage.task])
            random.Random(_stage_cycle_seed(seed, condition.name, stage.task, cycle)).shuffle(ordered)
            schedule.extend(deepcopy(ordered))
    if Counter(row["task"] for row in schedule) != Counter(condition.exposures):
        raise ValueError("Phase C schedule exposure differs from its frozen contract")
    windows = [schedule[start:start + 4] for start in range(0, len(schedule), 4)]
    if any(len(window) != 4 or len({row["task"] for row in window}) != 1 for window in windows):
        raise ValueError("A Phase C accumulation window crosses the task boundary")
    boundary_position = sum(stage.examples for stage in condition.stages[:1])
    if boundary_position // 4 != condition.boundary_update:
        raise ValueError("Phase C example and optimizer boundaries differ")
    return schedule


def _verify_phase_c_inputs(root: Path, config: PhaseCConfig):
    _verify_hash(root, config.base_training_config, config.base_training_config_sha256)
    for relative, expected in (
        (config.prerequisites.ablation_config, config.prerequisites.ablation_config_sha256),
        (config.prerequisites.phase_b_evaluation_config,
         config.prerequisites.phase_b_evaluation_config_sha256),
        (config.prerequisites.gradient_result, config.prerequisites.gradient_result_sha256),
    ):
        _verify_hash(root, relative, expected)
    gradient = json.loads((root / config.prerequisites.gradient_result).read_text(encoding="utf-8"))
    if gradient.get("status") != "complete" or gradient.get("safety", {}).get("optimizer_steps") != 0:
        raise ValueError("The no-update Phase C prerequisite is incomplete or unsafe")


def plan_phase_c(config: PhaseCConfig) -> dict:
    return {
        "phase": config.phase, "study_type": config.study_type,
        "conditions": {name: {"stages": [stage.model_dump() for stage in condition.stages],
                              "planned_exposure": condition.exposures,
                              "boundary_update": condition.boundary_update,
                              "endpoint_updates": condition.endpoint_updates,
                              "train_sampling_strategy": condition.train_sampling_strategy,
                              "seeds": config.seeds,
                              "output_pattern": f"{config.output_root}/{name}_seed<seed>"}
                       for name, condition in config.conditions.items()},
        "gate_c": config.gate_c.model_dump(),
        "independent_test_split": config.independent_test_split,
    }


def prepare_phase_c(root: Path, config_path: Path, config: PhaseCConfig,
                    condition_name: str, seed: int, *, offline: bool = False):
    if seed not in config.seeds or condition_name not in config.conditions:
        raise ValueError("Condition or seed is outside the frozen Phase C design")
    _verify_phase_c_inputs(root, config)
    base = read_config(root / config.base_training_config)
    condition = config.conditions[condition_name]
    if base.gradient_accumulation_steps != condition.gradient_accumulation_steps:
        raise ValueError("Base gradient accumulation differs from Phase C")
    tokenizer, records, validation, base_summary = prepare_training(root, base, offline=offline)
    schedule = phase_c_schedule(records, condition, seed)
    values = base.model_dump()
    values.update(method="structure_aware", seed=seed, max_steps=condition.endpoint_updates,
                  save_steps=condition.checkpoint_interval,
                  validation_steps=condition.checkpoint_interval,
                  output_dir=f"{config.output_root}/{condition_name}_seed{seed}")
    training_config = TrainingConfig.model_validate(values)
    identity = [{"position": index, "example_id": row["example_id"], "task": row["task"],
                 "tokens": len(row["input_ids"])} for index, row in enumerate(schedule)]
    papers = Counter(row["example_id"].split(":")[0] for row in schedule)
    followup = {
        "version": config.version, "phase": "followup_phase_c",
        "study_type": config.study_type, "condition": condition_name,
        "family": condition.family, "description": condition.description, "seed": seed,
        "stages": [stage.model_dump() for stage in condition.stages],
        "planned_exposure": condition.exposures, "total_examples": len(schedule),
        "boundary_update": condition.boundary_update,
        "endpoint_updates": condition.endpoint_updates,
        "per_paper_exposure": dict(sorted(papers.items())),
        "schedule_sha256": text_hash(json.dumps(identity, sort_keys=True)),
        "selection_policy": config.selection_policy, "endpoint_policy": "fixed_exposure",
        "independent_test_split": config.independent_test_split,
        "gate_c_sha256": text_hash(json.dumps(config.gate_c.model_dump(), sort_keys=True)),
        "gradient_result_sha256": config.prerequisites.gradient_result_sha256,
        "train_sampling_strategy": condition.train_sampling_strategy,
        "phase_config": str(config_path.relative_to(root)),
        "phase_config_sha256": file_hash(config_path),
    }
    summary = {**base_summary, "method": "followup_phase_c",
        "training_examples": len(schedule), "training_papers": len(papers),
        "task_counts": dict(sorted(Counter(row["task"] for row in schedule).items())),
        "longest_training_sequence": max(len(row["input_ids"]) for row in schedule),
        "supervised_tokens_per_pass": sum(sum(token != -100 for token in row["labels"])
                                          for row in schedule),
        "total_tokens_per_pass": sum(len(row["input_ids"]) for row in schedule),
        "followup": followup}
    return training_config, tokenizer, schedule, validation, summary
