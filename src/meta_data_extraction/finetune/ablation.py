"""Controlled D/E/R exposure schedules and final-shaped reference supervision."""
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

from .data import canonical_json
from .parsing import text_hash
from .tokenization import encode_record
from .training_data import TrainingConfig, prepare_training, read_config


Task = Literal["D", "E", "R", "P"]


class AblationCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    family: Literal["shared_control", "matched_d", "matched_compute", "reference_redesign"]
    description: str = Field(min_length=1)
    exposures: dict[Task, int]

    @model_validator(mode="after")
    def valid_exposure(self):
        if not self.exposures or self.exposures.get("D", 0) <= 0:
            raise ValueError("Every ablation condition requires positive D exposure")
        if any(type(value) is not int or value <= 0 for value in self.exposures.values()):
            raise ValueError("Task exposures must be positive integers")
        if sum(self.exposures.values()) % 4:
            raise ValueError("Total exposure must fill complete four-example accumulation windows")
        return self


class AblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    base_training_config: str
    output_root: str
    seeds: list[int]
    selection_policy: Literal["schema_reference_loss"]
    independent_test_split: str | None
    allow_current_test_evaluation: Literal[False]
    conditions: dict[str, AblationCondition]

    @model_validator(mode="after")
    def complete_design(self):
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Use at least three distinct predeclared seeds")
        required = {"d_only", "matched_d_de", "matched_d_dr", "matched_d_der",
                    "matched_compute_de", "matched_compute_dr", "matched_compute_der",
                    "matched_d_drp"}
        if set(self.conditions) != required:
            raise ValueError("Ablation matrix is incomplete or has unreviewed conditions")
        return self


def read_ablation_config(path: Path) -> AblationConfig:
    return AblationConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def reference_patch_record(reference_record: dict) -> dict:
    """Convert R into a final-shaped dataset-reference patch without reading gold."""
    if reference_record["task"] != "R" or reference_record["split"] != "train":
        raise ValueError("Reference patches derive only from frozen training R records")
    references = json.loads(reference_record["completion"][0]["content"])["references"]
    datasets: list[dict] = []
    for index in range(0, len(references), 2):
        pair = references[index:index + 2]
        if (len(pair) != 2 or pair[0]["dataset_id"] != pair[1]["dataset_id"]
                or [row["field"] for row in pair] != ["sample_refs", "facility_refs"]):
            raise ValueError("Frozen R target no longer has two ordered fields per dataset")
        datasets.append({"id": pair[0]["dataset_id"],
                         "sample_refs": list(pair[0]["target_ids"]),
                         "facility_refs": list(pair[1]["target_ids"])})

    prompt = reference_record["prompt"][0]["content"]
    task_marker, schema_marker = "## Task\n", "\n\n## Full source schema"
    if task_marker not in prompt or schema_marker not in prompt:
        raise ValueError("Frozen R prompt structure changed")
    _, after_task = prompt.split(task_marker, 1)
    _, tail = after_task.split(schema_marker, 1)
    system_line = "You are an expert system for extracting metadata from scientific documents."
    task_text = (
        "Resolve dataset-to-sample and dataset-to-facility links using the document and supplied "
        "entity inventory. Return a datasets list with exactly one object per inventory dataset, "
        "in order. Every object must contain exactly id, sample_refs, and facility_refs. Preserve "
        "the final schema field names and nesting; use [] when no link is supported. Do not add "
        "entities, change IDs, or return non-reference dataset attributes."
    )
    rebuilt = task_marker + system_line + "\n\n" + task_text + schema_marker + tail
    shape_start = "## Output shape\n\n"
    inventory_start = "\n\n## Entity inventory (reference fields removed)"
    before_shape, shape_tail = rebuilt.split(shape_start, 1)
    _, inventory = shape_tail.split(inventory_start, 1)
    output_shape = ('{"datasets": [{"id": "dataset_1", "sample_refs": [], '
                    '"facility_refs": []}]}')
    rebuilt = before_shape + shape_start + output_shape + inventory_start + inventory
    result = deepcopy(reference_record)
    result.update(example_id=f"{reference_record['paper_id']}:P", task="P",
                  prompt=[{"role": "user", "content": rebuilt}],
                  completion=[{"role": "assistant",
                               "content": canonical_json({"datasets": datasets}),
                               "reasoning_content": ""}])
    result["provenance"] = {**result["provenance"],
                            "ablation_task": "reference_patch_v1",
                            "source_example_id": reference_record["example_id"]}
    result.pop("token_counts", None)
    return result


def tokenize_patch(tokenizer, record: dict, cap: int) -> dict:
    ids, start, end = encode_record(tokenizer, record)
    if len(ids) > cap:
        raise ValueError(f"Reference patch exceeds context cap: {record['example_id']}")
    labels = [-100] * len(ids)
    labels[start:end] = ids[start:end]
    return {"input_ids": ids, "labels": labels, "task": "P",
            "example_id": record["example_id"]}


def _task_seed(seed: int, task: str) -> int:
    digest = hashlib.sha256(f"{seed}:{task}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def exposure_schedule(records: list[dict], exposures: dict[str, int], seed: int) -> list[dict]:
    """Repeat full task cycles, then choose a deterministic balanced remainder."""
    by_task = {task: sorted((row for row in records if row["task"] == task),
                            key=lambda row: row["example_id"]) for task in exposures}
    schedule = []
    for task, count in exposures.items():
        source = by_task[task]
        if len(source) != 8 or len({row["example_id"] for row in source}) != 8:
            raise ValueError(f"Task {task} must contain one source record per training paper")
        complete, remainder = divmod(count, len(source))
        chosen = source * complete
        shuffled = list(source)
        random.Random(_task_seed(seed, task)).shuffle(shuffled)
        chosen += shuffled[:remainder]
        schedule.extend(deepcopy(chosen))
    random.Random(seed).shuffle(schedule)
    if Counter(row["task"] for row in schedule) != Counter(exposures):
        raise ValueError("Scheduled exposure differs from the condition contract")
    return schedule


def prepare_ablation(root: Path, matrix: AblationConfig, condition_name: str, seed: int,
                     *, offline=False):
    if condition_name not in matrix.conditions or seed not in matrix.seeds:
        raise ValueError("Condition or seed is not in the frozen ablation matrix")
    base = read_config(root / matrix.base_training_config)
    tokenizer, records, validation, base_summary = prepare_training(root, base, offline=offline)
    raw = [json.loads(line) for line in
           (root / base.prepared_dir / "train_structure_aware.jsonl").read_text(encoding="utf-8").splitlines()]
    patches = [tokenize_patch(tokenizer, reference_patch_record(row), base.max_seq_length)
               for row in raw if row["task"] == "R"]
    all_records = records + patches
    condition = matrix.conditions[condition_name]
    schedule = exposure_schedule(all_records, condition.exposures, seed)
    total = len(schedule)
    updates = total // base.gradient_accumulation_steps
    values = base.model_dump()
    values.update(method="structure_aware", seed=seed, max_steps=updates,
                  save_steps=min(6, updates), validation_steps=min(6, updates),
                  output_dir=f"{matrix.output_root}/{condition_name}_seed{seed}")
    training_config = TrainingConfig.model_validate(values)
    papers = Counter(row["example_id"].split(":")[0] for row in schedule)
    schedule_identity = [{"example_id": row["example_id"], "task": row["task"],
                          "tokens": len(row["input_ids"])} for row in schedule]
    task_counts = Counter(row["task"] for row in schedule)
    summary = {**base_summary,
        "method": "ablation",
        "training_examples": total,
        "training_papers": len(papers),
        "task_counts": dict(sorted(task_counts.items())),
        "longest_training_sequence": max(len(row["input_ids"]) for row in schedule),
        "supervised_tokens_per_pass": sum(
            sum(token != -100 for token in row["labels"]) for row in schedule),
        "total_tokens_per_pass": sum(len(row["input_ids"]) for row in schedule),
        "ablation": {
            "version": matrix.version, "condition": condition_name, "family": condition.family,
            "description": condition.description, "seed": seed,
            "planned_exposure": condition.exposures, "total_examples": total,
            "endpoint_updates": updates, "per_paper_exposure": dict(sorted(papers.items())),
            "schedule_sha256": text_hash(json.dumps(schedule_identity, sort_keys=True)),
            "selection_policy": matrix.selection_policy,
            "current_test_evaluation_allowed": False,
            "independent_test_split": matrix.independent_test_split,
        },
    }
    return training_config, tokenizer, schedule, validation, summary


def plan_matrix(matrix: AblationConfig) -> dict:
    return {name: {"family": condition.family, "exposures": condition.exposures,
                   "total_examples": sum(condition.exposures.values()),
                   "endpoint_updates": sum(condition.exposures.values()) // 4,
                   "seeds": matrix.seeds,
                   "output_pattern": f"{matrix.output_root}/{name}_seed<seed>"}
            for name, condition in matrix.conditions.items()}

