"""Frozen Stage-A pseudo supervision and Stage-B GoldD24 calibration contracts."""
from __future__ import annotations

from collections import Counter
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator
import yaml

from .ablation import exposure_schedule
from .parsing import text_hash
from .pseudo_training_data import (
    DEFAULT_PREPARED_DIR,
    LENGTH_FILTER_MANIFEST,
    TRAINING_SEQUENCE_CUTOFF,
    apply_length_filter,
    file_hash,
    prepare_pseudo_training,
    read_json,
)
from .training_data import TrainingConfig, prepare_training, read_config


STAGE_A_EXAMPLES = 442
STAGE_A_ENDPOINT = 111
STAGE_A_CHECKPOINTS = [32, 64, 96, STAGE_A_ENDPOINT]
SHARED_TRAINING_FIELDS = (
    "learning_rate", "lr_scheduler_type", "weight_decay", "max_grad_norm",
    "microbatch_size", "gradient_accumulation_steps", "gradient_checkpointing",
    "offload_activations", "mlp_chunk_size", "packing", "max_seq_length", "dtype",
    "load_in_4bit", "attn_implementation", "lora_rank", "lora_alpha", "lora_dropout",
)


class PseudoThenGoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    pseudo_only_config: str
    gold_base_config: str
    output_dir: str
    seed: Literal[42]
    pseudo_endpoint_step: Literal[111]
    gold_exposure: Literal[24]
    endpoint_updates: Literal[6]
    checkpoint_interval: Literal[6]
    selection_policy: Literal["fixed_endpoint"]
    optimizer_policy: Literal["fresh_adamw_torch"]
    scheduler_policy: Literal["fresh_constant"]
    reuse_pseudo_optimizer: Literal[False]

    @model_validator(mode="after")
    def exact_stage_boundary(self):
        if self.gold_exposure // 4 != self.endpoint_updates:
            raise ValueError("GoldD24 must produce exactly six four-example updates")
        return self


def read_pseudo_then_gold_config(path: Path) -> PseudoThenGoldConfig:
    return PseudoThenGoldConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def _assert_stage_a_config(config: TrainingConfig) -> None:
    if (config.method != "json_only"
            or config.prepared_dir != DEFAULT_PREPARED_DIR
            or config.seed != 42
            or config.max_steps != STAGE_A_ENDPOINT
            or config.save_steps != 32
            or config.validation_steps != 32
            or config.gradient_accumulation_steps != 4
            or config.max_seq_length != 65_536
            or config.validation_max_new_tokens != 16_384
            or not config.load_in_4bit):
        raise ValueError("Stage A differs from the frozen length-filtered QLoRA pilot")


def prepare_pseudo_stage_a(root: Path, config_path: Path, *, offline=False):
    config = read_config(config_path)
    _assert_stage_a_config(config)
    tokenizer, records, validation, summary = prepare_pseudo_training(
        root, prepared_dir=config.prepared_dir, offline=offline
    )
    records, summary, length_filter = apply_length_filter(
        root, records, summary, prepared_dir=config.prepared_dir
    )
    expected_steps = (
        len(records) + config.gradient_accumulation_steps - 1
    ) // config.gradient_accumulation_steps
    if (len(records) != STAGE_A_EXAMPLES or expected_steps != config.max_steps
            or len({row["paper_id"] for row in records}) != len(records)):
        raise ValueError("Stage A is not exactly one pass over 442 unique papers")
    experiment = {
        "version": "1.0.0",
        "phase": "pseudo_stage_a",
        "condition": "pseudo_length_filtered",
        "family": "pseudo_label_supervision",
        "description": "One fixed pass of complete-JSON teacher pseudo-labels within the reviewed memory envelope.",
        "seed": config.seed,
        "planned_exposure": {"D": STAGE_A_EXAMPLES},
        "total_examples": STAGE_A_EXAMPLES,
        "endpoint_updates": STAGE_A_ENDPOINT,
        "diagnostic_checkpoints": STAGE_A_CHECKPOINTS,
        "selection_policy": "fixed_endpoint",
        "endpoint_policy": "fixed_exposure",
        "target_source": "teacher_pseudo_label",
        "human_gold_exposure": 0,
        "schedule_policy": "one_randomly_shuffled_pass_without_replacement_with_final_two_example_update",
        "final_update_examples": 2,
        "length_filter": {
            "manifest": LENGTH_FILTER_MANIFEST,
            "manifest_sha256": file_hash(root / LENGTH_FILTER_MANIFEST),
            **length_filter["policy"],
            **length_filter["counts"],
            **length_filter["summary"],
        },
        "training_ids_sha256": text_hash(json.dumps(
            sorted(row["paper_id"] for row in records), separators=(",", ":")
        )),
        "config": str(config_path.relative_to(root)),
        "config_sha256": file_hash(config_path),
    }
    summary = {**summary, "pseudo_experiment": experiment}
    return config, tokenizer, records, validation, summary, experiment


def _assert_shared_schedule(stage_a: TrainingConfig, gold: TrainingConfig) -> None:
    differences = {
        field: (getattr(stage_a, field), getattr(gold, field))
        for field in SHARED_TRAINING_FIELDS
        if getattr(stage_a, field) != getattr(gold, field)
    }
    if differences:
        raise ValueError(f"Pseudo and gold training policies differ: {differences}")


def prepare_pseudo_stage_b(root: Path, config_path: Path, *, offline=False):
    plan = read_pseudo_then_gold_config(config_path)
    stage_a_path = root / plan.pseudo_only_config
    gold_path = root / plan.gold_base_config
    stage_a = read_config(stage_a_path)
    gold = read_config(gold_path)
    _assert_stage_a_config(stage_a)
    _assert_shared_schedule(stage_a, gold)
    tokenizer, records, validation, base_summary = prepare_training(
        root, gold, offline=offline
    )
    schedule = exposure_schedule(records, {"D": plan.gold_exposure}, plan.seed)
    if (len(schedule) != 24
            or Counter(row["task"] for row in schedule) != Counter({"D": 24})
            or Counter(row["example_id"] for row in schedule)
            != Counter({row["example_id"]: 3 for row in records})):
        raise ValueError("Stage B schedule is not exactly three cycles over eight gold D papers")

    values = gold.model_dump()
    values.update(
        seed=plan.seed,
        max_steps=plan.endpoint_updates,
        save_steps=plan.checkpoint_interval,
        validation_steps=plan.checkpoint_interval,
        output_dir=plan.output_dir,
    )
    config = TrainingConfig.model_validate(values)
    identity = [
        {"example_id": row["example_id"], "task": row["task"],
         "tokens": len(row["input_ids"])}
        for row in schedule
    ]
    experiment = {
        "version": plan.version,
        "phase": "pseudo_stage_b",
        "condition": "pseudo_length_filtered_then_gold_d24",
        "family": "pseudo_label_supervision",
        "description": "Continue the fixed pseudo endpoint on GoldD24 with fresh optimization state.",
        "seed": plan.seed,
        "planned_exposure": {"D": plan.gold_exposure},
        "total_examples": len(schedule),
        "endpoint_updates": plan.endpoint_updates,
        "selection_policy": plan.selection_policy,
        "endpoint_policy": "fixed_exposure",
        "optimizer_policy": plan.optimizer_policy,
        "scheduler_policy": plan.scheduler_policy,
        "reuse_pseudo_optimizer": plan.reuse_pseudo_optimizer,
        "stage_a_expected_step": plan.pseudo_endpoint_step,
        "schedule_sha256": text_hash(json.dumps(identity, sort_keys=True)),
        "per_paper_exposure": dict(sorted(Counter(
            row["example_id"].split(":")[0] for row in schedule
        ).items())),
        "config": str(config_path.relative_to(root)),
        "config_sha256": file_hash(config_path),
        "gold_base_config_sha256": file_hash(gold_path),
        "pseudo_only_config_sha256": file_hash(stage_a_path),
    }
    summary = {
        **base_summary,
        "method": "pseudo_length_filtered_then_gold_d24",
        "training_examples": len(schedule),
        "training_papers": 8,
        "task_counts": {"D": 24},
        "longest_training_sequence": max(len(row["input_ids"]) for row in schedule),
        "supervised_tokens_per_pass": sum(
            sum(token != -100 for token in row["labels"]) for row in schedule
        ),
        "total_tokens_per_pass": sum(len(row["input_ids"]) for row in schedule),
        "pseudo_experiment": experiment,
    }
    return config, tokenizer, schedule, validation, summary, experiment


def resolve_stage_a_endpoint(root: Path, plan: PseudoThenGoldConfig) -> tuple[Path, dict]:
    stage_a = read_config(root / plan.pseudo_only_config)
    _assert_stage_a_config(stage_a)
    output = (root / stage_a.output_dir).resolve()
    selection_path = output / "selection.json"
    summary_path = output / "training_summary.json"
    manifest_path = output / "run_manifest.json"
    if not all(path.is_file() for path in (selection_path, summary_path, manifest_path)):
        raise ValueError("Stage A must complete before Stage B training")
    selection = read_json(selection_path)
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    expected_checkpoint = f"checkpoint-{plan.pseudo_endpoint_step}"
    if (selection.get("status") != "complete"
            or selection.get("metric") != "fixed_exposure_endpoint"
            or selection.get("endpoint_policy") != "fixed_exposure"
            or selection.get("kind") != "adapter"
            or selection.get("step") != plan.pseudo_endpoint_step
            or selection.get("checkpoint") != expected_checkpoint
            or selection.get("exposure", {}).get("examples") != {"D": STAGE_A_EXAMPLES}
            or summary.get("global_step") != plan.pseudo_endpoint_step
            or summary.get("exposure", {}).get("examples") != {"D": STAGE_A_EXAMPLES}
            or manifest.get("experiment", {}).get("phase") != "pseudo_stage_a"
            or manifest.get("config") != stage_a.model_dump()):
        raise ValueError("Stage A endpoint does not satisfy the frozen boundary contract")
    endpoint = output / expected_checkpoint
    if file_hash(endpoint / "adapter_model.safetensors") != selection["adapter_sha256"]:
        raise ValueError("Stage A selected adapter hash changed")
    provenance = {
        "output_dir": stage_a.output_dir,
        "checkpoint": expected_checkpoint,
        "adapter_sha256": selection["adapter_sha256"],
        "selection_sha256": file_hash(selection_path),
        "training_summary_sha256": file_hash(summary_path),
        "run_manifest_sha256": file_hash(manifest_path),
    }
    return endpoint, provenance


def build_stage_a_contract(root: Path, config_path: Path, config: TrainingConfig,
                           summary: dict, architecture: dict) -> dict:
    directory = root / config.prepared_dir
    prepared = read_json(directory / "preparation_manifest.json")
    split = read_json(root / prepared["inputs"]["split_manifest"])
    parse_summary = read_json(directory / "parse_summary.json")
    source_files = [
        "pyproject.toml", "uv.lock", str(config_path.relative_to(root)),
        "scripts/prepare_pseudo_length_filter.py",
        "scripts/run_pseudo_finetune.py",
        "src/meta_data_extraction/finetune/pseudo_experiment.py",
        "src/meta_data_extraction/finetune/pseudo_training_data.py",
        "src/meta_data_extraction/finetune/training.py",
        "src/meta_data_extraction/finetune/training_data.py",
        "src/meta_data_extraction/finetune/memory.py",
        "src/meta_data_extraction/finetune/validation.py",
        "src/meta_data_extraction/finetune/schema.py",
        "src/meta_data_extraction/finetune/tracking.py",
        config.tracking_config,
        "src/meta_data_extraction/evaluation.py",
        "src/meta_data_extraction/alignment.py",
        "src/meta_data_extraction/normalization.py",
        "src/meta_data_extraction/config_validation.py",
        "configs/evaluation_metrics.yaml", "configs/alignment_helper.yaml",
        LENGTH_FILTER_MANIFEST,
    ]

    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    optional_versions = {}
    for package in ("bitsandbytes", "flash-linear-attention", "fla-core", "einops",
                    "causal-conv1d"):
        try:
            optional_versions[package] = version(package)
        except PackageNotFoundError:
            optional_versions[package] = None
    runtime = {"python": sys.version.split()[0], **optional_versions}
    runtime.update({
        package: version(package)
        for package in ("torch", "transformers", "tokenizers", "peft", "trl",
                        "accelerate", "mlflow")
    })
    return {
        "config": config.model_dump(),
        "inference_precision": config.precision_policy(),
        "data": summary,
        "architecture": architecture,
        "preparation": {
            "inputs": prepared["inputs"],
            "policy": prepared["policy"],
            "counts": prepared["counts"],
            "parser": parse_summary["parser"],
        },
        "split": {
            key: split[key]
            for key in ("pseudo_train", "pseudo_validation", "future_test",
                        "excluded_missing_prediction")
        },
        "code": {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        },
        "sources": {name: file_hash(root / name) for name in source_files},
        "runtime": runtime,
    }


def plan_pseudo_experiment(root: Path, stage_a_path: Path,
                           stage_b_path: Path) -> dict:
    stage_a = read_config(stage_a_path)
    _assert_stage_a_config(stage_a)
    stage_b = read_pseudo_then_gold_config(stage_b_path)
    if stage_b.pseudo_only_config != str(stage_a_path.relative_to(root)):
        raise ValueError("Stage B does not point to the reviewed Stage-A config")
    return {
        "stage_a": {
            "condition": "pseudo_length_filtered", "seed": stage_a.seed,
            "examples": STAGE_A_EXAMPLES, "optimizer_updates": stage_a.max_steps,
            "diagnostic_checkpoints": STAGE_A_CHECKPOINTS,
            "output_dir": stage_a.output_dir,
        },
        "stage_b": {
            "condition": "pseudo_length_filtered_then_gold_d24", "seed": stage_b.seed,
            "examples": stage_b.gold_exposure,
            "optimizer_updates": stage_b.endpoint_updates,
            "initial_adapter": f"{stage_a.output_dir}/checkpoint-{stage_b.pseudo_endpoint_step}",
            "optimizer_policy": stage_b.optimizer_policy,
            "scheduler_policy": stage_b.scheduler_policy,
            "output_dir": stage_b.output_dir,
        },
        "training_started": False,
    }
