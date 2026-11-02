"""Exploratory 30-paper development evaluation for fixed ablation endpoints."""
from __future__ import annotations

from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import statistics
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator
import torch
from transformers import set_seed
import yaml

from meta_data_extraction.evaluation import build_evaluator_metadata
from .ablation import read_ablation_config
from .evaluation import (
    EvaluationConfig, generate_condition, prepare_condition_output, prepare_test_rows,
    read_evaluation_config, runtime_versions, score_condition,
)
from .tracking import checksum
from .training import load_base, reload_adapter
from .training_data import TrainingConfig, file_hash
from .validation import write_json


SHORTLIST = ("d_only", "matched_d_dr", "matched_d_der", "matched_d_drp")
SEEDS = (42, 123, 456)
QUALITY_METRICS = (
    "json_valid_rate", "schema_valid_rate", "facilities_f1", "samples_f1",
    "datasets_f1", "sample_refs_f1", "facility_refs_f1",
    "generic_metadata_macro_score", "mean_latency_ms", "mean_generated_tokens",
)


class AblationEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    evaluation_config: str
    ablation_config: str
    output_dir: str
    corpus_role: Literal["exploratory_reused_step7_test_as_development"]
    endpoint_policy: Literal["fixed_exposure"]
    attach_mlflow: Literal[False]
    conditions: list[str]
    seeds: list[int]

    @model_validator(mode="after")
    def frozen_shortlist(self):
        if self.conditions != list(SHORTLIST) or self.seeds != list(SEEDS):
            raise ValueError("The exploratory evaluation shortlist or seed order changed")
        if Path(self.output_dir) == Path("outputs/finetune/evaluation"):
            raise ValueError("Ablation development evaluation cannot overwrite Step 7")
        return self


def read_ablation_evaluation_config(path: Path) -> AblationEvaluationConfig:
    return AblationEvaluationConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def label(condition: str, seed: int) -> str:
    return f"{condition}_seed{seed}"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def endpoint_adapter(root: Path, matrix, condition: str, seed: int) -> tuple[TrainingConfig, Path, dict]:
    if condition not in SHORTLIST or seed not in SEEDS:
        raise ValueError("Condition/seed is outside the frozen evaluation shortlist")
    output = root / matrix.output_root / label(condition, seed)
    manifest = _read(output / "run_manifest.json")
    summary = _read(output / "training_summary.json")
    training = TrainingConfig.model_validate(manifest["config"])
    declaration = matrix.conditions[condition]
    endpoint = sum(declaration.exposures.values()) // training.gradient_accumulation_steps
    ablation = summary.get("ablation") or {}
    if (training.seed, training.max_steps, summary.get("global_step"),
            ablation.get("condition"), ablation.get("seed"),
            ablation.get("planned_exposure"),
            (summary.get("exposure") or {}).get("examples")) != (
                seed, endpoint, endpoint, condition, seed,
                declaration.exposures, declaration.exposures):
        raise ValueError(f"Training endpoint differs from frozen matrix: {label(condition, seed)}")
    checkpoint = output / f"checkpoint-{endpoint}"
    required = ("adapter_config.json", "adapter_model.safetensors", "inference_precision.json",
                "trainer_state.json", "exposure.json", "validation_state.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"Endpoint checkpoint is incomplete ({label(condition, seed)}): {missing}")
    state = _read(checkpoint / "trainer_state.json")
    exposure = _read(checkpoint / "exposure.json")
    if state["global_step"] != endpoint or exposure["global_step"] != endpoint:
        raise ValueError(f"Endpoint checkpoint step differs: {label(condition, seed)}")
    identity = {
        "condition": condition, "family": declaration.family, "seed": seed,
        "label": label(condition, seed), "endpoint_step": endpoint,
        "planned_exposure": declaration.exposures,
        "adapter_sha256": checksum(checkpoint / "adapter_model.safetensors"),
        "training_mlflow_run_id": summary["mlflow_run_id"],
        "training_runtime_seconds": summary["metrics"]["train_runtime"],
    }
    return training, checkpoint, identity


def evaluation_contract(root: Path, config_path: Path, config: AblationEvaluationConfig,
                        base_config: EvaluationConfig, matrix, condition: str, seed: int,
                        rows: list[dict], data: dict, tokenizer_identity: dict) -> tuple[dict, TrainingConfig, Path]:
    training, checkpoint, identity = endpoint_adapter(root, matrix, condition, seed)
    prepare = yaml.safe_load((root / base_config.prepare_config).read_text(encoding="utf-8"))
    precision = _read(checkpoint / "inference_precision.json")
    if precision != training.precision_policy():
        raise ValueError(f"Endpoint precision contract changed: {identity['label']}")
    source_paths = [
        config_path, root / config.evaluation_config, root / config.ablation_config,
        root / base_config.metrics_config, root / base_config.alignment_config,
        root / "scripts/evaluate_finetune_ablation.py",
        root / "src/meta_data_extraction/finetune/ablation_evaluation.py",
        root / "src/meta_data_extraction/finetune/evaluation.py",
        root / "src/meta_data_extraction/finetune/schema.py",
        root / "src/meta_data_extraction/evaluation.py",
        root / "src/meta_data_extraction/alignment.py",
    ]
    contract = {
        "version": config.version, "study_type": "exploratory",
        "corpus_role": config.corpus_role, "condition": identity["label"],
        "method": "ablation_fixed_endpoint", "endpoint_policy": config.endpoint_policy,
        "training": identity,
        "model": {"model_id": prepare["model_id"], "revision": prepare["revision"],
                  "precision": precision, "adapter": str(checkpoint.resolve()),
                  "adapter_sha256": identity["adapter_sha256"],
                  "selected_by_two_paper_validation": False},
        "decoding": {"method": base_config.decoding, "enable_thinking": base_config.enable_thinking,
                     "max_new_tokens": base_config.max_new_tokens,
                     "max_seq_length": base_config.max_seq_length,
                     "seed": seed, "num_beams": 1, "repetition_penalty": 1.0},
        "data": {"source_split": "step7_test", "analysis_role": "exploratory_development",
                 "paper_ids": [row["paper_id"] for row in rows],
                 "prompts": [{key: row[key] for key in
                              ("paper_id", "prompt_tokens", "prompt_sha256", "text_sha256")}
                             for row in rows],
                 "preparation_manifest_sha256": data["preparation_manifest_sha256"],
                 "gold_read_during_generation": False,
                 "prior_results_influenced_method_design": True},
        "tokenizer": tokenizer_identity,
        "evaluator": build_evaluator_metadata(root / base_config.metrics_config),
        "mlflow_attachment_allowed": False,
        "sources": {str(path.relative_to(root)): file_hash(path) for path in source_paths},
    }
    if contract["evaluator"]["schema_version_mismatch"]:
        raise ValueError("Evaluator schema versions do not match current schemas")
    return contract, training, checkpoint


def run_one(root: Path, config_path: Path, config: AblationEvaluationConfig,
            base_config: EvaluationConfig, matrix, condition: str, seed: int,
            tokenizer, tokenizer_identity: dict, rows: list[dict], data: dict, *,
            offline: bool, score_only: bool) -> dict:
    name = label(condition, seed)
    output = root / config.output_dir / name
    contract, training, checkpoint = evaluation_contract(
        root, config_path, config, base_config, matrix, condition, seed,
        rows, data, tokenizer_identity)
    contract["runtime"] = runtime_versions()
    prepare_condition_output(output, contract)
    if not score_only:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("Generation requires one CUDA GPU with native BF16 support")
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU for evaluation")
        prepare = yaml.safe_load((root / base_config.prepare_config).read_text(encoding="utf-8"))
        set_seed(seed)
        model = reload_adapter(load_base(prepare, training, offline=offline), checkpoint,
                               prepare, config=training)
        try:
            generate_condition(model, tokenizer, rows, base_config, name, output, data["schemas"])
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    return score_condition(root, base_config, name, output)


def _reference_complete(output: Path) -> int:
    rows = [json.loads(line) for line in (output / "generation.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 30:
        raise ValueError(f"Incomplete generation accounting: {output}")
    return sum(row["status"] == "generated" and not any(
        ".sample_refs:" in error or ".facility_refs:" in error
        for error in row.get("schema_errors", [])) for row in rows)


def _summary(values: dict[int, float]) -> dict:
    if set(values) != set(SEEDS):
        raise ValueError("Comparison requires all three seeds")
    numbers = list(values.values())
    return {"mean": math.fsum(numbers) / len(numbers), "sample_sd": statistics.stdev(numbers),
            "min": min(numbers), "max": max(numbers),
            "by_seed": {str(seed): value for seed, value in sorted(values.items())}}


def compare(root: Path, config: AblationEvaluationConfig) -> dict:
    output_root = root / config.output_dir
    conditions = {}
    common_order = None
    for condition in config.conditions:
        per_seed = {}
        for seed in config.seeds:
            name = label(condition, seed)
            output = output_root / name
            metrics = _read(output / "metrics.json")
            order = [row["paper_id"] for row in metrics["papers"]]
            common_order = order if common_order is None else common_order
            if order != common_order or len(order) != 30 or len(set(order)) != 30:
                raise ValueError("Evaluated paper order differs across ablation runs")
            per_seed[seed] = {
                **metrics["summary"], "papers_reference_complete": _reference_complete(output),
                "adapter_sha256": _read(output / "manifest.json")["model"]["adapter_sha256"],
            }
        numeric = (*QUALITY_METRICS, "papers_json_valid", "papers_schema_valid",
                   "papers_reference_complete", "generation_failures", "output_limit_failures")
        conditions[condition] = {
            "runs": {str(seed): per_seed[seed] for seed in config.seeds},
            "aggregate": {metric: _summary({seed: float(per_seed[seed][metric] or 0.0)
                                            for seed in config.seeds}) for metric in numeric},
        }
    historical = {}
    for name in ("m0", "m1", "m2"):
        path = root / "outputs/finetune/evaluation" / name / "metrics.json"
        if path.is_file():
            historical[name] = _read(path)["summary"]
    comparison = {
        "version": config.version, "study_type": "exploratory",
        "corpus_role": config.corpus_role, "paper_ids": common_order,
        "fixed_endpoint_conditions": conditions,
        "historical_step7_reference": historical,
        "caveats": [
            "These thirty papers influenced method design and are development data, not an independent test.",
            "All fixed endpoints and all three seeds are reported; no checkpoint is revised from these results.",
            "Historical M0/M1/M2 summaries are context only and were produced before this follow-up.",
        ],
    }
    write_json(output_root / "comparison.json", comparison)
    lines = [
        "# Exploratory ablation development evaluation", "",
        "The thirty Step 7 papers are reused as development data. These results are not an independent test.", "",
        "| Condition | JSON valid | Full schema | Reference complete | Facilities F1 | Samples F1 | Datasets F1 | Sample refs F1 | Facility refs F1 | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, values in conditions.items():
        agg = values["aggregate"]
        def show(metric, scale=1.0, digits=3):
            return f"{agg[metric]['mean'] / scale:.{digits}f} ± {agg[metric]['sample_sd'] / scale:.{digits}f}"
        lines.append(
            f"| `{condition}` | {show('papers_json_valid', digits=1)}/30 | "
            f"{show('papers_schema_valid', digits=1)}/30 | {show('papers_reference_complete', digits=1)}/30 | "
            f"{show('facilities_f1')} | {show('samples_f1')} | {show('datasets_f1')} | "
            f"{show('sample_refs_f1')} | {show('facility_refs_f1')} | {show('mean_latency_ms', 1000, 1)} |"
        )
    lines += ["", "Values are mean ± sample standard deviation across seeds 42, 123 and 456.",
              "Every row uses the declared fixed-exposure endpoint, not post-evaluation checkpoint selection.", ""]
    temporary = (output_root / "comparison.md").with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output_root / "comparison.md")
    return comparison


def plan(root: Path, config: AblationEvaluationConfig) -> list[dict]:
    matrix = read_ablation_config(root / config.ablation_config)
    result = []
    for condition in config.conditions:
        for seed in config.seeds:
            _, checkpoint, identity = endpoint_adapter(root, matrix, condition, seed)
            result.append({**identity, "checkpoint": str(checkpoint.relative_to(root)),
                           "output": f"{config.output_dir}/{identity['label']}"})
    return result
