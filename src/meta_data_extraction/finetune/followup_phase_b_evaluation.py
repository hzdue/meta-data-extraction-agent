"""Phase B D32 development evaluation and frozen Gate B decision."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import statistics
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch
from transformers import set_seed
import yaml

from meta_data_extraction.evaluation import build_evaluator_metadata
from .evaluation import (
    EvaluationConfig, generate_condition, prepare_condition_output, runtime_versions,
    score_condition,
)
from .followup_analysis import (
    FIELD_SPECS, add_counts, prf, report_field_counts, report_field_f1,
)
from .followup_training import GateB, PhaseBConfig, read_phase_b_config
from .tracking import checksum
from .training import load_base, reload_adapter
from .training_data import TrainingConfig, file_hash
from .validation import write_json


SEEDS = (42, 123, 456)
CONDITIONS = ("d_only", "matched_d_dr", "d32")
SUMMARY_METRICS = (
    "json_valid_rate", "schema_valid_rate", "facilities_f1", "samples_f1",
    "datasets_f1", "sample_refs_f1", "facility_refs_f1",
    "generic_metadata_macro_score", "mean_latency_ms", "mean_generated_tokens",
    "papers_json_valid", "papers_schema_valid", "generation_failures",
    "output_limit_failures",
)
CONTRAST_METRICS = ("facilities", "samples", "datasets", "sample_refs", "facility_refs")


class PhaseBEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    phase: Literal["phase_b_evaluation"]
    study_type: Literal["exploratory"]
    training_config: str
    training_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_config: str
    ablation_evaluation_config: str
    ablation_evaluation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_evaluation_dir: str
    output_dir: str
    corpus_role: Literal["exploratory_reused_step7_test_as_development"]
    endpoint_policy: Literal["fixed_exposure"]
    d32_condition: Literal["d32"]
    controls: list[str]
    seeds: list[int]
    expected_papers: Literal[30]
    attach_mlflow: Literal[False]

    @model_validator(mode="after")
    def frozen_contract(self):
        if self.controls != ["d_only", "matched_d_dr"] or self.seeds != list(SEEDS):
            raise ValueError("Phase B controls or seed order changed")
        if Path(self.output_dir) == Path(self.control_evaluation_dir):
            raise ValueError("D32 evaluation must not overwrite existing controls")
        return self


def read_phase_b_evaluation_config(path: Path) -> PhaseBEvaluationConfig:
    return PhaseBEvaluationConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_hash(root: Path, relative: str, expected: str) -> None:
    path = root / relative
    if not path.is_file() or file_hash(path) != expected:
        raise ValueError(f"Frozen Phase B evaluation input changed: {relative}")


def label(seed: int) -> str:
    return f"d32_seed{seed}"


def d32_endpoint(root: Path, phase_b: PhaseBConfig, config: PhaseBEvaluationConfig,
                 seed: int) -> tuple[TrainingConfig, Path, dict]:
    if seed not in SEEDS:
        raise ValueError("Seed is outside the frozen Phase B evaluation")
    output = root / phase_b.output_root / label(seed)
    manifest = _read(output / "run_manifest.json")
    summary = _read(output / "training_summary.json")
    training = TrainingConfig.model_validate(manifest["config"])
    manifest_experiment = manifest.get("experiment") or {}
    summary_experiment = summary.get("experiment") or {}
    expected = (
        seed, 8, 8, "followup_phase_b", "d32", seed, {"D": 32}, {"D": 32},
        config.training_config_sha256,
    )
    actual = (
        training.seed, training.max_steps, summary.get("global_step"),
        manifest.get("phase"), manifest_experiment.get("condition"),
        manifest_experiment.get("seed"), manifest_experiment.get("planned_exposure"),
        (summary.get("exposure") or {}).get("examples"),
        manifest_experiment.get("phase_config_sha256"),
    )
    if actual != expected or summary_experiment != manifest_experiment:
        raise ValueError(f"D32 training endpoint differs from its frozen contract: {label(seed)}")
    if set((manifest_experiment.get("per_paper_exposure") or {}).values()) != {4}:
        raise ValueError(f"D32 per-paper exposure differs: {label(seed)}")
    checkpoint = output / "checkpoint-8"
    required = (
        "adapter_config.json", "adapter_model.safetensors", "inference_precision.json",
        "trainer_state.json", "exposure.json", "validation_state.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"D32 endpoint is incomplete ({label(seed)}): {missing}")
    if (_read(checkpoint / "trainer_state.json")["global_step"] != 8
            or _read(checkpoint / "exposure.json")["global_step"] != 8):
        raise ValueError(f"D32 endpoint step differs: {label(seed)}")
    identity = {
        "condition": "d32", "family": "additional_training_control", "seed": seed,
        "label": label(seed), "endpoint_step": 8, "planned_exposure": {"D": 32},
        "adapter_sha256": checksum(checkpoint / "adapter_model.safetensors"),
        "training_mlflow_run_id": summary["mlflow_run_id"],
        "training_runtime_seconds": summary["metrics"]["train_runtime"],
        "schedule_sha256": manifest_experiment["schedule_sha256"],
        "gate_b_sha256": manifest_experiment["gate_b_sha256"],
    }
    return training, checkpoint, identity


def evaluation_contract(root: Path, config_path: Path, config: PhaseBEvaluationConfig,
                        phase_b: PhaseBConfig, base_config: EvaluationConfig, seed: int,
                        rows: list[dict], data: dict, tokenizer_identity: dict):
    training, checkpoint, identity = d32_endpoint(root, phase_b, config, seed)
    prepare = yaml.safe_load((root / base_config.prepare_config).read_text(encoding="utf-8"))
    precision = _read(checkpoint / "inference_precision.json")
    if precision != training.precision_policy():
        raise ValueError(f"D32 precision contract changed: {label(seed)}")
    source_paths = [
        config_path, root / config.training_config, root / config.evaluation_config,
        root / config.ablation_evaluation_config, root / base_config.metrics_config,
        root / base_config.alignment_config,
        root / "scripts/evaluate_finetune_followup_phase_b.py",
        root / "src/meta_data_extraction/finetune/followup_phase_b_evaluation.py",
        root / "src/meta_data_extraction/finetune/evaluation.py",
        root / "src/meta_data_extraction/finetune/schema.py",
        root / "src/meta_data_extraction/evaluation.py",
        root / "src/meta_data_extraction/alignment.py",
    ]
    contract = {
        "version": config.version, "study_type": config.study_type,
        "corpus_role": config.corpus_role, "condition": identity["label"],
        "method": "followup_phase_b_fixed_endpoint", "endpoint_policy": config.endpoint_policy,
        "training": identity,
        "model": {
            "model_id": prepare["model_id"], "revision": prepare["revision"],
            "precision": precision, "adapter": str(checkpoint.resolve()),
            "adapter_sha256": identity["adapter_sha256"],
            "selected_by_two_paper_validation": False,
        },
        "decoding": {
            "method": base_config.decoding, "enable_thinking": base_config.enable_thinking,
            "max_new_tokens": base_config.max_new_tokens,
            "max_seq_length": base_config.max_seq_length, "seed": seed,
            "num_beams": 1, "repetition_penalty": 1.0,
        },
        "data": {
            "source_split": "step7_test", "analysis_role": "exploratory_development",
            "paper_ids": [row["paper_id"] for row in rows],
            "prompts": [{key: row[key] for key in
                         ("paper_id", "prompt_tokens", "prompt_sha256", "text_sha256")}
                        for row in rows],
            "preparation_manifest_sha256": data["preparation_manifest_sha256"],
            "gold_read_during_generation": False,
            "prior_results_influenced_method_design": True,
        },
        "tokenizer": tokenizer_identity,
        "evaluator": build_evaluator_metadata(root / base_config.metrics_config),
        "mlflow_attachment_allowed": False,
        "sources": {str(path.relative_to(root)): file_hash(path) for path in source_paths},
    }
    if contract["evaluator"]["schema_version_mismatch"]:
        raise ValueError("Evaluator schema versions do not match current schemas")
    return contract, training, checkpoint


def _generation_complete(output: Path, paper_ids: list[str]) -> bool:
    path = output / "generation.jsonl"
    if not path.is_file():
        return False
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row.get("paper_id") for row in rows] == paper_ids


def run_one(root: Path, config_path: Path, config: PhaseBEvaluationConfig,
            phase_b: PhaseBConfig, base_config: EvaluationConfig, seed: int,
            tokenizer, tokenizer_identity: dict, rows: list[dict], data: dict, *,
            offline: bool, score_only: bool) -> dict:
    name = label(seed)
    output = root / config.output_dir / name
    contract, training, checkpoint = evaluation_contract(
        root, config_path, config, phase_b, base_config, seed, rows, data, tokenizer_identity)
    contract["runtime"] = runtime_versions()
    prepare_condition_output(output, contract)
    paper_ids = [row["paper_id"] for row in rows]
    if score_only:
        return score_condition(root, base_config, name, output)
    if _generation_complete(output, paper_ids):
        generate_condition(None, tokenizer, rows, base_config, name, output, data["schemas"])
        return score_condition(root, base_config, name, output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("D32 generation requires one CUDA GPU with native BF16 support")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU for D32 evaluation")
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


def _summary(values: dict[int, float]) -> dict:
    if set(values) != set(SEEDS):
        raise ValueError("Phase B comparison requires all three seeds")
    numbers = list(values.values())
    return {
        "mean": math.fsum(numbers) / len(numbers), "sample_sd": statistics.stdev(numbers),
        "min": min(numbers), "max": max(numbers),
        "by_seed": {str(seed): value for seed, value in sorted(values.items())},
    }


def _reference_complete(output: Path) -> int:
    rows = [json.loads(line) for line in
            (output / "generation.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 30:
        raise ValueError(f"Incomplete generation accounting: {output}")
    return sum(row["status"] == "generated" and not any(
        ".sample_refs:" in error or ".facility_refs:" in error
        for error in row.get("schema_errors", [])) for row in rows)


def _load_condition(root: Path, config: PhaseBEvaluationConfig, condition: str) -> dict:
    directory = (root / config.output_dir if condition == "d32"
                 else root / config.control_evaluation_dir)
    saved_comparison = (_read(directory / "comparison.json")
                        if condition != "d32" else None)
    runs = {}
    paper_ids = evaluator = None
    for seed in SEEDS:
        run_dir = directory / (label(seed) if condition == "d32" else f"{condition}_seed{seed}")
        manifest, metrics = _read(run_dir / "manifest.json"), _read(run_dir / "metrics.json")
        order = [row["paper_id"] for row in metrics["papers"]]
        if len(order) != config.expected_papers or len(set(order)) != config.expected_papers:
            raise ValueError(f"Paper coverage differs: {condition} seed {seed}")
        paper_ids = order if paper_ids is None else paper_ids
        if order != paper_ids or manifest["data"]["paper_ids"] != paper_ids:
            raise ValueError("Paper order differs across Phase B evaluation runs")
        current_evaluator = metrics["evaluator"]
        evaluator = current_evaluator if evaluator is None else evaluator
        if current_evaluator != evaluator or manifest["evaluator"] != evaluator:
            raise ValueError("Evaluator metadata differs across Phase B evaluation runs")
        expected_label = label(seed) if condition == "d32" else f"{condition}_seed{seed}"
        if (manifest.get("condition"), manifest.get("endpoint_policy")) != (
                expected_label, config.endpoint_policy):
            raise ValueError(f"Evaluation manifest differs: {expected_label}")
        reports = {paper_id: _read(run_dir / "reports" / f"{paper_id}.json")
                   for paper_id in paper_ids}
        for field in CONTRAST_METRICS:
            pooled = prf(add_counts([
                report_field_counts(reports[paper_id], field) for paper_id in paper_ids
            ]))["f1"]
            stored = metrics["summary"][f"{field}_f1"]
            if stored is None or not math.isclose(pooled, float(stored), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Report/metric counts differ: {condition} seed {seed} {field}")
        if saved_comparison is not None:
            saved = saved_comparison["fixed_endpoint_conditions"][condition]["runs"][str(seed)]
            if manifest["model"]["adapter_sha256"] != saved["adapter_sha256"]:
                raise ValueError(f"Reused adapter checksum differs: {condition} seed {seed}")
            for key, value in metrics["summary"].items():
                if key in saved and saved[key] != value:
                    raise ValueError(f"Reused metric differs: {condition} seed {seed} {key}")
        runs[seed] = {
            "directory": run_dir, "manifest": manifest, "metrics": metrics,
            "reports": reports, "reference_complete": _reference_complete(run_dir),
        }
    return {"paper_ids": paper_ids, "evaluator": evaluator, "runs": runs}


def _paired_contrast(treatment: dict, control: dict, paper_ids: list[str], field: str) -> dict:
    summary_key = f"{field}_f1"
    by_seed = {}
    per_paper = {paper_id: {} for paper_id in paper_ids}
    for seed in SEEDS:
        treatment_value = treatment["runs"][seed]["metrics"]["summary"][summary_key]
        control_value = control["runs"][seed]["metrics"]["summary"][summary_key]
        if treatment_value is None or control_value is None:
            raise ValueError(f"Pooled Gate B metric is undefined: {field} seed {seed}")
        by_seed[seed] = float(treatment_value) - float(control_value)
        for paper_id in paper_ids:
            first = report_field_f1(treatment["runs"][seed]["reports"][paper_id], field)
            second = report_field_f1(control["runs"][seed]["reports"][paper_id], field)
            per_paper[paper_id][str(seed)] = (
                first - second if first is not None and second is not None else None)
    paper_rows = []
    for paper_id in paper_ids:
        values = [value for value in per_paper[paper_id].values() if value is not None]
        paper_rows.append({
            "paper_id": paper_id, "delta_f1_by_seed": per_paper[paper_id],
            "mean_delta_f1": math.fsum(values) / len(values) if values else None,
            "comparable_seeds": len(values),
        })
    comparable = [row for row in paper_rows if row["mean_delta_f1"] is not None]
    improved = sum(row["mean_delta_f1"] > 1e-12 for row in comparable)
    tied = sum(abs(row["mean_delta_f1"]) <= 1e-12 for row in comparable)
    worsened = sum(row["mean_delta_f1"] < -1e-12 for row in comparable)
    absolute_total = math.fsum(abs(row["mean_delta_f1"]) for row in comparable)
    largest_share = (max((abs(row["mean_delta_f1"]) for row in comparable), default=0.0)
                     / absolute_total if absolute_total else 0.0)
    return {
        "treatment": "matched_d_dr", "control": "d32",
        "pooled_f1_delta_across_seeds": _summary(by_seed),
        "positive_seeds": sum(value > 1e-12 for value in by_seed.values()),
        "paper_direction": {
            "improved": improved, "tied": tied, "worsened": worsened,
            "not_comparable": len(paper_ids) - len(comparable),
        },
        "largest_absolute_paper_share": largest_share,
        "papers": paper_rows,
    }


def apply_gate_b(contrasts: dict, gate: GateB) -> dict:
    means = {field: contrasts[field]["pooled_f1_delta_across_seeds"]["mean"]
             for field in CONTRAST_METRICS}
    prefer_checks = {
        "facilities_within_margin": means["facilities"] <= gate.equivalence_margin_f1,
        "facility_refs_within_margin": means["facility_refs"] <= gate.equivalence_margin_f1,
        "d32_samples_recovery": means["samples"] <= -gate.sample_recovery_margin_f1,
        "d32_sample_refs_recovery": means["sample_refs"] <= -gate.sample_recovery_margin_f1,
    }
    proceed_checks = {
        "facilities_margin": means["facilities"] >= gate.r_specific_facilities_margin_f1,
        "facility_refs_margin": means["facility_refs"] >= gate.r_specific_facility_refs_margin_f1,
    }
    for field in ("facilities", "facility_refs"):
        values = contrasts[field]
        proceed_checks[f"{field}_positive_seeds"] = (
            values["positive_seeds"] >= gate.minimum_positive_seeds)
        direction = values["paper_direction"]
        proceed_checks[f"{field}_paper_direction"] = (
            direction["improved"] - direction["worsened"]
            >= gate.minimum_net_improved_papers)
        proceed_checks[f"{field}_concentration"] = (
            values["largest_absolute_paper_share"]
            <= gate.maximum_largest_paper_absolute_share)
    prefer = all(prefer_checks.values())
    proceed = all(proceed_checks.values())
    if prefer and proceed:
        raise ValueError("Frozen Gate B rules produced overlapping outcomes")
    outcome = "prefer_d32" if prefer else "proceed_phase_c" if proceed else "inconclusive"
    return {
        "outcome": outcome, "comparison": gate.comparison,
        "thresholds": gate.model_dump(exclude={"decision_rules"}),
        "prefer_d32_checks": prefer_checks, "proceed_phase_c_checks": proceed_checks,
        "decision_rule_text": gate.decision_rules,
    }


def compare(root: Path, config: PhaseBEvaluationConfig, phase_b: PhaseBConfig) -> dict:
    loaded = {condition: _load_condition(root, config, condition) for condition in CONDITIONS}
    paper_ids = loaded["d32"]["paper_ids"]
    evaluator = loaded["d32"]["evaluator"]
    if any(values["paper_ids"] != paper_ids or values["evaluator"] != evaluator
           for values in loaded.values()):
        raise ValueError("D24, D24+R8 and D32 do not share paper/evaluator provenance")
    conditions = {}
    for condition, values in loaded.items():
        runs = {}
        for seed in SEEDS:
            summary = values["runs"][seed]["metrics"]["summary"]
            runs[str(seed)] = {**summary,
                               "papers_reference_complete": values["runs"][seed]["reference_complete"]}
        numeric = (*SUMMARY_METRICS, "papers_reference_complete")
        conditions[condition] = {
            "runs": runs,
            "aggregate": {metric: _summary({seed: float(runs[str(seed)][metric] or 0.0)
                                             for seed in SEEDS}) for metric in numeric},
        }
    contrasts = {
        field: _paired_contrast(loaded["matched_d_dr"], loaded["d32"], paper_ids, field)
        for field in CONTRAST_METRICS
    }
    gate_result = apply_gate_b(contrasts, phase_b.gate_b)
    result = {
        "version": config.version, "phase": config.phase, "study_type": config.study_type,
        "corpus_role": config.corpus_role, "paper_ids": paper_ids, "evaluator": evaluator,
        "conditions": conditions, "contrast_definition": "D24+R8 minus D32",
        "contrasts": contrasts, "gate_b": gate_result,
        "accounting": {"new_generations": 90, "reused_control_generations": 180,
                       "independent_test_papers": 0},
        "caveats": [
            "The thirty papers are reused development data, not an independent test.",
            "D32 and D24+R8 match examples and optimizer updates, not supervised tokens or compute.",
            "Gate B thresholds were frozen before D32 evaluation and are applied without revision.",
        ],
    }
    output = root / config.output_dir
    write_json(output / "comparison.json", result)
    _write_markdown(output / "comparison.md", result)
    return result


def _write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Step 9 Phase B: D32 additional-training control", "",
        "Exploratory results on thirty reused development papers; values are mean ± sample standard deviation over seeds 42, 123 and 456.", "",
        "| Condition | Full schema | Reference complete | Facilities | Samples | Datasets | Sample refs | Facility refs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        aggregate = result["conditions"][condition]["aggregate"]
        def show(metric: str, digits: int = 3) -> str:
            return f"{aggregate[metric]['mean']:.{digits}f} ± {aggregate[metric]['sample_sd']:.{digits}f}"
        lines.append(
            f"| `{condition}` | {show('papers_schema_valid', 1)}/30 | "
            f"{show('papers_reference_complete', 1)}/30 | {show('facilities_f1')} | "
            f"{show('samples_f1')} | {show('datasets_f1')} | {show('sample_refs_f1')} | "
            f"{show('facility_refs_f1')} |"
        )
    lines += ["", "## D24+R8 minus D32 paired contrasts", "",
              "| Field | Mean ΔF1 | Seed SD | Positive seeds | Papers +/=/-/NA | Largest-paper share |",
              "|---|---:|---:|---:|---:|---:|"]
    for field in CONTRAST_METRICS:
        values = result["contrasts"][field]
        summary, direction = values["pooled_f1_delta_across_seeds"], values["paper_direction"]
        lines.append(
            f"| {FIELD_SPECS[field]['label']} | {summary['mean']:+.3f} | {summary['sample_sd']:.3f} | "
            f"{values['positive_seeds']}/3 | {direction['improved']}/{direction['tied']}/"
            f"{direction['worsened']}/{direction['not_comparable']} | "
            f"{values['largest_absolute_paper_share']:.1%} |"
        )
    gate = result["gate_b"]
    lines += ["", "## Frozen Gate B decision", "", f"**Outcome: `{gate['outcome']}`.**", "",
              "The decision uses the thresholds frozen before D32 evaluation. Failed and passed checks are retained in `comparison.json`.", "",
              "## Accounting", "", "- New D32 generations: 90.",
              "- Reused D24/D24+R8 generations: 180.", "- Independent test papers: 0.", ""]
    lines.extend(f"- {item}" for item in result["caveats"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def plan(root: Path, config: PhaseBEvaluationConfig, phase_b: PhaseBConfig) -> list[dict]:
    result = []
    for seed in SEEDS:
        _, checkpoint, identity = d32_endpoint(root, phase_b, config, seed)
        result.append({**identity, "checkpoint": str(checkpoint.relative_to(root)),
                       "output": f"{config.output_dir}/{label(seed)}"})
    return result
