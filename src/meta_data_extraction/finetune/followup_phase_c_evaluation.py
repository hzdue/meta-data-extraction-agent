"""Post-hoc Phase C order evaluation and frozen Gate C interpretation."""
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
from .evaluation import (EvaluationConfig, generate_condition, prepare_condition_output,
                         runtime_versions, score_condition)
from .followup_analysis import FIELD_SPECS, add_counts, prf, report_field_counts, report_field_f1
from .followup_phase_b_evaluation import CONTRAST_METRICS, SUMMARY_METRICS
from .followup_training import GateC, PhaseCConfig
from .tracking import checksum
from .training import load_base, reload_adapter
from .training_data import TrainingConfig, file_hash
from .validation import write_json


SEEDS = (42, 123, 456)
NEW_CONDITIONS = ("r_then_d", "d_then_r")
CONTROLS = ("matched_d_dr", "d32")
CONDITIONS = (*CONTROLS, *NEW_CONDITIONS)


class PhaseCEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    phase: Literal["phase_c_evaluation"]
    study_type: Literal["post_hoc_exploratory"]
    training_config: str
    training_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_config: str
    evaluation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mixed_evaluation_dir: str
    mixed_comparison_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    d32_evaluation_dir: str
    d32_comparison_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_dir: str
    corpus_role: Literal["exploratory_reused_step7_test_as_development"]
    endpoint_policy: Literal["fixed_exposure"]
    new_conditions: list[str]
    controls: list[str]
    seeds: list[int]
    expected_papers: Literal[30]
    attach_mlflow: Literal[False]
    model_selection_allowed: Literal[False]

    @model_validator(mode="after")
    def frozen_contract(self):
        if (self.new_conditions != list(NEW_CONDITIONS) or self.controls != list(CONTROLS)
                or self.seeds != list(SEEDS)):
            raise ValueError("Phase C evaluation conditions or seed order changed")
        if len({self.output_dir, self.mixed_evaluation_dir, self.d32_evaluation_dir}) != 3:
            raise ValueError("Phase C outputs may not overwrite existing controls")
        return self


def read_phase_c_evaluation_config(path: Path) -> PhaseCEvaluationConfig:
    return PhaseCEvaluationConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def label(condition: str, seed: int) -> str:
    return f"{condition}_seed{seed}"


def verify_inputs(root: Path, config: PhaseCEvaluationConfig):
    frozen = (
        (config.training_config, config.training_config_sha256),
        (config.evaluation_config, config.evaluation_config_sha256),
        (f"{config.mixed_evaluation_dir}/comparison.json", config.mixed_comparison_sha256),
        (f"{config.d32_evaluation_dir}/comparison.json", config.d32_comparison_sha256),
    )
    for relative, expected in frozen:
        path = root / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Frozen Phase C evaluation input changed: {relative}")


def phase_c_endpoint(root: Path, phase: PhaseCConfig, config: PhaseCEvaluationConfig,
                     condition: str, seed: int):
    if condition not in NEW_CONDITIONS or seed not in SEEDS:
        raise ValueError("Endpoint is outside the frozen Phase C evaluation")
    declaration = phase.conditions[condition]
    output = root / phase.output_root / label(condition, seed)
    manifest, summary = _read(output / "run_manifest.json"), _read(output / "training_summary.json")
    training = TrainingConfig.model_validate(manifest["config"])
    experiment = manifest.get("experiment") or {}
    if summary.get("experiment") != experiment:
        raise ValueError(f"Phase C manifest/summary experiment differs: {label(condition, seed)}")
    expected = (seed, 8, 8, "followup_phase_c", condition, seed, {"D": 24, "R": 8},
                {"D": 24, "R": 8}, declaration.boundary_update,
                config.training_config_sha256)
    actual = (training.seed, training.max_steps, summary.get("global_step"), manifest.get("phase"),
              experiment.get("condition"), experiment.get("seed"),
              experiment.get("planned_exposure"), (summary.get("exposure") or {}).get("examples"),
              experiment.get("boundary_update"), experiment.get("phase_config_sha256"))
    if actual != expected or set((experiment.get("per_paper_exposure") or {}).values()) != {4}:
        raise ValueError(f"Phase C endpoint differs from its frozen contract: {label(condition, seed)}")
    boundary = output / f"checkpoint-{declaration.boundary_update}"
    endpoint = output / "checkpoint-8"
    required = ("adapter_config.json", "adapter_model.safetensors", "inference_precision.json",
                "trainer_state.json", "exposure.json", "validation_state.json")
    for checkpoint in (boundary, endpoint):
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise ValueError(f"Incomplete Phase C checkpoint {checkpoint}: {missing}")
    boundary_exposure = _read(boundary / "exposure.json")
    endpoint_exposure = _read(endpoint / "exposure.json")
    expected_boundary = ({"R": 8} if condition == "r_then_d" else {"D": 24})
    if (boundary_exposure.get("global_step"), boundary_exposure.get("examples")) != (
            declaration.boundary_update, expected_boundary):
        raise ValueError(f"Phase C boundary exposure differs: {label(condition, seed)}")
    if (endpoint_exposure.get("global_step"), endpoint_exposure.get("examples")) != (
            8, {"D": 24, "R": 8}):
        raise ValueError(f"Phase C endpoint exposure differs: {label(condition, seed)}")
    identity = {"condition": condition, "family": declaration.family, "seed": seed,
                "label": label(condition, seed), "endpoint_step": 8,
                "boundary_step": declaration.boundary_update,
                "planned_exposure": declaration.exposures,
                "adapter_sha256": checksum(endpoint / "adapter_model.safetensors"),
                "boundary_adapter_sha256": checksum(boundary / "adapter_model.safetensors"),
                "training_mlflow_run_id": summary["mlflow_run_id"],
                "training_runtime_seconds": summary["metrics"]["train_runtime"],
                "schedule_sha256": experiment["schedule_sha256"],
                "gate_c_sha256": experiment["gate_c_sha256"]}
    return training, endpoint, identity


def evaluation_contract(root: Path, config_path: Path, config: PhaseCEvaluationConfig,
                        phase: PhaseCConfig, base: EvaluationConfig, condition: str, seed: int,
                        rows: list[dict], data: dict, tokenizer_identity: dict):
    training, checkpoint, identity = phase_c_endpoint(root, phase, config, condition, seed)
    preparation = yaml.safe_load((root / base.prepare_config).read_text(encoding="utf-8"))
    precision = _read(checkpoint / "inference_precision.json")
    if precision != training.precision_policy():
        raise ValueError(f"Phase C precision contract changed: {label(condition, seed)}")
    sources = [config_path, root / config.training_config, root / config.evaluation_config,
               root / base.metrics_config, root / base.alignment_config,
               root / "scripts/evaluate_finetune_followup_phase_c.py",
               root / "src/meta_data_extraction/finetune/followup_phase_c_evaluation.py",
               root / "src/meta_data_extraction/finetune/evaluation.py",
               root / "src/meta_data_extraction/finetune/schema.py",
               root / "src/meta_data_extraction/evaluation.py", root / "src/meta_data_extraction/alignment.py"]
    contract = {"version": config.version, "study_type": config.study_type,
        "corpus_role": config.corpus_role, "condition": identity["label"],
        "method": "followup_phase_c_fixed_endpoint", "endpoint_policy": config.endpoint_policy,
        "training": identity,
        "model": {"model_id": preparation["model_id"], "revision": preparation["revision"],
                  "precision": precision, "adapter": str(checkpoint.resolve()),
                  "adapter_sha256": identity["adapter_sha256"],
                  "selected_by_two_paper_validation": False},
        "decoding": {"method": base.decoding, "enable_thinking": base.enable_thinking,
                     "max_new_tokens": base.max_new_tokens, "max_seq_length": base.max_seq_length,
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
        "evaluator": build_evaluator_metadata(root / base.metrics_config),
        "mlflow_attachment_allowed": False, "model_selection_allowed": False,
        "sources": {str(path.relative_to(root)): file_hash(path) for path in sources}}
    if contract["evaluator"]["schema_version_mismatch"]:
        raise ValueError("Evaluator schema versions do not match")
    return contract, training, checkpoint


def _generation_complete(output: Path, paper_ids: list[str]) -> bool:
    path = output / "generation.jsonl"
    if not path.is_file():
        return False
    return [json.loads(line).get("paper_id") for line in path.read_text(encoding="utf-8").splitlines()] == paper_ids


def run_one(root: Path, config_path: Path, config: PhaseCEvaluationConfig,
            phase: PhaseCConfig, base: EvaluationConfig, condition: str, seed: int,
            tokenizer, tokenizer_identity: dict, rows: list[dict], data: dict, *,
            offline: bool, score_only: bool):
    name, output = label(condition, seed), root / config.output_dir / label(condition, seed)
    contract, training, checkpoint = evaluation_contract(
        root, config_path, config, phase, base, condition, seed, rows, data, tokenizer_identity)
    contract["runtime"] = runtime_versions()
    prepare_condition_output(output, contract)
    paper_ids = [row["paper_id"] for row in rows]
    if score_only:
        return score_condition(root, base, name, output)
    if _generation_complete(output, paper_ids):
        generate_condition(None, tokenizer, rows, base, name, output, data["schemas"])
        return score_condition(root, base, name, output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("Phase C evaluation requires one CUDA GPU with native BF16 support")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU for Phase C evaluation")
    preparation = yaml.safe_load((root / base.prepare_config).read_text(encoding="utf-8"))
    set_seed(seed)
    model = reload_adapter(load_base(preparation, training, offline=offline), checkpoint,
                           preparation, config=training)
    try:
        generate_condition(model, tokenizer, rows, base, name, output, data["schemas"])
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return score_condition(root, base, name, output)


def _summary(values: dict[int, float]) -> dict:
    numbers = list(values.values())
    if set(values) != set(SEEDS):
        raise ValueError("Phase C requires all three seeds")
    return {"mean": math.fsum(numbers) / 3, "sample_sd": statistics.stdev(numbers),
            "min": min(numbers), "max": max(numbers),
            "by_seed": {str(seed): values[seed] for seed in SEEDS}}


def _reference_complete(output: Path) -> int:
    rows = [json.loads(line) for line in (output / "generation.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 30:
        raise ValueError(f"Incomplete generation accounting: {output}")
    return sum(row["status"] == "generated" and not any(
        ".sample_refs:" in error or ".facility_refs:" in error
        for error in row.get("schema_errors", [])) for row in rows)


def _saved_control(root: Path, config: PhaseCEvaluationConfig, condition: str) -> dict:
    if condition == "matched_d_dr":
        comparison = _read(root / config.mixed_evaluation_dir / "comparison.json")
        return comparison["fixed_endpoint_conditions"][condition]
    comparison = _read(root / config.d32_evaluation_dir / "comparison.json")
    return comparison["conditions"][condition]


def _verify_reused_adapter(root: Path, manifest: dict, previous: dict,
                           condition: str, seed: int):
    recorded = manifest["model"]["adapter_sha256"]
    expected = previous.get("adapter_sha256")
    adapter = Path(manifest["model"]["adapter"])
    if not adapter.is_absolute():
        adapter = root / adapter
    if expected is None:
        summary = _read(adapter.parent / "training_summary.json")
        expected = (summary.get("selection") or {}).get("adapter_sha256")
    adapter_file = adapter / "adapter_model.safetensors"
    if expected is None or recorded != expected or checksum(adapter_file) != expected:
        raise ValueError(f"Reused adapter checksum differs: {condition} seed {seed}")


def _load_condition(root: Path, config: PhaseCEvaluationConfig, condition: str) -> dict:
    if condition in NEW_CONDITIONS:
        directory = root / config.output_dir
    elif condition == "matched_d_dr":
        directory = root / config.mixed_evaluation_dir
    else:
        directory = root / config.d32_evaluation_dir
    saved = _saved_control(root, config, condition) if condition in CONTROLS else None
    runs, paper_ids, evaluator = {}, None, None
    for seed in SEEDS:
        run_dir = directory / label(condition, seed)
        manifest, metrics = _read(run_dir / "manifest.json"), _read(run_dir / "metrics.json")
        order = [row["paper_id"] for row in metrics["papers"]]
        if len(order) != config.expected_papers or len(set(order)) != config.expected_papers:
            raise ValueError(f"Paper coverage differs: {condition} seed {seed}")
        paper_ids = order if paper_ids is None else paper_ids
        evaluator = metrics["evaluator"] if evaluator is None else evaluator
        if order != paper_ids or manifest["data"]["paper_ids"] != paper_ids:
            raise ValueError("Paper order differs across Phase C runs")
        if metrics["evaluator"] != evaluator or manifest["evaluator"] != evaluator:
            raise ValueError("Evaluator provenance differs across Phase C runs")
        if (manifest.get("condition"), manifest.get("endpoint_policy")) != (
                label(condition, seed), config.endpoint_policy):
            raise ValueError(f"Evaluation manifest differs: {label(condition, seed)}")
        reports = {paper: _read(run_dir / "reports" / f"{paper}.json") for paper in paper_ids}
        for field in CONTRAST_METRICS:
            pooled = prf(add_counts([report_field_counts(reports[paper], field)
                                    for paper in paper_ids]))["f1"]
            stored = metrics["summary"][f"{field}_f1"]
            if stored is None or not math.isclose(pooled, float(stored), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Report/metric counts differ: {condition} {seed} {field}")
        if saved is not None:
            previous = saved["runs"][str(seed)]
            _verify_reused_adapter(root, manifest, previous, condition, seed)
            for key, value in metrics["summary"].items():
                if key in previous and previous[key] != value:
                    raise ValueError(f"Reused metric differs: {condition} seed {seed} {key}")
        runs[seed] = {"directory": run_dir, "manifest": manifest, "metrics": metrics,
                      "reports": reports, "reference_complete": _reference_complete(run_dir)}
    return {"paper_ids": paper_ids, "evaluator": evaluator, "runs": runs}


def paired_contrast(treatment: str, control: str, loaded: dict, paper_ids: list[str], field: str) -> dict:
    by_seed, per_paper = {}, {paper: {} for paper in paper_ids}
    for seed in SEEDS:
        first = loaded[treatment]["runs"][seed]
        second = loaded[control]["runs"][seed]
        by_seed[seed] = (float(first["metrics"]["summary"][f"{field}_f1"])
                         - float(second["metrics"]["summary"][f"{field}_f1"]))
        for paper in paper_ids:
            a, b = report_field_f1(first["reports"][paper], field), report_field_f1(second["reports"][paper], field)
            per_paper[paper][str(seed)] = a - b if a is not None and b is not None else None
    paper_rows = []
    for paper, values_by_seed in per_paper.items():
        values = [value for value in values_by_seed.values() if value is not None]
        paper_rows.append({"paper_id": paper, "delta_f1_by_seed": values_by_seed,
                           "mean_delta_f1": math.fsum(values) / len(values) if values else None,
                           "comparable_seeds": len(values)})
    comparable = [row for row in paper_rows if row["mean_delta_f1"] is not None]
    improved = sum(row["mean_delta_f1"] > 1e-12 for row in comparable)
    tied = sum(abs(row["mean_delta_f1"]) <= 1e-12 for row in comparable)
    worsened = sum(row["mean_delta_f1"] < -1e-12 for row in comparable)
    absolute = math.fsum(abs(row["mean_delta_f1"]) for row in comparable)
    share = max((abs(row["mean_delta_f1"]) for row in comparable), default=0) / absolute if absolute else 0
    return {"treatment": treatment, "control": control,
            "pooled_f1_delta_across_seeds": _summary(by_seed),
            "positive_seeds": sum(value > 1e-12 for value in by_seed.values()),
            "negative_seeds": sum(value < -1e-12 for value in by_seed.values()),
            "paper_direction": {"improved": improved, "tied": tied, "worsened": worsened,
                                "not_comparable": len(paper_ids) - len(comparable)},
            "largest_absolute_paper_share": share, "papers": paper_rows}


def _distributed(values: dict, gate: GateC, *, positive: bool) -> dict:
    direction = values["paper_direction"]
    return {"seed_support": (values["positive_seeds"] if positive else values["negative_seeds"])
                            >= gate.minimum_positive_seeds,
            "paper_direction": ((direction["improved"] - direction["worsened"])
                                if positive else (direction["worsened"] - direction["improved"]))
                               >= gate.minimum_net_improved_papers,
            "concentration": values["largest_absolute_paper_share"]
                             <= gate.maximum_largest_paper_absolute_share}


def apply_gate_c(contrasts: dict, gate: GateC) -> dict:
    mixed = contrasts["r_then_d_minus_matched_d_dr"]
    d32 = contrasts["r_then_d_minus_d32"]
    order = contrasts["d_then_r_minus_r_then_d"]
    shortlist = {
        "mixed_samples_recovery": mixed["samples"]["pooled_f1_delta_across_seeds"]["mean"]
                                  >= gate.sample_recovery_margin_f1,
        "mixed_sample_refs_recovery": mixed["sample_refs"]["pooled_f1_delta_across_seeds"]["mean"]
                                      >= gate.sample_recovery_margin_f1,
    }
    for field in ("facilities", "datasets", "facility_refs"):
        shortlist[f"mixed_{field}_equivalence"] = mixed[field]["pooled_f1_delta_across_seeds"]["mean"] >= -gate.equivalence_margin_f1
    for field in CONTRAST_METRICS:
        shortlist[f"d32_{field}_equivalence"] = d32[field]["pooled_f1_delta_across_seeds"]["mean"] >= -gate.equivalence_margin_f1
    for field in ("samples", "sample_refs"):
        for key, value in _distributed(mixed[field], gate, positive=True).items():
            shortlist[f"mixed_{field}_{key}"] = value
    late_fields = {}
    for field in ("samples", "sample_refs"):
        checks = {"margin": order[field]["pooled_f1_delta_across_seeds"]["mean"]
                            <= -gate.late_interference_margin_f1,
                  **_distributed(order[field], gate, positive=False)}
        late_fields[field] = {"checks": checks, "passes": all(checks.values())}
    shortlist_passes = all(shortlist.values())
    late_passes = any(values["passes"] for values in late_fields.values())
    outcome = ("shortlist_r_then_d" if shortlist_passes else
               "late_interference_only" if late_passes else "stop_order_experiments")
    return {"outcome": outcome, "shortlist_r_then_d": shortlist_passes,
            "late_interference": late_passes, "shortlist_checks": shortlist,
            "late_interference_fields": late_fields,
            "thresholds": gate.model_dump(exclude={"decision_rules"}),
            "decision_rule_text": gate.decision_rules}


def compare(root: Path, config: PhaseCEvaluationConfig, phase: PhaseCConfig) -> dict:
    loaded = {condition: _load_condition(root, config, condition) for condition in CONDITIONS}
    paper_ids, evaluator = loaded["d32"]["paper_ids"], loaded["d32"]["evaluator"]
    if any(item["paper_ids"] != paper_ids or item["evaluator"] != evaluator for item in loaded.values()):
        raise ValueError("Phase C conditions do not share paper/evaluator provenance")
    conditions = {}
    for condition, values in loaded.items():
        runs = {str(seed): {**values["runs"][seed]["metrics"]["summary"],
                            "papers_reference_complete": values["runs"][seed]["reference_complete"],
                            "adapter_sha256": values["runs"][seed]["manifest"]["model"]["adapter_sha256"]}
                for seed in SEEDS}
        conditions[condition] = {"runs": runs,
            "aggregate": {metric: _summary({seed: float(runs[str(seed)][metric] or 0)
                                             for seed in SEEDS})
                          for metric in (*SUMMARY_METRICS, "papers_reference_complete")}}
    definitions = (("r_then_d", "matched_d_dr"), ("r_then_d", "d32"),
                   ("d_then_r", "r_then_d"))
    contrasts = {f"{first}_minus_{second}": {
        field: paired_contrast(first, second, loaded, paper_ids, field)
        for field in CONTRAST_METRICS} for first, second in definitions}
    gate = apply_gate_c(contrasts, phase.gate_c)
    result = {"version": config.version, "phase": config.phase, "study_type": config.study_type,
              "corpus_role": config.corpus_role, "paper_ids": paper_ids, "evaluator": evaluator,
              "conditions": conditions, "contrasts": contrasts, "gate_c": gate,
              "accounting": {"new_generations": 180, "reused_control_generations": 180,
                             "independent_test_papers": 0},
              "caveats": ["The thirty papers are reused development data, not an independent test.",
                          "Phase C was separately authorized post-hoc after frozen Gate B was inconclusive.",
                          "Gate C thresholds were frozen before sequential training.",
                          "No Phase C result changes the original M1 or Gate B decisions."]}
    output = root / config.output_dir
    write_json(output / "comparison.json", result)
    write_markdown(output / "comparison.md", result)
    return result


def write_markdown(path: Path, result: dict):
    lines = ["# Step 9 Phase C: R/D training order", "",
             "Post-hoc exploratory results on thirty reused development papers; values are mean ± sample standard deviation over three seeds.", "",
             "| Condition | Full schema | Facilities | Samples | Datasets | Sample refs | Facility refs |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        aggregate = result["conditions"][condition]["aggregate"]
        show = lambda key, digits=3: f"{aggregate[key]['mean']:.{digits}f} ± {aggregate[key]['sample_sd']:.{digits}f}"
        lines.append(f"| `{condition}` | {show('papers_schema_valid', 1)}/30 | {show('facilities_f1')} | "
                     f"{show('samples_f1')} | {show('datasets_f1')} | {show('sample_refs_f1')} | "
                     f"{show('facility_refs_f1')} |")
    for name, contrast in result["contrasts"].items():
        lines += ["", f"## `{name}`", "", "| Field | Mean ΔF1 | Seed SD | Positive/negative seeds | Papers +/=/-/NA | Largest share |",
                  "|---|---:|---:|---:|---:|---:|"]
        for field in CONTRAST_METRICS:
            value, direction = contrast[field], contrast[field]["paper_direction"]
            summary = value["pooled_f1_delta_across_seeds"]
            lines.append(f"| {FIELD_SPECS[field]['label']} | {summary['mean']:+.3f} | {summary['sample_sd']:.3f} | "
                         f"{value['positive_seeds']}/{value['negative_seeds']} | {direction['improved']}/{direction['tied']}/"
                         f"{direction['worsened']}/{direction['not_comparable']} | {value['largest_absolute_paper_share']:.1%} |")
    gate = result["gate_c"]
    lines += ["", "## Frozen Gate C interpretation", "", f"**Outcome: `{gate['outcome']}`.**", "",
              "All checks and thresholds are retained in `comparison.json`.", ""]
    lines.extend(f"- {item}" for item in result["caveats"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def plan(config: PhaseCEvaluationConfig, phase: PhaseCConfig) -> list[dict]:
    return [{"condition": condition, "seed": seed,
             "checkpoint": f"{phase.output_root}/{label(condition, seed)}/checkpoint-8",
             "output": f"{config.output_dir}/{label(condition, seed)}", "papers": config.expected_papers}
            for condition in NEW_CONDITIONS for seed in SEEDS]
