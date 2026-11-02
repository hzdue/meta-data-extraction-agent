#!/usr/bin/env python
"""Aggregate the completed exploratory ablation matrix without test-set evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics

from mlflow import MlflowClient
import yaml

from meta_data_extraction.finetune.validation import write_json


ROOT = Path(__file__).resolve().parents[1]
ENTITY_FIELDS = ("facilities", "samples", "datasets")
REFERENCE_FIELDS = ("sample_refs", "facility_refs")
CONTRASTS = {
    "matched_d_de_minus_d_only": ("matched_d_de", "d_only"),
    "matched_d_dr_minus_d_only": ("matched_d_dr", "d_only"),
    "matched_d_der_minus_d_only": ("matched_d_der", "d_only"),
    "matched_d_drp_minus_d_only": ("matched_d_drp", "d_only"),
    "matched_d_drp_minus_matched_d_dr": ("matched_d_drp", "matched_d_dr"),
    "matched_compute_de_minus_d_only": ("matched_compute_de", "d_only"),
    "matched_compute_dr_minus_d_only": ("matched_compute_dr", "d_only"),
    "matched_compute_der_minus_d_only": ("matched_compute_der", "d_only"),
}
OUTCOME_METRICS = (
    "schema_valid_papers", "reference_complete_papers", "mean_reference_f1",
    "sample_refs_f1", "facility_refs_f1",
    "facilities_f1", "samples_f1", "datasets_f1", "validation_loss",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty metric")
    return math.fsum(values) / len(values)


def endpoint_metrics(entry: dict) -> dict:
    diagnostics = entry["diagnostics"]
    if len(diagnostics) != 2 or len({row["paper_id"] for row in diagnostics}) != 2:
        raise ValueError("Every endpoint must contain two distinct validation papers")
    reference = {field: [] for field in REFERENCE_FIELDS}
    entities = {field: [] for field in ENTITY_FIELDS}
    schema_errors = []
    reference_complete_papers = 0
    for row in diagnostics:
        row_errors = row.get("schema_errors") or []
        schema_errors.extend(row_errors)
        reference_errors = [error for error in row_errors
                            if ".sample_refs:" in error or ".facility_refs:" in error]
        reference_complete_papers += int(row.get("status") == "scored" and not reference_errors)
        scores = row.get("conditional_reference_f1") or {}
        for field in REFERENCE_FIELDS:
            value = (scores.get(field) or {}).get("f1")
            reference[field].append(0.0 if value is None else float(value))
        scores = row.get("entity_f1") or {}
        for field in ENTITY_FIELDS:
            entities[field].append(float(scores.get(field, 0.0)))
    reference_values = [value for values in reference.values() for value in values]
    return {
        "validation_loss": float(entry["loss"]),
        "json_valid_papers": sum(row.get("json_valid") is True for row in diagnostics),
        "schema_valid_papers": sum(row.get("status") == "scored" and row.get("schema_valid") is True
                                   for row in diagnostics),
        "reference_complete_papers": reference_complete_papers,
        "scored_papers": sum(row.get("status") == "scored" for row in diagnostics),
        "generation_failures": sum(row.get("status") != "scored" for row in diagnostics),
        "hit_output_limit_papers": sum(row.get("hit_output_limit") is True for row in diagnostics),
        "mean_reference_f1": mean(reference_values),
        **{f"{field}_f1": mean(values) for field, values in reference.items()},
        **{f"{field}_f1": mean(values) for field, values in entities.items()},
        "generated_tokens": sum(int(row["generated_tokens"]) for row in diagnostics),
        "schema_errors": schema_errors,
    }


def summarize(by_seed: dict[int, float]) -> dict:
    if len(by_seed) != 3:
        raise ValueError("Every condition requires exactly three seeds")
    values = list(by_seed.values())
    return {
        "mean": mean(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "by_seed": {str(seed): value for seed, value in sorted(by_seed.items())},
    }


def mlflow_resources(record: dict) -> dict:
    run = MlflowClient(tracking_uri=record["settings"]["tracking_uri"]).get_run(record["run_id"])
    if run.info.status != "FINISHED":
        raise ValueError(f"MLflow run is not FINISHED: {record['run_id']}")
    peaks = [value for key, value in run.data.metrics.items()
             if key.endswith("/final/peak_allocated_bytes")]
    reserved = [value for key, value in run.data.metrics.items()
                if key.endswith("/final/peak_reserved_bytes")]
    if not peaks or not reserved:
        raise ValueError(f"MLflow resource metrics are incomplete: {record['run_id']}")
    return {"run_id": record["run_id"], "status": run.info.status,
            "peak_allocated_gib": max(peaks) / 2**30,
            "peak_reserved_gib": max(reserved) / 2**30}


def load_run(root: Path, condition: str, family: str, exposures: dict, seed: int) -> dict:
    directory = root / "outputs/finetune/ablations" / f"{condition}_seed{seed}"
    summary = read_json(directory / "training_summary.json")
    state = read_json(directory / "validation_history.json")
    selection = read_json(directory / "selection.json")
    record = read_json(directory / "mlflow_run.json")
    endpoint = sum(exposures.values()) // 4
    ablation = summary.get("ablation") or {}
    if (summary.get("global_step"), ablation.get("condition"), ablation.get("seed"),
            ablation.get("planned_exposure")) != (endpoint, condition, seed, exposures):
        raise ValueError(f"Run contract differs from matrix: {condition} seed {seed}")
    if (summary.get("exposure") or {}).get("examples") != exposures:
        raise ValueError(f"Actual exposure differs from matrix: {condition} seed {seed}")
    history = {row["step"]: row for row in state["history"]}
    if 0 not in history or endpoint not in history:
        raise ValueError(f"Base or endpoint validation is missing: {condition} seed {seed}")
    if selection.get("status") != "complete" or selection.get("step") not in {0, endpoint}:
        raise ValueError(f"Selection is incomplete or not an eligible endpoint: {condition} seed {seed}")
    selected = history[selection["step"]]
    if not math.isclose(selection["loss"], selected["loss"], rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Selection/history loss differs: {condition} seed {seed}")
    resources = mlflow_resources(record)
    endpoint_result = endpoint_metrics(history[endpoint])
    base_result = endpoint_metrics(history[0])
    selected_result = endpoint_metrics(selected)
    exposure = summary["exposure"]
    return {
        "condition": condition, "family": family, "seed": seed,
        "endpoint_step": endpoint, "planned_exposure": exposures,
        "endpoint_selected": selection["step"] == endpoint,
        "selected_step": selection["step"], "selection_kind": selection["kind"],
        "base": base_result, "endpoint": endpoint_result, "selected": selected_result,
        "cost": {
            "runtime_seconds": float(summary["metrics"]["train_runtime"]),
            "supervised_tokens": sum(exposure["supervised_tokens"].values()),
            "total_tokens": sum(exposure["total_tokens"].values()),
            **resources,
        },
    }


def aggregate_condition(runs: list[dict]) -> dict:
    seeds = {run["seed"] for run in runs}
    if seeds != {42, 123, 456}:
        raise ValueError("Condition does not contain the three frozen seeds")
    endpoint_metrics_names = [key for key, value in runs[0]["endpoint"].items()
                              if isinstance(value, (int, float))]
    selected_metrics_names = [key for key, value in runs[0]["selected"].items()
                              if isinstance(value, (int, float))]
    cost_metrics_names = [key for key, value in runs[0]["cost"].items()
                          if isinstance(value, (int, float))]
    return {
        "family": runs[0]["family"], "planned_exposure": runs[0]["planned_exposure"],
        "endpoint_step": runs[0]["endpoint_step"],
        "endpoint_selected_count": sum(run["endpoint_selected"] for run in runs),
        "endpoint_selected_seeds": sorted(run["seed"] for run in runs if run["endpoint_selected"]),
        "endpoint": {metric: summarize({run["seed"]: run["endpoint"][metric] for run in runs})
                     for metric in endpoint_metrics_names},
        "selected": {metric: summarize({run["seed"]: run["selected"][metric] for run in runs})
                     for metric in selected_metrics_names},
        "cost": {metric: summarize({run["seed"]: run["cost"][metric] for run in runs})
                 for metric in cost_metrics_names},
    }


def paired_contrast(runs_by_condition: dict[str, list[dict]], treatment: str, control: str) -> dict:
    treatment_runs = {run["seed"]: run for run in runs_by_condition[treatment]}
    control_runs = {run["seed"]: run for run in runs_by_condition[control]}
    if set(treatment_runs) != set(control_runs) or set(treatment_runs) != {42, 123, 456}:
        raise ValueError("Paired contrast seeds differ")
    result = {}
    for metric in OUTCOME_METRICS:
        deltas = {seed: treatment_runs[seed]["endpoint"][metric]
                  - control_runs[seed]["endpoint"][metric] for seed in treatment_runs}
        result[metric] = summarize(deltas)
    return {"treatment": treatment, "control": control, "definition": "fixed endpoint, paired by seed",
            "metrics": result}


def normalized_schema_errors(runs: list[dict]) -> dict:
    errors = Counter()
    for run in runs:
        for error in run["endpoint"]["schema_errors"]:
            errors[re.sub(r"\[[0-9]+\]", "[]", error)] += 1
    return dict(errors.most_common())


def schema_errors_by_condition(runs_by_condition: dict[str, list[dict]]) -> dict:
    return {condition: normalized_schema_errors(runs)
            for condition, runs in runs_by_condition.items()}


def fmt(summary: dict, digits=3) -> str:
    return f"{summary['mean']:.{digits}f} ± {summary['sample_sd']:.{digits}f}"


def exposure_text(exposure: dict) -> str:
    return "+".join(f"{task}{count}" for task, count in exposure.items())


def markdown_report(analysis: dict) -> str:
    lines = [
        "# Exploratory fine-tuning ablation matrix",
        "",
        "All values below use the predeclared fixed-exposure endpoint. Results are descriptive over "
        "three seeds and two validation papers; the thirty-paper Step 7 test set was not evaluated.",
        "",
        "## Fixed-endpoint outcomes",
        "",
        "| Condition | Exposure | Step | Endpoint selected | Full schema | Reference fields | Reference F1 | Facilities F1 | Samples F1 | Datasets F1 | Validation NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, row in analysis["conditions"].items():
        endpoint = row["endpoint"]
        schema_total = sum(endpoint["schema_valid_papers"]["by_seed"].values())
        reference_total = sum(endpoint["reference_complete_papers"]["by_seed"].values())
        lines.append(
            f"| `{condition}` | {exposure_text(row['planned_exposure'])} | {row['endpoint_step']} | "
            f"{row['endpoint_selected_count']}/3 | {schema_total:.0f}/6 | {reference_total:.0f}/6 | "
            f"{fmt(endpoint['mean_reference_f1'])} | {fmt(endpoint['facilities_f1'])} | "
            f"{fmt(endpoint['samples_f1'])} | {fmt(endpoint['datasets_f1'])} | "
            f"{fmt(endpoint['validation_loss'], 4)} |"
        )
    lines += [
        "", "Reference F1 is the same selection-compatible macro used during training: missing or "
        "undefined paper/field scores count as zero.",
        "", "## Cost", "",
        "| Condition | Runtime (min) | Supervised tokens | Total tokens | Peak allocated VRAM (GiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, row in analysis["conditions"].items():
        cost = row["cost"]
        runtime = {**cost["runtime_seconds"],
                   "mean": cost["runtime_seconds"]["mean"] / 60,
                   "sample_sd": cost["runtime_seconds"]["sample_sd"] / 60}
        lines.append(
            f"| `{condition}` | {fmt(runtime, 1)} | {cost['supervised_tokens']['mean']:.0f} | "
            f"{cost['total_tokens']['mean']:.0f} | {fmt(cost['peak_allocated_gib'], 2)} |"
        )
    lines += [
        "", "## Paired fixed-endpoint contrasts", "",
        "| Contrast | Δ schema-valid papers | Δ reference-complete papers | Δ reference F1 | Δ facilities F1 | Δ samples F1 | Δ datasets F1 | Δ validation NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in analysis["paired_contrasts"].items():
        metrics = row["metrics"]
        lines.append(
            f"| `{name}` | {metrics['schema_valid_papers']['mean']:+.3f} | "
            f"{metrics['reference_complete_papers']['mean']:+.3f} | "
            f"{metrics['mean_reference_f1']['mean']:+.3f} | "
            f"{metrics['facilities_f1']['mean']:+.3f} | {metrics['samples_f1']['mean']:+.3f} | "
            f"{metrics['datasets_f1']['mean']:+.3f} | {metrics['validation_loss']['mean']:+.4f} |"
        )
    lines += [
        "", "## Accounting", "",
        f"- Runs: {analysis['accounting']['runs']} complete ({analysis['accounting']['conditions']} conditions × "
        f"{analysis['accounting']['seeds_per_condition']} seeds).",
        f"- Endpoint validations: {analysis['accounting']['endpoint_validation_outputs']} paper outputs.",
        f"- JSON-valid endpoint outputs: {analysis['accounting']['json_valid_endpoint_outputs']}.",
        f"- Reference-complete endpoint outputs: {analysis['accounting']['reference_complete_endpoint_outputs']}.",
        f"- Generation failures: {analysis['accounting']['generation_failures']}.",
        f"- Output-limit failures: {analysis['accounting']['output_limit_failures']}.",
        f"- Trained endpoint selected: {analysis['accounting']['endpoint_selected_runs']}/24 runs.",
        "- Independent test evaluations: 0.", "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/finetune_ablation.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/finetune/ablations/analysis.json"))
    parser.add_argument("--markdown", type=Path,
                        default=Path("outputs/finetune/ablations/analysis.md"))
    args = parser.parse_args(argv)
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if config.get("independent_test_split") is not None or config.get("allow_current_test_evaluation") is not False:
        raise ValueError("This analyzer is restricted to the declared exploratory validation matrix")
    seeds = config["seeds"]
    if seeds != [42, 123, 456]:
        raise ValueError("Seed contract changed")
    runs = []
    runs_by_condition = {}
    for condition, declaration in config["conditions"].items():
        condition_runs = [load_run(ROOT, condition, declaration["family"], declaration["exposures"], seed)
                          for seed in seeds]
        runs.extend(condition_runs)
        runs_by_condition[condition] = condition_runs
    if len(runs) != 24:
        raise ValueError("Expected the complete 24-run matrix")
    base = runs[0]["base"]
    base_numeric = {key: value for key, value in base.items() if isinstance(value, (int, float))}
    for run in runs[1:]:
        candidate = {key: value for key, value in run["base"].items() if isinstance(value, (int, float))}
        if candidate.keys() != base_numeric.keys() or any(
                not math.isclose(candidate[key], base_numeric[key], rel_tol=0, abs_tol=1e-12)
                for key in base_numeric):
            raise ValueError("Step-zero baseline differs across runs")
    analysis = {
        "version": "1.0.0", "study_type": "exploratory",
        "scope": "two held-out validation papers; no Step 7 test evaluation",
        "seeds": seeds, "baseline": base,
        "conditions": {condition: aggregate_condition(condition_runs)
                       for condition, condition_runs in runs_by_condition.items()},
        "paired_contrasts": {name: paired_contrast(runs_by_condition, treatment, control)
                             for name, (treatment, control) in CONTRASTS.items()},
        "schema_error_classes": normalized_schema_errors(runs),
        "schema_error_classes_by_condition": schema_errors_by_condition(runs_by_condition),
        "accounting": {
            "runs": len(runs), "conditions": len(runs_by_condition), "seeds_per_condition": len(seeds),
            "endpoint_validation_outputs": 2 * len(runs),
            "json_valid_endpoint_outputs": sum(run["endpoint"]["json_valid_papers"] for run in runs),
            "schema_valid_endpoint_outputs": sum(run["endpoint"]["schema_valid_papers"] for run in runs),
            "reference_complete_endpoint_outputs": sum(
                run["endpoint"]["reference_complete_papers"] for run in runs),
            "generation_failures": sum(run["endpoint"]["generation_failures"] for run in runs),
            "output_limit_failures": sum(run["endpoint"]["hit_output_limit_papers"] for run in runs),
            "endpoint_selected_runs": sum(run["endpoint_selected"] for run in runs),
            "independent_test_evaluations": 0,
        },
        "runs": runs,
    }
    write_json(ROOT / args.output, analysis)
    markdown = markdown_report(analysis)
    markdown_path = ROOT / args.markdown
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    temporary.write_text(markdown + "\n", encoding="utf-8")
    temporary.replace(markdown_path)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
