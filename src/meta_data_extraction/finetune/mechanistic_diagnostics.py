"""Exploratory reference-error decomposition and equal-update oracle comparison."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from .ablation import read_ablation_config
from .ablation_evaluation import endpoint_adapter
from .followup_analysis import add_counts, combined_sha256, prf
from .followup_evaluation import (
    REFERENCE_FIELDS, _match_map, prepare_context, prepare_rows,
)
from .followup_phase_b_evaluation import (
    d32_endpoint, read_phase_b_evaluation_config,
)
from .followup_training import read_phase_b_config
from .tracking import checksum
from .training_data import file_hash
from .validation import write_json


SEEDS = (42, 123, 456)
CONDITIONS = ("d_only", "matched_d_dr", "d32")
FIELDS = ("sample_refs", "facility_refs")


class ErrorDecompositionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    conditions: list[str]
    fields: list[str]

    @model_validator(mode="after")
    def fixed_scope(self):
        if self.conditions != list(CONDITIONS) or self.fields != list(FIELDS):
            raise ValueError("Mechanistic error decomposition scope changed")
        return self


class OracleComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reused_condition: Literal["matched_d_dr"]
    generated_conditions: list[str]
    primary_contrast: list[str]
    secondary_contrast: list[str]
    inventory: Literal["oracle"]
    endpoint_policy: Literal["fixed_exposure"]
    bootstrap_samples: int
    bootstrap_seed: int
    attach_mlflow: Literal[False]
    model_selection_allowed: Literal[False]

    @model_validator(mode="after")
    def fixed_scope(self):
        if self.generated_conditions != ["d32", "d_only"]:
            raise ValueError("Oracle generation must add only D32 and D24")
        if self.primary_contrast != ["matched_d_dr", "d32"]:
            raise ValueError("Primary oracle contrast must be D24+R8 minus D32")
        if self.secondary_contrast != ["matched_d_dr", "d_only"]:
            raise ValueError("Secondary oracle contrast must be D24+R8 minus D24")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        return self


class MechanisticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    study_type: Literal["exploratory_mechanistic_diagnostic"]
    corpus_role: Literal["exploratory_reused_step7_test_as_development"]
    seeds: list[int]
    expected_papers: Literal[30]
    phase_a2_config: str
    phase_a2_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ablation_config: str
    ablation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_b_config: str
    phase_b_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_b_evaluation_config: str
    phase_b_evaluation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_evaluation_dir: str
    d32_evaluation_dir: str
    existing_oracle_dir: str
    output_dir: str
    error_decomposition: ErrorDecompositionConfig
    oracle_comparison: OracleComparisonConfig

    @model_validator(mode="after")
    def fixed_scope(self):
        if self.seeds != list(SEEDS):
            raise ValueError("Mechanistic diagnostic seed order changed")
        return self


def read_config(path: Path) -> MechanisticConfig:
    return MechanisticConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def verify_frozen_configs(root: Path, config: MechanisticConfig) -> None:
    for path, expected in (
        (config.phase_a2_config, config.phase_a2_config_sha256),
        (config.ablation_config, config.ablation_config_sha256),
        (config.phase_b_config, config.phase_b_config_sha256),
        (config.phase_b_evaluation_config, config.phase_b_evaluation_config_sha256),
    ):
        if file_hash(root / path) != expected:
            raise ValueError(f"Frozen mechanistic input changed: {path}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _condition_directory(root: Path, config: MechanisticConfig,
                         condition: str, seed: int) -> Path:
    base = config.d32_evaluation_dir if condition == "d32" else config.control_evaluation_dir
    return root / base / f"{condition}_seed{seed}"


def reference_error_decomposition(gold: dict, prediction: dict, report: dict,
                                  field: str) -> dict:
    """Partition end-to-end link errors into inventory and linking causes."""
    if field not in FIELDS:
        raise ValueError(f"Unknown reference field: {field}")
    output = prediction.get("output", prediction)
    target_collection = "samples" if field == "sample_refs" else "facilities"
    datasets = output.get("datasets")
    targets = output.get(target_collection)
    if not isinstance(datasets, list) or not isinstance(targets, list):
        raise ValueError("Prediction is missing entity collections")

    dataset_map = _match_map(report, "datasets")
    target_map = _match_map(report, target_collection)
    aligned_gold_datasets = set(dataset_map.values())
    aligned_gold_targets = set(target_map.values())
    gold_links = {
        (dataset["id"], target)
        for dataset in gold.get("datasets", [])
        for target in dataset.get(field, [])
    }

    mapped_links = set()
    fp = {"unaligned_dataset": 0, "unaligned_target": 0, "wrong_link": 0}
    seen_predicted = set()
    for dataset in datasets:
        dataset_id = dataset.get("id")
        values = dataset.get(field)
        if not isinstance(values, list):
            raise ValueError(f"Prediction dataset is missing list field {field}")
        for target_id in values:
            predicted = (dataset_id, target_id)
            if predicted in seen_predicted:
                continue
            seen_predicted.add(predicted)
            if dataset_id not in dataset_map:
                fp["unaligned_dataset"] += 1
            elif target_id not in target_map:
                fp["unaligned_target"] += 1
            else:
                mapped = (dataset_map[dataset_id], target_map[target_id])
                mapped_links.add(mapped)
                if mapped not in gold_links:
                    fp["wrong_link"] += 1

    true_positives = len(mapped_links & gold_links)
    fn = {
        "dataset_and_target_missing": 0,
        "dataset_missing_only": 0,
        "target_missing_only": 0,
        "link_error_with_entities_present": 0,
    }
    for dataset_id, target_id in gold_links - mapped_links:
        dataset_missing = dataset_id not in aligned_gold_datasets
        target_missing = target_id not in aligned_gold_targets
        if dataset_missing and target_missing:
            fn["dataset_and_target_missing"] += 1
        elif dataset_missing:
            fn["dataset_missing_only"] += 1
        elif target_missing:
            fn["target_missing_only"] += 1
        else:
            fn["link_error_with_entities_present"] += 1

    inventory_limited = sum(value for key, value in fn.items()
                            if key != "link_error_with_entities_present")
    false_negatives = sum(fn.values())
    false_positives = sum(fp.values())
    total_gold = len(gold_links)
    return {
        "gold_links": total_gold,
        "predicted_unique_links": len(seen_predicted),
        "true_positive_links": true_positives,
        "false_positive_links": false_positives,
        "false_negative_links": false_negatives,
        "fp_unaligned_dataset": fp["unaligned_dataset"],
        "fp_unaligned_target": fp["unaligned_target"],
        "fp_wrong_link": fp["wrong_link"],
        "fn_dataset_and_target_missing": fn["dataset_and_target_missing"],
        "fn_dataset_missing_only": fn["dataset_missing_only"],
        "fn_target_missing_only": fn["target_missing_only"],
        "fn_link_error_with_entities_present": fn["link_error_with_entities_present"],
        "fn_inventory_limited": inventory_limited,
        "recoverable_gold_links": total_gold - inventory_limited,
    }


COUNT_KEYS = (
    "gold_links", "predicted_unique_links", "true_positive_links",
    "false_positive_links", "false_negative_links", "fp_unaligned_dataset",
    "fp_unaligned_target", "fp_wrong_link", "fn_dataset_and_target_missing",
    "fn_dataset_missing_only", "fn_target_missing_only",
    "fn_link_error_with_entities_present", "fn_inventory_limited",
    "recoverable_gold_links",
)


def _summarize_decomposition(rows: list[dict]) -> dict:
    totals = {key: sum(row[key] for row in rows) for key in COUNT_KEYS}
    gold = totals["gold_links"]
    recoverable = totals["recoverable_gold_links"]
    totals.update({
        "observed_recall": totals["true_positive_links"] / gold if gold else 1.0,
        "inventory_recall_ceiling": recoverable / gold if gold else 1.0,
        "inventory_limited_fraction_of_false_negatives": (
            totals["fn_inventory_limited"] / totals["false_negative_links"]
            if totals["false_negative_links"] else 0.0
        ),
        "link_recall_when_entities_available": (
            totals["true_positive_links"] / recoverable if recoverable else 1.0
        ),
    })
    return totals


def build_error_decomposition(root: Path, config_path: Path,
                              config: MechanisticConfig) -> dict:
    verify_frozen_configs(root, config)
    papers, source_paths = [], []
    aggregates: dict[str, dict[str, list[dict]]] = {
        condition: {field: [] for field in FIELDS} for condition in CONDITIONS
    }
    expected_ids = expected_evaluator = None
    for condition in CONDITIONS:
        for seed in SEEDS:
            directory = _condition_directory(root, config, condition, seed)
            manifest = read_json(directory / "manifest.json")
            generation = read_jsonl(directory / "generation.jsonl")
            paper_ids = manifest["data"]["paper_ids"]
            if (manifest.get("condition") != f"{condition}_seed{seed}"
                    or [row["paper_id"] for row in generation] != paper_ids
                    or len(paper_ids) != config.expected_papers):
                raise ValueError(f"Incomplete or mismatched diagnostic source: {condition} seed {seed}")
            expected_ids = paper_ids if expected_ids is None else expected_ids
            if paper_ids != expected_ids:
                raise ValueError("Diagnostic conditions use different papers")
            current_evaluator = manifest.get("evaluator")
            expected_evaluator = current_evaluator if expected_evaluator is None else expected_evaluator
            if current_evaluator != expected_evaluator:
                raise ValueError("Diagnostic conditions use different evaluator provenance")
            generated = {row["paper_id"]: row for row in generation}
            for paper_id in paper_ids:
                prediction_path = directory / generated[paper_id]["prediction_file"]
                report_path = directory / "reports" / f"{paper_id}.json"
                gold_path = root / "data/gold" / f"{paper_id}.json"
                if checksum(prediction_path) != generated[paper_id]["prediction_sha256"]:
                    raise ValueError(f"Saved prediction changed: {condition} {seed} {paper_id}")
                prediction, report, gold = map(read_json, (prediction_path, report_path, gold_path))
                reported_prediction = Path(report["pred"])
                if not reported_prediction.is_absolute():
                    reported_prediction = root / reported_prediction
                if reported_prediction.resolve() != prediction_path.resolve():
                    raise ValueError(f"Report/prediction mismatch: {condition} {seed} {paper_id}")
                fields = {}
                for field in FIELDS:
                    row = reference_error_decomposition(gold, prediction, report, field)
                    aggregates[condition][field].append(row)
                    fields[field] = row
                papers.append({"condition": condition, "seed": seed,
                               "paper_id": paper_id, "fields": fields})
                source_paths.extend((prediction_path, report_path, gold_path))
            source_paths.extend((directory / "manifest.json", directory / "generation.jsonl"))

    summary = {
        condition: {field: _summarize_decomposition(aggregates[condition][field])
                    for field in FIELDS}
        for condition in CONDITIONS
    }
    return {
        "version": config.version,
        "study_type": config.study_type,
        "corpus_role": config.corpus_role,
        "analysis": "existing one-call prediction reference-error decomposition",
        "conditions": list(CONDITIONS), "seeds": list(SEEDS),
        "paper_ids": expected_ids, "summary": summary, "papers": papers,
        "accounting": {"run_paper_outputs": len(papers), "new_generations": 0,
                       "independent_test_papers": 0},
        "definitions": {
            "inventory_limited": "gold link cannot be recovered because its dataset, target entity, or both have no evaluator-aligned prediction",
            "link_error_with_entities_present": "gold dataset and target entity are both aligned, but their correct link is absent",
            "inventory_recall_ceiling": "maximum recall if all links among the currently aligned entities were predicted correctly",
        },
        "provenance": {
            "config": str(config_path.relative_to(root)),
            "config_sha256": file_hash(config_path),
            "source_files": len(set(source_paths)),
            "source_bundle_sha256": combined_sha256(source_paths, root=root),
        },
        "caveats": [
            "The thirty papers are reused development data, not an independent test.",
            "Evaluator alignment is treated as the entity correspondence authority.",
            "The mutually exclusive false-negative categories describe necessary missing components, not causal attribution.",
            "Gold-inventory performance may also benefit from training-like IDs, ordering and attributes.",
        ],
    }


def error_markdown(result: dict) -> str:
    lines = [
        "# Mechanistic reference-error decomposition", "",
        "This exploratory analysis uses existing one-call predictions and performs no generation or training.", "",
        "| Condition | Field | Gold links | Observed recall | Inventory ceiling | FN inventory-limited | FN link error with entities present | FP unaligned dataset | FP unaligned target | FP wrong link |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"d_only": "D24", "matched_d_dr": "D24+R8", "d32": "D32",
              "sample_refs": "Sample refs", "facility_refs": "Facility refs"}
    for condition in CONDITIONS:
        for field in FIELDS:
            row = result["summary"][condition][field]
            lines.append(
                f"| {labels[condition]} | {labels[field]} | {row['gold_links']} | "
                f"{row['observed_recall']:.3f} | {row['inventory_recall_ceiling']:.3f} | "
                f"{row['fn_inventory_limited']} | {row['fn_link_error_with_entities_present']} | "
                f"{row['fp_unaligned_dataset']} | {row['fp_unaligned_target']} | "
                f"{row['fp_wrong_link']} |"
            )
    lines += ["", "False-negative inventory categories are mutually exclusive: both dataset and target missing, dataset only, target only, or link error with both present.", "",
              "## Caveats", ""]
    lines.extend(f"- {item}" for item in result["caveats"])
    return "\n".join(lines) + "\n"


def endpoint(root: Path, config: MechanisticConfig, condition: str, seed: int):
    matrix = read_ablation_config(root / config.ablation_config)
    if condition == "d_only":
        return endpoint_adapter(root, matrix, condition, seed)
    if condition == "d32":
        phase_b = read_phase_b_config(root / config.phase_b_config)
        evaluation = read_phase_b_evaluation_config(root / config.phase_b_evaluation_config)
        return d32_endpoint(root, phase_b, evaluation, seed)
    raise ValueError(f"Oracle generation is not allowed for condition: {condition}")


def oracle_output(root: Path, config: MechanisticConfig, condition: str, seed: int) -> Path:
    return root / config.output_dir / "oracle" / f"{condition}_seed{seed}"


def oracle_metrics_path(root: Path, config: MechanisticConfig,
                        condition: str, seed: int) -> Path:
    if condition == config.oracle_comparison.reused_condition:
        return root / config.existing_oracle_dir / f"oracle_seed{seed}" / "metrics.json"
    return oracle_output(root, config, condition, seed) / "metrics.json"


def oracle_contract(root: Path, config_path: Path, config: MechanisticConfig,
                    phase_a2, context: dict, tokenizer_identity: dict,
                    condition: str, seed: int, rows: list[dict],
                    training, checkpoint: Path, identity: dict,
                    evaluator: dict) -> dict:
    sources = [config_path, root / config.phase_a2_config, root / config.ablation_config,
               root / config.phase_b_config, root / config.phase_b_evaluation_config,
               root / "src/meta_data_extraction/finetune/mechanistic_diagnostics.py",
               root / "src/meta_data_extraction/finetune/followup_evaluation.py",
               root / "src/meta_data_extraction/evaluation.py",
               root / "src/meta_data_extraction/alignment.py"]
    return {
        "version": config.version, "phase": "mechanistic_oracle",
        "study_type": config.study_type, "corpus_role": config.corpus_role,
        "condition": f"oracle_{condition}_seed{seed}", "inventory_type": "oracle",
        "endpoint_policy": config.oracle_comparison.endpoint_policy,
        "training": identity,
        "model": {"model_id": context["prepare"]["model_id"],
                  "revision": context["prepare"]["revision"],
                  "precision": training.precision_policy(),
                  "adapter": str(checkpoint.resolve()),
                  "adapter_sha256": identity["adapter_sha256"]},
        "decoding": {"method": phase_a2.decoding,
                     "enable_thinking": phase_a2.enable_thinking,
                     "max_new_tokens": phase_a2.max_new_tokens,
                     "max_seq_length": phase_a2.max_seq_length,
                     "seed": seed, "num_beams": 1, "repetition_penalty": 1.0},
        "data": {"source_split": "step7_test",
                 "analysis_role": "exploratory_development",
                 "paper_ids": context["paper_ids"],
                 "gold_entity_inventory_in_prompt": True,
                 "gold_read_for_scoring": True,
                 "prompts": [{key: row[key] for key in
                              ("paper_id", "prompt_tokens", "prompt_sha256",
                               "text_sha256", "inventory_sha256")}
                             for row in rows],
                 "preparation_manifest_sha256": context["preparation_manifest_sha256"],
                 "prior_results_influenced_design": True},
        "tokenizer": tokenizer_identity,
        "evaluator": evaluator,
        "mlflow_attachment_allowed": False,
        "model_selection_allowed": False,
        "sources": {str(path.relative_to(root)): file_hash(path) for path in sources},
    }


def relabel_oracle_predictions(output: Path, condition: str, seed: int) -> None:
    """Replace the reused A2 generator label with this diagnostic's model label."""
    path = output / "generation.jsonl"
    rows = read_jsonl(path)
    label = f"oracle_{condition}_seed{seed}"
    for row in rows:
        prediction_path = output / row["reconstructed_file"]
        prediction = read_json(prediction_path)
        prediction["provenance"]["config_name"] = label
        write_json(prediction_path, prediction)
        row["reconstructed_sha256"] = checksum(prediction_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                 for row in rows), encoding="utf-8")
    temporary.replace(path)


def percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def _paper_counts(metrics: dict, field: str) -> dict[str, dict]:
    return {row["paper_id"]: row["end_to_end_reference"][field]
            for row in metrics["papers"]}


def _pooled_f1(rows: list[dict]) -> float:
    return prf(add_counts(rows))["f1"]


def paired_oracle_bootstrap(metrics: dict[str, dict[int, dict]], contrast: list[str],
                            field: str, samples: int, seed: int) -> dict:
    treatment, control = contrast
    paper_ids = list(_paper_counts(metrics[treatment][SEEDS[0]], field))
    counts = {condition: {run_seed: _paper_counts(metrics[condition][run_seed], field)
                          for run_seed in SEEDS} for condition in contrast}

    def statistic(selected: list[str]) -> float:
        deltas = []
        for run_seed in SEEDS:
            treatment_f1 = _pooled_f1([counts[treatment][run_seed][paper] for paper in selected])
            control_f1 = _pooled_f1([counts[control][run_seed][paper] for paper in selected])
            deltas.append(treatment_f1 - control_f1)
        return math.fsum(deltas) / len(deltas)

    rng = random.Random(seed)
    observed = statistic(paper_ids)
    estimates = [statistic([paper_ids[rng.randrange(len(paper_ids))]
                            for _ in paper_ids]) for _ in range(samples)]
    return {"mean_seed_delta_f1": observed,
            "paired_paper_cluster_bootstrap_95_interval":
                [percentile(estimates, .025), percentile(estimates, .975)],
            "probability_delta_above_zero": sum(value > 0 for value in estimates) / samples,
            "samples": samples, "seed": seed,
            "definition": "paired paper-cluster bootstrap retaining all three seeds"}


def compare_oracle(root: Path, config_path: Path, config: MechanisticConfig) -> dict:
    verify_frozen_configs(root, config)
    metrics: dict[str, dict[int, dict]] = {condition: {} for condition in CONDITIONS}
    evaluator = paper_ids = None
    for condition in CONDITIONS:
        for seed in SEEDS:
            value = read_json(oracle_metrics_path(root, config, condition, seed))
            if value.get("seed") != seed or value.get("inventory_type") != "oracle":
                raise ValueError(f"Oracle metric identity changed: {condition} seed {seed}")
            if condition != config.oracle_comparison.reused_condition:
                if value.get("model_condition") != condition:
                    raise ValueError(f"Oracle model identity changed: {condition} seed {seed}")
            current_ids = [row["paper_id"] for row in value["papers"]]
            if len(current_ids) != config.expected_papers:
                raise ValueError(f"Incomplete oracle metrics: {condition} seed {seed}")
            if paper_ids is not None and current_ids != paper_ids:
                raise ValueError("Oracle conditions use different papers")
            paper_ids = current_ids
            if evaluator is not None and value["evaluator"] != evaluator:
                raise ValueError("Oracle conditions use different evaluator provenance")
            evaluator = value["evaluator"]
            metrics[condition][seed] = value

    conditions = {}
    for condition in CONDITIONS:
        by_seed = {}
        for seed in SEEDS:
            value = metrics[condition][seed]
            by_seed[str(seed)] = {
                field: _pooled_f1(list(_paper_counts(value, field).values()))
                for field in FIELDS
            }
            by_seed[str(seed)]["contract_valid_papers"] = value["summary"]["papers_reference_contract_valid"]
        conditions[condition] = {
            "by_seed": by_seed,
            "aggregate": {
                field: {"mean": statistics.mean(by_seed[str(seed)][field] for seed in SEEDS),
                        "sample_sd": statistics.stdev(by_seed[str(seed)][field] for seed in SEEDS)}
                for field in FIELDS
            },
        }
        valid = [by_seed[str(seed)]["contract_valid_papers"] for seed in SEEDS]
        conditions[condition]["aggregate"]["contract_valid_papers"] = {
            "mean": statistics.mean(valid), "sample_sd": statistics.stdev(valid)
        }
    contrasts = {}
    for index, contrast in enumerate((config.oracle_comparison.primary_contrast,
                                      config.oracle_comparison.secondary_contrast)):
        name = f"{contrast[0]}_minus_{contrast[1]}"
        contrasts[name] = {
            field: paired_oracle_bootstrap(
                metrics, contrast, field, config.oracle_comparison.bootstrap_samples,
                config.oracle_comparison.bootstrap_seed + index * 10 + field_index)
            for field_index, field in enumerate(FIELDS)
        }
    return {
        "version": config.version, "study_type": config.study_type,
        "corpus_role": config.corpus_role,
        "question": "Does explicit R supervision improve relationship linking under the same gold entity inventory?",
        "conditions": conditions, "contrasts": contrasts,
        "primary_contrast": config.oracle_comparison.primary_contrast,
        "secondary_contrast": config.oracle_comparison.secondary_contrast,
        "paper_ids": paper_ids, "evaluator": evaluator,
        "accounting": {"conditions": 3, "seeds": 3, "papers": 30,
                       "reused_generations": 90, "new_generations": 180,
                       "independent_test_papers": 0},
        "provenance": {"config": str(config_path.relative_to(root)),
                       "config_sha256": file_hash(config_path)},
        "caveats": [
            "This is an exploratory diagnostic on reused development papers.",
            "D32 versus D24+R8 matches optimizer updates and examples, not supervised tokens or compute.",
            "The R prompt is familiar only to R-trained models, so a difference may include task-format familiarity.",
            "Gold inventories differ from predicted inventories in correctness, IDs, ordering and attribute completeness.",
            "These results cannot select or replace the original pilot winner.",
        ],
    }


def oracle_markdown(result: dict) -> str:
    labels = {"d_only": "D24", "matched_d_dr": "D24+R8", "d32": "D32",
              "sample_refs": "Sample refs", "facility_refs": "Facility refs"}
    lines = ["# Equal-inventory oracle relationship diagnostic", "",
             "All models receive the same gold entity inventory. Scores are end-to-end relation F1 reconstructed over that inventory.", "",
             "| Condition | R contract valid | Sample refs F1 | Facility refs F1 |",
             "|---|---:|---:|---:|"]
    for condition in CONDITIONS:
        agg = result["conditions"][condition]["aggregate"]
        lines.append(f"| {labels[condition]} | {agg['contract_valid_papers']['mean']:.1f} ± {agg['contract_valid_papers']['sample_sd']:.1f} / 30 | "
                     f"{agg['sample_refs']['mean']:.3f} ± {agg['sample_refs']['sample_sd']:.3f} | "
                     f"{agg['facility_refs']['mean']:.3f} ± {agg['facility_refs']['sample_sd']:.3f} |")
    lines += ["", "## Paired contrasts", "",
              "| Contrast | Field | Mean ΔF1 | Paired 95% interval | P(Δ>0) |",
              "|---|---|---:|---:|---:|"]
    for name, fields in result["contrasts"].items():
        for field, row in fields.items():
            interval = row["paired_paper_cluster_bootstrap_95_interval"]
            lines.append(f"| `{name}` | {labels[field]} | {row['mean_seed_delta_f1']:+.3f} | "
                         f"[{interval[0]:+.3f}, {interval[1]:+.3f}] | "
                         f"{row['probability_delta_above_zero']:.3f} |")
    lines += ["", "## Caveats", ""]
    lines.extend(f"- {item}" for item in result["caveats"])
    return "\n".join(lines) + "\n"


def write_result(path: Path, value: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        write_json(path, value)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
