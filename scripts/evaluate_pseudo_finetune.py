#!/usr/bin/env python
"""Evaluate a fixed pseudo-supervision endpoint on the frozen 30-paper corpus."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import torch
from transformers import set_seed
import yaml

from meta_data_extraction.evaluation import build_evaluator_metadata
from meta_data_extraction.finetune.evaluation import (
    generate_condition,
    prepare_condition_output,
    prepare_test_rows,
    read_evaluation_config,
    runtime_versions,
    score_condition,
)
from meta_data_extraction.finetune.pseudo_experiment import resolve_stage_a_endpoint
from meta_data_extraction.finetune.pseudo_training_data import file_hash
from meta_data_extraction.finetune.tokenization import load_tokenizer
from meta_data_extraction.finetune.tracking import checksum
from meta_data_extraction.finetune.training import load_base, reload_adapter
from meta_data_extraction.finetune.training_data import (
    TrainingConfig,
    read_config as read_training_config,
)
from meta_data_extraction.finetune.validation import write_json


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_FIELDS = (
    "papers_total",
    "papers_json_valid",
    "papers_schema_valid",
    "generation_failures",
    "output_limit_failures",
    "scoring_fallbacks",
    "json_valid_rate",
    "schema_valid_rate",
    "generic_metadata_macro_score",
    "facilities_f1",
    "samples_f1",
    "datasets_f1",
    "sample_refs_f1",
    "facility_refs_f1",
    "mean_latency_ms",
    "mean_generated_tokens",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)


class PseudoEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    evaluation_config: str
    pseudo_only_config: str
    output_dir: str
    base_metrics: str
    condition: str = Field(min_length=1)
    stage: Literal["a", "b"] = "a"
    corpus_role: Literal["exploratory_reused_step7_test_as_development"]
    endpoint_policy: Literal["fixed_exposure"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_pseudo_evaluation_config(path: Path) -> PseudoEvaluationConfig:
    return PseudoEvaluationConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def verify_no_benchmark_overlap(root: Path) -> dict:
    benchmark = read_json(root / "data/finetune/splits.json")
    pseudo = read_json(root / "data/finetune/pseudo_label_supervision/splits.json")
    test_ids = set(benchmark["test"])
    checked = ("pseudo_train", "pseudo_validation", "future_test")
    overlap = {
        name: sorted(test_ids.intersection(pseudo[name]))
        for name in checked
    }
    if any(overlap.values()):
        raise ValueError(f"Pseudo data overlaps the frozen 30-paper benchmark: {overlap}")
    if pseudo.get("leakage_checks", {}).get("existing_benchmark_overlap") != []:
        raise ValueError("Frozen pseudo split does not declare zero benchmark overlap")
    return {
        "benchmark_test_papers": len(test_ids),
        "pseudo_train_papers": len(pseudo["pseudo_train"]),
        "pseudo_validation_papers": len(pseudo["pseudo_validation"]),
        "future_test_papers": len(pseudo["future_test"]),
        "overlap": overlap,
        "benchmark_split_sha256": file_hash(root / "data/finetune/splits.json"),
        "pseudo_split_sha256": file_hash(
            root / "data/finetune/pseudo_label_supervision/splits.json"
        ),
    }


def evaluation_contract(
    root: Path,
    config_path: Path,
    config: PseudoEvaluationConfig,
    base_config,
    rows: list[dict],
    data: dict,
    tokenizer_identity: dict,
) -> tuple[dict, object, Path]:
    from meta_data_extraction.finetune.pseudo_experiment import (
        read_pseudo_then_gold_config,
    )
    stage_plan = read_pseudo_then_gold_config(
        root / "configs/finetune_pseudo_then_gold.yaml"
    )
    stage_a_checkpoint, stage_a_identity = resolve_stage_a_endpoint(root, stage_plan)
    if config.stage == "a":
        training_config = read_training_config(root / config.pseudo_only_config)
        output = root / training_config.output_dir
        checkpoint = stage_a_checkpoint
        phase = "pseudo_stage_a"
    else:
        output = root / stage_plan.output_dir
        manifest = read_json(output / "run_manifest.json")
        training_config = TrainingConfig.model_validate(manifest["config"])
        summary = read_json(output / "training_summary.json")
        experiment = manifest.get("experiment", {})
        if (experiment.get("phase") != "pseudo_stage_b"
                or experiment.get("stage_a") != stage_a_identity
                or summary.get("global_step") != stage_plan.endpoint_updates
                or summary.get("exposure", {}).get("examples") != {"D": stage_plan.gold_exposure}
                or training_config.output_dir != stage_plan.output_dir):
            raise ValueError("Stage-B training provenance differs from the frozen plan")
        checkpoint = output / f"checkpoint-{stage_plan.endpoint_updates}"
        phase = "pseudo_stage_b"
    selection = read_json(output / "selection.json")
    precision = read_json(checkpoint / "inference_precision.json")
    if precision != training_config.precision_policy():
        raise ValueError(f"Stage-{config.stage.upper()} adapter precision differs from its training contract")
    if checksum(checkpoint / "adapter_model.safetensors") != selection["adapter_sha256"]:
        raise ValueError(f"Stage-{config.stage.upper()} adapter checksum changed")
    expected_step = 111 if config.stage == "a" else stage_plan.endpoint_updates
    if (selection.get("status") != "complete"
            or selection.get("kind") != "adapter"
            or selection.get("metric") != "fixed_exposure_endpoint"
            or selection.get("endpoint_policy") != "fixed_exposure"
            or selection.get("step") != expected_step
            or selection.get("checkpoint") != f"checkpoint-{expected_step}"):
        raise ValueError(f"Evaluation requires the fixed Stage-{config.stage.upper()} endpoint")

    prepare = yaml.safe_load((root / base_config.prepare_config).read_text(encoding="utf-8"))
    source_paths = [
        config_path,
        root / config.evaluation_config,
        root / config.pseudo_only_config,
        root / "configs/finetune_pseudo_then_gold.yaml",
        root / base_config.metrics_config,
        root / base_config.alignment_config,
        root / "scripts/evaluate_pseudo_finetune.py",
        root / "src/meta_data_extraction/finetune/evaluation.py",
        root / "src/meta_data_extraction/finetune/pseudo_experiment.py",
        root / "src/meta_data_extraction/finetune/schema.py",
        root / "src/meta_data_extraction/evaluation.py",
        root / "src/meta_data_extraction/alignment.py",
    ]
    contract = {
        "version": config.version,
        "study_type": "exploratory",
        "corpus_role": config.corpus_role,
        "condition": config.condition,
        "method": "pseudo_label_supervision_fixed_endpoint",
        "endpoint_policy": config.endpoint_policy,
        "training": {
            "phase": phase,
            "seed": training_config.seed,
            "selected_step": selection["step"],
            "selected_exposure": selection["exposure"],
            "selection_sha256": file_hash(output / "selection.json"),
            "training_summary_sha256": file_hash(output / "training_summary.json"),
            "training_mlflow_run_id": read_json(
                output / "training_summary.json"
            )["mlflow_run_id"],
            "stage_a": stage_a_identity,
        },
        "model": {
            "model_id": prepare["model_id"],
            "revision": prepare["revision"],
            "precision": precision,
            "adapter": str(checkpoint.resolve()),
            "adapter_sha256": selection["adapter_sha256"],
        },
        "decoding": {
            "method": base_config.decoding,
            "enable_thinking": base_config.enable_thinking,
            "max_new_tokens": base_config.max_new_tokens,
            "max_seq_length": base_config.max_seq_length,
            "seed": base_config.seed,
            "num_beams": 1,
            "repetition_penalty": 1.0,
        },
        "data": {
            "source_split": "step7_test",
            "analysis_role": "exploratory_development",
            "paper_ids": [row["paper_id"] for row in rows],
            "prompts": [
                {key: row[key] for key in (
                    "paper_id", "prompt_tokens", "prompt_sha256", "text_sha256"
                )}
                for row in rows
            ],
            "preparation_manifest_sha256": data["preparation_manifest_sha256"],
            "gold_read_during_generation": False,
            "leakage_audit": verify_no_benchmark_overlap(root),
        },
        "tokenizer": tokenizer_identity,
        "evaluator": build_evaluator_metadata(root / base_config.metrics_config),
        "sources": {
            str(path.relative_to(root)): file_hash(path) for path in source_paths
        },
    }
    if contract["evaluator"]["schema_version_mismatch"]:
        raise ValueError("Evaluator schema versions do not match current schemas")
    return contract, training_config, checkpoint


def metric_delta(treatment: dict, control: dict) -> dict:
    return {
        key: (None if control.get(key) is None or treatment.get(key) is None
              else treatment[key] - control[key])
        for key in SUMMARY_FIELDS
    }


def compare(root: Path, config: PseudoEvaluationConfig, pseudo_metrics: dict) -> dict:
    base_path = root / config.base_metrics
    base = read_json(base_path)
    base_order = [row["paper_id"] for row in base["papers"]]
    pseudo_order = [row["paper_id"] for row in pseudo_metrics["papers"]]
    if base_order != pseudo_order or len(base_order) != 30 or len(set(base_order)) != 30:
        raise ValueError("Base and pseudo evaluations do not cover the same ordered papers")
    if base["evaluator"] != pseudo_metrics["evaluator"]:
        raise ValueError("Base and pseudo evaluations used different evaluator contracts")
    base_summary = base["summary"]
    pseudo_summary = pseudo_metrics["summary"]
    delta = metric_delta(pseudo_summary, base_summary)
    if config.stage == "b":
        gold_path = root / "outputs/finetune/ablation-development-evaluation/d_only_seed42/metrics.json"
        stage_a_path = root / config.output_dir / "pseudo_length_filtered_seed42/metrics.json"
        gold = read_json(gold_path)
        stage_a = read_json(stage_a_path)
        for name, metrics in (("GoldD24", gold), ("Stage A", stage_a)):
            order = [row["paper_id"] for row in metrics["papers"]]
            if order != base_order or metrics["evaluator"] != base["evaluator"]:
                raise ValueError(f"{name} comparison contract differs")
        comparison_conditions = {
            "base": base_summary,
            "gold_d24_seed42": gold["summary"],
            "pseudo_length_filtered_seed42": stage_a["summary"],
            config.condition: pseudo_summary,
        }
        deltas = {
            "delta_stage_b_minus_base": delta,
            "delta_stage_b_minus_gold_d24": metric_delta(pseudo_summary, gold["summary"]),
            "delta_stage_b_minus_stage_a": metric_delta(pseudo_summary, stage_a["summary"]),
        }
    else:
        comparison_conditions = {"base": base_summary, config.condition: pseudo_summary}
        deltas = {"delta_pseudo_minus_base": delta}
    result = {
        "version": config.version,
        "study_type": "exploratory",
        "corpus_role": config.corpus_role,
        "paper_ids": base_order,
        "base_metrics": str(base_path.relative_to(root)),
        "conditions": comparison_conditions,
        **deltas,
        "caveats": [
            "These thirty papers are reused development data, not an independent test.",
            f"The fixed Stage-{config.stage.upper()} endpoint was chosen without using these thirty evaluation labels.",
            "Base generations are reused from the canonical frozen evaluation because the decoding and evaluator contracts match.",
        ],
    }
    output = root / config.output_dir
    comparison_name = "comparison.json" if config.stage == "a" else "comparison_stage_b.json"
    write_json(output / comparison_name, result)

    keys = (
        ("JSON valid", "json_valid_rate"),
        ("Schema valid", "schema_valid_rate"),
        ("Facilities F1", "facilities_f1"),
        ("Samples F1", "samples_f1"),
        ("Datasets F1", "datasets_f1"),
        ("Sample refs F1", "sample_refs_f1"),
        ("Facility refs F1", "facility_refs_f1"),
        ("Generic macro", "generic_metadata_macro_score"),
    )
    if config.stage == "a":
        lines = [
            "# Stage-A pseudo-label evaluation against base", "",
            "This is an exploratory comparison on the reused 30-paper development corpus.", "",
            "| Metric | Base | Pseudo Stage A | Delta |",
            "|---|---:|---:|---:|",
        ]
        for label, key in keys:
            old, new, change = base_summary[key], pseudo_summary[key], delta[key]
            show = lambda value: "n/a" if value is None else f"{value:.3f}"
            lines.append(f"| {label} | {show(old)} | {show(new)} | {show(change)} |")
        lines += ["", "Delta is pseudo Stage A minus base.", ""]
        markdown_name = "comparison.md"
    else:
        labels = (("Base", "base"), ("GoldD24", "gold_d24_seed42"),
                  ("Pseudo Stage A", "pseudo_length_filtered_seed42"),
                  ("Pseudo Stage B", config.condition))
        lines = [
            "# Stage-B pseudo-label evaluation", "",
            "This is an exploratory comparison on the reused 30-paper development corpus.", "",
            "| Metric | " + " | ".join(label for label, _ in labels) + " |",
            "|---|" + "---:|" * len(labels),
        ]
        show = lambda value: "n/a" if value is None else f"{value:.3f}"
        for label, key in keys:
            lines.append("| " + label + " | " + " | ".join(
                show(comparison_conditions[name][key]) for _, name in labels
            ) + " |")
        lines += ["", "All adapter endpoints were fixed before this evaluation.", ""]
        markdown_name = "comparison_stage_b.md"
    (output / markdown_name).write_text("\n".join(lines), encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/finetune_pseudo_evaluation.yaml"),
    )
    parser.add_argument("--offline", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        config_path = (ROOT / args.config).resolve()
        config = read_pseudo_evaluation_config(config_path)
        base_config = read_evaluation_config(ROOT / config.evaluation_config)
        prepare = yaml.safe_load(
            (ROOT / base_config.prepare_config).read_text(encoding="utf-8")
        )
        tokenizer, identity = load_tokenizer(prepare, local_files_only=args.offline)
        rows, data = prepare_test_rows(ROOT, base_config, tokenizer, identity)
        contract, training, checkpoint = evaluation_contract(
            ROOT, config_path, config, base_config, rows, data, identity
        )
        contract["runtime"] = runtime_versions()
        output = ROOT / config.output_dir / config.condition
        if args.dry_run:
            print(json.dumps({
                "status": "dry_run_passed",
                "condition": config.condition,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "adapter_sha256": contract["model"]["adapter_sha256"],
                "papers": len(rows),
                "leakage_audit": contract["data"]["leakage_audit"],
                "output": str(output.relative_to(ROOT)),
            }, indent=2))
            return 0

        prepare_condition_output(output, contract)
        if not args.score_only:
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(
                including_emulation=False
            ):
                raise ValueError("Generation requires one CUDA GPU with native BF16 support")
            if torch.cuda.device_count() != 1:
                raise ValueError("Expose exactly one GPU for evaluation")
            set_seed(base_config.seed)
            model = reload_adapter(
                load_base(prepare, training, offline=args.offline),
                checkpoint,
                prepare,
                config=training,
            )
            try:
                generate_condition(
                    model, tokenizer, rows, base_config, config.condition,
                    output, data["schemas"],
                )
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
        metrics = score_condition(ROOT, base_config, config.condition, output)
        metrics["method"] = contract["method"]
        write_json(output / "metrics.json", metrics)
        result = compare(ROOT, config, metrics)
        print(json.dumps({
            "status": "complete",
            "summary": metrics["summary"],
            "deltas": {key: value for key, value in result.items() if key.startswith("delta_")},
        }, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Pseudo evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
