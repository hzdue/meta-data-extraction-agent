#!/usr/bin/env python
"""Plan, verify, or train one predeclared controlled D/E/R ablation run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from meta_data_extraction.finetune.ablation import (
    plan_matrix, prepare_ablation, read_ablation_config,
)
from meta_data_extraction.finetune.training_data import file_hash


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/finetune_ablation.yaml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Print the frozen matrix; load no tokenizer/model.")
    mode.add_argument("--dry-run", action="store_true", help="Verify one schedule and architecture; load no weights.")
    mode.add_argument("--train", action="store_true", help="Train one condition/seed on one CUDA GPU.")
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", type=Path,
                        help="Resume --train from an existing complete checkpoint directory.")
    args = parser.parse_args(argv)
    try:
        path = (ROOT / args.config).resolve()
        matrix = read_ablation_config(path)
        if args.plan:
            if args.condition is not None or args.seed is not None or args.resume is not None:
                raise ValueError("--plan does not accept --condition, --seed or --resume")
            print(json.dumps({"status": "planned_not_executed", "matrix": plan_matrix(matrix),
                              "independent_test_required": matrix.independent_test_split is None,
                              "current_test_evaluation_allowed": False}, indent=2))
            return 0
        if args.condition is None or args.seed is None:
            raise ValueError("--dry-run/--train require --condition and --seed")
        if args.resume is not None and not args.train:
            raise ValueError("--resume is valid only with --train")
        config, tokenizer, records, validation, summary = prepare_ablation(
            ROOT, matrix, args.condition, args.seed, offline=args.offline)
        preparation = yaml.safe_load((ROOT / config.prepare_config).read_text(encoding="utf-8"))
        from meta_data_extraction.finetune.training import inspect_model, run_training
        from meta_data_extraction.finetune.validation import validate_rows
        architecture = inspect_model(preparation, offline=args.offline)
        validate_rows(validation, {row["example_id"].split(":")[0] for row in records},
                      config.max_seq_length, config.validation_max_new_tokens)
        if args.dry_run:
            print(json.dumps({"status": "dry_run_passed", "config": config.model_dump(),
                              "ablation": summary["ablation"],
                              "actual_task_counts": dict(sorted(__import__("collections").Counter(
                                  row["task"] for row in records).items())),
                              "longest_sequence": max(len(row["input_ids"]) for row in records),
                              "architecture": {key: value for key, value in architecture.items()
                                               if key != "target_modules"},
                              "gpu_feasibility": "not tested", "test_evaluation": "blocked"}, indent=2))
        else:
            experiment = {**summary["ablation"],
                          "matrix_config": str(path.relative_to(ROOT)),
                          "matrix_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                          "source_sha256": {name: file_hash(ROOT / name) for name in
                                            ("scripts/run_finetune_ablation.py",
                                             "src/meta_data_extraction/finetune/ablation.py",
                                             "src/meta_data_extraction/finetune/schema.py")}}
            print(json.dumps(run_training(
                ROOT, config, tokenizer, records, summary, architecture, preparation,
                validation=validation, offline=args.offline,
                resume=(ROOT / args.resume).resolve() if args.resume else None,
                selection_policy=matrix.selection_policy, experiment=experiment), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(f"Ablation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
