#!/usr/bin/env python
"""Plan, verify, or train one frozen Step 9 follow-up condition."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

import yaml

from meta_data_extraction.finetune.followup_training import (
    plan_phase_b, prepare_phase_b, read_phase_b_config,
)
from meta_data_extraction.finetune.training_data import file_hash


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("b",))
    parser.add_argument("--config", type=Path, default=Path("configs/finetune_followup_b.yaml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Print the frozen Phase B design only.")
    mode.add_argument("--dry-run", action="store_true", help="Verify one D32 schedule; load no weights.")
    mode.add_argument("--train", action="store_true", help="Train one D32 seed on one CUDA GPU.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", type=Path,
                        help="Resume --train from the latest complete checkpoint.")
    args = parser.parse_args(argv)
    try:
        config_path = (ROOT / args.config).resolve()
        phase = read_phase_b_config(config_path)
        if args.plan:
            if args.seed is not None or args.resume is not None:
                raise ValueError("--plan does not accept --seed or --resume")
            print(json.dumps({"status": "planned_not_executed", **plan_phase_b(phase)}, indent=2))
            return 0
        if args.seed is None:
            raise ValueError("--dry-run/--train require --seed")
        if args.resume is not None and not args.train:
            raise ValueError("--resume is valid only with --train")
        config, tokenizer, records, validation, summary = prepare_phase_b(
            ROOT, config_path, phase, args.seed, offline=args.offline)
        preparation = yaml.safe_load((ROOT / config.prepare_config).read_text(encoding="utf-8"))
        from meta_data_extraction.finetune.training import inspect_model, run_training
        from meta_data_extraction.finetune.validation import validate_rows
        architecture = inspect_model(preparation, offline=args.offline)
        validate_rows(validation, {row["example_id"].split(":")[0] for row in records},
                      config.max_seq_length, config.validation_max_new_tokens)
        if args.dry_run:
            print(json.dumps({
                "status": "dry_run_passed",
                "config": config.model_dump(),
                "followup": summary["followup"],
                "actual_task_counts": dict(sorted(collections.Counter(
                    row["task"] for row in records).items())),
                "per_paper_counts": dict(sorted(collections.Counter(
                    row["example_id"].split(":")[0] for row in records).items())),
                "longest_sequence": max(len(row["input_ids"]) for row in records),
                "architecture": {key: value for key, value in architecture.items()
                                 if key != "target_modules"},
                "gpu_feasibility": "inherited_from_step_8_not_retested",
                "development_evaluation": "not_run",
            }, indent=2))
        else:
            experiment = {
                **summary["followup"],
                "source_sha256": {
                    name: file_hash(ROOT / name) for name in (
                        "scripts/run_finetune_followup_training.py",
                        "src/meta_data_extraction/finetune/followup_training.py",
                        "src/meta_data_extraction/finetune/training.py",
                        "src/meta_data_extraction/finetune/tracking.py",
                    )
                },
            }
            resume = (ROOT / args.resume).resolve() if args.resume else None
            print(json.dumps(run_training(
                ROOT, config, tokenizer, records, summary, architecture, preparation,
                validation=validation, offline=args.offline, resume=resume,
                selection_policy=phase.selection_policy, experiment=experiment), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(f"Follow-up training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
