"""Verify data on CPU, or train on the server with validation-based selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from meta_data_extraction.finetune.training_data import TrainingConfig, prepare_training, read_config

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Verify data, labels and architecture; no weights/training")
    mode.add_argument("--train", action="store_true", help="Run on one CUDA GPU with validation-based checkpoint selection")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Explicit finite override, e.g. for a server smoke run")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.dry_run and args.resume_from_checkpoint:
            raise ValueError("Resume is only supported with --train")
        config = read_config(ROOT / args.config)
        values = config.model_dump()
        if args.max_steps is not None:
            values["max_steps"] = args.max_steps
        if args.output_dir is not None:
            values["output_dir"] = args.output_dir
        config = TrainingConfig.model_validate(values)
        tokenizer, records, validation, summary = prepare_training(ROOT, config, offline=args.offline)
        from meta_data_extraction.finetune.training import inspect_model, run_training
        from meta_data_extraction.finetune.validation import validate_rows
        from meta_data_extraction.finetune.tracking import read_tracking_config

        tracking = read_tracking_config(ROOT / config.tracking_config)
        preparation = yaml.safe_load((ROOT / config.prepare_config).read_text(encoding="utf-8"))
        validate_rows(validation, {r["example_id"].split(":")[0] for r in records},
                      config.max_seq_length, config.validation_max_new_tokens)
        architecture = inspect_model(preparation, offline=args.offline)
        if args.dry_run:
            print(json.dumps({"status": "dry_run_passed", **summary,
                              "architecture": {k: v for k, v in architecture.items() if k != "target_modules"},
                              "max_steps": config.max_steps, "validation_steps": config.validation_steps,
                              "validation_max_new_tokens": config.validation_max_new_tokens,
                              "selection_metric": "mean_of_per_paper_target_token_nll",
                              "tracking": {"experiment_name": tracking.experiment_name,
                                           "configured": tracking.tracking_uri is not None,
                                           "connected": False},
                              "gpu_feasibility": "not tested"}, indent=2))
        else:
            resume = (ROOT / args.resume_from_checkpoint).resolve() if args.resume_from_checkpoint else None
            print(json.dumps(run_training(ROOT, config, tokenizer, records, summary, architecture,
                                          preparation, validation=validation, offline=args.offline, resume=resume), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(f"Fine-tuning failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
