#!/usr/bin/env python
"""Plan, verify, or run the frozen pseudo-only and pseudo-then-GoldD24 stages."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import yaml

from meta_data_extraction.finetune.pseudo_experiment import (
    build_stage_a_contract,
    plan_pseudo_experiment,
    prepare_pseudo_stage_a,
    prepare_pseudo_stage_b,
    read_pseudo_then_gold_config,
    resolve_stage_a_endpoint,
)
from meta_data_extraction.finetune.pseudo_training_data import file_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_A = Path("configs/finetune_pseudo_only.yaml")
DEFAULT_STAGE_B = Path("configs/finetune_pseudo_then_gold.yaml")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b"))
    parser.add_argument("--stage-a-config", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--stage-b-config", type=Path, default=DEFAULT_STAGE_B)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args(argv)

    try:
        stage_a_path = (ROOT / args.stage_a_config).resolve()
        stage_b_path = (ROOT / args.stage_b_config).resolve()
        if args.plan:
            if args.stage is not None or args.resume is not None or args.stop_after_step is not None:
                raise ValueError("--plan does not accept stage, resume or stop arguments")
            print(json.dumps({
                "status": "planned_not_executed",
                **plan_pseudo_experiment(ROOT, stage_a_path, stage_b_path),
            }, indent=2))
            return 0
        if args.stage is None:
            raise ValueError("--dry-run and --train require --stage")
        if args.resume is not None and not args.train:
            raise ValueError("--resume is valid only with --train")
        if args.stop_after_step is not None and not args.train:
            raise ValueError("--stop-after-step is valid only with --train")

        if args.stage == "a":
            config, tokenizer, records, validation, summary, experiment = (
                prepare_pseudo_stage_a(ROOT, stage_a_path, offline=args.offline)
            )
        else:
            config, tokenizer, records, validation, summary, experiment = (
                prepare_pseudo_stage_b(ROOT, stage_b_path, offline=args.offline)
            )

        preparation = yaml.safe_load(
            (ROOT / config.prepare_config).read_text(encoding="utf-8")
        )
        from meta_data_extraction.finetune.training import inspect_model, run_training

        architecture = inspect_model(preparation, offline=args.offline)
        stage_a_contract = (build_stage_a_contract(
            ROOT, stage_a_path, config, summary, architecture
        ) if args.stage == "a" else None)
        stage_a_endpoint = None
        stage_a_provenance = None
        stage_a_ready = None
        if args.stage == "b":
            plan = read_pseudo_then_gold_config(stage_b_path)
            try:
                stage_a_endpoint, stage_a_provenance = resolve_stage_a_endpoint(ROOT, plan)
                stage_a_ready = True
            except ValueError as exc:
                if args.train:
                    raise
                stage_a_ready = False
                stage_a_provenance = {"pending_reason": str(exc)}

        if args.dry_run:
            print(json.dumps({
                "status": "dry_run_passed",
                "stage": args.stage,
                "config": config.model_dump(),
                "experiment": experiment,
                "actual_task_counts": dict(sorted(Counter(
                    row["task"] for row in records
                ).items())),
                "training_papers": len({
                    row.get("paper_id", row["example_id"].split(":")[0])
                    for row in records
                }),
                "validation_papers": len(validation),
                "longest_training_sequence": max(len(row["input_ids"]) for row in records),
                "architecture": {
                    key: value for key, value in architecture.items()
                    if key != "target_modules"
                },
                **({"stage_a_ready": stage_a_ready,
                    "stage_a_provenance": stage_a_provenance}
                   if args.stage == "b" else {}),
                **({"run_contract_ready": True,
                    "run_contract_sources": len(stage_a_contract["sources"])}
                   if args.stage == "a" else {}),
                "gpu_feasibility": "not tested",
                "optimizer_updates": 0,
            }, indent=2))
            return 0

        resume = (ROOT / args.resume).resolve() if args.resume else None
        experiment = {**experiment, "source_sha256": {
            name: file_hash(ROOT / name)
            for name in (
                "scripts/run_pseudo_finetune.py",
                "src/meta_data_extraction/finetune/pseudo_experiment.py",
                "src/meta_data_extraction/finetune/pseudo_training_data.py",
                "src/meta_data_extraction/finetune/training.py",
                "src/meta_data_extraction/finetune/validation.py",
                "configs/finetune_pseudo_only.yaml",
                "configs/finetune_pseudo_then_gold.yaml",
            )
        }}
        if args.stage == "a":
            contract = stage_a_contract
            initial_adapter = None
            step_zero_base = True
            expected_exposure = {"D": 442}
        else:
            experiment = {**experiment, "stage_a": stage_a_provenance}
            contract = None
            initial_adapter = stage_a_endpoint
            step_zero_base = False
            expected_exposure = {"D": 24}
        result = run_training(
            ROOT, config, tokenizer, records, summary, architecture, preparation,
            validation=validation, offline=args.offline, resume=resume,
            stop_after_step=args.stop_after_step,
            selection_policy="fixed_endpoint", experiment=experiment,
            validation_step_zero_base=step_zero_base,
            initial_adapter=initial_adapter, precomputed_contract=contract,
            expected_final_exposure=expected_exposure,
        )
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Pseudo fine-tuning failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
