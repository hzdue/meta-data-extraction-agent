#!/usr/bin/env python
"""Evaluate the six Phase C endpoints and apply the frozen order interpretation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meta_data_extraction.finetune.evaluation import prepare_test_rows, read_evaluation_config
from meta_data_extraction.finetune.followup_phase_c_evaluation import (
    NEW_CONDITIONS, compare, evaluation_contract, plan,
    read_phase_c_evaluation_config, run_one, verify_inputs,
)
from meta_data_extraction.finetune.followup_training import read_phase_c_config
from meta_data_extraction.finetune.tokenization import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/finetune_followup_c_evaluation.yaml"))
    parser.add_argument("--offline", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    mode.add_argument("--compare-only", action="store_true")
    args = parser.parse_args(argv)
    config_path = (ROOT / args.config).resolve()
    config = read_phase_c_evaluation_config(config_path)
    verify_inputs(ROOT, config)
    phase = read_phase_c_config(ROOT / config.training_config)
    if args.plan:
        print(json.dumps({"status": "planned_not_executed", "study_type": config.study_type,
                          "new_runs": plan(config, phase), "reused_controls": config.controls,
                          "gate_c": phase.gate_c.model_dump()}, indent=2))
        return 0
    if args.compare_only:
        print(json.dumps({"gate_c": compare(ROOT, config, phase)["gate_c"]}, indent=2))
        return 0
    base = read_evaluation_config(ROOT / config.evaluation_config)
    preparation = yaml.safe_load((ROOT / base.prepare_config).read_text(encoding="utf-8"))
    tokenizer, identity = load_tokenizer(preparation, local_files_only=args.offline)
    rows, data = prepare_test_rows(ROOT, base, tokenizer, identity)
    if args.dry_run:
        contracts = []
        for condition in NEW_CONDITIONS:
            for seed in config.seeds:
                contract, _, checkpoint = evaluation_contract(
                    ROOT, config_path, config, phase, base, condition, seed,
                    rows, data, identity)
                contracts.append({"condition": contract["condition"],
                                  "adapter_sha256": contract["model"]["adapter_sha256"],
                                  "checkpoint": str(checkpoint.relative_to(ROOT)),
                                  "papers": len(contract["data"]["paper_ids"])})
        print(json.dumps({"status": "dry_run_passed", "contracts": contracts,
                          "reused_controls": config.controls}, indent=2))
        return 0
    for condition in NEW_CONDITIONS:
        for seed in config.seeds:
            print(f"phase_c_evaluation_start condition={condition} seed={seed}", flush=True)
            metrics = run_one(ROOT, config_path, config, phase, base, condition, seed,
                              tokenizer, identity, rows, data, offline=args.offline,
                              score_only=args.score_only)
            print(json.dumps({"condition": condition, "seed": seed,
                              "summary": metrics["summary"]}, indent=2), flush=True)
    result = compare(ROOT, config, phase)
    print(json.dumps({"gate_c": result["gate_c"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
