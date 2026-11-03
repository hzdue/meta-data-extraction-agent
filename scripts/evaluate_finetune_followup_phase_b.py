#!/usr/bin/env python
"""Evaluate the three D32 endpoints and apply the frozen Gate B decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meta_data_extraction.finetune.evaluation import prepare_test_rows, read_evaluation_config
from meta_data_extraction.finetune.followup_phase_b_evaluation import (
    compare, evaluation_contract, plan, read_phase_b_evaluation_config, run_one,
)
from meta_data_extraction.finetune.followup_training import read_phase_b_config
from meta_data_extraction.finetune.tokenization import load_tokenizer
from meta_data_extraction.finetune.training_data import file_hash


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/finetune_followup_b_evaluation.yaml"))
    parser.add_argument("--offline", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Verify the three D32 endpoints only.")
    mode.add_argument("--dry-run", action="store_true",
                      help="Verify prompts and immutable contracts without loading weights.")
    mode.add_argument("--score-only", action="store_true",
                      help="Score complete cached D32 generations only.")
    mode.add_argument("--compare-only", action="store_true",
                      help="Rebuild comparison and Gate B from complete metrics files.")
    args = parser.parse_args(argv)
    config_path = (ROOT / args.config).resolve()
    config = read_phase_b_evaluation_config(config_path)
    if file_hash(ROOT / config.training_config) != config.training_config_sha256:
        raise ValueError("Frozen Phase B training configuration changed")
    if file_hash(ROOT / config.ablation_evaluation_config) != config.ablation_evaluation_config_sha256:
        raise ValueError("Frozen ablation evaluation configuration changed")
    phase_b = read_phase_b_config(ROOT / config.training_config)
    if args.plan:
        print(json.dumps({"status": "planned_not_executed", "study_type": config.study_type,
                          "corpus_role": config.corpus_role,
                          "new_runs": plan(ROOT, config, phase_b),
                          "reused_controls": config.controls}, indent=2))
        return 0
    if args.compare_only:
        result = compare(ROOT, config, phase_b)
        print(json.dumps({"gate_b": result["gate_b"]}, indent=2))
        return 0
    base_config = read_evaluation_config(ROOT / config.evaluation_config)
    preparation = yaml.safe_load((ROOT / base_config.prepare_config).read_text(encoding="utf-8"))
    tokenizer, identity = load_tokenizer(preparation, local_files_only=args.offline)
    rows, data = prepare_test_rows(ROOT, base_config, tokenizer, identity)
    if args.dry_run:
        contracts = []
        for seed in config.seeds:
            contract, _, checkpoint = evaluation_contract(
                ROOT, config_path, config, phase_b, base_config, seed,
                rows, data, identity)
            contracts.append({
                "condition": contract["condition"],
                "adapter_sha256": contract["model"]["adapter_sha256"],
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "papers": len(contract["data"]["paper_ids"]),
                "corpus_role": contract["corpus_role"],
            })
        print(json.dumps({"status": "dry_run_passed", "contracts": contracts,
                          "reused_controls": config.controls}, indent=2))
        return 0
    for seed in config.seeds:
        print(f"phase_b_evaluation_start condition=d32 seed={seed}", flush=True)
        metrics = run_one(
            ROOT, config_path, config, phase_b, base_config, seed,
            tokenizer, identity, rows, data, offline=args.offline,
            score_only=args.score_only)
        print(json.dumps({"condition": "d32", "seed": seed,
                          "summary": metrics["summary"]}, indent=2), flush=True)
    result = compare(ROOT, config, phase_b)
    print(json.dumps({"gate_b": result["gate_b"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
