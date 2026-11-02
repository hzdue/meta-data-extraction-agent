#!/usr/bin/env python
"""Run the frozen exploratory 30-paper evaluation for twelve ablation endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meta_data_extraction.finetune.ablation import read_ablation_config
from meta_data_extraction.finetune.ablation_evaluation import (
    SHORTLIST, compare, evaluation_contract, plan, read_ablation_evaluation_config, run_one,
)
from meta_data_extraction.finetune.evaluation import prepare_test_rows, read_evaluation_config
from meta_data_extraction.finetune.tokenization import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/finetune_ablation_evaluation.yaml"))
    parser.add_argument("--condition", choices=(*SHORTLIST, "all"), default="all")
    parser.add_argument("--offline", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Verify and print the 12 fixed endpoints only.")
    mode.add_argument("--dry-run", action="store_true",
                      help="Verify prompts and immutable contracts without loading model weights.")
    mode.add_argument("--score-only", action="store_true", help="Score complete cached generations only.")
    mode.add_argument("--compare-only", action="store_true", help="Rebuild comparison from 12 metrics files.")
    args = parser.parse_args(argv)
    config_path = (ROOT / args.config).resolve()
    config = read_ablation_evaluation_config(config_path)
    if args.plan:
        print(json.dumps({"status": "planned_not_executed", "study_type": "exploratory",
                          "corpus_role": config.corpus_role, "runs": plan(ROOT, config)}, indent=2))
        return 0
    if args.compare_only:
        if args.condition != "all":
            raise ValueError("--compare-only requires --condition all")
        compare(ROOT, config)
        return 0
    base_config = read_evaluation_config(ROOT / config.evaluation_config)
    matrix = read_ablation_config(ROOT / config.ablation_config)
    prepare = yaml.safe_load((ROOT / base_config.prepare_config).read_text(encoding="utf-8"))
    tokenizer, identity = load_tokenizer(prepare, local_files_only=args.offline)
    rows, data = prepare_test_rows(ROOT, base_config, tokenizer, identity)
    chosen = config.conditions if args.condition == "all" else [args.condition]
    if args.dry_run:
        contracts = []
        for condition in chosen:
            for seed in config.seeds:
                contract, _, checkpoint = evaluation_contract(
                    ROOT, config_path, config, base_config, matrix, condition, seed,
                    rows, data, identity)
                contracts.append({"condition": contract["condition"],
                                  "adapter_sha256": contract["model"]["adapter_sha256"],
                                  "checkpoint": str(checkpoint.relative_to(ROOT)),
                                  "papers": len(contract["data"]["paper_ids"]),
                                  "corpus_role": contract["corpus_role"]})
        print(json.dumps({"status": "dry_run_passed", "contracts": contracts}, indent=2))
        return 0
    for condition in chosen:
        for seed in config.seeds:
            print(f"evaluation_start condition={condition} seed={seed}", flush=True)
            metrics = run_one(ROOT, config_path, config, base_config, matrix, condition, seed,
                              tokenizer, identity, rows, data,
                              offline=args.offline, score_only=args.score_only)
            print(json.dumps({"condition": condition, "seed": seed,
                              "summary": metrics["summary"]}, indent=2), flush=True)
    if args.condition == "all":
        compare(ROOT, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
