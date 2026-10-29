#!/usr/bin/env python
"""Generate and score the frozen 30-paper M0/M1/M2 fine-tuning comparison."""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from meta_data_extraction.finetune.evaluation import (
    CONDITIONS,
    attach_all_to_mlflow,
    compare_conditions,
    prepare_test_rows,
    read_evaluation_config,
    run_condition,
)
from meta_data_extraction.finetune.tokenization import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/finetune_evaluation.yaml"))
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), default="all")
    parser.add_argument("--offline", action="store_true", help="Require already-cached model/tokenizer files.")
    parser.add_argument("--score-only", action="store_true", help="Score a complete saved generation without loading a model.")
    parser.add_argument("--compare-only", action="store_true", help="Rebuild comparison files from three complete metrics files.")
    parser.add_argument("--attach-mlflow", action="store_true", help="Attach a complete three-condition comparison to existing runs.")
    return parser.parse_args()


def main():
    args = arguments()
    config_path = (ROOT / args.config).resolve()
    config = read_evaluation_config(config_path)
    if args.attach_mlflow and args.condition != "all":
        raise ValueError("MLflow attachment requires --condition all so every run receives the same comparison")
    if args.compare_only:
        if args.condition != "all":
            raise ValueError("--compare-only requires --condition all")
        comparison = compare_conditions(ROOT, config)
    else:
        prepare = yaml.safe_load((ROOT / config.prepare_config).read_text(encoding="utf-8"))
        tokenizer, identity = load_tokenizer(prepare, local_files_only=args.offline)
        rows, data = prepare_test_rows(ROOT, config, tokenizer, identity)
        chosen = CONDITIONS if args.condition == "all" else (args.condition,)
        for condition in chosen:
            run_condition(ROOT, config_path, config, condition, tokenizer, identity, rows, data,
                          offline=args.offline, score_only=args.score_only)
        comparison = compare_conditions(ROOT, config) if args.condition == "all" else None
    if args.attach_mlflow:
        attach_all_to_mlflow(ROOT, config, ROOT / config.output_dir / "comparison.json")
    if comparison is not None:
        for condition, values in comparison["conditions"].items():
            print(condition, values["quality"])


if __name__ == "__main__":
    main()
