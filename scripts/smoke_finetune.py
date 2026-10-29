"""Plan or run one isolated Step 6 GPU feasibility check; never launch comparison runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from meta_data_extraction.finetune.smoke import CASES, probe_plan, run_probe
from meta_data_extraction.finetune.training import inspect_model
from meta_data_extraction.finetune.training_data import prepare_training, read_config

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", choices=("plan", *CASES), default="plan")
    parser.add_argument("--output-dir", type=Path, help="Required for GPU cases; must be new/empty")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.case != "plan" and args.output_dir is None:
        parser.error("GPU cases require --output-dir on persistent storage")
    try:
        config = read_config(ROOT / args.config)
        tok, records, validation, summary = prepare_training(ROOT, config, offline=args.offline)
        preparation = yaml.safe_load((ROOT / config.prepare_config).read_text(encoding="utf-8"))
        architecture = inspect_model(preparation, offline=args.offline)
        if args.case == "plan":
            result = {"status": "plan_only", "gpu_feasibility": "not tested", "data": summary,
                      "cases": probe_plan(records, validation, config)}
        else:
            result = run_probe(ROOT, config, tok, records, validation, summary, architecture,
                               preparation, args.case, (ROOT / args.output_dir).resolve(), offline=args.offline)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"Smoke check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
