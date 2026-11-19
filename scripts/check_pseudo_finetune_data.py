"""Verify and tokenize frozen pseudo D records on CPU; never load model weights or train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from meta_data_extraction.finetune.pseudo_training_data import (
    DEFAULT_PREPARED_DIR,
    prepare_pseudo_training,
)


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default=DEFAULT_PREPARED_DIR)
    parser.add_argument(
        "--offline", action="store_true", help="Use only the cached pinned tokenizer"
    )
    args = parser.parse_args(argv)
    try:
        _, _, _, summary = prepare_pseudo_training(
            ROOT, prepared_dir=args.prepared_dir, offline=args.offline
        )
        print(json.dumps({
            "status": "data_preflight_passed",
            **summary,
            "gpu_feasibility": "not tested",
            "optimizer_steps": 0,
        }, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Pseudo data preflight failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
