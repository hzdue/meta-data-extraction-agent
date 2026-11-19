#!/usr/bin/env python
"""Freeze the approved <=32,660-token pseudo-training population."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from meta_data_extraction.finetune.pseudo_training_data import (
    LENGTH_FILTER_MANIFEST,
    build_length_filter_manifest,
    verify_pseudo_prepared,
)
from meta_data_extraction.finetune.validation import write_json


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Write the derived manifest after all source checks pass")
    args = parser.parse_args(argv)
    try:
        datasets, _, _ = verify_pseudo_prepared(ROOT)
        manifest = build_length_filter_manifest(ROOT, datasets["pseudo_train"])
        if args.write:
            write_json(ROOT / LENGTH_FILTER_MANIFEST, manifest)
        print(json.dumps({
            "status": "written" if args.write else "preview",
            "path": LENGTH_FILTER_MANIFEST,
            "policy": manifest["policy"],
            "counts": manifest["counts"],
            "summary": manifest["summary"],
        }, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(f"Length-filter preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
