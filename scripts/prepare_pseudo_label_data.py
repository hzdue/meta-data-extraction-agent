"""Create the pseudo-label split and parse the 600 source PDFs; never train."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from meta_data_extraction.finetune.pseudo_label_preparation import (
    check_config,
    create_split_manifest,
    parse_all,
    read_json,
)


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/finetune_pseudo_prepare.yaml")
    parser.add_argument("--stage", choices=("split", "parse", "all"), default="all")
    parser.add_argument(
        "--repair-stale-cache", action="store_true",
        help="Rebuild only stale or incomplete per-paper Markdown cache pairs")
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        check_config(config)
        if args.stage in {"split", "all"}:
            split = create_split_manifest(ROOT, config)
            print("Created deterministic 508/57/30 pseudo-label split", flush=True)
        else:
            split = read_json(ROOT / config["split_manifest"])
        if args.stage in {"parse", "all"}:
            summary = parse_all(
                ROOT, config, split,
                repair_stale_cache=args.repair_stale_cache)
            print(f"Parsed and cached {summary['papers']} complete PDFs", flush=True)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"Pseudo-label preparation failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

