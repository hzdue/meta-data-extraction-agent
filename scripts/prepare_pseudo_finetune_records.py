"""Validate pseudo-labels and create complete-output D fine-tuning records."""
from __future__ import annotations

import argparse
from pathlib import Path

from meta_data_extraction.finetune.pseudo_label_records import prepare


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/finetune/pseudo_label_supervision/prepared")
    args = parser.parse_args(argv)
    try:
        manifest = prepare(ROOT, args.output.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Pseudo-label record preparation failed: {exc}")
        return 1
    print(
        "Prepared "
        f"{manifest['counts']['pseudo_train']} train and "
        f"{manifest['counts']['pseudo_validation']} validation D records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
