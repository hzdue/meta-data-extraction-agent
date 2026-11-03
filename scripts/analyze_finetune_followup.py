#!/usr/bin/env python
"""Run one frozen Step 9 follow-up analysis phase."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from meta_data_extraction.finetune.followup_analysis import (
    build_phase_a1, markdown_report, read_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("a1",))
    parser.add_argument("--config", type=Path, default=Path("configs/finetune_followup.yaml"))
    args = parser.parse_args(argv)
    config_path = (ROOT / args.config).resolve()
    config = read_config(config_path)
    analysis = build_phase_a1(ROOT, config_path, config)
    output_json = ROOT / config.phase_a1.output_json
    output_markdown = ROOT / config.phase_a1.output_markdown
    _write_text(output_json, json.dumps(
        analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    report = markdown_report(analysis)
    _write_text(output_markdown, report)
    print(report)
    print(f"JSON: {output_json.relative_to(ROOT)}")
    print(f"Markdown: {output_markdown.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
