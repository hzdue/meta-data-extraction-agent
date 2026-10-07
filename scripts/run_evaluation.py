"""Evaluate all 22 paper conditions on the 30 evaluation papers.

Reports are saved under outputs/evaluation/<condition>/.

Usage:
    uv run python scripts/run_evaluation.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

if __package__:
    from .corpus import paper_ids
else:
    from corpus import paper_ids

from meta_data_extraction.components.tracker import LocalTracker
from meta_data_extraction.evaluation import (
    DEFAULT_PLAN_PATH,
    build_evaluator_metadata,
    evaluate,
    validate_evaluator_provenance,
)

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("configs")
GOLD_DIR = Path("data/gold")
PROCESSED_DIR = Path("data/experiment")
OUTPUT_DIR = Path("outputs/evaluation")
PLAN_PATH = DEFAULT_PLAN_PATH


def run_condition(
    extraction_dir: Path,
    *,
    gold_dir: Path,
    output_dir: Path,
    papers: list[str],
    plan_path: Path,
    expected_evaluator: dict,
) -> tuple[int, int]:
    """Write fresh reports for one condition; return scored and failed counts."""
    report_dir = output_dir / extraction_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    scored = failed = 0
    for paper in papers:
        report_path = report_dir / f"{paper}_evaluation.json"
        try:
            # A failed rerun must not leave an earlier successful report in place.
            report_path.unlink(missing_ok=True)
            report = evaluate(
                gold_dir / f"{paper}.json", extraction_dir / f"{paper}.json",
                plan_path=plan_path,
            )
            validate_evaluator_provenance(
                report.get("evaluator", {}), expected_evaluator, source=str(report_path),
            )
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("%s/%s: %s", extraction_dir.name, paper, exc)
            failed += 1
        else:
            scored += 1
    print(f"{extraction_dir.name}: {scored}/{len(papers)} scored, {failed} failed.")
    return scored, failed


def main(argv: list[str] | None = None) -> int:
    """Evaluate the complete paper experiment, returning nonzero on failure."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    for directory in (GOLD_DIR, PROCESSED_DIR):
        if not directory.is_dir():
            logger.error("Input directory not found: %s", directory)
            return 1
    try:
        papers = list(paper_ids("evaluation"))
        configs = sorted(CONFIG_DIR.glob("pan_*.yaml"))
        if len(configs) != 22:
            raise ValueError(f"Expected 22 experiment configs in {CONFIG_DIR}, found {len(configs)}.")
        tracker = LocalTracker(output_dir=str(PROCESSED_DIR))
        extraction_dirs = [tracker.config_dir(config.stem) for config in configs]
        expected_evaluator = build_evaluator_metadata(PLAN_PATH)
        if expected_evaluator["schema_version_mismatch"]:
            raise ValueError("The scoring plan's declared schema versions do not match the schema files.")
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    if not papers:
        logger.error("No papers selected.")
        return 1

    scored = failed = 0
    for extraction_dir in extraction_dirs:
        if not extraction_dir.is_dir():
            logger.error("Prediction condition not found: %s", extraction_dir)
            return 1
        try:
            condition_scored, condition_failed = run_condition(
                extraction_dir, gold_dir=GOLD_DIR, output_dir=OUTPUT_DIR,
                papers=papers, plan_path=PLAN_PATH, expected_evaluator=expected_evaluator,
            )
        except OSError as exc:
            logger.error("Cannot write reports for %s: %s", extraction_dir.name, exc)
            return 1
        scored += condition_scored
        failed += condition_failed
    print(f"Total: {len(extraction_dirs)} conditions, {scored} scored, {failed} failed. Reports: {OUTPUT_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
