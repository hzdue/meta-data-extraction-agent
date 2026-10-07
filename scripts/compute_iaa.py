"""Compute inter-annotator agreement for all 30 evaluation papers.

Compare each paper's two annotations, using reconciled gold identities for
funding, samples, and datasets. Save per-paper reports and one pooled summary
under outputs/iaa/.

Usage:
    uv run python scripts/compute_iaa.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

if __package__:
    from .corpus import DEFAULT_CORPUS_PATH, annotation_batches
else:
    from corpus import DEFAULT_CORPUS_PATH, annotation_batches

from meta_data_extraction.evaluation import (
    aggregate_reports,
    build_evaluator_metadata,
    evaluate,
    validate_evaluator_provenance,
)

logger = logging.getLogger(__name__)

CORPUS_PATH = DEFAULT_CORPUS_PATH
ANNOTATIONS_DIR = Path("data/annotations")
IDENTITY_GOLD_DIR = Path("data/gold")
OUTPUT_DIR = Path("outputs/iaa")
PLAN_PATH = Path("configs/evaluation_metrics.yaml")


def compute_batch(
    batch_name: str,
    annotators: list[str],
    papers: list[str],
    expected_evaluator: dict,
) -> tuple[list[dict], int, int]:
    """Write fresh reports for one pair; return reports, missing and failed counts."""
    first, second = annotators
    batch_dir = OUTPUT_DIR / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    missing = failed = 0
    for paper in papers:
        report_path = batch_dir / f"{paper}_evaluation.json"
        try:
            # A failed rerun must not leave an earlier successful report in place.
            report_path.unlink(missing_ok=True)
            report = evaluate(
                ANNOTATIONS_DIR / first / f"{paper}.json",
                ANNOTATIONS_DIR / second / f"{paper}.json",
                plan_path=PLAN_PATH,
                identity_gold_path=IDENTITY_GOLD_DIR / f"{paper}.json",
            )
            validate_evaluator_provenance(
                report.get("evaluator", {}), expected_evaluator, source=str(report_path),
            )
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except FileNotFoundError as exc:
            logger.error("%s/%s: missing input: %s", batch_name, paper, exc)
            missing += 1
        except Exception as exc:
            logger.error("%s/%s: %s", batch_name, paper, exc)
            failed += 1
        else:
            reports.append(report)
    print(f"{batch_name}: {len(reports)}/{len(papers)} scored, {missing} missing, {failed} failed.")
    return reports, missing, failed


def main(argv: list[str] | None = None) -> int:
    """Evaluate all three annotation batches and write their pooled agreement."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    try:
        batches = annotation_batches(CORPUS_PATH)
        total = sum(len(batch["papers"]) for batch in batches.values())
        if len(batches) != 3 or total != 30:
            raise ValueError("Expected three annotation batches covering 30 evaluation papers.")
        expected_evaluator = build_evaluator_metadata(PLAN_PATH)
        if expected_evaluator["schema_version_mismatch"]:
            raise ValueError("The scoring plan's declared schema versions do not match the schema files.")
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    reports = []
    missing = failed = 0
    summary_path = OUTPUT_DIR / "summary.json"
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path.unlink(missing_ok=True)
        for batch_name, batch in sorted(batches.items()):
            batch_reports, batch_missing, batch_failed = compute_batch(
                batch_name, batch["annotators"], batch["papers"], expected_evaluator,
            )
            reports.extend(batch_reports)
            missing += batch_missing
            failed += batch_failed

        summary = {
            "n_papers_total": total,
            "n_papers_scored": len(reports),
            "n_papers_missing": missing,
            "n_papers_failed": failed,
            "evaluator": expected_evaluator,
            # Pool paper-level counts before computing F1, rather than averaging batch F1s.
            "aggregated": aggregate_reports(reports) if reports else {},
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error("Cannot write IAA reports: %s", exc)
        return 1
    print(f"Total: {len(reports)}/{total} scored, {missing} missing, {failed} failed. Summary: {summary_path}")
    return 1 if missing or failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
