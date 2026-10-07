"""Independent count-based expectations for current-report integration tests.

Read per-paper files directly: using the reporting collectors or evaluator's
aggregator here would let the same selection/aggregation bug pass both sides.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def formal_reports(run_name: str, evaluation_dir: Path) -> list[dict]:
    corpus = json.loads((ROOT / "data/corpus.json").read_text(encoding="utf8"))
    paper_ids = sorted(
        paper["arxiv_id"]
        for paper in corpus["papers"]
        if paper["split"] == "evaluation"
    )
    directory = evaluation_dir / run_name
    return [
        json.loads((directory / f"{paper}_evaluation.json").read_text(encoding="utf8"))
        for paper in paper_ids
    ]


def micro_f1(reports: list[dict], path: str) -> float:
    tp, fp, fn = (
        sum(report["fields"][path][key] for report in reports)
        for key in ("tp", "fp", "fn")
    )
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
