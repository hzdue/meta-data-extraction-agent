"""Run extraction on the evaluation split, or an explicit list of papers.

Read PDFs from data/raw/<arxiv_id>/paper.pdf. Write results and status.md under
<tracker.output_dir>/<sanitized_config_name>/, skipping existing results unless
--force is given.

Usage:
    uv run python scripts/run_experiment.py --config configs/my_run.yaml
    uv run python scripts/run_experiment.py --config configs/my_run.yaml --papers 2405.14324
    uv run python scripts/run_experiment.py --config configs/my_run.yaml --limit 3
"""

from __future__ import annotations

import argparse
import collections
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .corpus import DEFAULT_CORPUS_PATH, paper_ids
else:
    from corpus import DEFAULT_CORPUS_PATH, paper_ids

from meta_data_extraction.app_config import AppConfig, LLMConfig
from meta_data_extraction.components.tracker import LocalTracker
from meta_data_extraction.pipeline import Pipeline
from meta_data_extraction.utils import load_config

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
CORPUS_PATH = DEFAULT_CORPUS_PATH


def select_papers(
    config: AppConfig, explicit: list[str] | None = None, limit: int | None = None,
) -> list[str]:
    """Select evaluation IDs or explicit IDs, limit them, and reject exemplar targets."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer")
    ids = list(explicit) if explicit is not None else list(paper_ids("evaluation", CORPUS_PATH))
    ids = ids[:limit]
    if config.extractor.shot_mode == "few_shot":
        few_shot = config.extractor.few_shot
        assert few_shot is not None
        leaked = set(ids) & set(few_shot.example_ids)
        if leaked:
            raise ValueError(
                f"Exemplar leakage: {sorted(leaked)} would be extracted with "
                "their own gold answers in the prompt. Refusing to run."
            )
    return ids


def render_status_md(
    config_path: Path,
    llm_config: LLMConfig,
    config_dir: Path,
    started_at: str,
    elapsed_s: float,
    outcomes: list[dict],
) -> str:
    """Summarize this invocation's outcomes, usage, and warnings."""
    counts = collections.Counter(o["status"] for o in outcomes)
    lines = [
        f"# Extraction status — {config_path.stem}",
        "",
        f"- Config: `{config_path}`",
        f"- Model: `{llm_config.model}` @ `{llm_config.base_url or 'openai default'}`",
        f"- Folder: `{config_dir}`",
        f"- Run at: {started_at}",
        f"- Duration: {elapsed_s:.1f} s",
        (
            f"- Attempted: {len(outcomes)} "
            f"(extracted: {counts.get('extracted', 0)}, "
            f"skipped: {counts.get('skipped', 0)}, "
            f"failed: {counts.get('failed', 0)}, "
            f"missing PDF: {counts.get('missing_pdf', 0)})"
        ),
        "",
        "| arxiv_id | status | tokens (in/out) | latency (ms) | notes |",
        "|---|---|---:|---:|---|",
    ]
    for o in outcomes:
        tokens = ""
        if o.get("input_tokens") is not None:
            tokens = f"{o.get('input_tokens')}/{o.get('output_tokens')}"
        latency = (
            f"{o['latency_ms']:.0f}" if o.get("latency_ms") is not None else ""
        )
        notes = (o.get("notes") or "").replace("|", "\\|")
        lines.append(
            f"| `{o['arxiv_id']}` | {o['status']} | {tokens} | {latency} | {notes} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the selected papers and return a nonzero status for incomplete runs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run a metadata extraction experiment over the paper corpus."
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="CONFIG",
        help="Path to the YAML config file (e.g. configs/my_run.yaml).",
    )
    parser.add_argument(
        "--papers",
        nargs="+",
        metavar="ARXIV_ID",
        default=None,
        help="Run on these arXiv IDs instead of the 30 evaluation papers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Select the first N papers (positive integer, before skipping existing results).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract papers whose result JSON already exists; by default, skip them.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    try:
        config = load_config(config_path)
        requested_ids = select_papers(config, args.papers, args.limit)
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    config_name = config_path.stem
    if not requested_ids:
        logger.error("No papers selected.")
        return 1
    logger.info("Config: %s | model: %s", config_path.name, config.llm.model)
    logger.info("Papers to process: %d", len(requested_ids))

    tracker = LocalTracker(output_dir=config.tracker.output_dir)
    config_dir = tracker.config_dir(config_name)
    logger.info("Output folder: %s", config_dir)

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_start = time.monotonic()
    outcomes: list[dict] = []
    pipeline = None

    for arxiv_id in requested_ids:
        result_path = tracker.result_path(arxiv_id, config_name)
        if result_path.exists() and not args.force:
            logger.info(
                "%s: already extracted at %s; skipping (use --force to re-run)",
                arxiv_id, result_path,
            )
            outcomes.append({
                "arxiv_id": arxiv_id,
                "status": "skipped",
                "notes": "result already exists",
            })
            continue

        pdf_path = RAW_DIR / arxiv_id / "paper.pdf"
        if not pdf_path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            outcomes.append({
                "arxiv_id": arxiv_id,
                "status": "missing_pdf",
                "notes": f"PDF not found at {pdf_path}",
            })
            continue

        if pipeline is None:
            try:
                pipeline = Pipeline.from_config(config, config_path=config_path)
            except (OSError, ValueError) as exc:
                logger.error("%s", exc)
                return 1

        logger.info("Processing %s ...", arxiv_id)
        try:
            result = pipeline.run(pdf_path)
        except Exception as exc:
            logger.error("%s failed: %s", arxiv_id, exc)
            outcomes.append({
                "arxiv_id": arxiv_id,
                "status": "failed",
                "notes": f"{type(exc).__name__}: {exc}",
            })
            continue

        for warning in result.warnings:
            logger.warning("%s: %s", arxiv_id, warning)

        logger.info(
            "%s done — %d input tokens, %d output tokens, %.0f ms",
            arxiv_id,
            result.provenance.input_tokens or 0,
            result.provenance.output_tokens or 0,
            result.provenance.latency_ms or 0,
        )
        outcomes.append({
            "arxiv_id": arxiv_id,
            "status": "extracted",
            "input_tokens": result.provenance.input_tokens,
            "output_tokens": result.provenance.output_tokens,
            "latency_ms": result.provenance.latency_ms,
            "notes": "; ".join(result.warnings) if result.warnings else "",
        })

    elapsed = time.monotonic() - t_start
    counts = collections.Counter(o["status"] for o in outcomes)

    config_dir.mkdir(parents=True, exist_ok=True)
    status_md = render_status_md(
        config_path=config_path,
        llm_config=config.llm,
        config_dir=config_dir,
        started_at=started_at,
        elapsed_s=elapsed,
        outcomes=outcomes,
    )
    status_path = config_dir / "status.md"
    status_path.write_text(status_md, encoding="utf-8")
    logger.info("Status: %s", status_path)

    logger.info(
        "Done: %d extracted, %d skipped, %d failed, %d missing PDF in %.1f s. Folder: %s",
        counts.get("extracted", 0), counts.get("skipped", 0),
        counts.get("failed", 0), counts.get("missing_pdf", 0),
        elapsed, config_dir,
    )

    return 1 if counts["failed"] or counts["missing_pdf"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
