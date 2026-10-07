"""The released paper index, evaluation split, and annotation pairs."""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus.json"


def load_corpus(path: Path | str = DEFAULT_CORPUS_PATH) -> dict:
    """Read the index and reject ambiguous split or annotation membership."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = data["annotation_pairs"]
    for batch, pair in pairs.items():
        if not isinstance(pair, list) or len(pair) != 2 or len(set(pair)) != 2:
            raise ValueError(f"{path}: {batch} must name two distinct annotators")

    seen = set()
    for paper in data["papers"]:
        paper_id = paper["arxiv_id"]
        if not isinstance(paper_id, str) or not paper_id or paper_id in seen:
            raise ValueError(f"{path}: invalid or duplicate paper ID {paper_id!r}")
        seen.add(paper_id)
        split = paper["split"]
        if split not in {"calibration", "evaluation"}:
            raise ValueError(f"{path}: invalid split for {paper_id}: {split!r}")
        if split == "evaluation" and paper.get("annotation_batch") not in pairs:
            raise ValueError(f"{path}: missing annotation pair for {paper_id}")
        if split == "calibration" and "annotation_batch" in paper:
            raise ValueError(f"{path}: calibration paper {paper_id} has an evaluation batch")
    return data


def paper_ids(
    split: str | None = None, corpus_path: Path | str = DEFAULT_CORPUS_PATH
) -> tuple[str, ...]:
    """Return sorted paper IDs, optionally restricted to one explicit split."""
    if split not in {None, "calibration", "evaluation"}:
        raise ValueError(f"Unknown corpus split: {split!r}")
    return tuple(sorted(
        paper["arxiv_id"]
        for paper in load_corpus(corpus_path)["papers"]
        if split is None or paper["split"] == split
    ))


def annotation_batches(corpus_path: Path | str = DEFAULT_CORPUS_PATH) -> dict[str, dict]:
    """Group evaluation IDs by their ordered annotation pair for IAA."""
    data = load_corpus(corpus_path)
    return {
        batch: {
            "annotators": pair,
            "papers": sorted(
                paper["arxiv_id"] for paper in data["papers"]
                if paper.get("annotation_batch") == batch
            ),
        }
        for batch, pair in data["annotation_pairs"].items()
    }
