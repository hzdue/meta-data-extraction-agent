"""Offline M01 source/evidence inspection; outputs stay in a fresh local run."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from meta_data_extraction.agent.config import load_offline_config
from meta_data_extraction.agent.paths import RunDirectory
from meta_data_extraction.agent.retrieval import SourceCorpus
from meta_data_extraction.agent.sources import prepare_sources


@dataclass(frozen=True)
class InspectionRun:
    path: Path
    failed_sources: int


def inspect_sources(root: Path | str, config_path: str, *, query: str,
                    run_id: str | None = None, max_chars: int = 2000,
                    limit: int = 5, source_max_chars: int | None = None) -> InspectionRun:
    config = load_offline_config(config_path, root)
    prepared = prepare_sources(root, config.sources, max_chars=source_max_chars)
    failed = sum(result.status == "failed" for result in prepared)
    full = []
    targeted = None
    if not failed:
        corpus = SourceCorpus(bundles=tuple(result.bundle for result in prepared))
        full = [corpus.full_text(bundle.source_id, max_chars=max_chars).model_dump(mode="json")
                for bundle in corpus.bundles]
        targeted = corpus.search(query, max_chars=max_chars, limit=limit).model_dump(mode="json")
    payload = {
        "kind": "m01_source_inspection", "status": "preparation_failed" if failed else "prepared",
        "model_calls": 0, "query": query, "sources": [r.model_dump(mode="json") for r in prepared],
        "full_text": full, "targeted": targeted,
        "budget_unit": "Unicode code points, not model tokens",
    }
    run = RunDirectory.create(root, config.run_id if run_id is None else run_id)
    return InspectionRun(path=run.write_json("inspection.json", payload), failed_sources=failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/agent/evidence_offline.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--source-max-chars", type=int)
    args = parser.parse_args(argv)
    try:
        result = inspect_sources(Path.cwd(), args.config, query=args.query, run_id=args.run_id,
                                 max_chars=args.max_chars, limit=args.limit,
                                 source_max_chars=args.source_max_chars)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"M01 inspection failed: {exc}\n")
    print(result.path)
    return 2 if result.failed_sources else 0


if __name__ == "__main__":
    raise SystemExit(main())
