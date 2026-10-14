"""Offline input/schema/run-isolation check. Does not extract or evaluate records."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from meta_data_extraction.agent.config import load_offline_config
from meta_data_extraction.agent.paths import RunDirectory, workspace_root
from meta_data_extraction.utils import get_schema_refs, load_merged_schema


def run_smoke(root: Path | str, config_path: str, *, run_id: str | None = None) -> Path:
    root = workspace_root(root)
    config = load_offline_config(config_path, root)
    sources = config.source_files(root)
    schemas = config.schema_files(root)
    # Existing schema loaders accept absolute filenames as well as schema names.
    schema_paths = [str(path) for path in schemas]
    merged = load_merged_schema(schema_paths)
    versions = get_schema_refs(schema_paths)
    source_checks = []
    for path in sources:
        raw = path.read_bytes()
        source_checks.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        })
    payload = {
        "kind": "m00_input_smoke", "status": "checks_passed", "mode": config.mode,
        "model_calls": 0, "extraction_performed": False,
        "sources": source_checks,
        "schemas": [
            {"name": path.name, "version": versions[str(path)],
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in schemas
        ],
        "schema_top_level_keys": sorted(merged),
    }
    run = RunDirectory.create(root, config.run_id if run_id is None else run_id)
    return run.write_json("smoke.json", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/agent/offline.yaml")
    parser.add_argument("--run-id", help="Fresh portable run ID; an existing run is refused")
    args = parser.parse_args(argv)
    try:
        path = run_smoke(Path.cwd(), args.config, run_id=args.run_id)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"M00 smoke failed: {exc}\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
