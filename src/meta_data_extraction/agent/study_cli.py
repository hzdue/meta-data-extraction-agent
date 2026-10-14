"""Dry-run and execute the frozen M09 synthetic study without live inference.

The runtime command reads only an explicitly named study config. Offline
scoring and report collection are separate commands and imports.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .paths import checked_path, input_file, validate_run_id, workspace_root
from .storage import RunConflictError, RunIncompleteError
from .study_runner import dry_run, run_study
from .study_types import StudySpec

MAX_CONFIG_BYTES = 1_000_000
MAX_CONFIG_DEPTH = 32


def load_config(root: Path | str, relative: str) -> StudySpec:
    """Read one bounded strict JSON object from the runtime config allowlist."""
    path = input_file(workspace_root(root), relative, allowed_roots=("configs/agent",))
    if path.suffix.lower() != ".json":
        raise ValueError("A study config must be an explicitly named JSON file")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("Study config exceeds the size bound")
    with path.open("rb") as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("Study config exceeds the size bound")

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Study config has duplicate object keys")
            value[key] = item
        return value

    def constant(value):
        raise ValueError("Study config contains a nonfinite value")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Study config contains a nonfinite value")
        return number

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=constant, parse_float=finite_float)
        if not isinstance(value, dict):
            raise ValueError("A study config must contain one JSON object")
        pending = [(value, 0)]
        while pending:
            item, depth = pending.pop()
            if depth > MAX_CONFIG_DEPTH:
                raise ValueError("Study config exceeds the JSON structure bound")
            if isinstance(item, dict):
                pending.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
        return StudySpec.model_validate(value)
    except (RecursionError, OverflowError) as error:
        raise ValueError("Study config exceeds the JSON structure bound") from error


def execute(root: Path | str, relative: str, *, run_id: str) -> dict:
    """Claim a fresh study run; historical runs are never implicitly replayed."""
    root = workspace_root(root)
    run_id = validate_run_id(run_id)
    path = checked_path(root, f"runs/agent/{run_id}", allowed_roots=("runs/agent",))
    if path.exists():
        raise FileExistsError("Study CLI requires a fresh run ID")
    return run_study(root, load_config(root, relative), run_id=run_id)


def replay(root: Path | str, relative: str, *, run_id: str) -> dict:
    """Verify and return an existing completion without permitting dispatch."""
    root = workspace_root(root)
    run_id = validate_run_id(run_id)
    directory = checked_path(root, f"runs/agent/{run_id}", allowed_roots=("runs/agent",))
    completion = checked_path(root, f"runs/agent/{run_id}/result.json", allowed_roots=("runs/agent",))
    if not directory.is_dir() or not completion.is_file():
        raise RunIncompleteError("Replay requires an existing completed study run")
    report = run_study(root, load_config(root, relative), run_id=run_id, replay_only=True)
    if report.get("execution") != "cached_replay" or any(
        type(report.get(field)) is not int or report[field] != 0
        for field in ("new_model_calls", "new_live_requests")
    ):
        raise RunConflictError("Replay returned an invalid execution observation")
    return report


def dry_run_summary(report: dict) -> dict:
    """Show the frozen scope and ceilings without dumping code/dependency pins."""
    spec = report["spec"]
    return {
        "version": report["version"], "readiness": report["readiness"],
        "scope": spec["scope"], "authorized_mode": spec["authorized_mode"],
        "spec_sha256": report["spec_sha256"],
        "evaluation_protocol_sha256": spec["evaluation_protocol_sha256"],
        "primary_outcome": spec["primary_outcome"], "counting_policy": spec["counting_policy"],
        "publication_count": len(spec["publications"]), "publications": spec["publications"],
        "variants": spec["variants"], "caps": spec["caps"], "repeats": spec["repeats"],
        "estimated_cells": report["estimated_cells"],
        "maximum_model_calls": report["maximum_model_calls"],
        "max_total_model_calls": report["max_total_model_calls"],
        "live_requests": report["live_requests"], "limitations": report["limitations"],
    }


def run_summary(report: dict) -> dict:
    """Account for every cell while leaving detailed predictions in the run."""
    result = report["result"]
    cells = result["cells"]
    canonical = [cell["canonical"]["sha256"] for cell in cells if cell.get("canonical")]
    return {
        "version": result["version"], "execution": report["execution"],
        "scope": result["spec"]["scope"], "spec_sha256": result["spec_sha256"],
        "new_model_calls": report["new_model_calls"], "new_live_requests": report["new_live_requests"],
        "total_cells": len(cells), "cell_statuses": dict(sorted(Counter(cell["status"] for cell in cells).items())),
        "workflow_statuses": dict(sorted(Counter(cell["workflow_status"] for cell in cells
                                                  if cell.get("workflow_status") is not None).items())),
        "canonical_exports": len(canonical), "canonical_variants": len(set(canonical)),
        "failed_cells": [{key: cell[key] for key in (
            "cell_id", "publication_id", "variant", "cap", "repeat", "error_code",
        )} for cell in cells if cell["status"] != "completed"],
        "budget": result["budget"], "accounting": result["accounting"],
        "limitations": result["limitations"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("dry-run", "Validate inputs and report bounds without writing artifacts or calling a provider."),
        ("run", "Execute an invented offline comparison in a fresh run directory."),
        ("replay", "Verify a completed study without writing artifacts or dispatching providers."),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--root", default=".", help="Repository workspace (default: current directory).")
        command.add_argument("--config", required=True, help="Study JSON under configs/agent/.")
        if name != "dry-run":
            command.add_argument("--run-id", required=True, help=(
                "Fresh runs/agent/ directory name." if name == "run" else "Existing completed study run ID."
            ))
    args = parser.parse_args()
    try:
        if args.command == "dry-run":
            result = dry_run_summary(dry_run(args.root, load_config(args.root, args.config)))
        elif args.command == "replay":
            result = run_summary(replay(args.root, args.config, run_id=args.run_id))
        else:
            result = run_summary(execute(args.root, args.config, run_id=args.run_id))
        output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as error:
        # Exception strings can contain source/config values, filesystem paths,
        # or credentials. Only the exception's implementation-owned type is shown.
        parser.exit(2, f"Study command failed ({type(error).__name__}); inspect the config and selected run's completion state.\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
