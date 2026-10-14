"""Focused local human-review commands for an existing pending workflow run.

Reviewer identity is supplied by the caller and is not authenticated here.
Simulated responses remain simulations. Pending review never implies approval.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .paths import input_file, validate_run_id, workspace_root
from .review_store import ReviewStore
from .review_types import ReviewError, ReviewResponse

MAX_RESPONSE_BYTES = 1_000_000


def load_response(root: Path | str, run_id: str, relative: str) -> ReviewResponse:
    """Read one explicitly named JSON response, never discover reviewer files."""
    root = workspace_root(root)
    run_id = validate_run_id(run_id)
    path = input_file(root, relative, allowed_roots=(
        "tests/agent/fixtures", f"runs/agent/{run_id}/review-responses",
    ))
    if path.suffix.lower() != ".json":
        raise ValueError("A review response must be an explicitly named JSON file")
    if path.stat().st_size > MAX_RESPONSE_BYTES:
        raise ValueError("Review response exceeds the size bound")
    raw = path.read_bytes()
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Review response exceeds the size bound")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Review response has duplicate object keys")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Review response contains a nonfinite value")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
        if not isinstance(value, dict):
            raise ValueError("A review response must be one JSON object")
        return ReviewResponse.model_validate(value)
    except (RecursionError, OverflowError) as error:
        raise ValueError("Review response exceeds the JSON structure bound") from error


def show_review(root: Path | str, run_id: str) -> dict:
    with ReviewStore.open(workspace_root(root), validate_run_id(run_id)) as store:
        return store.view()


def submit_review(root: Path | str, run_id: str, response_path: str) -> dict:
    """Commit once, then export; resume/export can recover a failed publication."""
    root = workspace_root(root)
    run_id = validate_run_id(run_id)
    response = load_response(root, run_id, response_path)
    with ReviewStore.open(root, run_id) as store:
        result = store.submit(response)
        state = result["state"]
        manifest = store.export()
        return {
            "execution": result["execution"], "state_id": state.state_id,
            "record_revision": state.current.revision_id, "status": state.status,
            "pending_cases": sum(case.status == "pending" for case in state.cases),
            "response_provenance": response.provenance, "response_outcome": response.outcome,
            "reviewer_identity": "self_declared_unverified", "export": manifest,
        }


def export_review(root: Path | str, run_id: str) -> dict:
    """Export the committed state without dispatching tools or adding a decision."""
    with ReviewStore.open(workspace_root(root), validate_run_id(run_id)) as store:
        return store.export()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    help_text = {
        "show": "Show focused questions, source evidence, and current review bindings.",
        "submit": "Submit one explicit response and export the committed result.",
        "export": "Export the current committed review state without a new decision.",
        "resume": "Resume by exporting the current committed state; no response is invented.",
    }
    for command, description in help_text.items():
        subparser = commands.add_parser(command, help=description)
        subparser.add_argument("--root", default=".", help="Repository workspace (default: current directory).")
        subparser.add_argument("--run-id", required=True, help="Existing pending workflow run ID.")
        if command == "submit":
            subparser.add_argument("--response", required=True, help=(
                "JSON under this run's review-responses/ or tests/agent/fixtures/. "
                "Declare reviewer identity and simulated/user_supplied provenance explicitly."
            ))
    args = parser.parse_args()
    try:
        if args.command == "show":
            result = show_review(args.root, args.run_id)
        elif args.command == "submit":
            result = submit_review(args.root, args.run_id, args.response)
        else:
            result = export_review(args.root, args.run_id)
    except ReviewError as error:
        code = error.code if isinstance(error.code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,99}", error.code) else "review_rejected"
        parser.exit(2, f"Review rejected ({code}); use show to inspect current state and review bindings.\n")
    except Exception as error:
        # No response excerpts, reviewer values, source text, or credentials in
        # parser/SQLite/provider exception strings may reach the terminal.
        parser.exit(2, f"Review command failed ({type(error).__name__}); use show to inspect state or resume to retry export.\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
