"""Offline M07 repair-policy mechanics on explicitly pinned invented fixtures.

The initial M06 judgment is interpreted without dispatch. Fixed, issue-aware and
generic mock policies share the same initial record, issues, tools and budgets.
This diagnostic is not a natural-paper performance or independent-label study.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
from typing import Literal

from pydantic import Field, model_validator

from meta_data_extraction.config_validation import load_yaml_mapping
from .contracts import digest
from .evidence import FrozenModel, sha256
from .inspect_record import InputPin, RecordRunConfig, _targets
from .paths import input_file, validate_run_id, workspace_root
from .profiles import load_profile
from .providers import ProviderResult
from .record_checks import check_structure
from .record_patches import build_candidate
from .sources import prepare_sources
from .storage import RunIncompleteError
from .support_checks import interpret_support_response, verify_report
from .workflow import WorkflowMockProvider, WorkflowRuntime
from .workflow_graph import run_workflow_graph
from .workflow_tools import ToolContext
from .workflow_types import WorkflowBudget, WorkflowIssue, WorkflowState

MAX_INPUT_BYTES = 1_000_000
CHECKER_MODEL = "m07-synthetic-checker"
POLICY_MODEL = "m07-synthetic-policy"
SEED_MODEL = "m06-synthetic-checker"
PROPOSAL_ID = "correct_dataset_a"
RETRIEVAL_QUERY = "Dataset A Sample A 300 500"


class WorkflowRunConfig(FrozenModel):
    mode: Literal["synthetic_offline"] = "synthetic_offline"
    run_id: str
    policy: Literal["fixed", "issue_aware", "generic_llm"]
    seed: InputPin
    checker_responses: tuple[InputPin, ...] = Field(min_length=2, max_length=2)
    policy_responses: tuple[InputPin, ...] = Field(default=(), max_length=32)
    budget: WorkflowBudget
    raw_retention: Literal["hash_only"]

    @model_validator(mode="after")
    def _scope(self) -> WorkflowRunConfig:
        validate_run_id(self.run_id)
        if bool(self.policy_responses) != (self.policy == "generic_llm"):
            raise ValueError("Only a generic policy requires explicit mock policy responses")
        return self


def _bytes(root: Path, relative: str, *, roots: tuple[str, ...]) -> bytes:
    path = input_file(root, relative, allowed_roots=roots)
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Workflow inspection input exceeds the size bound")
    raw = path.read_bytes()
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Workflow inspection input exceeds the size bound")
    return raw


def load_config(root: Path | str, relative: str) -> WorkflowRunConfig:
    raw = _bytes(workspace_root(root), relative, roots=("configs/agent",))
    return WorkflowRunConfig.model_validate(load_yaml_mapping(raw.decode("utf-8"), source=relative))


def _implementation(root: Path) -> dict[str, str]:
    names = (
        "agent/inspect_workflow.py", "agent/inspect_record.py", "agent/sources.py", "agent/config.py",
        "config_validation.py", "components/parser.py", "components/llm.py", "base/llm.py",
        "utils.py", "data_models.py",
    )
    return {name: sha256(input_file(root, "src/meta_data_extraction/" + name,
                                   allowed_roots=("src/meta_data_extraction",)).read_bytes()) for name in names}


def inspect_workflow(root: Path | str, config: WorkflowRunConfig, *, run_id: str | None = None,
                     max_steps: int | None = None) -> dict:
    """Prepare one frozen synthetic study and delegate execution/resumption to the graph."""
    root = workspace_root(root)
    config = WorkflowRunConfig.model_validate(config.model_dump())
    run_id = validate_run_id(run_id or config.run_id)
    pins: dict[str, str] = {}
    raw_inputs: dict[str, bytes] = {}

    def pinned(pin: InputPin, *, roots=("tests/agent/fixtures",)) -> bytes:
        # Validate each path's allowed scope even when a prior read was cached.
        input_file(root, pin.path, allowed_roots=roots)
        if pin.path not in raw_inputs:
            raw_inputs[pin.path] = _bytes(root, pin.path, roots=roots)
        raw = raw_inputs[pin.path]
        if sha256(raw) != pin.sha256:
            raise ValueError("Workflow fixture differs from its declared hash")
        pins[pin.path] = pin.sha256
        return raw

    raw_seed = pinned(config.seed, roots=("configs/agent",))
    seed = RecordRunConfig.model_validate(load_yaml_mapping(raw_seed.decode("utf-8"), source="pinned M06 seed"))
    if seed.patch is None or seed.before_response is None or not seed.patch_policy.accept_model_judgments:
        raise ValueError("Synthetic seed requires an explicit correction and checker fixture")
    pinned(seed.source)
    raw_record = pinned(seed.record)
    raw_seed_response = pinned(seed.before_response).decode("utf-8")
    checker_results = tuple(ProviderResult(status="ok", mode="mock", finish_reason="stop",
                            content=pinned(pin).decode("utf-8")) for pin in config.checker_responses)
    policy_results = tuple(ProviderResult(status="ok", mode="mock", finish_reason="stop",
                           content=pinned(pin).decode("utf-8")) for pin in config.policy_responses)
    prepared = prepare_sources(root, [seed.source.path])[0]
    if prepared.bundle is None or prepared.bundle.source_sha256 != seed.source.sha256:
        raise ValueError("Pinned source changed or could not be prepared")
    bundle = prepared.bundle
    profile = load_profile(root, seed.schema_names)
    canonical = load_yaml_mapping(raw_record.decode("utf-8"), source="pinned synthetic record")
    structure = check_structure(profile, bundle, canonical)
    if structure.snapshot is None:
        raise ValueError("Workflow seed must be structurally valid")
    record = structure.snapshot
    response = ProviderResult(status="ok", mode="mock", finish_reason="stop", content=raw_seed_response)
    initial_support = interpret_support_response(profile, record, bundle, _targets(bundle, seed.targets),
                                                response, model_name=SEED_MODEL, raw_retention="hash_only")
    verify_report(initial_support, profile, record, bundle, root=root)
    candidate = build_candidate(profile, bundle, record, seed.patch)
    if (candidate.prepared is None or initial_support.status != "succeeded"
            or len(initial_support.judgments) != 2
            or any(j.verdict != "contradicted" for j in initial_support.judgments)
            or {j.path for j in initial_support.judgments} != set(candidate.prepared.changed_paths)):
        raise ValueError("Synthetic seed requires two complete contradicted assertions and a bound correction")
    issues = tuple(WorkflowIssue(id=f"issue-{index + 1}", path=judgment.path,
                                diagnosis=judgment.diagnosis, summary=judgment.summary,
                                query=RETRIEVAL_QUERY, proposal_ids=(PROPOSAL_ID,), definition_keys=(), evidence=())
                   for index, judgment in enumerate(initial_support.judgments))
    initial = WorkflowState(initial_revision=record.revision_id, record=record, issues=issues)
    checker = WorkflowMockProvider(checker_results, model_name=CHECKER_MODEL)
    policy_provider = WorkflowMockProvider(policy_results, model_name=POLICY_MODEL) if policy_results else None
    tools = ToolContext(root=root, profile=profile, bundle=bundle, proposals={PROPOSAL_ID: seed.patch},
                        checker=checker, patch_policy=seed.patch_policy)
    runtime = WorkflowRuntime(tools=tools, budget=config.budget, policy=config.policy,
                              policy_provider=policy_provider)
    request = {
        "version": "m07-inspection-1", "mode": "synthetic_offline",
        "config": config.model_dump(mode="json"), "seed_config": seed.model_dump(mode="json"),
        "input_sha256": pins, "source_snapshot_sha256": digest(bundle.model_dump(mode="json")),
        "seed_support": initial_support.model_dump(mode="json"),
        "seed_execution": "pinned_fixture_interpretation", "seed_new_model_calls": 0,
        "implementation": _implementation(root),
        "dependencies": {name: version(name) for name in ("PyYAML", "openai", "httpx", "pymupdf4llm", "pymupdf")},
        "standards_resources": [], "external_tracing": "disabled", "live_requests": 0,
        "comparison_scope": "shared_initial_state_repair_policy_mechanics",
        "independent_labels": "not_supplied",
    }
    return run_workflow_graph(root, run_id, request, initial, runtime, max_steps=max_steps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    try:
        result = inspect_workflow(args.root, load_config(args.root, args.config),
                                  run_id=args.run_id, max_steps=args.max_steps)
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error, RunIncompleteError) as error:
        parser.exit(2, f"Workflow inspection failed ({type(error).__name__}); no successful workflow report.\n")
    state = result["state"]
    print(json.dumps({
        "execution": result["execution"], "status": state.status,
        "new_model_calls": result["new_model_calls"], "new_live_requests": result["new_live_requests"],
        "total_model_calls": state.model_calls, "uncertain_model_calls": state.uncertain_model_calls,
        "actions": len(state.events), "pending_reviews": len(state.pending_reviews),
        "unresolved_issues": sum(issue.status != "resolved" for issue in state.issues),
        "usage_unknown": state.usage_unknown, "coverage_status": "unsupported",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
