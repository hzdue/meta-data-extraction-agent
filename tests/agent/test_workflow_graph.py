"""Offline SQLite restart, dispatch replay, and run-boundary diagnostics for M07."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from test_workflow import workflow_case, tool_case, selected, retrieval
from test_workflow_tools import response
from meta_data_extraction.agent.contracts import digest
from meta_data_extraction.agent.providers import MockProvider, ProviderResult
from meta_data_extraction.agent.storage import RunIncompleteError, _digest, _json_bytes
from meta_data_extraction.agent.workflow import WorkflowMockProvider, WorkflowRuntime
from meta_data_extraction.agent.workflow_types import WorkflowBudget, WorkflowState
import meta_data_extraction.agent.workflow_graph as graph

REQUEST = {"diagnostic": "explicitly-invented-m07", "split": "synthetic"}
RUN_ID = "m07-graph-unit"


def execution(case, *, policy="issue_aware", actions=None, budget=None):
    context, _, prepared = case
    checker = WorkflowMockProvider((response(prepared.changed_paths, verdict="contradicted"),
                                    response(prepared.required_support_paths)), model_name="mock-workflow-checker")
    context = replace(context, checker=checker)
    provider = None if policy != "generic_llm" else WorkflowMockProvider(tuple(selected(a) for a in actions), model_name="mock-workflow-policy")
    return WorkflowRuntime(context, budget or WorkflowBudget(timeout_seconds=300), policy, policy_provider=provider)


def run(case, runtime=None, *, run_id=RUN_ID, request=None, **kwargs):
    context, state, _ = case
    return graph.run_workflow_graph(context.root, run_id, REQUEST if request is None else request, state,
                                     runtime or execution(case), **kwargs)


def run_dir(case):
    return case[0].root / "runs/agent" / RUN_ID


def test_complete_run_and_replay_never_repeat_model_calls(workflow_case, monkeypatch):
    execution1 = execution(workflow_case)
    first = run(workflow_case, execution1)
    assert first["execution"] == "executed" and first["state"].status == "finished"
    assert first["state"].record == workflow_case[2].candidate
    assert first["new_model_calls"] == 2 and first["new_live_requests"] == 0
    assert len(execution1.tools.checker.requests) == 2
    canonical = json.loads((run_dir(workflow_case) / "canonical.json").read_bytes())
    assert canonical == workflow_case[2].candidate.canonical
    before = {p.name: p.read_bytes() for p in run_dir(workflow_case).iterdir() if p.is_file()}
    execution2 = execution(workflow_case)
    monkeypatch.setattr(execution2.tools.checker, "generate", lambda *a: pytest.fail("Completed replay must not call a model"))
    replay = run(workflow_case, execution2)
    assert replay["execution"] == "cached_replay" and replay["state"] == first["state"]
    assert replay["new_model_calls"] == replay["new_live_requests"] == 0
    assert {p.name: p.read_bytes() for p in run_dir(workflow_case).iterdir() if p.is_file()} == before


@pytest.mark.parametrize("steps,calls_before,calls_after", [(1, 0, 2), (2, 2, 0)])
def test_committed_checkpoint_reopens_with_fresh_provider_objects(workflow_case, steps, calls_before, calls_after):
    first_runtime = execution(workflow_case)
    paused = run(workflow_case, first_runtime, max_steps=steps)
    assert paused["execution"] == "checkpointed" and paused["state"].status == "running"
    assert paused["new_model_calls"] == calls_before
    assert not (run_dir(workflow_case) / "result.json").exists()
    assert (run_dir(workflow_case) / "checkpoints.sqlite").is_file()
    assert (run_dir(workflow_case) / "dispatch.sqlite").is_file()
    second_runtime = execution(workflow_case)
    resumed = run(workflow_case, second_runtime)
    assert resumed["execution"] == "resumed" and resumed["state"].status == "finished"
    assert resumed["new_model_calls"] == calls_after
    assert len(second_runtime.tools.checker.requests) == calls_after
    assert resumed["state"].model_calls == 2 and len(resumed["state"].events) == 3
    assert resumed["state"].events[:steps] == paused["state"].events


def test_generic_provider_resume_positions_frozen_script_without_fake_calls(workflow_case):
    actions = [retrieval(), {"kind": "propose_update", "issue_id": "issue-0", "proposal_id": "correction"},
               {"kind": "finish", "outcome": "finished", "summary": "All supplied issues resolved"}]
    first = run(workflow_case, execution(workflow_case, policy="generic_llm", actions=actions), max_steps=2)
    assert first["execution"] == "checkpointed" and first["new_model_calls"] == 4
    second_runtime = execution(workflow_case, policy="generic_llm", actions=actions)
    second = run(workflow_case, second_runtime)
    assert second["execution"] == "resumed" and second["state"].status == "finished"
    assert second["new_model_calls"] == 1 and second["state"].model_calls == 5
    assert len(second_runtime.policy_provider.requests) == 1 and second_runtime.policy_provider.position == 3
    assert not second_runtime.tools.checker.requests and second_runtime.tools.checker.position == 2


def test_real_subprocess_resume_reads_closed_sqlite_and_preserves_checkpoint(workflow_case, tmp_path):
    first_runtime = execution(workflow_case)
    paused = run(workflow_case, first_runtime, max_steps=1)
    context, initial, prepared = workflow_case
    payload = {
        "root": str(context.root), "run_id": RUN_ID, "request": REQUEST,
        "profile": context.profile.model_dump(mode="json"), "bundle": context.bundle.model_dump(mode="json"),
        "proposals": {key: value.model_dump(mode="json") for key, value in context.proposals.items()},
        "policy": context.patch_policy.model_dump(mode="json"), "initial": initial.model_dump(mode="json"),
        "results": [response(prepared.changed_paths, verdict="contradicted").model_dump(mode="json"),
                    response(prepared.required_support_paths).model_dump(mode="json")],
    }
    payload_path = tmp_path / "subprocess-input.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    script = r'''
import json, socket, sys
from pathlib import Path
from meta_data_extraction.agent.evidence import DocumentBundle
from meta_data_extraction.agent.profiles import Profile
from meta_data_extraction.agent.providers import ProviderResult
from meta_data_extraction.agent.record_patches import PatchPolicy, PatchProposal
from meta_data_extraction.agent.workflow import WorkflowRuntime, WorkflowMockProvider
from meta_data_extraction.agent.workflow_tools import ToolContext
from meta_data_extraction.agent.workflow_types import WorkflowBudget, WorkflowState
from meta_data_extraction.agent.workflow_graph import run_workflow_graph

def denied(*args, **kwargs):
    raise AssertionError("Subprocess network is forbidden")
socket.socket.connect = denied
socket.create_connection = denied
socket.getaddrinfo = denied
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checker = WorkflowMockProvider(tuple(ProviderResult.model_validate(x) for x in p["results"]), model_name="mock-workflow-checker")
context = ToolContext(p["root"], Profile.model_validate(p["profile"]), DocumentBundle.model_validate(p["bundle"]),
                      {k: PatchProposal.model_validate(v) for k, v in p["proposals"].items()}, checker,
                      PatchPolicy.model_validate(p["policy"]))
runtime = WorkflowRuntime(context, WorkflowBudget(timeout_seconds=300), "issue_aware")
result = run_workflow_graph(p["root"], p["run_id"], p["request"], WorkflowState.model_validate(p["initial"]), runtime)
print(json.dumps({"execution": result["execution"], "state": result["state"].model_dump(mode="json"),
                  "new_model_calls": result["new_model_calls"], "actual_checker_calls": len(checker.requests)}))
'''
    completed = subprocess.run([sys.executable, "-c", script, str(payload_path)], check=True,
                               capture_output=True, text=True, timeout=40,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    result = json.loads(completed.stdout)
    resumed = WorkflowState.model_validate(result["state"])
    assert result["execution"] == "resumed" and resumed.status == "finished"
    assert result["new_model_calls"] == result["actual_checker_calls"] == 2
    assert resumed.events[0] == paused["state"].events[0]
    assert run(workflow_case)["state"] == resumed


def test_journal_completed_step_replays_after_checkpoint_write_failure(workflow_case, monkeypatch):
    original_complete = graph.DispatchJournal.complete
    once = []

    def crash_after_commit(self, before, after):
        original_complete(self, before, after)
        if not once:
            once.append(True)
            raise RuntimeError("Invented crash after durable transition")

    monkeypatch.setattr(graph.DispatchJournal, "complete", crash_after_commit)
    with pytest.raises(RuntimeError, match="Invented crash"):
        run(workflow_case)
    monkeypatch.setattr(graph.DispatchJournal, "complete", original_complete)
    result = run(workflow_case)
    assert result["execution"] == "resumed" and result["state"].status == "finished"
    assert len(result["state"].events) == 3 and result["state"].model_calls == 2
    with sqlite3.connect(run_dir(workflow_case) / "dispatch.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 3


def test_claimed_transition_with_unknown_outcome_never_retries_a_model(workflow_case, monkeypatch):
    original_advance = graph.advance
    actions = [{"kind": "propose_update", "issue_id": "issue-0", "proposal_id": "correction"}]
    first_runtime = execution(workflow_case, policy="generic_llm", actions=actions)

    def crash_after_call(state, runtime):
        original_advance(state, runtime)
        raise RuntimeError("Invented crash before journaling model outcome")

    monkeypatch.setattr(graph, "advance", crash_after_call)
    with pytest.raises(RuntimeError, match="Invented crash"):
        run(workflow_case, first_runtime)
    assert len(first_runtime.policy_provider.requests) == 1 and len(first_runtime.tools.checker.requests) == 2
    monkeypatch.setattr(graph, "advance", original_advance)
    second_runtime = execution(workflow_case, policy="generic_llm", actions=actions)
    result = run(workflow_case, second_runtime)
    assert result["execution"] == "resumed" and result["state"].status == "interrupted"
    assert result["state"].stop_reason == "dispatch_outcome_unknown"
    assert result["state"].uncertain_model_calls == 3 and result["state"].usage_unknown
    assert result["new_model_calls"] == 0 and not second_runtime.policy_provider.requests
    assert not second_runtime.tools.checker.requests and result["state"].record == workflow_case[1].record
    replay = run(workflow_case, execution(workflow_case, policy="generic_llm", actions=actions))
    assert replay["execution"] == "cached_replay" and replay["state"] == result["state"]


def test_pending_review_persists_across_replay_without_becoming_approval(workflow_case):
    context, initial, prepared = workflow_case
    issue = initial.issues[0].model_copy(update={"diagnosis": "context_conflict", "proposal_ids": ()})
    case = context, initial.model_copy(update={"issues": (issue,)}), prepared
    result = run(case)
    assert result["state"].status == "pending_review" and result["new_model_calls"] == 0
    assert result["state"].pending_reviews[0].status == "pending"
    replay = run(case)
    assert replay["execution"] == "cached_replay" and replay["state"] == result["state"]
    assert replay["state"].record == initial.record


def test_cancel_file_on_resume_prevents_remaining_checker_calls(workflow_case):
    paused = run(workflow_case, max_steps=1)
    (run_dir(workflow_case) / "cancel.request").write_text("User requested cancellation", encoding="utf-8")
    runtime = execution(workflow_case)
    result = run(workflow_case, runtime)
    assert result["execution"] == "resumed" and result["state"].status == "cancelled"
    assert result["state"].record == paused["state"].record
    assert result["new_model_calls"] == 0 and not runtime.tools.checker.requests
    assert run(workflow_case)["state"] == result["state"]


def test_original_deadline_persists_across_restart(workflow_case, monkeypatch):
    paused = run(workflow_case, max_steps=1)
    actual = graph.time.time
    later = actual() + 301
    monkeypatch.setattr(graph.time, "time", lambda: later)
    resumed = run(workflow_case)
    assert resumed["state"].status == "timed_out" and resumed["new_model_calls"] == 0
    assert resumed["state"].record == paused["state"].record
    assert run(workflow_case)["execution"] == "cached_replay"


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "1"])
def test_checkpoint_step_limit_is_strict_before_run_claim(workflow_case, bad):
    with pytest.raises(ValueError):
        run(workflow_case, max_steps=bad)
    assert not run_dir(workflow_case).exists()


@pytest.mark.parametrize("bad", ["../outputs", "outputs/result", "_control", "C:/outside", "../paper", ""])
def test_unsafe_run_ids_never_create_or_overwrite_output(workflow_case, bad):
    with pytest.raises(ValueError):
        run(workflow_case, run_id=bad)
    assert not (workflow_case[0].root / "outputs").exists()


@pytest.mark.parametrize("change", ["request", "budget", "code", "provider"])
def test_changed_frozen_bindings_cannot_resume_dispatch(workflow_case, monkeypatch, change):
    run(workflow_case, max_steps=1)
    runtime = execution(workflow_case)
    request = REQUEST
    if change == "request": request = {**REQUEST, "changed": True}
    elif change == "budget": runtime.budget = WorkflowBudget(timeout_seconds=299)
    elif change == "code": monkeypatch.setattr(graph, "_implementation", lambda: {"changed-code": "a" * 64})
    else:
        runtime.tools = replace(runtime.tools, checker=WorkflowMockProvider((), model_name="changed-checker"))
    with pytest.raises(ValueError):
        run(workflow_case, runtime, request=request)
    assert not runtime.tools.checker.requests


def test_incomplete_claim_without_checkpoint_is_not_implicitly_restarted(workflow_case):
    run(workflow_case, max_steps=1)
    checkpoint = run_dir(workflow_case) / "checkpoints.sqlite"
    checkpoint.rename(checkpoint.with_suffix(".saved.sqlite"))
    runtime = execution(workflow_case)
    with pytest.raises(RunIncompleteError):
        run(workflow_case, runtime)
    assert not runtime.tools.checker.requests


def test_second_worker_cannot_dispatch_while_run_lease_is_held(workflow_case):
    run(workflow_case, max_steps=1)
    run_location = graph.RunDirectory(workflow_case[0].root, RUN_ID)
    runtime = execution(workflow_case)
    with graph._lease(run_location):
        with pytest.raises(graph.WorkflowBusyError):
            run(workflow_case, runtime)
    assert not runtime.tools.checker.requests
    assert run(workflow_case)["state"].status == "finished"


def test_env_and_inherited_tracing_context_are_disabled_during_graph_nodes(workflow_case, monkeypatch):
    from langsmith import tracing_context
    from langsmith.run_helpers import get_tracing_context
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    seen = []
    original = graph.advance

    def inspect_trace(state, runtime):
        seen.append(get_tracing_context()["enabled"])
        return original(state, runtime)

    monkeypatch.setattr(graph, "advance", inspect_trace)
    with tracing_context(enabled=True, tags=["inherited-synthetic-trace"]):
        result = run(workflow_case)
    assert result["state"].status == "finished" and seen and all(value is False for value in seen)
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_rehashed_completion_state_forgery_rejected_against_durable_history(workflow_case):
    first = run(workflow_case)
    directory = run_dir(workflow_case)
    envelope = json.loads((directory / "result.json").read_bytes())
    sidecar = json.loads((directory / "sidecar.json").read_bytes())
    forged = dict(envelope["result"]["state"])
    forged.update(status="unresolved", stop_reason="invented terminal state")
    WorkflowState.model_validate(forged)
    envelope["result"]["state"] = forged
    sidecar["workflow"] = forged
    sidecar_bytes = _json_bytes(sidecar)
    (directory / "sidecar.json").write_bytes(sidecar_bytes)
    envelope["artifacts"]["sidecar.json"] = {"sha256": _digest(sidecar_bytes), "bytes": len(sidecar_bytes)}
    envelope["completion_sha256"] = _digest(_json_bytes({k: v for k, v in envelope.items() if k != "completion_sha256"}))
    (directory / "result.json").write_bytes(_json_bytes(envelope))
    runtime = execution(workflow_case)
    with pytest.raises(ValueError):
        run(workflow_case, runtime)
    assert not runtime.tools.checker.requests


def test_checkpoint_payload_changed_outside_journal_is_rejected_before_dispatch(workflow_case):
    run(workflow_case, max_steps=1)
    serde = graph.JsonPlusSerializer(pickle_fallback=False, allowed_json_modules=[], allowed_msgpack_modules=[])
    with sqlite3.connect(run_dir(workflow_case) / "checkpoints.sqlite") as conn:
        identifier, kind, raw = conn.execute(
            "SELECT checkpoint_id,type,checkpoint FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
        checkpoint = serde.loads_typed((kind, raw))
        checkpoint["channel_values"]["payload"]["elapsed_ms"] += 1
        new_kind, new_raw = serde.dumps_typed(checkpoint)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=? WHERE checkpoint_id=?", (new_kind, new_raw, identifier))
    runtime = execution(workflow_case)
    with pytest.raises(ValueError):
        run(workflow_case, runtime)
    assert not runtime.tools.checker.requests


def test_journal_reservation_tampering_is_rejected_before_dispatch(workflow_case):
    run(workflow_case, max_steps=1)
    with sqlite3.connect(run_dir(workflow_case) / "dispatch.sqlite") as conn:
        conn.execute("UPDATE steps SET reservation=999")
    runtime = execution(workflow_case)
    with pytest.raises(ValueError):
        run(workflow_case, runtime)
    assert not runtime.tools.checker.requests


def test_cancel_file_cannot_follow_an_external_hardlink(workflow_case):
    run(workflow_case, max_steps=1)
    target = workflow_case[0].root / "tests/agent/fixtures/cancel-marker.txt"
    target.write_text("Invented external marker", encoding="utf-8")
    os.link(target, run_dir(workflow_case) / "cancel.request")
    runtime = execution(workflow_case)
    with pytest.raises(ValueError):
        run(workflow_case, runtime)
    assert not runtime.tools.checker.requests
    assert target.read_text(encoding="utf-8") == "Invented external marker"


@pytest.mark.parametrize("tamper", ["status", "issues", "history"])
def test_terminal_checkpoint_after_export_crash_must_match_durable_journal(workflow_case, monkeypatch, tamper):
    original_complete = graph.RunStore.complete

    def crash_before_export(self, *args, **kwargs):
        raise RuntimeError("Invented crash before terminal export")

    monkeypatch.setattr(graph.RunStore, "complete", crash_before_export)
    with pytest.raises(RuntimeError, match="Invented crash"):
        run(workflow_case)
    assert not (run_dir(workflow_case) / "result.json").exists()
    monkeypatch.setattr(graph.RunStore, "complete", original_complete)
    serde = graph.JsonPlusSerializer(pickle_fallback=False, allowed_json_modules=[], allowed_msgpack_modules=[])
    with sqlite3.connect(run_dir(workflow_case) / "checkpoints.sqlite") as conn:
        identifier, kind, raw = conn.execute(
            "SELECT checkpoint_id,type,checkpoint FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
        checkpoint = serde.loads_typed((kind, raw))
        payload = checkpoint["channel_values"]["payload"]
        assert payload["status"] == "finished"
        if tamper == "status":
            payload.update(status="unresolved", stop_reason="Fabricated terminal state")
        elif tamper == "issues":
            payload["issues"][0]["status"] = "open"
            payload.update(status="unresolved", stop_reason="Fabricated issue status")
        else:
            payload["events"][-1]["summary"] = "Fabricated event history"
        WorkflowState.model_validate(payload)
        new_kind, new_raw = serde.dumps_typed(checkpoint)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=? WHERE checkpoint_id=?", (new_kind, new_raw, identifier))
    runtime = execution(workflow_case)
    with pytest.raises(ValueError):
        run(workflow_case, runtime)
    assert not runtime.tools.checker.requests and not (run_dir(workflow_case) / "result.json").exists()


def test_unchanged_terminal_checkpoint_after_export_crash_finishes_without_dispatch(workflow_case, monkeypatch):
    original_complete = graph.RunStore.complete
    saved = []

    def crash_before_export(self, result, **kwargs):
        saved.append(WorkflowState.model_validate(result["state"]))
        raise RuntimeError("Invented crash before terminal export")

    monkeypatch.setattr(graph.RunStore, "complete", crash_before_export)
    with pytest.raises(RuntimeError, match="Invented crash"):
        run(workflow_case)
    monkeypatch.setattr(graph.RunStore, "complete", original_complete)
    runtime = execution(workflow_case)
    result = run(workflow_case, runtime)
    assert result["execution"] == "resumed" and result["state"] == saved[0]
    assert result["new_model_calls"] == 0 and not runtime.tools.checker.requests
    assert (run_dir(workflow_case) / "result.json").is_file()


@pytest.mark.parametrize("mode,status,request_id,requests,invocations,unknown", [
    ("live", "blocked", None, 0, 1, False),
    ("live", "ok", "synthetic-reservation", 1, 1, False),
    ("live", "failed", "synthetic-reservation", 1, 1, False),
    ("live", "timeout", "synthetic-reservation", 1, 1, False),
    ("live", "failed", None, 0, 1, True),
    ("live", "timeout", None, 0, 1, True),
    ("live", "rate_limited", None, 0, 1, True),
    ("mock", "ok", None, 0, 0, False),
])
def test_resource_report_separates_live_dispatch_from_provider_invocation(
        workflow_case, mode, status, request_id, requests, invocations, unknown):
    # Fabricated serialized observations only: no live adapter, ledger, or service.
    from types import SimpleNamespace
    from meta_data_extraction.agent.contracts import ProviderObservation
    from meta_data_extraction.agent.workflow_types import ActionEvent
    initial = workflow_case[1]
    observation = ProviderObservation(status=status, mode=mode, model="synthetic-accounting-observation",
        finish_reason=None, input_tokens=None, output_tokens=None, latency_ms=None,
        error_code=None, request_id=request_id, response_sha256=None)
    event = ActionEvent(index=0, policy="generic_llm", status="failed", code="synthetic_observation",
        revision_before=initial.record.revision_id, revision_after=initial.record.revision_id,
        progress=False, providers=(observation,), summary="Fabricated resource-accounting observation")
    state = initial.model_copy(update={"events": (event,), "status": "failed", "model_calls": 1, "usage_unknown": True})
    configured = SimpleNamespace(policy="generic_llm", tools=SimpleNamespace(checker=SimpleNamespace(mode="mock")),
                                 policy_provider=SimpleNamespace(mode=mode))
    result = graph._result(state, configured)
    assert result["live_requests"] == requests
    assert result["live_provider_invocations"] == invocations
    assert result["live_dispatch_unknown"] is unknown
    assert result["cost"] is None
    assert result["cost_status"] == ("unknown" if mode == "live" else "not_applicable_mock")


@pytest.mark.parametrize("configured_live", [False, True])
def test_interrupted_reservation_reports_potential_live_dispatch_separately(workflow_case, configured_live):
    from types import SimpleNamespace
    initial = workflow_case[1]
    state = initial.model_copy(update={"status": "interrupted", "uncertain_model_calls": 2,
                                       "usage_unknown": True, "stop_reason": "dispatch_outcome_unknown"})
    configured = SimpleNamespace(policy="fixed", tools=SimpleNamespace(checker=SimpleNamespace(
        mode="live" if configured_live else "mock")), policy_provider=None)
    result = graph._result(state, configured)
    assert result["live_requests"] == result["live_provider_invocations"] == 0
    assert result["live_dispatch_unknown"] is configured_live
    assert result["uncertain_model_calls"] == 2 and result["usage_unknown"]
