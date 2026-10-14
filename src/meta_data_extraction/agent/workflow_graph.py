"""Thin local LangGraph/SQLite adapter with an at-most-once dispatch journal.

A completed transition can replay after a checkpoint crash. A claimed transition
without a result stops as interrupted: uncertain work is never retried implicitly.
This is local integrity checking, not authentication of an untrusted filesystem.
"""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sqlite3
import time
from typing import TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from .contracts import canonical_json, digest
from .evidence import sha256
from .paths import RunDirectory, checked_path
from .storage import RunIncompleteError, RunStore, _check_request, _digest, _json_bytes
from .workflow import WorkflowRuntime, _replace, advance, verify_state
from .workflow_types import WorkflowState

MAX_STATE_BYTES = 8_000_000
JOURNAL_VERSION = "m07-dispatch-1"


class WorkflowBusyError(ValueError):
    """Another process owns this run's dispatch lease."""


def _path(run: RunDirectory, name: str) -> Path:
    return checked_path(run.root, f"runs/agent/{run.run_id}/{name}", allowed_roots=("runs/agent",))


@contextmanager
def _lease(run: RunDirectory):
    path = _path(run, "workflow.lock")
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise WorkflowBusyError("Workflow run is already active") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _state_json(state: WorkflowState) -> str:
    raw = canonical_json(state.model_dump(mode="json"))
    if len(raw.encode("utf-8")) > MAX_STATE_BYTES:
        raise ValueError("Workflow state exceeds the persistence bound")
    return raw


def _database_path(run: RunDirectory, name: str) -> Path:
    for suffix in ("", "-wal", "-shm", "-journal"):
        _path(run, name + suffix)
    return _path(run, name)


class DispatchJournal:
    def __init__(self, run: RunDirectory, request_hash: str, initial: WorkflowState, *, fresh: bool):
        path = _database_path(run, "dispatch.sqlite")
        if fresh:
            with path.open("xb"):
                pass
        elif not path.is_file():
            raise RunIncompleteError("Dispatch journal is missing; no implicit reinitialization")
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA trusted_schema=OFF")
        self.conn.execute("PRAGMA synchronous=FULL")
        if fresh:
            with self.conn:
                self.conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                self.conn.execute("CREATE TABLE steps (state_id TEXT PRIMARY KEY, before_json TEXT NOT NULL, reservation INTEGER NOT NULL, after_json TEXT, after_id TEXT)")
                self.conn.executemany("INSERT INTO metadata VALUES (?, ?)", (
                    ("version", JOURNAL_VERSION), ("request_hash", request_hash),
                    ("initial_state", _state_json(initial)), ("started_unix", str(time.time())),
                ))
        try:
            meta = dict(self.conn.execute("SELECT key, value FROM metadata"))
            if set(meta) != {"version", "request_hash", "initial_state", "started_unix"} or (
                meta["version"], meta["request_hash"], meta["initial_state"]
            ) != (JOURNAL_VERSION, request_hash, _state_json(initial)):
                raise ValueError("Dispatch journal belongs to different inputs or state schema")
            self.started_unix = float(meta["started_unix"])
            if not 0 < self.started_unix <= time.time() + 5:
                raise ValueError("Invalid workflow start time")
        except Exception:
            self.conn.close()
            raise

    def close(self) -> None:
        self.conn.close()

    def claim(self, state: WorkflowState, reservation: int) -> tuple[str, WorkflowState | None]:
        before = _state_json(state)
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            row = self.conn.execute("SELECT before_json,reservation,after_json,after_id FROM steps WHERE state_id=?", (state.state_id,)).fetchone()
            if row is None:
                self.conn.execute("INSERT INTO steps VALUES (?, ?, ?, NULL, NULL)", (state.state_id, before, reservation))
                return "fresh", None
            if row[:2] != (before, reservation):
                raise ValueError("Dispatch claim differs from the current transition")
            if row[2] is None:
                return "interrupted", None
            after = WorkflowState.model_validate_json(row[2])
            if after.state_id != row[3] or _state_json(after) != row[2]:
                raise ValueError("Saved transition content changed")
            _transition(state, after)
            return "replay", after

    def complete(self, before: WorkflowState, after: WorkflowState) -> None:
        _transition(before, after)
        raw = _state_json(after)
        with self.conn:
            cursor = self.conn.execute("UPDATE steps SET after_json=?,after_id=? WHERE state_id=? AND after_json IS NULL", (raw, after.state_id, before.state_id))
            if cursor.rowcount != 1:
                row = self.conn.execute("SELECT after_json,after_id FROM steps WHERE state_id=?", (before.state_id,)).fetchone()
                if row != (raw, after.state_id):
                    raise ValueError("A completed transition cannot be replaced")


def _transition(before: WorkflowState, after: WorkflowState) -> None:
    if (after.initial_revision != before.initial_revision or after.record.record_id != before.record.record_id
        or after.events[:len(before.events)] != before.events
        or len(after.events) not in {len(before.events), len(before.events) + 1}
        or after.model_calls < before.model_calls or after.elapsed_ms < before.elapsed_ms):
        raise ValueError("Saved transition does not extend its exact prior state")
    if len(after.events) == len(before.events) and after.record != before.record:
        raise ValueError("A stop cannot silently modify the canonical record")


def _journal_head(conn: sqlite3.Connection, initial: WorkflowState, runtime: WorkflowRuntime) -> tuple[WorkflowState, set[str]]:
    """Validate the complete committed chain; checkpoint payloads must occur in it."""
    current = initial
    known = {current.state_id}
    for _ in range(runtime.budget.total_actions + 2):
        row = conn.execute("SELECT before_json,reservation,after_json,after_id FROM steps WHERE state_id=?", (current.state_id,)).fetchone()
        if row is None:
            return current, known
        expected_reservation = min(runtime.budget.max_model_calls - current.model_calls - current.uncertain_model_calls,
                                   3 if runtime.policy == "generic_llm" else 2)
        if row[0] != _state_json(current) or row[1] != expected_reservation:
            raise ValueError("Journal transition differs from its frozen state/budget")
        if row[2] is None:
            return current, known
        following = WorkflowState.model_validate_json(row[2])
        if following.state_id != row[3] or _state_json(following) != row[2]:
            raise ValueError("Journal result fingerprint changed")
        _transition(current, following)
        verify_state(following, runtime)
        if following.state_id in known:
            raise ValueError("Journal contains a state cycle")
        known.add(following.state_id)
        current = following
        if current.status != "running":
            return current, known
    raise ValueError("Journal exceeds its transition bound")


def _implementation() -> dict[str, str]:
    base = Path(__file__).parent
    names = ("workflow_graph", "workflow", "workflow_types", "workflow_policies", "workflow_tools",
             "record_checks", "record_patches", "support_checks", "contracts", "evidence", "profiles",
             "providers", "storage", "paths", "retrieval", "nexus_registry", "nexus_resources")
    paths = ["agent/" + name + ".py" for name in names]
    paths += ["utils.py", "config_validation.py", "components/llm.py", "base/llm.py", "data_models.py"]
    return {name: sha256((base.parent / name).read_bytes()) for name in paths}


class _GraphState(TypedDict):
    payload: dict


def run_workflow_graph(root: Path | str, run_id: str, request: dict, initial: WorkflowState,
                       runtime: WorkflowRuntime, *, max_steps: int | None = None) -> dict:
    """Execute/resume local checkpoints; a pending review remains a terminal pending case.

    max_steps intentionally pauses after committed transitions for restart tests or
    local operation. It does not supply a review answer or reset any run budgets.
    """
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    verify_state(initial, runtime)
    if initial.events or initial.status != "running" or initial.pending_reviews or initial.model_calls or initial.uncertain_model_calls:
        raise ValueError("Supply the original unexecuted workflow state")
    provider = runtime.policy_provider
    full_request = {"version": "m07-run-1", "declared": request,
        "initial_state": initial.model_dump(mode="json"), "tool_scope": runtime.tools._scope_id,
        "policy": runtime.policy, "budget": runtime.budget.model_dump(mode="json"),
        "policy_provider": None if provider is None else {"model": provider.model_name, "mode": provider.mode,
            "configuration": getattr(provider, "configuration_fingerprint", None),
            "scope": provider.spec.model_dump(mode="json") if hasattr(provider, "spec") else None},
        "implementation": _implementation(),
        "dependencies": {name: version(name) for name in ("langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite", "langchain-core", "langsmith", "pydantic")},
        "raw_retention": "hash_only", "external_tracing": "disabled", "python": platform.python_version()}
    request_hash = _digest(_json_bytes(full_request))
    resumed = False
    try:
        store = RunStore.begin(root, run_id, full_request)
    except RunIncompleteError:
        run = RunDirectory(Path(root).resolve(), run_id)
        _check_request(run, full_request, request_hash)
        store = RunStore(run, None, full_request, request_hash)
        resumed = True
    with _lease(store.run):
        if store.completed is not None:
            saved = WorkflowState.model_validate(store.completed["state"])
            verify_state(saved, runtime)
            journal_path = _database_path(store.run, "dispatch.sqlite")
            if not journal_path.is_file():
                raise ValueError("Completed workflow is missing its dispatch journal")
            audit = sqlite3.connect(journal_path.as_uri() + "?mode=ro", uri=True)
            try:
                audit.execute("PRAGMA trusted_schema=OFF")
                meta = dict(audit.execute("SELECT key,value FROM metadata"))
                if (meta.get("version"), meta.get("request_hash"), meta.get("initial_state")) != (JOURNAL_VERSION, request_hash, _state_json(initial)):
                    raise ValueError("Completed workflow journal inputs differ")
                terminal, _ = _journal_head(audit, initial, runtime)
                if terminal != saved or terminal.status == "running":
                    raise ValueError("Completion differs from the committed terminal journal state")
            finally:
                audit.close()
            expected = _result(saved, runtime)
            if expected != store.completed:
                raise ValueError("Workflow completion metadata changed")
            store.complete(expected, canonical=saved.record.canonical, sidecar={"workflow": saved.model_dump(mode="json"),
                "source": runtime.tools.bundle.model_dump(mode="json"), "profile": runtime.tools.profile.model_dump(mode="json")},
                raw_response=None, retain_raw=False)
            return {"execution": "cached_replay", "state": saved, "new_model_calls": 0, "new_live_requests": 0}
        journal = DispatchJournal(store.run, request_hash, initial, fresh=not resumed)
        checkpoint_path = _database_path(store.run, "checkpoints.sqlite")
        if resumed and not checkpoint_path.is_file():
            journal.close()
            raise RunIncompleteError("LangGraph checkpoint database is missing")
        if not resumed:
            with checkpoint_path.open("xb"):
                pass
        conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
        conn.execute("PRAGMA trusted_schema=OFF")
        conn.execute("PRAGMA synchronous=FULL")
        runtime.deadline_unix = journal.started_unix + runtime.budget.timeout_seconds
        original_cancel = runtime.cancelled
        runtime.cancelled = lambda: original_cancel() or _path(store.run, "cancel.request").is_file()
        new_calls = 0
        new_live = 0
        try:
            serde = JsonPlusSerializer(pickle_fallback=False, allowed_json_modules=[], allowed_msgpack_modules=[])
            saver = SqliteSaver(conn, serde=serde)

            def node(inputs: _GraphState) -> _GraphState:
                nonlocal new_calls, new_live
                before = WorkflowState.model_validate(inputs["payload"])
                verify_state(before, runtime)
                _, known_states = _journal_head(journal.conn, initial, runtime)
                if before.state_id not in known_states:
                    raise ValueError("Checkpoint state is absent from the dispatch journal")
                remaining = runtime.budget.max_model_calls - before.model_calls - before.uncertain_model_calls
                reservation = min(remaining, 3 if runtime.policy == "generic_llm" else 2)
                mode, after = journal.claim(before, reservation)
                if mode == "replay":
                    verify_state(after, runtime)
                    return {"payload": after.model_dump(mode="json")}
                if mode == "interrupted":
                    after = _replace(before, status="interrupted", stop_reason="dispatch_outcome_unknown",
                        uncertain_model_calls=before.uncertain_model_calls + reservation,
                        usage_unknown=before.usage_unknown or bool(reservation))
                else:
                    runtime.position_mocks(before)
                    after = advance(before, runtime)
                    new_calls += after.model_calls - before.model_calls
                    new_live += sum(p.mode == "live" and p.request_id is not None for e in after.events[len(before.events):] for p in e.providers)
                verify_state(after, runtime)
                journal.complete(before, after)
                return {"payload": after.model_dump(mode="json")}

            builder = StateGraph(_GraphState)
            builder.add_node("advance", node)
            builder.add_edge(START, "advance")
            builder.add_conditional_edges("advance", lambda s: "advance" if s["payload"]["status"] == "running" else END)
            graph = builder.compile(checkpointer=saver, interrupt_after=["advance"])
            config = {"configurable": {"thread_id": run_id}, "recursion_limit": runtime.budget.total_actions + 4, "callbacks": []}

            def execute():
                with tracing_context(enabled=False):
                    checkpoint = graph.get_state(config)
                    if resumed and not checkpoint.values:
                        raise RunIncompleteError("No saved LangGraph state; no implicit restart")
                    start = {"payload": initial.model_dump(mode="json")} if not checkpoint.values else None
                    if checkpoint.values:
                        restored = WorkflowState.model_validate(checkpoint.values["payload"])
                        verify_state(restored, runtime)
                        head, known = _journal_head(journal.conn, initial, runtime)
                        if restored.state_id not in known:
                            raise ValueError("Checkpoint state is absent from the dispatch journal")
                        if restored.status != "running":
                            if restored != head:
                                raise ValueError("Terminal checkpoint differs from committed journal head")
                            return restored, 0
                    count = 0
                    while True:
                        output = graph.invoke(start, config=config, durability="sync")
                        start = None
                        count += 1
                        current = WorkflowState.model_validate(output["payload"])
                        verify_state(current, runtime)
                        if current.status != "running" or (max_steps is not None and count >= max_steps):
                            return current, count

            # Clear inherited tracer callbacks as well as env-driven tracing.
            state, _ = contextvars.Context().run(execute)
            verify_state(state, runtime)
            head, _ = _journal_head(journal.conn, initial, runtime)
            if state.status != "running" and state != head:
                raise ValueError("Final checkpoint differs from committed journal head")
        finally:
            runtime.cancelled = original_cancel
            conn.close()
            journal.close()
        if state.status == "running":
            return {"execution": "checkpointed", "state": state, "new_model_calls": new_calls, "new_live_requests": new_live}
        result = _result(state, runtime)
        store.complete(result, canonical=state.record.canonical, sidecar={"workflow": state.model_dump(mode="json"),
            "source": runtime.tools.bundle.model_dump(mode="json"), "profile": runtime.tools.profile.model_dump(mode="json")},
            raw_response=None, retain_raw=False)
        return {"execution": "resumed" if resumed else "executed", "state": state,
                "new_model_calls": new_calls, "new_live_requests": new_live}


def _result(state: WorkflowState, runtime: WorkflowRuntime) -> dict:
    return {"version": "m07-result-1", "state": state.model_dump(mode="json"), "policy": runtime.policy,
        "coverage_status": "unsupported", "expert_approval": "not_supplied", "cost": None,
        "cost_status": "unknown" if runtime.tools.checker.mode == "live" or (runtime.policy_provider is not None and runtime.policy_provider.mode == "live") else "not_applicable_mock",
        "model_calls": state.model_calls, "uncertain_model_calls": state.uncertain_model_calls,
        "live_requests": sum(p.mode == "live" and p.request_id is not None for e in state.events for p in e.providers),
        "live_provider_invocations": sum(p.mode == "live" for e in state.events for p in e.providers),
        "live_dispatch_unknown": bool(state.uncertain_model_calls and (runtime.tools.checker.mode == "live" or
            (runtime.policy_provider is not None and runtime.policy_provider.mode == "live"))) or any(
                p.mode == "live" and p.request_id is None and p.status != "blocked" for e in state.events for p in e.providers),
        "usage_unknown": state.usage_unknown, "raw_retention": "hash_only"}
