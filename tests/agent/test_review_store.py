"""Durable M08 review engineering diagnostics using simulated reviewers only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from pydantic import ValidationError

from test_workflow_graph import workflow_case, tool_case, execution, run, run_dir
from meta_data_extraction.agent.contracts import at_pointer, canonical_json
from meta_data_extraction.agent.review_store import ReviewStore
from meta_data_extraction.agent.review_types import ReviewAttestation, ReviewResponse


@pytest.fixture
def review_run(workflow_case):
    context, initial, prepared = workflow_case
    initial = initial.model_copy(update={"issues": tuple(issue.model_copy(update={
        "diagnosis": "context_conflict", "proposal_ids": (),
    }) for issue in initial.issues)})
    case = context, initial, prepared
    result = run(case)
    assert result["state"].status == "pending_review" and result["new_model_calls"] == 0
    return case, result["state"]


def opened(review_run):
    case, _ = review_run
    return ReviewStore.open(case[0].root, run_dir(case).name)


def parent_bytes(review_run):
    return {path.name: path.read_bytes() for path in run_dir(review_run[0]).iterdir() if path.is_file()}


def unresolved(store, key="synthetic-review-1", *, case_ids=None):
    state = store.state
    return ReviewResponse(idempotency_key=key, expected_state_id=state.state_id,
        expected_revision=state.current.revision_id,
        case_ids=case_ids or tuple(case.case_id for case in state.cases if case.status == "pending"),
        outcome="unresolved", provenance="simulated", reviewer_id="simulated-reviewer",
        rationale="Invented review diagnostic; the supplied excerpts do not resolve this case.")


def approved(store, review_run, key="synthetic-approval-1"):
    state = store.state
    case, _ = review_run
    selected = tuple(item for item in state.cases if item.status == "pending")
    paths = {item.path for item in selected}
    paths.update(ref.path for ref in state.current.references
                 if any(item.path.startswith("/datasets/0/") for item in selected)
                 and ref.path.startswith("/datasets/0/"))
    evidence = case[1].issues[0].evidence
    refs = {ref.path for ref in state.current.references}
    attestations = tuple(ReviewAttestation(path=path, value_json=canonical_json(at_pointer(state.current.canonical, path)),
        role="attachment" if path in refs else "measurement_acquisition", evidence=evidence) for path in sorted(paths))
    return ReviewResponse(idempotency_key=key, expected_state_id=state.state_id,
        expected_revision=state.current.revision_id, case_ids=tuple(item.case_id for item in selected),
        outcome="approve", provenance="simulated", reviewer_id="simulated-reviewer",
        rationale="Invented approval fixture for persistence testing, not an expert judgment.", attestations=attestations)


def test_open_freezes_pending_review_without_changing_parent_or_calling_models(review_run, monkeypatch):
    before = parent_bytes(review_run)
    with opened(review_run) as store:
        state = store.state
        assert state.current == review_run[1].record
        assert state.initial_workflow == review_run[1]
        assert state.status == "pending"
        assert all(case.status == "pending" for case in state.cases)
        assert state.decisions == ()
        assert (run_dir(review_run[0]) / "review/review.sqlite").is_file()
    with opened(review_run) as reopened:
        assert reopened.state == state
    assert parent_bytes(review_run) == before
    replay = run(review_run[0])
    assert replay["execution"] == "cached_replay" and replay["new_model_calls"] == 0


def test_unresolved_response_is_durable_and_does_not_change_canonical_record(review_run):
    with opened(review_run) as store:
        initial = store.state
        response = unresolved(store)
        result = store.submit(response)
        assert result["execution"] == "applied"
        assert result["state"] == store.state and store.state != initial
        assert store.state.current == initial.current
        assert all(case.status == "unresolved" for case in store.state.cases)
        assert len(store.state.decisions) == 1
        expected = store.state
    with opened(review_run) as reopened:
        assert reopened.state == expected
        assert all(case.status != "approved" for case in reopened.state.cases)


def test_exact_duplicate_is_idempotent_even_after_a_later_response(review_run):
    with opened(review_run) as store:
        first_response = unresolved(store, case_ids=(store.state.cases[0].case_id,))
        first = store.submit(first_response)
        second = store.submit(unresolved(store, key="synthetic-review-2"))
        head = store.state
        replay = store.submit(first_response)
        assert replay["execution"] == "replayed"
        assert replay["state"] == head
        assert store.state == head == second["state"]
        assert len(store.state.decisions) == 2
    with opened(review_run) as reopened:
        assert reopened.submit(first_response)["execution"] == "replayed"
        assert reopened.state == head


def test_reused_idempotency_key_with_changed_payload_is_rejected(review_run):
    with opened(review_run) as store:
        response = unresolved(store)
        store.submit(response)
        before = store.state
        altered = response.model_copy(update={"rationale": "A different simulated response"})
        with pytest.raises(ValueError):
            store.submit(altered)
        assert store.state == before and len(store.state.decisions) == 1


@pytest.mark.parametrize("field", ["expected_state_id", "expected_revision"])
def test_stale_response_rejected_without_any_review_mutation(review_run, field):
    with opened(review_run) as store:
        response = unresolved(store).model_copy(update={field: "stale:" + "0" * 64})
        initial = store.state
        with pytest.raises(ValueError):
            store.submit(response)
        assert store.state == initial and store.state.decisions == ()
    with opened(review_run) as reopened:
        assert reopened.state == initial


def test_run_lease_serializes_writers_and_reopened_stale_window_is_rejected(review_run):
    from meta_data_extraction.agent.workflow_graph import WorkflowBusyError
    with opened(review_run) as first:
        first_response = unresolved(first, case_ids=(first.state.cases[0].case_id,))
        stale_response = unresolved(first, key="other-window")
        with pytest.raises(WorkflowBusyError):
            opened(review_run)
        accepted = first.submit(first_response)
    with opened(review_run) as reopened:
        with pytest.raises(ValueError):
            reopened.submit(stale_response)
        assert reopened.state == accepted["state"] and len(reopened.state.decisions) == 1


def test_invalid_approval_attestations_do_not_create_approval_or_receipt(review_run):
    with opened(review_run) as store:
        good = approved(store, review_run)
        initial = store.state
        bad = good.model_copy(update={"attestations": ()})
        with pytest.raises(ValueError):
            store.submit(bad)
        assert store.state == initial and store.state.decisions == ()
        assert all(case.status == "pending" for case in store.state.cases)


def test_simulated_approval_stays_labelled_as_simulated_and_keeps_parent_immutable(review_run):
    before = parent_bytes(review_run)
    with opened(review_run) as store:
        response = approved(store, review_run)
        result = store.submit(response)
        assert result["execution"] == "applied"
        assert all(case.status == "approved" and case.review_provenance == "simulated" for case in store.state.cases)
        assert store.state.current == review_run[1].record
        assert "simulated" in store.state.model_dump_json()
        saved = store.state
    with opened(review_run) as reopened:
        assert reopened.state == saved
    assert parent_bytes(review_run) == before


def test_real_second_process_submits_and_exits_then_duplicate_receipt_replays(review_run, tmp_path):
    with opened(review_run) as first_process:
        response = unresolved(first_process)
        initial_id = first_process.state.state_id
    payload = {"root": str(review_run[0][0].root), "run_id": run_dir(review_run[0]).name,
               "response": response.model_dump(mode="json")}
    payload_path = tmp_path / "review-subprocess.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    script = r'''
import json, socket, sys
from pathlib import Path

def denied(*args, **kwargs):
    raise AssertionError("Review subprocess network is forbidden")
socket.socket.connect = denied
socket.create_connection = denied
socket.getaddrinfo = denied
from meta_data_extraction.agent.review_store import ReviewStore
from meta_data_extraction.agent.review_types import ReviewResponse
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
with ReviewStore.open(p["root"], p["run_id"]) as store:
    result = store.submit(ReviewResponse.model_validate(p["response"]))
    print(json.dumps({"execution": result["execution"], "state_id": result["state"].state_id}), flush=True)
'''
    result = subprocess.run([sys.executable, "-c", script, str(payload_path)], check=True,
                            capture_output=True, text=True, timeout=40,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    report = json.loads(result.stdout)
    assert report["execution"] == "applied" and report["state_id"] != initial_id
    with opened(review_run) as parent:
        replay = parent.submit(response)
        assert replay["execution"] == "replayed" and replay["state"].state_id == report["state_id"]
        assert len(parent.state.decisions) == 1


def test_crash_after_committed_response_does_not_apply_it_twice(review_run, tmp_path):
    with opened(review_run) as first_process:
        response = unresolved(first_process)
    payload = {"root": str(review_run[0][0].root), "run_id": run_dir(review_run[0]).name,
               "response": response.model_dump(mode="json")}
    payload_path = tmp_path / "review-lost-ack.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    script = r'''
import json, os, socket, sys
from pathlib import Path

def denied(*args, **kwargs):
    raise AssertionError("Review subprocess network is forbidden")
socket.socket.connect = denied
socket.create_connection = denied
socket.getaddrinfo = denied
from meta_data_extraction.agent.review_store import ReviewStore
from meta_data_extraction.agent.review_types import ReviewResponse
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
with ReviewStore.open(p["root"], p["run_id"]) as store:
    store.submit(ReviewResponse.model_validate(p["response"]))
    os._exit(7)
'''
    result = subprocess.run([sys.executable, "-c", script, str(payload_path)], check=False,
                            capture_output=True, text=True, timeout=40,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 7
    with opened(review_run) as reopened:
        assert reopened.submit(response)["execution"] == "replayed"
        assert len(reopened.state.decisions) == 1


def test_export_is_versioned_idempotent_and_never_overwrites_parent(review_run):
    original_parent = parent_bytes(review_run)
    with opened(review_run) as store:
        store.submit(unresolved(store))
        first = store.export()
        export_root = run_dir(review_run[0]) / "review/exports"
        artifacts = {str(p.relative_to(export_root)): p.read_bytes() for p in export_root.rglob("*") if p.is_file()}
        manifests = list(export_root.rglob("result.json"))
        assert len(manifests) == 1
        canonical = json.loads((manifests[0].parent / "canonical.json").read_bytes())
        assert canonical == store.state.current.canonical
        second = store.export()
        assert second == first
        assert {str(p.relative_to(export_root)): p.read_bytes() for p in export_root.rglob("*") if p.is_file()} == artifacts
    assert parent_bytes(review_run) == original_parent


@pytest.mark.parametrize("parent_artifact", ["request.json", "result.json", "canonical.json", "sidecar.json", "dispatch.sqlite", "checkpoints.sqlite"])
def test_changed_parent_pins_block_existing_review_session(review_run, parent_artifact):
    with opened(review_run) as store:
        initial = store.state
    path = run_dir(review_run[0]) / parent_artifact
    path.write_bytes(path.read_bytes() + b"\nchanged-parent-pin")
    with pytest.raises((ValueError, sqlite3.Error, OSError)):
        with opened(review_run):
            pytest.fail("Changed frozen parent must not reopen")


def test_unsupported_review_sqlite_contents_are_not_reinitialized(review_run):
    directory = run_dir(review_run[0]) / "review"
    directory.mkdir()
    database = directory / "review.sqlite"
    database.write_bytes(b"unsupported or corrupt review state")
    before = database.read_bytes()
    with pytest.raises((ValueError, sqlite3.Error, OSError)):
        with opened(review_run):
            pytest.fail("Corrupt review state must not be reset")
    assert database.read_bytes() == before


def test_review_database_hardlink_is_rejected(review_run):
    with opened(review_run):
        pass
    database = run_dir(review_run[0]) / "review/review.sqlite"
    linked = review_run[0][0].root / "tests/agent/fixtures/review-database-alias.sqlite"
    os.link(database, linked)
    with pytest.raises(ValueError):
        with opened(review_run):
            pytest.fail("Review database cannot be hardlinked")


def test_nonpending_completed_workflow_cannot_be_promoted_to_review_session(workflow_case):
    result = run(workflow_case)
    assert result["state"].status == "finished"
    with pytest.raises(ValueError):
        with ReviewStore.open(workflow_case[0].root, run_dir(workflow_case).name):
            pytest.fail("Only explicit pending cases may initialize review")
    assert not (run_dir(workflow_case) / "review/review.sqlite").exists()


@pytest.mark.parametrize("stage,committed", [("_before_commit", False), ("_after_commit", True)])
def test_submit_fault_boundaries_are_atomic_and_recoverable(review_run, monkeypatch, stage, committed):
    with opened(review_run) as store:
        initial = store.state
        response = unresolved(store)

        def crash(state):
            raise RuntimeError("Invented transaction crash")

        monkeypatch.setattr(store, stage, crash)
        with pytest.raises(RuntimeError, match="Invented transaction crash"):
            store.submit(response)
    with opened(review_run) as reopened:
        assert (len(reopened.state.decisions) == 1) is committed
        if not committed:
            assert reopened.state == initial
        retry = reopened.submit(response)
        assert retry["execution"] == ("replayed" if committed else "applied")
        assert len(reopened.state.decisions) == 1


@pytest.mark.parametrize("artifact", ["canonical.json", "sidecar.json", "result.json"])
@pytest.mark.parametrize("after_publication", [False, True])
def test_interrupted_export_resumes_only_exact_committed_state(
        review_run, monkeypatch, artifact, after_publication):
    with opened(review_run) as store:
        store.submit(unresolved(store))
        original = store._publish_file
        saved = store.state
        faulted = []

        def crash(relative, payload):
            if relative.endswith("/" + artifact) and not faulted:
                faulted.append(True)
                if after_publication:
                    original(relative, payload)
                else:
                    pending = store._path(relative + ".pending")
                    pending.write_bytes(payload[:7])
                raise RuntimeError("Invented export crash")
            original(relative, payload)

        monkeypatch.setattr(store, "_publish_file", crash)
        with pytest.raises(RuntimeError, match="Invented export crash"):
            store.export()
    with opened(review_run) as reopened:
        assert reopened.state == saved and len(reopened.state.decisions) == 1
        manifest = reopened.export()
        directory = reopened.run.root / manifest["directory"]
        assert json.loads((directory / "canonical.json").read_bytes()) == saved.current.canonical
        assert not list(directory.glob("*.pending"))
        files = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        assert reopened.export() == manifest
        assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == files


@pytest.mark.parametrize("artifact", ["canonical.json", "sidecar.json", "result.json"])
def test_completed_review_export_corruption_is_never_overwritten(review_run, artifact):
    with opened(review_run) as store:
        store.submit(unresolved(store))
        manifest = store.export()
        path = store.run.root / manifest["directory"] / artifact
        path.write_bytes(path.read_bytes() + b"\nchanged")
        damaged = path.read_bytes()
        with pytest.raises(ValueError):
            store.export()
        assert path.read_bytes() == damaged
    with opened(review_run) as reopened:
        with pytest.raises(ValueError):
            reopened.export()


def test_conflicting_final_file_in_incomplete_export_is_never_replaced(review_run, monkeypatch):
    with opened(review_run) as store:
        store.submit(unresolved(store))
        original = store._publish_file
        marker = []

        def crash(relative, payload):
            if relative.endswith("/sidecar.json"):
                marker.append(store._path(relative))
                raise RuntimeError("Invented export crash")
            original(relative, payload)

        monkeypatch.setattr(store, "_publish_file", crash)
        with pytest.raises(RuntimeError):
            store.export()
        canonical = marker[0].parent / "canonical.json"
        canonical.write_bytes(b'{"forged":"canonical"}')
    with opened(review_run) as reopened:
        with pytest.raises(ValueError):
            reopened.export()
        assert canonical.read_bytes() == b'{"forged":"canonical"}'


@pytest.mark.parametrize("key", ["version", "implementation", "initial_state", "parent_pins"])
def test_review_metadata_changes_require_explicit_migration(review_run, key):
    with opened(review_run):
        pass
    database = run_dir(review_run[0]) / "review/review.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE metadata SET value=? WHERE key=?", ("unsupported-version-or-binding", key))
    with pytest.raises(ValueError):
        with opened(review_run):
            pytest.fail("Incompatible persisted review bindings must not reopen")


@pytest.mark.parametrize("field", ["response_json", "before_id", "after_json", "after_id", "idempotency_key"])
def test_corrupted_response_chain_cannot_reconstruct_a_valid_approval(review_run, field):
    with opened(review_run) as store:
        store.submit(unresolved(store))
    database = run_dir(review_run[0]) / "review/review.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(f"UPDATE responses SET {field}=? WHERE seq=1", ("changed-persisted-review",))
    with pytest.raises(ValueError):
        with opened(review_run):
            pytest.fail("Changed response history must not authorize anything")


def test_rehashed_stored_state_cannot_promote_simulated_review_to_user_approval(review_run):
    from meta_data_extraction.agent.contracts import digest
    from meta_data_extraction.agent.review_types import ReviewState
    with opened(review_run) as store:
        store.submit(approved(store, review_run))
    database = run_dir(review_run[0]) / "review/review.sqlite"
    with sqlite3.connect(database) as connection:
        raw = connection.execute("SELECT after_json FROM responses WHERE seq=1").fetchone()[0]
        state = json.loads(raw)
        state["cases"][0]["review_provenance"] = "user_supplied"
        forged = ReviewState.model_validate(state)
        connection.execute("UPDATE responses SET after_json=?,after_id=? WHERE seq=1",
                           (canonical_json(forged.model_dump(mode="json")), forged.state_id))
    with pytest.raises(ValueError):
        with opened(review_run):
            pytest.fail("Rehashed output cannot change the original simulated response")


@pytest.mark.parametrize("database", ["dispatch.sqlite", "checkpoints.sqlite"])
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_parent_sqlite_auxiliary_paths_cannot_escape_frozen_inputs(review_run, database, suffix):
    target = review_run[0][0].root / "tests/agent/fixtures/foreign-sqlite-marker"
    target.write_bytes(b"An unrelated private fixture file")
    alias = run_dir(review_run[0]) / (database + suffix)
    os.link(target, alias)
    with pytest.raises(ValueError):
        with opened(review_run):
            pytest.fail("SQLite auxiliary paths must be checked before SQLite opens them")
    assert target.read_bytes() == b"An unrelated private fixture file"


def test_private_export_staging_hardlink_is_rejected_without_overwriting_target(review_run, monkeypatch):
    with opened(review_run) as store:
        store.submit(unresolved(store))
        original = store._publish_file
        target = store.run.root / "tests/agent/fixtures/private-staging-target"
        target.write_bytes(b"Do not change this unrelated fixture")

        def add_alias(relative, payload):
            if relative.endswith("/canonical.json"):
                os.link(target, store._path(relative + ".pending"))
            return original(relative, payload)

        monkeypatch.setattr(store, "_publish_file", add_alias)
        with pytest.raises(ValueError):
            store.export()
        assert target.read_bytes() == b"Do not change this unrelated fixture"


def edited(store, review_run, key="synthetic-edit-1"):
    prepared = review_run[0][2]
    current = prepared.candidate
    response = approved(store, review_run, key=key)
    attestations = tuple(attestation.model_copy(update={
        "value_json": canonical_json(at_pointer(current.canonical, attestation.path)),
    }) for attestation in response.attestations)
    return ReviewResponse.model_validate({**response.model_dump(), "outcome": "edit",
        "proposal": prepared.proposal, "attestations": attestations})


def test_explicit_edit_reopens_persists_revision_and_keeps_prior_version_export(review_run):
    original_parent = parent_bytes(review_run)
    with opened(review_run) as store:
        initial_manifest = store.export()
        initial_directory = store.run.root / initial_manifest["directory"]
        initial_bytes = {p.name: p.read_bytes() for p in initial_directory.iterdir() if p.is_file()}
        result = store.submit(edited(store, review_run))
        assert result["execution"] == "applied"
        assert store.state.current == review_run[0][2].candidate
        assert store.state.original == review_run[1].record
        assert store.state.current.entities == store.state.original.entities
        assert len(store.state.current.canonical["samples"]) == 2
        assert store.state.initial_workflow.model_calls == review_run[1].model_calls == 0
        manifest = store.export()
        assert manifest["directory"] != initial_manifest["directory"]
        assert {p.name: p.read_bytes() for p in initial_directory.iterdir() if p.is_file()} == initial_bytes
        assert all(case.review_provenance == "simulated" for case in store.state.cases)
        saved = store.state
    with opened(review_run) as reopened:
        assert reopened.state == saved
        assert reopened.export() == manifest
    assert parent_bytes(review_run) == original_parent


def test_structurally_invalid_edit_is_rejected_before_durable_state_change(review_run):
    with opened(review_run) as store:
        response = edited(store, review_run)
        proposal = response.proposal
        operations = list(proposal.operations)
        operations[-1] = operations[-1].model_copy(update={"replacement_value_json": canonical_json("sample_999")})
        response = response.model_copy(update={"proposal": proposal.model_copy(update={"operations": tuple(operations)})})
        initial = store.state
        with pytest.raises(ValueError):
            store.submit(response)
        assert store.state == initial and not store.state.decisions
    with opened(review_run) as reopened:
        assert reopened.state == initial


@pytest.mark.parametrize("field,forged", [("model_calls", 9), ("live_requests", 9),
                                         ("live_provider_invocations", 9), ("usage_unknown", True)])
def test_rehashed_parent_accounting_cannot_override_actual_observations(review_run, field, forged):
    from meta_data_extraction.agent.storage import _digest, _json_bytes
    path = run_dir(review_run[0]) / "result.json"
    envelope = json.loads(path.read_bytes())
    envelope["result"][field] = forged
    envelope["completion_sha256"] = _digest(_json_bytes({key: value for key, value in envelope.items()
                                                        if key != "completion_sha256"}))
    path.write_bytes(_json_bytes(envelope))
    with pytest.raises(ValueError):
        with opened(review_run):
            pytest.fail("Original accounting must be derived from recorded observations")


def test_review_view_contains_targeted_entity_and_attachment_context(review_run):
    with opened(review_run) as store:
        view = store.view()
        assert view["new_model_calls"] == view["new_live_requests"] == 0
        assert view["reviewer_identity"] == "self_declared_not_authenticated"
        assert view["coverage_status"] == "unsupported"
        reference = next(case for case in view["cases"] if case["path"].endswith("/sample_refs/0"))
        assert reference["entity_record"]["id"] == "dataset_1"
        linked = reference["linked_entities"]
        assert linked and "sample_2" in json.dumps(linked)
        assert "Sample B" in json.dumps(linked) and "CaCO3" in json.dumps(linked)
        assert reference["candidate_corrections"] == []
        assert reference["candidate_availability"] == "not_recorded"
        for passage in reference["passages"]:
            assert passage["evidence"]["source_sha256"] == store.bundle.source_sha256
            assert passage["text"] == passage["evidence"]["quotation"]
        assert "canonical_json" not in json.dumps(view)


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_submit_checks_current_sqlite_auxiliary_paths_before_starting_transaction(review_run, suffix):
    with opened(review_run) as store:
        response = unresolved(store)
        target = store.run.root / "tests/agent/fixtures/new-sqlite-aux-target"
        target.write_bytes(b"Unrelated content must not be opened by SQLite")
        alias = run_dir(review_run[0]) / ("review/review.sqlite" + suffix)
        os.link(target, alias)
        sql = []
        store.conn.set_trace_callback(sql.append)
        with pytest.raises(ValueError):
            store.submit(response)
        assert not sql
        assert target.read_bytes() == b"Unrelated content must not be opened by SQLite"


def test_closed_store_rejects_submission_explicitly(review_run):
    with opened(review_run) as store:
        response = unresolved(store)
    with pytest.raises(ValueError):
        store.submit(response)
