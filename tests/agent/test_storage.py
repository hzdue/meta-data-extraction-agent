"""Synthetic-only storage invariants; no provider, gold, or live resources."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from meta_data_extraction.agent.paths import PathPolicyError
from meta_data_extraction.agent.storage import (
    RunConflictError,
    RunIncompleteError,
    RunStore,
)


REQUEST = {"source_sha256": "synthetic-hash", "profile": "v1", "mode": "mock"}
RESULT = {"status": "success", "record_revision": "revision-1"}
CANONICAL = {"samples": [{"id": "s1", "temperature": "80 K"}]}
SIDECAR = {"evidence": [{"entity_id": "s1", "quote": "80 K"}]}


def finish(store, **overrides):
    values = {"result": RESULT, "canonical": CANONICAL, "sidecar": SIDECAR,
              "raw_response": '{"temperature": "80 K"}', "retain_raw": True}
    values.update(overrides)
    return store.complete(**values)


def test_completed_replay_validates_separate_exports_and_hashes(workspace):
    store = RunStore.begin(workspace, "complete", REQUEST)
    assert store.completed is None
    assert finish(store) == RESULT
    paths = {path.name for path in store.run.path.iterdir()}
    assert paths == {"request.json", "canonical.json", "sidecar.json",
                     "raw-response.json", "result.json"}
    assert json.loads((store.run.path / "canonical.json").read_text()) == CANONICAL
    envelope = json.loads((store.run.path / "result.json").read_text())
    for name, metadata in envelope["artifacts"].items():
        content = (store.run.path / name).read_bytes()
        assert metadata == {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    replay = RunStore.begin(workspace, "complete", REQUEST)
    assert replay.completed == RESULT
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in store.run.path.iterdir()}
    assert finish(store) == finish(replay) == RESULT
    assert before == {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in store.run.path.iterdir()}


def test_inputs_and_completed_outputs_do_not_alias_caller_objects(workspace):
    request = {"profile": {"version": "original"}}
    store = RunStore.begin(workspace, "copies", request)
    request["profile"]["version"] = "changed"
    result = {"status": {"value": "success"}}
    returned = finish(store, result=result)
    result["status"]["value"] = "changed"
    returned["status"]["value"] = "also changed"
    assert store.completed == {"status": {"value": "success"}}
    replay = RunStore.begin(workspace, "copies", {"profile": {"version": "original"}})
    assert replay.completed == {"status": {"value": "success"}}


@pytest.mark.parametrize("retain,raw", [(False, "private discarded text"), (True, None)])
def test_raw_retention_is_explicit_and_absence_is_not_empty_text(workspace, retain, raw):
    store = RunStore.begin(workspace, "no-raw", REQUEST)
    finish(store, retain_raw=retain, raw_response=raw)
    assert not (store.run.path / "raw-response.json").exists()
    assert "private discarded text" not in (store.run.path / "result.json").read_text()
    assert RunStore.begin(workspace, "no-raw", REQUEST).completed == RESULT


def test_empty_raw_response_is_retained_when_authorized(workspace):
    store = RunStore.begin(workspace, "empty-raw", REQUEST)
    finish(store, raw_response="")
    assert json.loads((store.run.path / "raw-response.json").read_text()) == {"raw_response": ""}


def test_error_outcome_does_not_emit_canonical_record(workspace):
    store = RunStore.begin(workspace, "error", REQUEST)
    result = {"status": "parse_error", "message": "Invalid JSON"}
    finish(store, result=result, canonical=None)
    assert not (store.run.path / "canonical.json").exists()
    assert RunStore.begin(workspace, "error", REQUEST).completed == result


def test_explicit_empty_canonical_is_preserved(workspace):
    store = RunStore.begin(workspace, "empty", REQUEST)
    finish(store, canonical={})
    assert json.loads((store.run.path / "canonical.json").read_text()) == {}


@pytest.mark.parametrize("completed", [False, True])
def test_run_id_rejects_different_inputs(workspace, completed):
    store = RunStore.begin(workspace, "same-id", REQUEST)
    if completed:
        finish(store)
    with pytest.raises(RunConflictError):
        RunStore.begin(workspace, "same-id", {**REQUEST, "profile": "v2"})


@pytest.mark.parametrize("change", [
    {"result": {"status": "failure"}}, {"canonical": {}}, {"canonical": None},
    {"sidecar": {}}, {"raw_response": "different"}, {"retain_raw": False},
])
def test_completed_export_rejects_changed_payload_without_overwriting(workspace, change):
    store = RunStore.begin(workspace, "changed", REQUEST)
    finish(store)
    before = {path.name: path.read_bytes() for path in store.run.path.iterdir()}
    with pytest.raises(RunConflictError):
        finish(store, **change)
    assert before == {path.name: path.read_bytes() for path in store.run.path.iterdir()}


def test_incomplete_claim_never_resumes_automatically(workspace):
    store = RunStore.begin(workspace, "interrupted", REQUEST)
    with pytest.raises(RunIncompleteError):
        RunStore.begin(workspace, "interrupted", REQUEST)
    assert list(store.run.path.iterdir()) == [store.run.path / "request.json"]


def test_directory_without_request_is_also_an_execution_claim(workspace):
    (workspace / "runs/agent/bare-claim").mkdir(parents=True)
    with pytest.raises(RunIncompleteError):
        RunStore.begin(workspace, "bare-claim", REQUEST)


def test_concurrent_begin_has_exactly_one_owner(workspace):
    (workspace / "runs/agent").mkdir(parents=True)
    barrier = Barrier(8)

    def begin():
        barrier.wait(timeout=10)
        try:
            return RunStore.begin(workspace, "shared", REQUEST)
        except (RunIncompleteError, RunConflictError) as exc:
            # A competitor can observe request.json before its write completes;
            # this must fail closed just like any other incomplete claim.
            return exc

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: begin(), range(8)))
    assert sum(isinstance(result, RunStore) for result in results) == 1
    assert all(isinstance(result, (RunStore, RunIncompleteError, RunConflictError))
               for result in results)


def test_interrupted_export_never_becomes_completed_on_retry(workspace, monkeypatch):
    import meta_data_extraction.agent.storage as storage

    store = RunStore.begin(workspace, "partial", REQUEST)
    original = storage._write_exclusive

    def interrupted(run, name, payload):
        if name == "canonical.json":
            raise OSError("simulated disk failure")
        return original(run, name, payload)

    monkeypatch.setattr(storage, "_write_exclusive", interrupted)
    with pytest.raises(OSError, match="simulated disk failure"):
        finish(store)
    assert (store.run.path / "sidecar.json").exists()
    assert not (store.run.path / "result.json").exists()
    with pytest.raises(RunIncompleteError):
        RunStore.begin(workspace, "partial", REQUEST)
    with pytest.raises(RunIncompleteError):
        finish(store)


def test_invalid_payload_has_no_partial_export(workspace):
    store = RunStore.begin(workspace, "bad-payload", REQUEST)
    with pytest.raises(ValueError):
        finish(store, sidecar={"invalid": float("nan")})
    assert {path.name for path in store.run.path.iterdir()} == {"request.json"}


@pytest.mark.parametrize("invalid_request", [{1: "integer-key"}, {"tuple": (1, 2)}, ["not-object"]])
def test_invalid_request_cannot_claim_a_run(workspace, invalid_request):
    with pytest.raises(TypeError):
        RunStore.begin(workspace, "invalid", invalid_request)
    assert not (workspace / "runs").exists()


@pytest.mark.parametrize("filename", ["request.json", "result.json", "canonical.json",
                                     "sidecar.json", "raw-response.json"])
def test_tampered_saved_data_is_not_replayed(workspace, filename):
    store = RunStore.begin(workspace, "tampered", REQUEST)
    finish(store)
    (store.run.path / filename).write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RunConflictError):
        RunStore.begin(workspace, "tampered", REQUEST)


def test_manifest_cannot_supply_a_path(workspace):
    store = RunStore.begin(workspace, "bad-manifest", REQUEST)
    finish(store)
    path = store.run.path / "result.json"
    envelope = json.loads(path.read_text())
    envelope["artifacts"]["../../outputs/secret.json"] = {"bytes": 0, "sha256": "x"}
    # Even a recomputed checksum must not authorize a non-allowlisted path.
    from meta_data_extraction.agent.storage import _digest, _json_bytes
    envelope["completion_sha256"] = _digest(_json_bytes({
        key: value for key, value in envelope.items() if key != "completion_sha256"
    }))
    path.write_text(json.dumps(envelope))
    with pytest.raises(RunConflictError, match="manifest"):
        RunStore.begin(workspace, "bad-manifest", REQUEST)


def test_undeclared_artifact_is_not_accepted(workspace):
    store = RunStore.begin(workspace, "extra", REQUEST)
    finish(store, canonical=None, retain_raw=False)
    (store.run.path / "canonical.json").write_text("{}")
    with pytest.raises(RunConflictError, match="undeclared"):
        RunStore.begin(workspace, "extra", REQUEST)


@pytest.mark.parametrize("filename", ["request.json", "result.json", "canonical.json",
                                     "sidecar.json", "raw-response.json"])
def test_hardlinked_artifacts_are_rejected_before_reading(workspace, filename):
    store = RunStore.begin(workspace, "linked", REQUEST)
    finish(store)
    os.link(store.run.path / filename, workspace / "alias.json")
    with pytest.raises(PathPolicyError, match="Hard-linked"):
        RunStore.begin(workspace, "linked", REQUEST)


@pytest.mark.parametrize("filename", ["request.json", "result.json", "canonical.json",
                                     "sidecar.json", "raw-response.json"])
def test_linked_artifact_components_are_rejected(workspace, link_directory, filename):
    store = RunStore.begin(workspace, "linked-artifact", REQUEST)
    finish(store)
    path = store.run.path / filename
    path.unlink()  # Replace only this synthetic fixture artifact with a link.
    target = workspace / "outputs"
    target.mkdir()
    link_directory(path, target)
    try:
        with pytest.raises(PathPolicyError, match="Symlinks and junctions"):
            RunStore.begin(workspace, "linked-artifact", REQUEST)
        assert list(target.iterdir()) == []
    finally:
        path.rmdir() if os.name == "nt" else path.unlink()


def test_linked_run_directory_is_rejected(workspace, link_directory):
    released = workspace / "outputs"
    released.mkdir()
    sentinel = released / "sentinel.json"
    sentinel.write_text("released")
    (workspace / "runs/agent").mkdir(parents=True)
    link = workspace / "runs/agent/linked"
    link_directory(link, released)
    try:
        with pytest.raises(PathPolicyError):
            RunStore.begin(workspace, "linked", REQUEST)
        assert list(released.iterdir()) == [sentinel]
        assert sentinel.read_text() == "released"
    finally:
        link.rmdir() if os.name == "nt" else link.unlink()


@pytest.mark.parametrize("run_id", ["../outputs", "NUL", "CON", "C:outputs", "a:b", "a."])
def test_unsafe_windows_or_traversal_ids_cannot_claim_run(workspace, run_id):
    with pytest.raises(PathPolicyError):
        RunStore.begin(workspace, run_id, REQUEST)
    assert not (workspace / "runs").exists()


def test_completion_marker_is_last_and_fully_formed(workspace, monkeypatch):
    import meta_data_extraction.agent.storage as storage

    store = RunStore.begin(workspace, "publication", REQUEST)
    original = storage._write_exclusive
    observed = []

    def inspect(run, name, payload):
        assert not (run.path / "result.json").exists()
        observed.append(name)
        original(run, name, payload)

    monkeypatch.setattr(storage, "_write_exclusive", inspect)
    finish(store)
    assert observed[-1] == "result.pending.json"
    assert not (store.run.path / "result.pending.json").exists()
    assert RunStore.begin(workspace, "publication", REQUEST).completed == RESULT
