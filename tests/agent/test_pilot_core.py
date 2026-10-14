"""M10 facade diagnostics: invented source, mock calls, simulated reviews only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from meta_data_extraction.agent import pilot_core as core
from meta_data_extraction.agent.contracts import at_pointer, canonical_json
from meta_data_extraction.agent.pilot_bundle import BUNDLE_FILES, build_bundle
from meta_data_extraction.agent.record_patches import build_candidate
from meta_data_extraction.agent.review_store import ReviewStore
from meta_data_extraction.agent.paths import RunDirectory
from meta_data_extraction.agent.workflow_graph import _lease

RUN_ID = "m10-core-test"


def make_root(tmp_path_factory):
    repository = Path(__file__).resolve().parents[2]
    # Windows legacy MAX_PATH also applies to the unchanged M08 hashed export
    # layout. A short temporary root keeps the bundle plus child-run path valid.
    source = tmp_path_factory.mktemp("p") / "s"
    for relative in BUNDLE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, path)
    return Path(build_bundle(source, "m10-b")["runtime_root"])


@pytest.fixture
def pilot_root(tmp_path_factory):
    return make_root(tmp_path_factory)


@pytest.fixture(scope="module")
def validation_service(tmp_path_factory):
    # These rejection-only cases assert no run directory is ever created.
    return core.PilotService(make_root(tmp_path_factory))


@pytest.fixture
def service(pilot_root):
    return core.PilotService(pilot_root)


@pytest.fixture
def pending(service):
    initial = service.start(RUN_ID)
    assert initial["workflow_status"] == "pending_review"
    assert initial["new_model_calls"] == 4 and initial["new_live_requests"] == 0
    return service, initial


def response(view, **changes):
    return {"idempotency_key": "synthetic-response-1", "expected_state_id": view["state_id"],
            "expected_revision": view["record_revision"],
            "case_ids": [case["case_id"] for case in view["cases"] if case["status"] == "pending"],
            "outcome": "unresolved", "provenance": "simulated", "reviewer_id": "simulated-reviewer",
            "rationale": "Invented unresolved review for persistence testing only.",
            "proposal": None, "attestations": []} | changes


def run_bytes(root):
    directory = root / "runs/agent" / RUN_ID
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def test_health_is_read_only_and_explicit_about_scope(service):
    before = set(service.root.rglob("*"))
    result = service.health()
    assert result["status"] == "ready" and result["synthetic_only"]
    assert result["live_requests"] == 0 and result["empirical_status"] == "not_run"
    assert result["reviewer_identity"] == "self_declared_not_authenticated"
    assert result["limits"]["runs"] == 8
    assert set(service.root.rglob("*")) == before


def test_rejects_repository_root_without_bundle():
    with pytest.raises(core.PilotError):
        core.PilotService(Path(__file__).resolve().parents[2])


@pytest.mark.parametrize("run_id", ["m08-released", "../m10-x", "m10-x/y", "m10-UPPER", "m10-", "m10-service-lock", "m10-" + "a" * 61, "", None])
def test_rejects_unscoped_run_ids_before_writing(validation_service, run_id):
    service = validation_service
    with pytest.raises(core.PilotError, match="invalid_run_id"):
        service.start(run_id)
    assert not (service.root / "runs").exists()


@pytest.mark.parametrize("max_steps", [True, False, 0, -1, 13, "1", 1.0])
def test_rejects_step_coercion_and_unbounded_requests(validation_service, max_steps):
    service = validation_service
    with pytest.raises(core.PilotError, match="invalid_max_steps"):
        service.start(RUN_ID, max_steps=max_steps)
    assert not (service.root / "runs").exists()


def test_start_creates_pending_review_and_restart_preserves_every_byte(pending):
    service, initial = pending
    view = service.review(RUN_ID)
    assert initial["status"] == view["status"] == "pending"
    assert initial["pending_cases"] == 2 and view["decisions"] == 0
    assert view["reviewer_identity"] == "self_declared_not_authenticated"
    before = run_bytes(service.root)
    restarted = core.PilotService(service.root)
    replay = restarted.start(RUN_ID)
    assert replay["execution"] == "cached_replay" and replay["new_model_calls"] == 0
    assert restarted.status(RUN_ID)["state_id"] == view["state_id"]
    assert restarted.review(RUN_ID) == view
    assert before == run_bytes(service.root)


def test_pending_export_is_state_bound_and_does_not_become_approval(pending):
    service, initial = pending
    exported = service.export(RUN_ID, initial["state_id"])
    assert exported["manifest"]["status"] == "pending"
    assert exported["sidecar"]["expert_approval"] == "not_verified"
    assert exported["sidecar"]["review"]["decisions"] == []
    assert exported["canonical"]["datasets"][0]["sample_refs"] == ["sample_2"]
    before = run_bytes(service.root)
    assert service.export(RUN_ID, initial["state_id"]) == exported
    assert run_bytes(service.root) == before


def test_simulated_unresolved_is_durable_idempotent_and_unchanged_canonical(pending):
    service, initial = pending
    canonical = service.export(RUN_ID, initial["state_id"])["canonical"]
    payload = response(service.review(RUN_ID))
    first = service.submit(RUN_ID, payload)
    assert first["execution"] == "applied" and first["status"] == "unresolved"
    assert first["response_provenance"] == "simulated"
    assert first["pending_cases"] == first["approved_cases"] == 0
    assert first["unresolved_cases"] == 2
    assert first["new_model_calls"] == first["new_live_requests"] == 0
    before = run_bytes(service.root)
    restarted = core.PilotService(service.root)
    second = restarted.submit(RUN_ID, payload)
    assert second["execution"] == "replayed" and second["state_id"] == first["state_id"]
    assert run_bytes(service.root) == before
    assert restarted.export(RUN_ID, second["state_id"])["canonical"] == canonical


def test_explicit_simulated_edit_reuses_m08_candidate_attachment_and_evidence(pending):
    service, initial = pending
    view = service.review(RUN_ID)
    with ReviewStore.open(service.root, RUN_ID) as store:
        proposal = store.state.proposals[0].proposal
        candidate = build_candidate(store.profile, store.bundle, store.state.current, proposal).prepared.candidate
        evidence = view["cases"][0]["passages"][0]["evidence"]
        paths = {case["path"] for case in view["cases"]}
        paths.update(ref.path for ref in candidate.references if ref.path.startswith("/datasets/0/"))
        references = {ref.path for ref in candidate.references}
        attestations = [{"path": path, "value_json": canonical_json(at_pointer(candidate.canonical, path)),
                         "role": "attachment" if path in references else "measurement_acquisition",
                         "evidence": [evidence]} for path in sorted(paths)]
    result = service.submit(RUN_ID, response(view, outcome="edit", proposal=proposal.model_dump(mode="json"),
                                            attestations=attestations))
    assert result["status"] == "completed" and result["approved_cases"] == 2
    assert result["record_revision"] != initial["record_revision"]
    exported = service.export(RUN_ID, result["state_id"])
    dataset = exported["canonical"]["datasets"][0]
    assert dataset["dataset_description"] == "Measured at 300 K." and dataset["sample_refs"] == ["sample_1"]
    review = exported["sidecar"]["review"]
    assert review["expert_authentication"] == "not_authenticated"
    assert len(review["current_assertions"]) == 3
    assert all(item["provenance"] == "simulated" for item in review["current_assertions"])
    with pytest.raises(core.PilotError, match="stale_state"):
        service.export(RUN_ID, initial["state_id"])


def test_stale_review_rejected_and_duplicate_after_later_decision_returns_current_head(pending):
    service, _ = pending
    view = service.review(RUN_ID)
    first_payload = response(view, case_ids=[view["cases"][0]["case_id"]])
    stale = response(view, idempotency_key="stale-window")
    first = service.submit(RUN_ID, first_payload)
    before = run_bytes(service.root)
    with pytest.raises(core.PilotError, match="stale_state"):
        service.submit(RUN_ID, stale)
    assert run_bytes(service.root) == before
    second = service.submit(RUN_ID, response(service.review(RUN_ID), idempotency_key="second"))
    replay = service.submit(RUN_ID, first_payload)
    assert replay["execution"] == "replayed" and replay["state_id"] == second["state_id"] != first["state_id"]
    assert replay["decisions"] == 2


def test_conflicting_idempotency_key_rejected_without_mutation(pending):
    service, _ = pending
    payload = response(service.review(RUN_ID))
    service.submit(RUN_ID, payload)
    before = run_bytes(service.root)
    with pytest.raises(core.PilotError) as error:
        service.submit(RUN_ID, payload | {"rationale": "Changed simulated decision."})
    assert error.value.status_code == 409
    assert run_bytes(service.root) == before


def test_checkpoint_status_never_dispatches_and_explicit_restart_resumes_same_budget(service, monkeypatch):
    first = service.start(RUN_ID, max_steps=1)
    assert first["execution"] == "checkpointed" and first["new_model_calls"] == 1
    before = run_bytes(service.root)
    original = core.inspect_workflow
    monkeypatch.setattr(core, "inspect_workflow", lambda *a, **k: pytest.fail("Status dispatched workflow"))
    inspected = service.status(RUN_ID)
    assert inspected["workflow_status"] == "incomplete" and inspected["new_model_calls"] == 0
    assert run_bytes(service.root) == before
    with pytest.raises(core.PilotError, match="review_not_ready"):
        service.review(RUN_ID)
    monkeypatch.setattr(core, "inspect_workflow", original)
    resumed = core.PilotService(service.root).start(RUN_ID)
    assert resumed["execution"] == "resumed" and resumed["status"] == "pending"
    assert resumed["new_model_calls"] == 3 and resumed["total_model_calls"] == 4


def test_unknown_dispatch_crash_is_sanitized_and_resume_does_not_repeat_call(service, monkeypatch):
    from meta_data_extraction.agent import workflow_graph as graph
    original = graph.advance
    calls = []
    def crash(state, runtime):
        result = original(state, runtime)
        calls.append(result.model_calls)
        raise RuntimeError("private invented secret and source excerpt")
    monkeypatch.setattr(graph, "advance", crash)
    with pytest.raises(core.PilotError, match="operation_failed") as error:
        service.start(RUN_ID)
    assert "private" not in str(error.value) and calls == [1]
    monkeypatch.setattr(graph, "advance", lambda *a: pytest.fail("Uncertain dispatch was repeated"))
    result = core.PilotService(service.root).start(RUN_ID)
    assert result["workflow_status"] == "interrupted" and result["new_model_calls"] == 0
    assert result["uncertain_model_calls"] == 3 and result["usage_unknown"]
    assert service.status(RUN_ID)["workflow_status"] == "interrupted"


def test_export_failure_after_commit_recovers_through_same_response(pending, monkeypatch):
    service, _ = pending
    payload = response(service.review(RUN_ID))
    original = ReviewStore.export
    monkeypatch.setattr(ReviewStore, "export", lambda _: (_ for _ in ()).throw(OSError("private export path")))
    with pytest.raises(core.PilotError, match="export_pending_after_commit"):
        service.submit(RUN_ID, payload)
    assert service.review(RUN_ID)["decisions"] == 1
    monkeypatch.setattr(ReviewStore, "export", original)
    result = core.PilotService(service.root).submit(RUN_ID, payload)
    assert result["execution"] == "replayed" and result["decisions"] == 1


def test_shared_service_lease_prevents_competing_operations(pending):
    service, _ = pending
    with _lease(RunDirectory(service.root, core.LOCK_ID)):
        with pytest.raises(core.PilotError, match="pilot_busy"):
            core.PilotService(service.root).start("m10-other-run")
        with pytest.raises(core.PilotError, match="pilot_busy"):
            service.status(RUN_ID)
    assert not (service.root / "runs/agent/m10-other-run").exists()


def test_existing_review_run_lease_prevents_concurrent_submit(pending):
    service, _ = pending
    payload = response(service.review(RUN_ID))
    with ReviewStore.open(service.root, RUN_ID):
        with pytest.raises(core.PilotError, match="pilot_busy"):
            service.submit(RUN_ID, payload)
    assert service.review(RUN_ID)["decisions"] == 0


def test_capacity_exhaustion_prevents_dispatch(service, monkeypatch):
    for index in range(core.MAX_RUNS):
        (service.root / f"runs/agent/m10-occupied-{index}").mkdir(parents=True)
    monkeypatch.setattr(core, "inspect_workflow", lambda *a, **k: pytest.fail("Capacity dispatched"))
    with pytest.raises(core.PilotError, match="run_limit") as error:
        service.start(RUN_ID)
    assert error.value.status_code == 507 and not (service.root / f"runs/agent/{RUN_ID}").exists()


def test_storage_reserve_exhaustion_prevents_dispatch(service, monkeypatch):
    monkeypatch.setattr(core, "MAX_STORAGE_BYTES", core.MUTATION_RESERVE_BYTES - 1)
    monkeypatch.setattr(core, "inspect_workflow", lambda *a, **k: pytest.fail("Storage dispatched"))
    with pytest.raises(core.PilotError, match="storage_limit"):
        service.start(RUN_ID)


def test_missing_run_has_typed_not_found_error(service):
    with pytest.raises(core.PilotError, match="run_not_found") as error:
        service.status(RUN_ID)
    assert error.value.status_code == 404


@pytest.mark.parametrize("method", ["health", "start", "status", "review", "submit", "export"])
def test_every_operation_revalidates_bundle_before_dispatch(service, method):
    path = service.root / core.CONFIG_PATH
    path.write_text("{}", encoding="utf-8")
    kwargs = {"health": (), "start": (RUN_ID,), "status": (RUN_ID,), "review": (RUN_ID,),
              "submit": (RUN_ID, response({"state_id": "review-state:" + "0" * 64,
                                          "record_revision": "revision:" + "0" * 64,
                                          "cases": [{"case_id": "case", "status": "pending"}]})),
              "export": (RUN_ID, "review-state:" + "0" * 64)}[method]
    with pytest.raises(core.PilotError, match="bundle_invalid"):
        getattr(service, method)(*kwargs)


@pytest.mark.parametrize("payload", [[], None, {"unexpected": "private source content"}, {"a": float("nan")}, {1: "value"}])
def test_invalid_response_is_sanitized_before_review_access(validation_service, payload):
    service = validation_service
    with pytest.raises(core.PilotError) as error:
        service.submit(RUN_ID, payload)
    assert error.value.status_code == 400 and "private" not in str(error.value)
    assert not (service.root / "runs").exists()


def test_oversized_and_deep_responses_rejected_without_writes(validation_service):
    service = validation_service
    with pytest.raises(core.PilotError, match="response_too_large"):
        service.submit(RUN_ID, {"rationale": "x" * core.MAX_RESPONSE_BYTES})
    deep = {}
    for _ in range(34):
        deep = {"nested": deep}
    with pytest.raises(core.PilotError, match="json_too_deep"):
        service.submit(RUN_ID, deep)
    assert not (service.root / "runs").exists()


@pytest.mark.parametrize("state_id", [None, "old", "review-state:" + "A" * 64, ""])
def test_export_requires_explicit_valid_state_id_before_writing(validation_service, state_id):
    service = validation_service
    with pytest.raises(core.PilotError, match="invalid_state_id"):
        service.export(RUN_ID, state_id)
    assert not (service.root / "runs").exists()


def test_tampered_completed_artifact_cannot_be_reviewed_or_replayed(pending):
    service, _ = pending
    path = service.root / f"runs/agent/{RUN_ID}/canonical.json"
    path.write_text("{}", encoding="utf-8")
    for action in (lambda: service.status(RUN_ID), lambda: service.start(RUN_ID), lambda: service.review(RUN_ID)):
        with pytest.raises(core.PilotError):
            action()


def test_rejects_different_workflow_even_with_a_rehashed_request(pending):
    from meta_data_extraction.agent.storage import _digest, _json_bytes
    service, _ = pending
    path = service.root / f"runs/agent/{RUN_ID}/request.json"
    data = json.loads(path.read_bytes())
    data["request"]["declared"]["config"]["policy"] = "fixed"
    data["request_sha256"] = _digest(_json_bytes(data["request"]))
    path.write_bytes(_json_bytes(data))
    with pytest.raises(core.PilotError, match="run_scope_mismatch"):
        service.review(RUN_ID)


def test_linked_bundle_root_is_rejected_without_resolving_away_link(pilot_root, link_directory):
    link = pilot_root.parent / "linked-runtime"
    link_directory(link, pilot_root)
    with pytest.raises(core.PilotError):
        core.PilotService(link)


def test_internal_error_strings_cannot_escape_api(service, monkeypatch):
    monkeypatch.setattr(core, "inspect_workflow", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sk-private-test-value")))
    with pytest.raises(core.PilotError) as error:
        service.start(RUN_ID)
    assert str(error.value) == "operation_failed" and error.value.status_code == 500


def test_real_second_process_cannot_bypass_service_lease(pending):
    service, _ = pending
    script = """
import socket, sys
def denied(*args, **kwargs):
    raise AssertionError('Network forbidden in synthetic pilot subprocess')
socket.socket.connect = denied
socket.create_connection = denied
socket.getaddrinfo = denied
from meta_data_extraction.agent.pilot_core import PilotError, PilotService
service = PilotService(sys.argv[1])
try:
    service.start('m10-other-process')
except PilotError as error:
    assert error.code == 'pilot_busy'
    print(error.code)
else:
    raise AssertionError('Lease was bypassed')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(service.root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with _lease(RunDirectory(service.root, core.LOCK_ID)):
        result = subprocess.run([sys.executable, "-c", script, str(service.root)],
                                cwd=service.root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "pilot_busy"
    assert not (service.root / "runs/agent/m10-other-process").exists()


def test_windows_export_path_bound_is_checked_before_run_creation(validation_service, monkeypatch):
    service = validation_service
    original = service.root
    monkeypatch.setattr(service, "root", original / ("x" * 150))
    # Check the length rule directly using an otherwise valid path resolver; no
    # long directory or export needs to be created to exercise this boundary.
    monkeypatch.setattr(core.RunDirectory, "path", property(lambda run: run.root / run.run_id))
    monkeypatch.setattr(core, "WINDOWS_PATH_LIMIT", 260)
    with pytest.raises(core.PilotError, match="run_path_too_long"):
        service._run(RUN_ID)
