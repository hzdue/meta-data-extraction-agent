"""Offline publication facade tests; reviews are explicitly simulated."""
from contextlib import contextmanager
from dataclasses import replace
import io
import json
from types import SimpleNamespace
import threading

import pytest

from test_workflow_graph import workflow_case, tool_case, execution, run
from test_pilot_http import ephemeral_loopback_only, _ACTIVE_PORTS, request, TOKEN
from meta_data_extraction.agent import publication_service as service_module
from meta_data_extraction.agent import publication_cli as cli
from meta_data_extraction.agent.contracts import digest
from meta_data_extraction.agent.pilot_core import PilotError
from meta_data_extraction.agent.publication_service import PublicationService, make_publication_server
from meta_data_extraction.agent.publication_types import PublicationSpec
from meta_data_extraction.agent.record_patches import PatchPolicy
from meta_data_extraction.agent.review_store import ReviewStore
from meta_data_extraction.agent.review_types import ReviewResponse
from meta_data_extraction.agent.workflow_types import WorkflowBudget


def spec(**changes):
    return PublicationSpec.model_validate({"run_id": "m10-publication", "mode": "mock", "split": "synthetic",
        "source": {"path": "tests/agent/fixtures/synthetic_source.md", "sha256": "0" * 64},
        "mock_responses": [{"path": "tests/agent/fixtures/response.json", "sha256": "1" * 64}], **changes})


def live_spec(**changes):
    return spec(mode="live", split="development", mock_responses=[],
                live={"endpoint": "https://configured.example.invalid/v1"}, **changes)


@pytest.fixture
def backend(workspace, monkeypatch):
    frozen = spec()
    summary = {"run_id": frozen.run_id, "spec_sha256": digest(frozen.model_dump(mode="json")),
               "workflow_run_id": "publication-child", "status": "pending_review",
               "workflow_status": "pending_review", "new_model_calls": 0, "new_live_requests": 0}
    calls = []
    def start(root, selected, **kwargs):
        calls.append((root, selected, kwargs))
        return {**summary, "execution": "mock"}
    fake = SimpleNamespace(run_publication=start, read_publication=lambda root, run_id: dict(summary))
    monkeypatch.setattr(service_module, "_runner", lambda: fake)
    return PublicationService(workspace, frozen), fake, summary, calls


def test_health_and_status_do_not_dispatch(backend):
    service, _, _, calls = backend
    assert service.health()["interface"] == "authenticated_json_api"
    assert service.status(service.spec.run_id)["workflow_status"] == "pending_review"
    assert not calls


def test_start_forwards_only_frozen_spec_and_explicit_options(backend):
    service, _, _, calls = backend
    service.start(service.spec.run_id, max_steps=2)
    assert calls == [(service.root, service.spec, {
        "max_steps": 2, "configured_chatai_endpoint": None, "load_dotenv_file": False})]


@pytest.mark.parametrize("method", ["start", "status", "review"])
def test_other_run_is_rejected_before_read_or_dispatch(backend, method):
    service, _, _, calls = backend
    with pytest.raises(PilotError, match="run_outside_scope"):
        getattr(service, method)("m10-other")
    assert not calls


@pytest.mark.parametrize("steps", [0, 13, True, "1", -1])
def test_step_limits_precede_dispatch(backend, steps):
    service, _, _, calls = backend
    with pytest.raises(PilotError, match="max_steps_invalid"):
        service.start(service.spec.run_id, max_steps=steps)
    assert not calls


def test_saved_spec_mismatch_blocks_review(backend):
    service, _, summary, calls = backend
    summary["spec_sha256"] = "2" * 64
    with pytest.raises(PilotError, match="publication_scope_changed"):
        service.review(service.spec.run_id)
    assert not calls


@pytest.mark.parametrize("status", ["incomplete", "finished", "failed", "budget_exhausted", "interrupted"])
def test_only_pending_review_enters_store(backend, status):
    service, _, summary, _ = backend
    summary["workflow_status"] = status
    with pytest.raises(PilotError, match="review_not_ready"):
        service.review(service.spec.run_id)


def test_live_requires_explicit_matching_endpoint_before_start(workspace, monkeypatch):
    selected = live_spec()
    service = PublicationService(workspace, selected)
    monkeypatch.setattr(service_module, "_runner", lambda: pytest.fail("No dispatch permitted"))
    assert service.health()["mode"] == "live"
    with pytest.raises(PilotError, match="trusted_endpoint_required"):
        service.start(selected.run_id)
    with pytest.raises(ValueError, match="differs"):
        PublicationService(workspace, selected, configured_chatai_endpoint="https://other.example.invalid")


def test_live_endpoint_and_credential_choice_are_forwarded(workspace, monkeypatch):
    selected = live_spec()
    calls = []
    fake = SimpleNamespace(run_publication=lambda *args, **kwargs: calls.append(kwargs) or {"status": "blocked"})
    monkeypatch.setattr(service_module, "_runner", lambda: fake)
    service = PublicationService(workspace, selected,
        configured_chatai_endpoint=selected.live.endpoint, load_dotenv_file=True)
    service.start(selected.run_id)
    assert calls[0]["configured_chatai_endpoint"] == selected.live.endpoint
    assert calls[0]["load_dotenv_file"] is True


def test_backend_exception_is_redacted(backend):
    service, fake, _, _ = backend
    def fail(*args, **kwargs):
        raise RuntimeError("secret-token reviewer-text source-text")
    fake.read_publication = fail
    with pytest.raises(PilotError, match="operation_failed") as error:
        service.status(service.spec.run_id)
    assert "secret" not in str(error.value)


@pytest.fixture
def pending(workflow_case, monkeypatch):
    context, initial, prepared = workflow_case
    key = prepared.proposal.proposal_id
    context = replace(context, proposals={key: prepared.proposal})
    initial = initial.model_copy(update={"issues": tuple(issue.model_copy(update={
        "diagnosis": "context_conflict", "proposal_ids": (key,),
    }) for issue in initial.issues)})
    case = context, initial, prepared
    declared = {"version": "publication-workflow-1", "proposals": {key: prepared.proposal.model_dump(mode="json")}}
    result = run(case, request=declared, run_id="pub-child")
    assert result["state"].status == "pending_review"
    selected = spec()
    summary = {"run_id": selected.run_id, "spec_sha256": digest(selected.model_dump(mode="json")),
               "workflow_run_id": "pub-child", "workflow_status": "pending_review"}
    fake = SimpleNamespace(read_publication=lambda *a: dict(summary),
                           run_publication=lambda *a, **k: pytest.fail("Review cannot dispatch"))
    monkeypatch.setattr(service_module, "_runner", lambda: fake)
    return PublicationService(context.root, selected), case, declared


def response(view):
    return ReviewResponse(idempotency_key="simulated-publication-review", expected_state_id=view["state_id"],
        expected_revision=view["record_revision"], case_ids=tuple(item["case_id"] for item in view["cases"]),
        outcome="unresolved", provenance="simulated", reviewer_id="simulated-tester",
        rationale="Invented persistence check; not an expert judgment.").model_dump(mode="json")


def test_review_cycle(pending):
    service, _, _ = pending
    view = service.review(service.spec.run_id)
    assert view["publication_run_id"] == service.spec.run_id
    assert view["run_id"] == "pub-child"
    assert view["cases"][0]["candidate_availability"] == "recorded"
    before = service.export(service.spec.run_id, view["state_id"])
    answer = response(view)
    submitted = service.submit(service.spec.run_id, answer)
    assert submitted["execution"] == "applied"
    assert submitted["new_model_calls"] == submitted["new_live_requests"] == 0
    replay = service.submit(service.spec.run_id, answer)
    assert replay["execution"] == "replayed" and replay["decisions"] == 1
    with pytest.raises(PilotError, match="stale_state"):
        service.export(service.spec.run_id, view["state_id"])
    after = service.export(service.spec.run_id, submitted["state_id"])
    assert after["canonical"] == before["canonical"]
    assert after["sidecar"]["expert_approval"] == "not_verified"


def test_stale_review(pending):
    service, _, _ = pending
    answer = response(service.review(service.spec.run_id))
    service.submit(service.spec.run_id, answer)
    answer["idempotency_key"] = "different-simulated-review"
    with pytest.raises(PilotError, match="stale_state"):
        service.submit(service.spec.run_id, answer)


def test_export_commit_recovery(pending, monkeypatch):
    service, _, _ = pending
    answer = response(service.review(service.spec.run_id))
    original = ReviewStore.export
    monkeypatch.setattr(ReviewStore, "export", lambda self: (_ for _ in ()).throw(OSError("secret-error")))
    with pytest.raises(PilotError, match="export_pending_after_commit"):
        service.submit(service.spec.run_id, answer)
    monkeypatch.setattr(ReviewStore, "export", original)
    result = service.submit(service.spec.run_id, answer)
    assert result["execution"] == "replayed" and result["decisions"] == 1


@pytest.mark.parametrize("remaining", [0, 1])
def test_manual_policy_needs_no_checker_budget(workflow_case, remaining):
    context, initial, prepared = workflow_case
    context = replace(context, patch_policy=PatchPolicy(accept_model_judgments=False))
    case = context, initial, prepared
    runtime = execution(case, budget=WorkflowBudget(max_model_calls=remaining, timeout_seconds=300))
    result = run(case, runtime=runtime, run_id="manual-budget")
    assert result["state"].status == "pending_review"
    assert result["new_model_calls"] == 0
    assert not runtime.tools.checker.requests
    assert result["state"].record == initial.record


@pytest.mark.parametrize("alteration", ["key", "unknown", "extra", "shape", "too_many"])
def test_invalid_publication_catalog(workflow_case, alteration):
    context, initial, prepared = workflow_case
    key = prepared.proposal.proposal_id
    context = replace(context, proposals={key: prepared.proposal})
    initial = initial.model_copy(update={"issues": tuple(issue.model_copy(update={
        "diagnosis": "context_conflict", "proposal_ids": (key,),
    }) for issue in initial.issues)})
    catalog = {key: prepared.proposal.model_dump(mode="json")}
    declared = {"version": "publication-workflow-1", "proposals": catalog}
    if alteration == "key":
        declared["proposals"] = {"renamed": catalog[key]}
    elif alteration == "unknown":
        other = prepared.proposal.model_copy(update={"summary": "Changed proposal identity"})
        declared["proposals"] = {other.proposal_id: other.model_dump(mode="json")}
    elif alteration == "extra":
        declared["unrecognized"] = True
    elif alteration == "shape":
        declared["proposals"] = []
    else:
        declared["proposals"] = {f"key-{i}": catalog[key] for i in range(33)}
    case = context, initial, prepared
    run(case, request=declared, run_id="bad-catalog")
    with pytest.raises(ValueError, match="[Cc]atalog|[Pp]roposal"):
        ReviewStore.open(context.root, "bad-catalog")


@contextmanager
def serving(service):
    log = io.StringIO()
    with make_publication_server(service, TOKEN, port=0, log_stream=log) as server:
        _ACTIVE_PORTS.add(server.server_port)
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            yield server, log
        finally:
            server.shutdown()
            worker.join(timeout=3)
            _ACTIVE_PORTS.remove(server.server_port)
            assert not worker.is_alive()


def test_http_is_one_scoped_authenticated_json_service(backend):
    service, _, _, calls = backend
    with serving(service) as (server, logs):
        status, headers, body = request(server, "/", auth=False)
        assert status == 200 and b"Local publication API" in body
        assert b"<script" not in body and b"synthetic fixture" not in body
        assert "default-src 'none'" in headers["Content-Security-Policy"]
        assert request(server, auth=False)[0] == 401
        assert request(server)[0] == 200
        assert request(server, "/runs/m10-other")[0] == 404
        assert request(server, "/runs", "POST", {"run_id": service.spec.run_id, "source": "data/raw/other.pdf"})[0] == 400
        assert not calls
        assert request(server, "/runs", "POST", {"run_id": service.spec.run_id, "max_steps": 1})[0] == 200
        assert len(calls) == 1
    assert TOKEN not in logs.getvalue() and service.spec.run_id not in logs.getvalue()


def test_serve_validates_run_id_and_endpoint_before_binding(workspace):
    with pytest.raises(ValueError, match="m10-"):
        make_publication_server(PublicationService(workspace, spec(run_id="publication")), TOKEN, port=0)
    with pytest.raises(PilotError, match="trusted_endpoint_required"):
        make_publication_server(PublicationService(workspace, live_spec()), TOKEN, port=0)


def test_cli_status_and_start_share_explicit_spec(backend, monkeypatch, capsys):
    service, _, _, calls = backend
    monkeypatch.setattr(cli, "load_publication_spec", lambda *a: service.spec)
    args = ["--root", str(service.root), "--spec", "configs/agent/chosen.json"]
    assert cli.main(["status", *args]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == service.spec.run_id
    assert not calls
    assert cli.main(["run", *args, "--max-steps", "1"]) == 0
    assert len(calls) == 1 and calls[0][2]["max_steps"] == 1


def test_cli_error_does_not_print_secret(backend, monkeypatch, capsys):
    service, _, _, _ = backend
    def fail(*a):
        raise ValueError("secret-credential-and-document")
    monkeypatch.setattr(cli, "load_publication_spec", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(["status", "--root", str(service.root), "--spec", "configs/agent/chosen.json"])
    assert error.value.code == 2
    assert "secret-credential" not in capsys.readouterr().err
