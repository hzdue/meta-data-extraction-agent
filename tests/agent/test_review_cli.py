"""Offline M08 review CLI tests; every submitted reviewer response is simulated.

A loader-only test preserves user_supplied declarations without applying them.
Real subprocess tests disable networking explicitly through a temporary startup hook.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from meta_data_extraction.agent.inspect_workflow import load_config, inspect_workflow
from meta_data_extraction.agent.review_store import ReviewStore
import meta_data_extraction.agent.review_cli as cli

CONFIG_PATH = "configs/agent/workflow_review.json"
RUN_ID = "m08-cli-unit"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def response_payload(**changes):
    return {
        "idempotency_key": "simulated-cli-response-1",
        "expected_state_id": "review-state:" + "0" * 64,
        "expected_revision": "revision:" + "0" * 64,
        "case_ids": ["review:" + "0" * 64],
        "outcome": "unresolved", "provenance": "simulated",
        "reviewer_id": "simulated-cli-reviewer", "rationale": "Invented unresolved test decision.",
        "proposal": None, "attestations": [],
    } | changes


def pending_response(state, **changes):
    return response_payload(expected_state_id=state.state_id, expected_revision=state.current.revision_id,
                            case_ids=[case.case_id for case in state.cases if case.status == "pending"], **changes)


@pytest.fixture
def fixture_workspace(workspace):
    repository = Path(__file__).resolve().parents[2]
    names = ["schemas/prompt_template.md", "configs/agent/record_inspection.json", CONFIG_PATH]
    names.extend(str(path.relative_to(repository)) for prefix in ("m06_", "m07_", "m08_")
                 for path in (repository / "tests/agent/fixtures").glob(prefix + "*"))
    for name in names:
        destination = workspace / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, destination)
    # A real subprocess reads this explicit implementation set for request pins;
    # imports still point to the current checkout, never a copied editable package.
    from meta_data_extraction.agent.inspect_workflow import _implementation
    for name in _implementation(repository):
        destination = workspace / "src/meta_data_extraction" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / "src/meta_data_extraction" / name, destination)
    config = load_config(workspace, CONFIG_PATH)
    config = type(config).model_validate(config.model_dump() | {"run_id": RUN_ID})
    write_json(workspace / CONFIG_PATH, config.model_dump(mode="json"))
    return workspace, config


@pytest.fixture
def pending(fixture_workspace):
    root, config = fixture_workspace
    result = inspect_workflow(root, config)
    assert result["state"].status == "pending_review" and result["new_model_calls"] == 4
    assert all(case.status == "pending" for case in result["state"].pending_reviews)
    with ReviewStore.open(root, RUN_ID) as store:
        state = store.state
    return root, state


@pytest.mark.parametrize("relative", [
    "tests/agent/fixtures/m08_response.json", f"runs/agent/{RUN_ID}/review-responses/answer.json",
])
def test_loads_one_explicit_allowed_response_without_changing_identity(workspace, relative):
    path = workspace / relative
    write_json(path, response_payload())
    response = cli.load_response(workspace, RUN_ID, relative)
    assert response.provenance == "simulated"
    assert response.reviewer_id == "simulated-cli-reviewer"
    assert response.outcome == "unresolved" and not response.attestations


def test_loader_keeps_user_supplied_declaration_without_inventing_expert_identity(workspace):
    relative = "tests/agent/fixtures/m08_loader_only.json"
    write_json(workspace / relative, response_payload(provenance="user_supplied", reviewer_id="self-declared-reviewer"))
    response = cli.load_response(workspace, RUN_ID, relative)
    assert response.provenance == "user_supplied" and response.reviewer_id == "self-declared-reviewer"
    assert not (workspace / "runs").exists()  # no decision was submitted


@pytest.mark.parametrize("relative", [
    "data/gold/review.json", "outputs/review.json", ".env",
    "configs/agent/review.json", "runs/agent/another-run/review-responses/review.json",
    f"runs/agent/{RUN_ID}/response.json", f"runs/agent/{RUN_ID}/review-responses/response.txt",
])
def test_response_scope_cannot_read_protected_neighbor_or_nonjson_files(workspace, relative):
    path = workspace / relative
    write_json(path, response_payload())
    with pytest.raises(ValueError):
        cli.load_response(workspace, RUN_ID, relative)


@pytest.mark.parametrize("raw", [
    b"[]", b"null", b"42", b"not-json", b"outcome: unresolved\n",
    b'{"outcome":"unresolved","outcome":"approve"}',
    b'{"input":NaN}', b'{"input":Infinity}',
])
def test_response_requires_strict_unambiguous_json_object(workspace, raw):
    relative = "tests/agent/fixtures/m08_invalid.json"
    (workspace / relative).write_bytes(raw)
    with pytest.raises(ValueError):
        cli.load_response(workspace, RUN_ID, relative)


def test_response_size_bound_rejects_before_parsing(workspace):
    relative = "tests/agent/fixtures/m08_oversize.json"
    (workspace / relative).write_bytes(b" " * (cli.MAX_RESPONSE_BYTES + 1))
    with pytest.raises(ValueError, match="size bound"):
        cli.load_response(workspace, RUN_ID, relative)


def test_loader_reads_bytes_once_and_does_not_print_response(workspace, monkeypatch, capsys):
    relative = "tests/agent/fixtures/m08_response.json"
    path = workspace / relative
    write_json(path, response_payload())
    original = Path.read_bytes
    reads = []
    def observed(current):
        if current == path:
            reads.append(current)
        return original(current)
    monkeypatch.setattr(Path, "read_bytes", observed)
    cli.load_response(workspace, RUN_ID, relative)
    assert reads == [path]
    assert capsys.readouterr().out == ""


def test_show_has_focused_assertions_passages_uncertainty_and_no_whole_record(pending):
    root, state = pending
    view = cli.show_review(root, RUN_ID)
    assert view["state_id"] == state.state_id
    assert len(view["cases"]) == 2
    serialized = json.dumps(view)
    assert "/datasets/0/dataset_description" in serialized and "/datasets/0/sample_refs/0" in serialized
    assert "500 K" in serialized and "300 K" in serialized
    assert "sample_1" in serialized and "sample_2" in serialized
    assert "canonical_json" not in view and "record" not in view
    assert all(case.status == "pending" for case in state.cases)


def test_submit_unresolved_commits_once_then_exports_and_replay_has_no_side_effect(pending):
    root, state = pending
    relative = f"runs/agent/{RUN_ID}/review-responses/unresolved.json"
    write_json(root / relative, pending_response(state))
    first = cli.submit_review(root, RUN_ID, relative)
    assert first["execution"] == "applied" and first["pending_cases"] == 0
    assert first["reviewer_identity"] == "self_declared_unverified" and first["export"]
    assert first["response_provenance"] == "simulated" and first["response_outcome"] == "unresolved"
    second = cli.submit_review(root, RUN_ID, relative)
    assert second["execution"] == "replayed" and second["state_id"] == first["state_id"]
    with ReviewStore.open(root, RUN_ID) as store:
        assert len(store.state.decisions) == 1
        assert all(case.status == "unresolved" for case in store.state.cases)
        assert store.state.current == state.current
    assert cli.export_review(root, RUN_ID) == first["export"]


def test_stale_response_is_rejected_without_a_second_decision(pending):
    root, state = pending
    relative = f"runs/agent/{RUN_ID}/review-responses/unresolved.json"
    write_json(root / relative, pending_response(state))
    cli.submit_review(root, RUN_ID, relative)
    write_json(root / relative, pending_response(state, idempotency_key="different-stale-response"))
    with pytest.raises(ValueError):
        cli.submit_review(root, RUN_ID, relative)
    with ReviewStore.open(root, RUN_ID) as store:
        assert len(store.state.decisions) == 1 and store.state.current == state.current


def test_export_failure_after_commit_is_recovered_by_resume_without_resubmission(pending, monkeypatch):
    root, state = pending
    relative = f"runs/agent/{RUN_ID}/review-responses/unresolved.json"
    write_json(root / relative, pending_response(state))
    original = ReviewStore.export
    def interrupted(self):
        raise RuntimeError("Synthetic interruption after committed decision")
    monkeypatch.setattr(ReviewStore, "export", interrupted)
    with pytest.raises(RuntimeError, match="Synthetic interruption"):
        cli.submit_review(root, RUN_ID, relative)
    monkeypatch.setattr(ReviewStore, "export", original)
    with ReviewStore.open(root, RUN_ID) as store:
        assert len(store.state.decisions) == 1
        committed_id = store.state.state_id
    assert cli.export_review(root, RUN_ID)
    with ReviewStore.open(root, RUN_ID) as store:
        assert store.state.state_id == committed_id and len(store.state.decisions) == 1


def test_cli_errors_hide_response_or_reviewer_values(workspace, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["review_cli", "show", "--root", str(workspace), "--run-id", RUN_ID])
    def invalid(*args):
        raise RuntimeError("private-reviewer-value-must-not-be-printed")
    monkeypatch.setattr(cli, "show_review", invalid)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "RuntimeError" in output.err and "private-reviewer" not in output.err
    assert output.out == ""


def test_resume_is_export_alias_and_never_loads_or_invents_response(workspace, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "export_review", lambda root, run_id: calls.append((root, run_id)) or {"export": "fixture"})
    monkeypatch.setattr(cli, "load_response", lambda *a: pytest.fail("Resume cannot invent a response"))
    monkeypatch.setattr(sys, "argv", ["review_cli", "resume", "--root", str(workspace), "--run-id", RUN_ID])
    assert cli.main() == 0
    assert calls == [(str(workspace), RUN_ID)]
    assert json.loads(capsys.readouterr().out) == {"export": "fixture"}


def test_submit_requires_explicit_response_option(workspace, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["review_cli", "submit", "--root", str(workspace), "--run-id", RUN_ID])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and "--response" in capsys.readouterr().err


def test_real_process_restart_reads_same_review_and_replays_without_duplicates(fixture_workspace):
    root, _ = fixture_workspace
    repository = Path(__file__).resolve().parents[2]
    guard = root / "subprocess-network-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import socket\ndef denied(*a, **k):\n raise AssertionError('Network disabled in review subprocess test')\n"
        "socket.socket.connect=denied\nsocket.socket.connect_ex=denied\n"
        "socket.create_connection=denied\nsocket.getaddrinfo=denied\n", encoding="utf-8")
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(guard), str(repository / "src"))),
                   "PYTHONDONTWRITEBYTECODE": "1", "LANGSMITH_TRACING": "false"}
    def command(module, *arguments):
        result = subprocess.run([sys.executable, "-m", "meta_data_extraction.agent." + module, *arguments],
                                cwd=repository, env=environment, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    started = command("inspect_workflow", "--root", str(root), "--config", CONFIG_PATH)
    assert started["status"] == "pending_review" and started["total_model_calls"] == 4
    process_a = command("review_cli", "show", "--root", str(root), "--run-id", RUN_ID)
    with ReviewStore.open(root, RUN_ID) as store:
        state = store.state
    assert process_a["state_id"] == state.state_id
    relative = f"runs/agent/{RUN_ID}/review-responses/process-response.json"
    write_json(root / relative, pending_response(state))
    process_b = command("review_cli", "submit", "--root", str(root), "--run-id", RUN_ID, "--response", relative)
    assert process_b["execution"] == "applied" and process_b["pending_cases"] == 0
    replay = command("review_cli", "submit", "--root", str(root), "--run-id", RUN_ID, "--response", relative)
    assert replay["execution"] == "replayed" and replay["state_id"] == process_b["state_id"]
    resumed = command("review_cli", "resume", "--root", str(root), "--run-id", RUN_ID)
    assert resumed == process_b["export"]
    with ReviewStore.open(root, RUN_ID) as store:
        assert len(store.state.decisions) == 1
        assert store.state.initial_workflow.model_calls == 4
        assert store.state.current == state.current


@pytest.mark.parametrize("code,visible", [("stale_state", "stale_state"),
                                         ("attestation_role_mismatch", "attestation_role_mismatch"),
                                         ("private-reviewer-fragment", "review_rejected")])
def test_cli_exposes_only_safe_review_rejection_codes(workspace, monkeypatch, capsys, code, visible):
    from meta_data_extraction.agent.review_types import ReviewError

    monkeypatch.setattr(sys, "argv", ["review_cli", "show", "--root", str(workspace), "--run-id", RUN_ID])
    def rejected(*args):
        raise ReviewError(code)
    monkeypatch.setattr(cli, "show_review", rejected)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert f"Review rejected ({visible})" in output.err
    assert "private-reviewer-fragment" not in output.err and output.out == ""
