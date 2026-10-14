"""Runtime study CLI boundaries; every observation is an offline engineering test."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import meta_data_extraction.agent.study_cli as cli


def config_file(root, raw=b"{}", name="study.json"):
    path = root / "configs/agent" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return str(path.relative_to(root)).replace("\\", "/")


@pytest.fixture
def accept_spec(monkeypatch):
    received = []
    sentinel = object()

    def validate(value):
        received.append(value)
        return sentinel

    monkeypatch.setattr(cli, "StudySpec", SimpleNamespace(model_validate=validate))
    return received, sentinel


def test_loader_delegates_strict_object_to_typed_spec(workspace, accept_spec):
    relative = config_file(workspace, '{"label":"invented μ","cap":1.5}'.encode("utf-8"))
    received, sentinel = accept_spec
    assert cli.load_config(workspace, relative) is sentinel
    assert received == [{"label": "invented μ", "cap": 1.5}]
    assert not (workspace / "runs").exists()


@pytest.mark.parametrize("raw", [
    b'{"same":1,"same":2}', b'{"outer":{"same":1,"same":2}}',
    b'{"value":NaN}', b'{"value":Infinity}', b'{"value":-Infinity}',
    b'{"value":1e9999}', b'{"value":-1e9999}',
    b'[]', b'null', b'1', b'"object"', b'{', b'{"value":1} trailing',
    b'\xff', b'\xef\xbb\xbf{}',
])
def test_loader_rejects_ambiguous_or_non_object_json_before_validation(workspace, accept_spec, raw):
    relative = config_file(workspace, raw)
    with pytest.raises((ValueError, UnicodeError)):
        cli.load_config(workspace, relative)
    assert accept_spec[0] == []


def test_loader_rejects_excessive_structure_with_bounded_error(workspace, accept_spec):
    relative = config_file(workspace, b'{"nested":' + b'[' * 2000 + b'0' + b']' * 2000 + b'}')
    with pytest.raises(ValueError, match="structure bound"):
        cli.load_config(workspace, relative)
    assert accept_spec[0] == []


def test_loader_rejects_oversized_config_before_validation(workspace, accept_spec, monkeypatch):
    monkeypatch.setattr(cli, "MAX_CONFIG_BYTES", 10)
    relative = config_file(workspace, b'{"value":12345}')
    with pytest.raises(ValueError, match="size bound"):
        cli.load_config(workspace, relative)
    assert accept_spec[0] == []


def test_loader_requires_json_extension(workspace, accept_spec):
    relative = config_file(workspace, b'{}', name="study.yaml")
    with pytest.raises(ValueError, match="JSON file"):
        cli.load_config(workspace, relative)
    assert accept_spec[0] == []


@pytest.mark.parametrize("relative", [
    "data/gold/study.json", "configs/alignment_helper.yaml", ".env", "outputs/study.json",
    "configs/offline/study_synthetic_evaluation.json",
    "tests/agent/fixtures/study.json", "configs/agent/../../outputs/study.json",
    "configs/agent/study.json:alternate", "D:/elsewhere/study.json",
])
def test_loader_path_boundary_precedes_any_spec_or_provider(workspace, accept_spec, relative):
    with pytest.raises(ValueError):
        cli.load_config(workspace, relative)
    assert accept_spec[0] == []


def test_loader_rejects_linked_config(workspace, accept_spec):
    import os

    source = workspace / "source.json"
    source.write_bytes(b"{}")
    destination = workspace / "configs/agent/linked.json"
    os.link(source, destination)
    with pytest.raises(ValueError, match="Hard-linked"):
        cli.load_config(workspace, "configs/agent/linked.json")
    assert accept_spec[0] == []


def test_execute_requires_fresh_directory_before_read_or_dispatch(workspace, monkeypatch):
    path = workspace / "runs/agent/existing-study"
    path.mkdir(parents=True)
    marker = path / "marker.txt"
    marker.write_bytes(b"original")
    monkeypatch.setattr(cli, "load_config", lambda *args: pytest.fail("Must reject before input read"))
    monkeypatch.setattr(cli, "run_study", lambda *args, **kwargs: pytest.fail("Must not dispatch"))
    with pytest.raises(FileExistsError):
        cli.execute(workspace, "configs/agent/study.json", run_id="existing-study")
    assert marker.read_bytes() == b"original"


@pytest.mark.parametrize("run_id", ["../outputs", "", "CON", "slash/name", "x" * 81])
def test_execute_rejects_invalid_run_id_before_input_read(workspace, monkeypatch, run_id):
    monkeypatch.setattr(cli, "load_config", lambda *args: pytest.fail("Invalid run must not read config"))
    with pytest.raises(ValueError):
        cli.execute(workspace, "configs/agent/study.json", run_id=run_id)
    assert not (workspace / "runs").exists()


def test_execute_passes_exact_loaded_spec_without_live_options(workspace, monkeypatch):
    spec = object()
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda root, relative: spec)

    def run(root, value, *, run_id):
        calls.append((root, value, run_id))
        return {"execution": "executed", "new_live_requests": 0}

    monkeypatch.setattr(cli, "run_study", run)
    assert cli.execute(workspace, "configs/agent/study.json", run_id="fresh-study") == {
        "execution": "executed", "new_live_requests": 0,
    }
    assert calls == [(workspace.resolve(), spec, "fresh-study")]


@pytest.mark.parametrize("existing", ["missing", "empty", "request_only", "result_directory"])
def test_replay_rejects_missing_or_incomplete_run_before_read_or_dispatch(workspace, monkeypatch, existing):
    directory = workspace / "runs/agent/replay-study"
    if existing != "missing":
        directory.mkdir(parents=True)
    if existing == "request_only":
        (directory / "request.json").write_bytes(b"{}")
    if existing == "result_directory":
        (directory / "result.json").mkdir()
    before = {str(path.relative_to(workspace)): path.read_bytes()
              for path in workspace.rglob("*") if path.is_file()}
    monkeypatch.setattr(cli, "load_config", lambda *args: pytest.fail("Must reject before input read"))
    monkeypatch.setattr(cli, "run_study", lambda *args, **kwargs: pytest.fail("Must not dispatch"))
    with pytest.raises(cli.RunIncompleteError):
        cli.replay(workspace, "configs/agent/study.json", run_id="replay-study")
    assert before == {str(path.relative_to(workspace)): path.read_bytes()
                      for path in workspace.rglob("*") if path.is_file()}
    assert directory.exists() == (existing != "missing")


def test_replay_calls_only_read_only_library_mode_and_requires_zero_calls(workspace, monkeypatch):
    directory = workspace / "runs/agent/replay-study"
    directory.mkdir(parents=True)
    (directory / "result.json").write_bytes(b"{}")
    spec = object()
    calls = []
    report = {"execution": "cached_replay", "new_model_calls": 0, "new_live_requests": 0}
    monkeypatch.setattr(cli, "load_config", lambda root, relative: spec)

    def run(root, loaded, *, run_id, replay_only):
        calls.append((root, loaded, run_id, replay_only))
        return report

    monkeypatch.setattr(cli, "run_study", run)
    assert cli.replay(workspace, "configs/agent/study.json", run_id="replay-study") is report
    assert calls == [(workspace.resolve(), spec, "replay-study", True)]
    assert (directory / "result.json").read_bytes() == b"{}"


@pytest.mark.parametrize("change", [
    {"execution": "executed"}, {"new_model_calls": 1}, {"new_live_requests": 1},
    {"new_model_calls": None}, {"new_model_calls": False}, {"new_live_requests": "0"},
])
def test_replay_rejects_invalid_library_execution_observation(workspace, monkeypatch, change):
    directory = workspace / "runs/agent/replay-study"
    directory.mkdir(parents=True)
    (directory / "result.json").write_bytes(b"{}")
    monkeypatch.setattr(cli, "load_config", lambda *args: object())
    report = {"execution": "cached_replay", "new_model_calls": 0, "new_live_requests": 0} | change
    monkeypatch.setattr(cli, "run_study", lambda *args, **kwargs: report)
    with pytest.raises(cli.RunConflictError):
        cli.replay(workspace, "configs/agent/study.json", run_id="replay-study")


def test_failed_config_does_not_claim_run_or_call_provider(workspace, monkeypatch):
    monkeypatch.setattr(cli, "run_study", lambda *args, **kwargs: pytest.fail("Config must validate first"))
    with pytest.raises(ValueError):
        cli.execute(workspace, config_file(workspace, b"[]"), run_id="fresh-study")
    assert not (workspace / "runs").exists()


def test_dry_run_cli_dispatches_only_validation_and_estimate(monkeypatch, capsys):
    spec = object()
    calls = []
    monkeypatch.setattr(sys, "argv", ["study", "dry-run", "--config", "configs/agent/study.json"])
    monkeypatch.setattr(cli, "load_config", lambda root, relative: spec)

    report = {
        "version": "m09-study-dry-run-1", "readiness": "ready", "estimated_cells": 3,
        "live_requests": 0, "spec_sha256": "a" * 64, "maximum_model_calls": 9,
        "max_total_model_calls": 9, "limitations": ["Synthetic only"],
        "frozen": {"code": {"some.py": "b" * 64}},
        "spec": {"scope": "synthetic_repair_policy", "authorized_mode": "synthetic_offline",
                 "evaluation_protocol_sha256": "c" * 64, "primary_outcome": "Declared outcome",
                 "counting_policy": "Publication-level fixture", "publications": [
                     {"publication_id": "invented-a", "split": "calibration", "data_kind": "synthetic"},
                 ], "variants": ["fixed"], "caps": [3], "repeats": 1},
    }

    def estimate(root, loaded):
        calls.append((root, loaded))
        return report

    monkeypatch.setattr(cli, "dry_run", estimate)
    monkeypatch.setattr(cli, "run_study", lambda *args, **kwargs: pytest.fail("Dry-run must not execute"))
    assert cli.main() == 0
    assert calls == [(".", spec)]
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["readiness"] == "ready" and result["live_requests"] == 0
    assert result["publication_count"] == 1 and result["estimated_cells"] == 3
    assert result["evaluation_protocol_sha256"] == "c" * 64
    assert "frozen" not in result and "spec" not in result
    assert captured.err == ""


@pytest.mark.parametrize("command", ["run", "replay"])
def test_run_cli_requires_explicit_id_without_dispatch(monkeypatch, capsys, command):
    monkeypatch.setattr(sys, "argv", ["study", command, "--config", "configs/agent/study.json"])
    monkeypatch.setattr(cli, "execute", lambda *args, **kwargs: pytest.fail("No implicit run ID"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "--run-id" in capsys.readouterr().err


def run_report():
    base = {"status": "completed", "workflow_status": "finished", "canonical": {"sha256": "d" * 64}}
    failed = {"status": "failed", "workflow_status": None, "canonical": None,
              "cell_id": "cell-missing", "publication_id": "invented-a", "variant": "generic_llm",
              "cap": 5, "repeat": 0, "error_code": "cell_exports_incomplete"}
    return {"execution": "executed", "new_model_calls": None, "new_live_requests": 0,
            "result": {"version": "m09-study-result-1", "spec_sha256": "a" * 64,
                       "spec": {"scope": "synthetic_repair_policy"},
                       "cells": [base, {**base, "workflow_status": "budget_exhausted"}, failed],
                       "budget": {"max_total_model_calls": 15},
                       "accounting": {"new_model_calls_known": 4, "new_model_calls_unknown": True},
                       "limitations": ["Synthetic only"]}}


def test_run_summary_counts_failures_caps_and_unknown_usage_without_printing_predictions():
    result = cli.run_summary(run_report())
    assert result["total_cells"] == 3 and result["canonical_exports"] == 2
    assert result["canonical_variants"] == 1
    assert result["cell_statuses"] == {"completed": 2, "failed": 1}
    assert result["workflow_statuses"] == {"budget_exhausted": 1, "finished": 1}
    assert result["new_model_calls"] is None and result["new_live_requests"] == 0
    assert result["accounting"]["new_model_calls_unknown"] is True
    assert result["failed_cells"][0]["cell_id"] == "cell-missing"
    assert result["failed_cells"][0]["error_code"] == "cell_exports_incomplete"
    assert "cells" not in result and "result" not in result


def test_run_cli_dispatches_explicit_coordinates_and_prints_concise_summary(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sys, "argv", ["study", "run", "--root", "test-root", "--config",
                                    "configs/agent/study.json", "--run-id", "fresh"])

    def execute(root, relative, *, run_id):
        calls.append((root, relative, run_id))
        return run_report()

    monkeypatch.setattr(cli, "execute", execute)
    assert cli.main() == 0
    assert calls == [("test-root", "configs/agent/study.json", "fresh")]
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["new_live_requests"] == 0 and result["canonical_variants"] == 1
    assert captured.err == ""


def test_replay_cli_dispatches_explicit_completed_id_and_preserves_original_accounting(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sys, "argv", ["study", "replay", "--root", "test-root", "--config",
                                    "configs/agent/study.json", "--run-id", "completed"])
    report = run_report() | {"execution": "cached_replay", "new_model_calls": 0}

    def replay(root, relative, *, run_id):
        calls.append((root, relative, run_id))
        return report

    monkeypatch.setattr(cli, "replay", replay)
    monkeypatch.setattr(cli, "execute", lambda *args, **kwargs: pytest.fail("Replay must not create a study"))
    assert cli.main() == 0
    assert calls == [("test-root", "configs/agent/study.json", "completed")]
    result = json.loads(capsys.readouterr().out)
    assert result["execution"] == "cached_replay" and result["new_model_calls"] == 0
    assert result["new_live_requests"] == 0
    assert result["accounting"]["new_model_calls_known"] == 4
    assert result["accounting"]["new_model_calls_unknown"] is True


@pytest.mark.parametrize("option", ["--endpoint", "--model", "--gold", "--api-key"])
def test_cli_exposes_no_live_or_offline_scoring_options(monkeypatch, capsys, option):
    monkeypatch.setattr(sys, "argv", ["study", "dry-run", "--config", "configs/agent/study.json", option, "unused"])
    monkeypatch.setattr(cli, "dry_run", lambda *args: pytest.fail("Unknown option must not execute"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("kind", [ValueError, OSError, RuntimeError, KeyError, TypeError])
def test_cli_error_output_redacts_exception_content(monkeypatch, capsys, kind):
    secret = "private-test-credential-value"
    monkeypatch.setattr(sys, "argv", ["study", "run", "--config", "configs/agent/study.json", "--run-id", "fresh"])

    def fail(*args, **kwargs):
        raise kind(secret)

    monkeypatch.setattr(cli, "execute", fail)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert secret not in captured.err
    assert f"({kind.__name__})" in captured.err


def test_runtime_cli_has_no_offline_evaluation_imports():
    source = Path(cli.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [item.name for node in ast.walk(tree) if isinstance(node, ast.Import) for item in node.names]
    assert not any("offline" in module or "evaluation" in module for module in imports)


def test_actual_fixture_dry_run_in_read_only_offline_subprocess():
    import os
    import subprocess

    repository = Path(__file__).resolve().parents[2]
    script = r'''
import builtins
import json
from pathlib import Path
import socket
import sqlite3
import sys
from meta_data_extraction.agent import study_cli, providers, workflow
from meta_data_extraction.agent.storage import RunStore

def denied(*args, **kwargs):
    raise AssertionError("Dry-run attempted a write, dispatch, or network operation")

original_open = builtins.open
original_path_open = Path.open

def check(path, mode):
    if any(flag in mode for flag in "wax+"):
        denied()
    normalized = str(path).replace("\\", "/").lower()
    if any(fragment in normalized for fragment in (
        "/data/gold/", "/data/annotations/", "/configs/offline/",
        "/outputs/", "/paper/", "alignment_helper.yaml",
    )) or normalized.endswith("/.env"):
        raise AssertionError("Dry-run crossed runtime input boundary")

def guarded_open(path, mode="r", *args, **kwargs):
    check(path, mode)
    return original_open(path, mode, *args, **kwargs)

def guarded_path_open(path, mode="r", *args, **kwargs):
    check(path, mode)
    return original_path_open(path, mode, *args, **kwargs)

builtins.open = guarded_open
Path.open = guarded_path_open
Path.mkdir = denied
sqlite3.connect = denied
socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
socket.getaddrinfo = denied
providers.MockProvider.generate = denied
providers.ChatAIProvider.generate = denied
workflow.WorkflowMockProvider.generate = denied
RunStore.begin = denied
study_cli.run_study = denied
sys.argv = ["study", "dry-run", "--config", "configs/agent/study_synthetic.json"]
raise SystemExit(study_cli.main())
'''
    child = subprocess.run(
        [sys.executable, "-c", script], cwd=repository,
        env=os.environ | {"PYTHONPATH": str(repository / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        check=False, capture_output=True, text=True, encoding="utf-8", timeout=45,
    )
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result["readiness"] == "ready" and result["scope"] == "synthetic_repair_policy"
    assert result["publication_count"] == 1 and result["estimated_cells"] == 6
    assert result["maximum_model_calls"] == result["max_total_model_calls"] == 21
    assert result["caps"] == [2, 5] and result["repeats"] == 1
    assert result["live_requests"] == 0
    assert result["variants"] == ["fixed", "issue_aware", "generic_llm"]
    assert all(item["data_kind"] == "synthetic" for item in result["publications"])
