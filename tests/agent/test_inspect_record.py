"""M06 runner integration using invented documents and mock checker responses only."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.agent.storage import RunConflictError, RunIncompleteError
import meta_data_extraction.agent.inspect_record as runner

CONFIG_PATH = "configs/agent/record_inspection.json"
RUN_ID = "m06-unit"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def amended(config, **changes):
    return runner.RecordRunConfig.model_validate(config.model_dump() | changes)


def change_fixture(root, config, field, content):
    pin = getattr(config, field)
    path = root / pin.path
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        write_json(path, content)
    return amended(config, **{field: {"path": pin.path, "sha256": sha256(path.read_bytes())}})


def no_patch(config):
    return amended(config, patch=None, patch_policy={"accept_model_judgments": False},
                   after_targets=[], after_response=None)


@pytest.fixture
def scope(workspace, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    names = ["schemas/prompt_template.md", CONFIG_PATH,
             "configs/agent/record_structural_inspection.json"]
    names += [str(path.relative_to(repository)) for path in
              (repository / "tests/agent/fixtures").glob("m06_*")]
    for name in names:
        destination = workspace / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, destination)
    monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic-runtime": "a" * 64})
    config = amended(runner.load_config(workspace, CONFIG_PATH), run_id=RUN_ID)
    return workspace, config


def test_patch_run_exports_atomic_changed_record_and_replay_has_zero_dispatch(scope, monkeypatch):
    root, config = scope
    original = (root / config.record.path).read_bytes()
    first = runner.inspect_record(root, config)
    assert first["execution"] == "synthetic_inspection"
    assert first["new_mock_requests"] == 2 and first["new_live_requests"] == 0
    result = first["result"]
    assert result["structure"]["status"] == "valid"
    assert result["patch_state"]["status"] == "applied"
    assert result["expert_approval"] == "not_supplied"
    assert result["coverage_status"] == "unsupported"
    assert result["support_scope"] == "selected_assertions_only"
    assert result["cost"] is None and result["cost_status"] == "not_applicable_mock"
    assert {j["provenance"] for j in result["before"]["judgments"]} == {"mock_model_judgment"}
    assert result["final_revision"] != result["structure"]["revision_id"]
    directory = root / "runs/agent" / RUN_ID
    canonical = json.loads((directory / "canonical.json").read_bytes())
    assert canonical["datasets"][0]["dataset_description"] == "Measured at 300 K."
    assert canonical["datasets"][0]["sample_refs"] == ["sample_1"]
    assert len(canonical["samples"]) == 2
    assert (root / config.record.path).read_bytes() == original
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    monkeypatch.setattr(runner, "check_support", lambda *a, **k: pytest.fail("Replay must not dispatch"))
    replay = runner.inspect_record(root, config)
    assert replay["execution"] == "cached_replay"
    assert replay["new_mock_requests"] == replay["new_live_requests"] == 0
    assert replay["result"] == result
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_invalid_record_preserved_and_never_sent_to_checker_or_exported_canonical(scope, monkeypatch):
    root, config = scope
    record = json.loads((root / config.record.path).read_bytes())
    record["datasets"][0]["sample_refs"] = ["sample_999"]
    record["unexpected"] = {"kept": [None, "raw extra"]}
    config = change_fixture(root, config, "record", record)
    monkeypatch.setattr(runner, "check_support", lambda *a, **k: pytest.fail("Invalid structure cannot dispatch"))
    report = runner.inspect_record(root, config)
    result = report["result"]
    assert report["new_mock_requests"] == 0
    assert result["structure"]["status"] == "invalid"
    assert json.loads(result["structure"]["candidate_json"]) == record
    assert result["support_execution"] == "blocked_by_structure"
    assert result["patch_state"]["code"] == "invalid_original_structure"
    assert result["before"] is None and result["after"] is None and result["final_revision"] is None
    assert {p.name for p in (root / "runs/agent" / RUN_ID).iterdir()} == {
        "request.json", "result.json", "sidecar.json",
    }


def test_structure_only_run_makes_no_mock_request_and_no_support_claim(scope, monkeypatch):
    root, config = scope
    config = amended(no_patch(config), targets=[], before_response=None)
    monkeypatch.setattr(runner, "check_support", lambda *a, **k: pytest.fail("No checker configured"))
    report = runner.inspect_record(root, config)
    assert report["new_mock_requests"] == 0
    assert report["result"]["support_execution"] == "not_requested"
    assert report["result"]["structure"]["support_status"] == "not_assessed"
    assert report["result"]["patch_state"]["status"] == "not_requested"


@pytest.mark.parametrize("policy", [False, True])
def test_disabled_policy_or_stale_patch_keeps_original_record(scope, policy):
    root, config = scope
    if policy:
        patch = config.patch.model_dump()
        patch["expected_revision"] = "revision:" + "0" * 64
        config = amended(config, patch=patch)
        code = "stale_revision"
    else:
        config = amended(config, patch_policy={"accept_model_judgments": False})
        code = "model_judgment_policy_disabled"
    result = runner.inspect_record(root, config)["result"]
    assert result["patch_state"]["code"] == code
    assert result["after"] is None and result["patch"] is None
    canonical = json.loads((root / "runs/agent" / RUN_ID / "canonical.json").read_bytes())
    assert canonical == json.loads((root / config.record.path).read_bytes())
    assert result["final_revision"] == result["structure"]["revision_id"]


@pytest.mark.parametrize("retention", ["retain", "hash_only"])
def test_raw_response_policy_is_enforced_and_hashed_observation_retained(scope, retention):
    root, config = scope
    config = amended(config, raw_retention=retention)
    result = runner.inspect_record(root, config)["result"]
    directory = root / "runs/agent" / RUN_ID
    assert (directory / "raw-response.json").exists() == (retention == "retain")
    assert "raw_output" not in result["before"] and "raw_output" not in result["after"]
    assert result["before"]["provider"]["response_sha256"] == config.before_response.sha256
    if retention == "retain":
        raw = json.loads(json.loads((directory / "raw-response.json").read_bytes())["raw_response"])
        assert raw["before"] == (root / config.before_response.path).read_bytes().decode("utf-8")
    assert runner.inspect_record(root, config)["execution"] == "cached_replay"


@pytest.mark.parametrize("changes", [
    {"mode": "live"}, {"raw_retention": "unknown"}, {"run_id": "../outputs"},
    {"targets": []}, {"after_response": None}, {"unexpected": "forbidden"},
])
def test_configuration_rejects_unbounded_or_inconsistent_execution(scope, changes):
    _, config = scope
    with pytest.raises(ValueError):
        amended(config, **changes)


def test_duplicate_targets_and_nonsynthetic_patch_are_rejected(scope):
    _, config = scope
    with pytest.raises(ValueError, match="distinct"):
        amended(config, targets=config.targets + config.targets[:1])
    patch = config.patch.model_dump()
    patch["provenance"] = "model_proposal"
    with pytest.raises(ValueError, match="synthetic"):
        amended(config, patch=patch)


@pytest.mark.parametrize("field", ["source", "record", "before_response", "after_response"])
def test_every_declared_input_hash_is_checked_before_execution_claim(scope, field):
    root, config = scope
    pin = getattr(config, field)
    config = amended(config, **{field: {"path": pin.path, "sha256": "0" * 64}})
    with pytest.raises(ValueError, match="declared hash"):
        runner.inspect_record(root, config)
    assert not (root / "runs/agent" / RUN_ID).exists()


@pytest.mark.parametrize("field,path", [
    ("source", "data/gold/secret.md"), ("record", "outputs/released.json"),
    ("before_response", "configs/alignment_helper.yaml"), ("after_response", ".env"),
])
def test_forbidden_input_roots_are_rejected_before_read_or_run_claim(scope, field, path):
    root, config = scope
    config = amended(config, **{field: {"path": path, "sha256": "a" * 64}})
    with pytest.raises(ValueError):
        runner.inspect_record(root, config)
    assert not (root / "runs/agent" / RUN_ID).exists()


def test_evidence_span_must_fit_actual_source_region(scope):
    root, config = scope
    target = config.targets[0].model_dump()
    target["spans"] = [{"start": 0, "end": 100_000}]
    config = amended(config, targets=[target, config.targets[1].model_dump()])
    with pytest.raises(ValueError, match="actual source region"):
        runner.inspect_record(root, config)
    assert not (root / "runs/agent" / RUN_ID).exists()


@pytest.mark.parametrize("field", ["before_response", "after_response"])
def test_malformed_checker_response_is_failed_not_empty_success(scope, field):
    root, config = scope
    config = change_fixture(root, config, field, b"not-json")
    report = runner.inspect_record(root, config)
    result = report["result"]
    failed = result["before"] if field == "before_response" else result["after"]
    assert failed["status"] == "failed" and failed["judgments"] == []
    assert failed["parse_status"] == "failed" and failed["issues"]
    assert result["patch_state"]["status"] == "rejected"
    assert result["final_revision"] == result["structure"]["revision_id"]
    assert runner.inspect_record(root, config)["execution"] == "cached_replay"


def test_strict_config_loader_rejects_duplicate_keys_and_oversized_input(scope):
    root, config = scope
    assert runner.load_config(root, CONFIG_PATH).source == config.source
    path = root / CONFIG_PATH
    path.write_bytes(b"mode: synthetic_offline\nmode: live\n")
    with pytest.raises(ValueError):
        runner.load_config(root, CONFIG_PATH)
    path.write_bytes(b" " * (runner.MAX_INPUT_BYTES + 1))
    with pytest.raises(ValueError, match="size bound"):
        runner.load_config(root, CONFIG_PATH)


@pytest.mark.parametrize("change", ["implementation", "source", "record", "fixture", "configuration"])
def test_changed_execution_inputs_require_fresh_run(scope, monkeypatch, change):
    root, config = scope
    runner.inspect_record(root, config)
    if change == "implementation":
        monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic-runtime": "b" * 64})
    elif change == "source":
        config = change_fixture(root, config, "source", (root / config.source.path).read_bytes() + b"\nAdded source.\n")
    elif change == "record":
        record = json.loads((root / config.record.path).read_bytes())
        record["bibliographic"]["title"] = "Changed synthetic title"
        config = change_fixture(root, config, "record", record)
    elif change == "fixture":
        config = change_fixture(root, config, "before_response", (root / config.before_response.path).read_bytes() + b"\n")
    else:
        config = amended(config, raw_retention="hash_only")
    with pytest.raises(RunConflictError):
        runner.inspect_record(root, config)


def rewrite_completion(root, envelope, *, artifact=None, content=None):
    from meta_data_extraction.agent.storage import RunStore, _digest, _json_bytes
    directory = root / "runs/agent" / RUN_ID
    if artifact:
        raw = _json_bytes(content)
        (directory / artifact).write_bytes(raw)
        envelope["artifacts"][artifact] = {"sha256": _digest(raw), "bytes": len(raw)}
    envelope["completion_sha256"] = _digest(_json_bytes({
        key: value for key, value in envelope.items() if key != "completion_sha256"
    }))
    (directory / "result.json").write_bytes(_json_bytes(envelope))
    request = json.loads((directory / "request.json").read_bytes())["request"]
    assert RunStore.begin(root, RUN_ID, request).completed == envelope["result"]


@pytest.mark.parametrize("key,value", [
    ("coverage_status", "complete"), ("expert_approval", "approved"),
    ("live_requests", 1), ("support_scope", "whole_publication"),
    ("final_revision", "revision:forged"), ("extra", "untracked-content"),
])
def test_rehashed_metadata_forgery_rejected_by_full_semantic_replay(scope, key, value):
    root, config = scope
    runner.inspect_record(root, config)
    envelope = json.loads((root / "runs/agent" / RUN_ID / "result.json").read_bytes())
    envelope["result"][key] = value
    rewrite_completion(root, envelope)
    with pytest.raises(RunConflictError, match="different output"):
        runner.inspect_record(root, config)


@pytest.mark.parametrize("artifact", ["sidecar.json", "canonical.json", "raw-response.json"])
def test_rehashed_artifact_forgery_rejected_by_recomputed_expected_bytes(scope, artifact):
    root, config = scope
    runner.inspect_record(root, config)
    envelope = json.loads((root / "runs/agent" / RUN_ID / "result.json").read_bytes())
    rewrite_completion(root, envelope, artifact=artifact, content={"forged": "unapproved replacement"})
    with pytest.raises(RunConflictError, match="different output"):
        runner.inspect_record(root, config)


def test_interrupted_claim_never_dispatches_again_automatically(scope, monkeypatch):
    root, config = scope
    original = runner.check_support
    def interrupted(*args, **kwargs):
        raise RuntimeError("Synthetic interruption after claim")
    monkeypatch.setattr(runner, "check_support", interrupted)
    with pytest.raises(RuntimeError, match="Synthetic interruption"):
        runner.inspect_record(root, config)
    monkeypatch.setattr(runner, "check_support", original)
    with pytest.raises(RunIncompleteError):
        runner.inspect_record(root, config)


def test_cli_hides_private_error_values(scope, monkeypatch, capsys):
    root, _ = scope
    monkeypatch.setattr(sys, "argv", ["inspect_record", "--root", str(root), "--config", CONFIG_PATH])
    def invalid(*args):
        raise ValueError("private-document-fragment-never-log")
    monkeypatch.setattr(runner, "load_config", invalid)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "ValueError" in captured.err and "private-document" not in captured.err
    assert captured.out == ""


def test_cli_summary_labels_mock_calls_and_unsupported_coverage(scope, monkeypatch, capsys):
    root, config = scope
    write_json(root / CONFIG_PATH, config.model_dump(mode="json"))
    monkeypatch.setattr(sys, "argv", ["inspect_record", "--root", str(root), "--config", CONFIG_PATH])
    assert runner.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["new_mock_requests"] == 2 and summary["new_live_requests"] == 0
    assert summary["coverage_status"] == "unsupported"
    assert summary["patch"]["status"] == "applied"


def test_fixture_record_and_responses_are_read_once_for_hash_and_parse(scope, monkeypatch):
    root, config = scope
    paths = {root / getattr(config, field).path for field in ("record", "before_response", "after_response")}
    original = Path.read_bytes
    reads = []
    def observed(path):
        if path in paths:
            reads.append(path)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", observed)
    runner.inspect_record(root, config)
    assert len(reads) == len(paths) and set(reads) == paths


@pytest.mark.parametrize("verdict,diagnosis,code", [
    ("contradicted", "context_conflict", "source_conflict_requires_review"),
    ("insufficient_evidence", "context_conflict", "source_conflict_requires_review"),
    ("insufficient_evidence", "ambiguous_expression", "old_assertion_not_contradicted"),
])
def test_conflicting_or_insufficient_source_stops_before_candidate_checker(scope, monkeypatch,
                                                                          verdict, diagnosis, code):
    root, config = scope
    response = json.loads((root / config.before_response.path).read_bytes())
    response["judgments"][0].update(
        verdict=verdict, diagnosis=diagnosis,
        summary="Invented source contexts do not establish a safe correction.",
    )
    config = change_fixture(root, config, "before_response", response)
    original_record = (root / config.record.path).read_bytes()
    original_check = runner.check_support
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[2].revision_id)
        if len(calls) > 1:
            pytest.fail("Unresolved source conflict must not dispatch candidate checking")
        return original_check(*args, **kwargs)

    monkeypatch.setattr(runner, "check_support", observed)
    report = runner.inspect_record(root, config)
    result = report["result"]
    assert report["new_mock_requests"] == 1 and report["new_live_requests"] == 0
    assert calls == [result["structure"]["revision_id"]]
    assert result["before"]["status"] == "succeeded"
    assert result["before"]["judgments"][0]["diagnosis"] == diagnosis
    assert result["patch_state"] == {"status": "rejected", "code": code}
    assert result["after"] is None and result["patch"] is None
    assert result["final_revision"] == result["structure"]["revision_id"]
    canonical = root / "runs/agent" / RUN_ID / "canonical.json"
    assert json.loads(canonical.read_bytes()) == json.loads(original_record)
    assert (root / config.record.path).read_bytes() == original_record
    replay = runner.inspect_record(root, config)
    assert replay["execution"] == "cached_replay" and replay["new_mock_requests"] == 0
    assert replay["result"] == result and len(calls) == 1
