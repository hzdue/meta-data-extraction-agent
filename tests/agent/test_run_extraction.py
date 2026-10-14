"""Offline fixed-run integration and replay; all documents/responses are invented."""
import json
from pathlib import Path
import shutil

import pytest

from extraction_fixtures import SYNTHETIC_TEXT, evidence_response, synthetic_record
from meta_data_extraction.agent.contracts import FixedExtraction
from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.agent.extraction import render_prompt
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.agent.providers import LiveRunSpec, LiveSource, MockProvider, ProviderResult
import meta_data_extraction.agent.run_extraction as runner
from meta_data_extraction.agent.sources import prepare_sources
from meta_data_extraction.agent.storage import RunConflictError, RunIncompleteError, RunStore


SOURCE = "tests/agent/fixtures/m02-runner.md"


@pytest.fixture
def runner_workspace(workspace, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "schemas/prompt_template.md", workspace / "schemas/prompt_template.md")
    (workspace / SOURCE).write_text(SYNTHETIC_TEXT, encoding="utf-8")
    # Unit integrations use explicit synthetic implementation provenance. The
    # separate test below verifies real fixed-list code hashing without inference.
    monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic.py": "a" * 64})
    return workspace


@pytest.fixture
def raw_response(runner_workspace):
    profile = load_profile(runner_workspace)
    bundle = prepare_sources(runner_workspace, [SOURCE])[0].bundle
    return json.dumps(evidence_response(synthetic_record(profile), bundle), indent=2)


def mock_for(raw, *, model_name="mock"):
    return MockProvider([ProviderResult(status="ok", content=raw, finish_reason="stop")], model_name=model_name)


def run(workspace, mock, **overrides):
    arguments = {"run_id": "fixed", "source": SOURCE, "mock": mock, "retain_raw": True}
    arguments.update(overrides)
    return runner.run_fixed(workspace, **arguments)


def live_scope(workspace):
    bundle = prepare_sources(workspace, [SOURCE])[0].bundle
    profile = load_profile(workspace)
    return LiveRunSpec(
        endpoint="https://configured-chatai.example.test/v1",
        sources=(LiveSource(path=SOURCE, sha256=bundle.source_sha256),), split="synthetic-test-only",
        profile_sha256=profile.profile_id.removeprefix("profile:"),
        approved_prompt_sha256=(sha256(render_prompt(bundle, profile, "evidence").encode("utf-8")),),
        max_calls=1, max_output_tokens=512, max_total_output_tokens=512,
    )


def test_completed_run_replays_without_consuming_another_mock_response(runner_workspace, raw_response):
    mock = mock_for(raw_response)
    first = run(runner_workspace, mock)
    assert first.execution == "mock" and first.new_live_requests == 0
    assert first.extraction.status == "succeeded"
    assert first.extraction.raw_output == raw_response
    replay = run(runner_workspace, mock)
    assert replay.execution == "cached_replay" and replay.new_live_requests == 0
    assert replay.extraction.raw_output is None
    assert replay.extraction.model_dump() == first.extraction.model_dump()
    assert len(mock.requests) == 1
    output = runner_workspace / "runs/agent/fixed"
    canonical = json.loads((output / "canonical.json").read_text(encoding="utf-8"))
    assert canonical == first.extraction.snapshot.canonical
    assert "assertions" not in canonical and "revision_id" not in canonical
    sidecar = json.loads((output / "sidecar.json").read_text(encoding="utf-8"))
    assert sidecar["extraction"]["assertions"]
    assert sidecar["source"]["source_id"] == SOURCE
    assert json.loads((output / "result.json").read_text(encoding="utf-8"))["result"]["cost_status"] == "not_applicable"


@pytest.mark.parametrize("change", ["response", "model"])
def test_changed_mock_configuration_conflicts_before_dispatch(runner_workspace, raw_response, change):
    run(runner_workspace, mock_for(raw_response))
    next_mock = mock_for(raw_response + "\n" if change == "response" else raw_response,
                         model_name="changed" if change == "model" else "mock")
    with pytest.raises(RunConflictError):
        run(runner_workspace, next_mock)
    assert next_mock.requests == ()


@pytest.mark.parametrize("change", ["source", "profile", "prompt_version", "implementation", "dependency"])
def test_changed_provenance_conflicts_before_dispatch(runner_workspace, raw_response, monkeypatch, change):
    mock = mock_for(raw_response)
    run(runner_workspace, mock)
    if change == "source":
        with (runner_workspace / SOURCE).open("a", encoding="utf-8") as stream:
            stream.write("Changed synthetic source.\n")
    elif change == "profile":
        with (runner_workspace / "schemas/prompt_template.md").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif change == "prompt_version":
        monkeypatch.setattr(runner, "EVIDENCE_PROMPT_VERSION", "synthetic-next-version")
    elif change == "implementation":
        monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic.py": "b" * 64})
    else:
        monkeypatch.setattr(runner, "version", lambda name: "synthetic-new-dependency-version")
    with pytest.raises(RunConflictError):
        run(runner_workspace, mock)
    assert len(mock.requests) == 1


def test_retention_policy_is_explicit_and_part_of_request(runner_workspace, raw_response):
    mock = mock_for(raw_response)
    run(runner_workspace, mock, retain_raw=False)
    output = runner_workspace / "runs/agent/fixed"
    assert not (output / "raw-response.json").exists()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))["result"]
    assert result["raw_retention"] == "discarded"
    assert "raw_output" not in result["extraction"]
    assert "raw_output" not in json.loads((output / "sidecar.json").read_text(encoding="utf-8"))["extraction"]
    with pytest.raises(RunConflictError):
        run(runner_workspace, mock, retain_raw=True)
    assert len(mock.requests) == 1


def test_parse_failure_exports_diagnostics_without_canonical_and_replays(runner_workspace):
    mock = mock_for("This is invalid JSON")
    report = run(runner_workspace, mock)
    assert report.extraction.status == "failed" and report.extraction.parse_status == "failed"
    output = runner_workspace / "runs/agent/fixed"
    assert not (output / "canonical.json").exists()
    assert (output / "sidecar.json").exists() and (output / "raw-response.json").exists()
    replay = run(runner_workspace, mock)
    assert replay.execution == "cached_replay" and replay.extraction.status == "failed"
    assert len(mock.requests) == 1


def test_claim_interruption_prevents_any_automatic_inference(runner_workspace, raw_response, monkeypatch):
    mock = mock_for(raw_response)
    original = runner.extract_fixed

    def interrupted(*args, **kwargs):
        raise RuntimeError("synthetic interruption before dispatch")

    monkeypatch.setattr(runner, "extract_fixed", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run(runner_workspace, mock)
    monkeypatch.setattr(runner, "extract_fixed", original)
    with pytest.raises(RunIncompleteError):
        run(runner_workspace, mock)
    assert mock.requests == ()


def test_export_interruption_cannot_repeat_successful_inference(runner_workspace, raw_response, monkeypatch):
    mock = mock_for(raw_response)

    def interrupted(*args, **kwargs):
        raise OSError("synthetic export interruption")

    monkeypatch.setattr(RunStore, "complete", interrupted)
    with pytest.raises(OSError, match="synthetic export interruption"):
        run(runner_workspace, mock)
    assert len(mock.requests) == 1
    with pytest.raises(RunIncompleteError):
        run(runner_workspace, mock)
    assert len(mock.requests) == 1


@pytest.mark.parametrize("endpoint", [None, "https://different.example.test/v1"])
def test_missing_or_mismatched_configured_endpoint_rejects_before_factory(runner_workspace, monkeypatch, endpoint):
    monkeypatch.setattr(runner, "create_live_provider", lambda *a, **k: pytest.fail("No live provider allowed"))
    spec = live_scope(runner_workspace)
    with pytest.raises(ValueError, match="configured ChatAI endpoint"):
        run(runner_workspace, None, live_spec=spec, configured_chatai_endpoint=endpoint)
    assert not (runner_workspace / "runs").exists()


@pytest.mark.parametrize("change", ["profile_sha256", "approved_prompt_sha256", "sources"])
def test_live_scope_rejects_unmatched_resources_before_factory(runner_workspace, monkeypatch, change):
    monkeypatch.setattr(runner, "create_live_provider", lambda *a, **k: pytest.fail("No live provider allowed"))
    spec = live_scope(runner_workspace)
    values = {
        "profile_sha256": "0" * 64,
        "approved_prompt_sha256": ("0" * 64,),
        "sources": (LiveSource(path=SOURCE, sha256="0" * 64),),
    }
    spec = spec.model_copy(update={change: values[change]})
    with pytest.raises(ValueError, match="exactly its source, profile and prompt"):
        run(runner_workspace, None, live_spec=spec, configured_chatai_endpoint=spec.endpoint)
    assert not (runner_workspace / "runs").exists()


def test_changing_mock_to_live_never_constructs_provider_on_conflicting_run(runner_workspace, raw_response, monkeypatch):
    run(runner_workspace, mock_for(raw_response))
    monkeypatch.setattr(runner, "create_live_provider", lambda *a, **k: pytest.fail("No live provider allowed"))
    spec = live_scope(runner_workspace)
    with pytest.raises(RunConflictError):
        run(runner_workspace, None, live_spec=spec, configured_chatai_endpoint=spec.endpoint)


def test_verified_replay_resolves_each_evidence_link_again(runner_workspace, raw_response, monkeypatch):
    mock = mock_for(raw_response)
    first = run(runner_workspace, mock)
    original = runner.resolve_evidence
    calls = []

    def observed(bundle, ref):
        calls.append(ref)
        return original(bundle, ref)

    monkeypatch.setattr(runner, "resolve_evidence", observed)
    run(runner_workspace, mock)
    assert len(calls) == sum(len(item.evidence) for item in first.extraction.assertions)
    assert calls and len(mock.requests) == 1


def test_replay_rejects_location_forgery_even_with_valid_storage_checksums(runner_workspace, raw_response):
    # Simulate an imported/manually modified artifact with recomputed checksums:
    # disk integrity alone cannot establish a quotation's location.
    from meta_data_extraction.agent.storage import _digest, _json_bytes

    mock = mock_for(raw_response)
    run(runner_workspace, mock)
    path = runner_workspace / "runs/agent/fixed/result.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    extraction = envelope["result"]["extraction"]
    evidence = extraction["assertions"][0]["evidence"][0]
    evidence["quotation"] = "X" + evidence["quotation"][1:]
    # Record/value bindings remain valid; only resolution against actual text fails.
    FixedExtraction.model_validate(extraction)
    envelope["completion_sha256"] = _digest(_json_bytes({
        key: value for key, value in envelope.items() if key != "completion_sha256"
    }))
    path.write_bytes(_json_bytes(envelope))
    with pytest.raises(RunConflictError, match="no longer validates"):
        run(runner_workspace, mock)
    assert len(mock.requests) == 1


def test_runner_requires_one_provider_and_boolean_retention(runner_workspace, raw_response):
    with pytest.raises(ValueError, match="exactly one"):
        run(runner_workspace, None)
    with pytest.raises(ValueError, match="exactly one"):
        run(runner_workspace, mock_for(raw_response), live_spec=live_scope(runner_workspace))
    with pytest.raises(TypeError, match="retention"):
        run(runner_workspace, mock_for(raw_response), retain_raw="yes")
    assert not (runner_workspace / "runs").exists()


def test_implementation_hashes_only_explicit_code_resources(workspace):
    repository = Path(__file__).resolve().parents[2]
    expected = runner._implementation(repository)
    for relative in expected:
        target = workspace / "src/meta_data_extraction" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / "src/meta_data_extraction" / relative, target)
    assert runner._implementation(workspace) == expected
    unused = workspace / "src/meta_data_extraction/not_runtime_evidence.py"
    unused.write_text("raise RuntimeError('not imported or indexed')", encoding="utf-8")
    assert runner._implementation(workspace) == expected
    changed = workspace / "src/meta_data_extraction/agent/contracts.py"
    with changed.open("a", encoding="utf-8") as stream:
        stream.write("\n# synthetic version-change test\n")
    actual = runner._implementation(workspace)
    assert actual["agent/contracts.py"] != expected["agent/contracts.py"]
    assert {key: value for key, value in actual.items() if key != "agent/contracts.py"} == {
        key: value for key, value in expected.items() if key != "agent/contracts.py"
    }
