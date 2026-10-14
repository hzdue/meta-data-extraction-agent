"""Offline runner tests with a deliberately mocked authentic-resource loader.

The mock returns a real parsed *synthetic* ontology. These tests establish runner
control flow, not authentic PaNET content, expert mapping judgments, or quality.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest
from pydantic import ValidationError

from extraction_fixtures import synthetic_record
from test_panet_resources import resource as synthetic_resource
from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.agent.extraction import extract_fixed
from meta_data_extraction.agent.panet_resources import ResourceError, ResourceSpec, load_panet
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.agent.providers import MockProvider, ProviderResult
from meta_data_extraction.agent.sources import prepare_sources
from meta_data_extraction.agent.storage import RunConflictError, RunIncompleteError
import meta_data_extraction.agent.inspect_panet as runner


SOURCE = "tests/agent/fixtures/mention.md"
TEXT = "In this synthetic study, test diffraction was used on sample A.\n"
EXPRESSION = "test diffraction"
SPEC_PATH = "configs/agent/panet-resource.json"
CONFIG_PATH = "configs/agent/panet-run.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def scope(workspace, monkeypatch):
    spec, resource_path = synthetic_resource(workspace)
    snapshot = load_panet(workspace, spec)
    (workspace / SOURCE).write_bytes(TEXT.encode("utf-8"))
    # Explicit loader test double: no forged authentic ontology bytes are created.
    mock_spec = ResourceSpec(
        kind="authentic", path="runs/agent/unit-test-placeholder/PaNET.owl", sha256="a" * 64,
        source_url="https://raw.githubusercontent.com/pan-ontologies/PaNET/" + "a" * 40 + "/PaNET.owl",
        commit="a" * 40, release="mocked-authentic-loader-only",
        namespace="http://purl.org/pan-science/PaNET/", license_url="https://creativecommons.org/licenses/by/4.0/",
        license_name="CC-BY-4.0",
    )
    write_json(workspace / SPEC_PATH, mock_spec.model_dump(mode="json"))
    start = TEXT.index(EXPRESSION)
    config = runner.PaNETRunConfig(
        run_id="panet-unit", resource_spec=SPEC_PATH, source=SOURCE,
        source_sha256=sha256(TEXT.encode()), families=("urn:synthetic:parent",),
        mentions=(runner.MentionRequest(original_expression=EXPRESSION, start=start, end=start + len(EXPRESSION),
            context_start=0, context_end=len(TEXT.rstrip()), context_use="current_study"),),
    )
    monkeypatch.setattr(runner, "load_panet", lambda root, requested_spec: snapshot)
    monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic-unit-implementation": "a" * 64})
    return workspace, config, snapshot, resource_path


def amended(config, **changes):
    return runner.PaNETRunConfig.model_validate({**config.model_dump(), **changes})


def test_run_exports_only_mapping_sidecars_and_replay_preserves_bytes(scope):
    root, config, _, _ = scope
    report = runner.inspect_panet(root, config)
    assert report["execution"] == "lexical_lookup" and report["model_calls"] == 0
    output = root / "runs/agent/panet-unit"
    assert not (output / "canonical.json").exists()
    assert not (output / "raw-response.json").exists()
    assert (output / "sidecar.json").exists()
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    replay = runner.inspect_panet(root, config)
    assert replay["execution"] == "cached_replay" and replay["model_calls"] == 0
    assert replay["result"] == report["result"]
    assert before == {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert len(report["result"]["decisions"]) == 1


@pytest.mark.parametrize("changes", [
    {"mode": "live"}, {"unexpected": "forbidden"}, {"candidate_limit": True},
    {"candidate_limit": 0}, {"run_id": "../unsafe"}, {"families": []},
    {"families": ["one", "two", "three"]}, {"families": ["same", "same"]},
    {"source_sha256": "unknown"}, {"record_sidecar": "runs/agent/x/sidecar.json"},
    {"record_sidecar_sha256": "0" * 64},
])
def test_config_is_bounded_explicit_offline_and_strict(scope, changes):
    _, config, _, _ = scope
    with pytest.raises(ValueError):
        amended(config, **changes)


def test_config_loader_reads_only_explicit_agent_configuration(scope):
    root, config, _, _ = scope
    write_json(root / CONFIG_PATH, config.model_dump(mode="json"))
    assert runner.load_config(root, CONFIG_PATH) == config
    write_json(root / "data/gold/forbidden.json", config.model_dump(mode="json"))
    with pytest.raises(ValueError):
        runner.load_config(root, "data/gold/forbidden.json")


@pytest.mark.parametrize("field,value", [
    ("source", "data/gold/forbidden.md"),
    ("resource_spec", "configs/alignment_helper.yaml"),
    ("record_sidecar", "data/gold/forbidden.json"),
])
def test_offline_only_inputs_cannot_enter_runtime(scope, field, value):
    root, config, _, _ = scope
    extra = {field: value}
    if field == "record_sidecar":
        extra["record_sidecar_sha256"] = "0" * 64
        extra["mentions"] = ({**config.mentions[0].model_dump(),
                              "assertion_path": "/datasets/0/measurement_technique"},)
    with pytest.raises(ValueError):
        runner.inspect_panet(root, amended(config, **extra))
    assert not (root / "runs").exists()


def test_source_hash_mismatch_rejected_before_claim(scope):
    root, config, _, _ = scope
    with pytest.raises(ValueError, match="Source content"):
        runner.inspect_panet(root, amended(config, source_sha256="0" * 64))
    assert not (root / "runs").exists()


@pytest.mark.parametrize("changes", [
    {"original_expression": "other expression"}, {"end": 1000, "context_end": 1001},
    {"context_start": 26}, {"start": -1}, {"context_end": 0},
])
def test_invalid_source_spans_and_quotations_fail(scope, changes):
    root, config, _, _ = scope
    mention = {**config.mentions[0].model_dump(), **changes}
    with pytest.raises(ValueError):
        changed = amended(config, mentions=(mention,))
        runner.inspect_panet(root, changed)


def test_context_cannot_cross_actual_source_regions(scope):
    root, config, _, _ = scope
    text = TEXT + "\nA separate synthetic paragraph.\n"
    (root / SOURCE).write_bytes(text.encode("utf-8"))
    mention = {**config.mentions[0].model_dump(), "context_end": len(text.rstrip())}
    with pytest.raises(ValueError, match="actual source region"):
        runner.inspect_panet(root, amended(config, source_sha256=sha256(text.encode()), mentions=(mention,)))


@pytest.mark.parametrize("change", ["implementation", "resource", "context", "families", "limit", "source", "dependency"])
def test_cache_identity_changes_require_a_fresh_run(scope, monkeypatch, change):
    root, config, _, _ = scope
    runner.inspect_panet(root, config)
    if change == "implementation":
        monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic-unit-implementation": "b" * 64})
    elif change == "resource":
        data = json.loads((root / SPEC_PATH).read_text(encoding="utf-8"))
        data["release"] = "changed-mocked-resource-release"
        write_json(root / SPEC_PATH, data)
    elif change == "context":
        mention = {**config.mentions[0].model_dump(), "context_use": "related_work"}
        config = amended(config, mentions=(mention,))
    elif change == "families":
        config = amended(config, families=("urn:synthetic:parent", "urn:synthetic:child"))
    elif change == "limit":
        config = amended(config, candidate_limit=4)
    elif change == "source":
        text = TEXT + "Additional synthetic sentence.\n"
        (root / SOURCE).write_bytes(text.encode("utf-8"))
        config = amended(config, source_sha256=sha256(text.encode()))
    else:
        monkeypatch.setattr(runner, "version", lambda package: "synthetic-next-dependency")
    with pytest.raises(RunConflictError):
        runner.inspect_panet(root, config)


def test_changed_resource_after_processing_never_publishes_success(scope, monkeypatch):
    root, config, _, resource_path = scope
    original = runner.propose_mapping

    def changes_resource(*args, **kwargs):
        result = original(*args, **kwargs)
        resource_path.write_bytes(resource_path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(runner, "propose_mapping", changes_resource)
    with pytest.raises(ResourceError, match="SHA-256"):
        runner.inspect_panet(root, config)
    assert not (root / "runs/agent/panet-unit/result.json").exists()
    monkeypatch.setattr(runner, "propose_mapping", original)
    with pytest.raises(RunIncompleteError):
        runner.inspect_panet(root, config)


def test_interrupted_claim_cannot_implicitly_restart(scope, monkeypatch):
    root, config, _, _ = scope
    original = runner.propose_mapping

    def interrupted(*args, **kwargs):
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(runner, "propose_mapping", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.inspect_panet(root, config)
    monkeypatch.setattr(runner, "propose_mapping", original)
    with pytest.raises(RunIncompleteError):
        runner.inspect_panet(root, config)


def test_synthetic_resource_config_rejected_before_loader_or_claim(scope, monkeypatch):
    root, config, snapshot, _ = scope
    write_json(root / SPEC_PATH, snapshot.resource.model_dump(mode="json"))
    monkeypatch.setattr(runner, "load_panet", lambda *a: pytest.fail("Synthetic config must reject before loader"))
    with pytest.raises(ValueError, match="authentic"):
        runner.inspect_panet(root, config)
    assert not (root / "runs").exists()


def test_cli_synthetic_rejection_is_sanitized(scope, monkeypatch, capsys):
    root, config, snapshot, _ = scope
    write_json(root / SPEC_PATH, snapshot.resource.model_dump(mode="json"))
    write_json(root / CONFIG_PATH, config.model_dump(mode="json"))
    monkeypatch.setattr(sys, "argv", ["inspect-panet", "--root", str(root), "--config", CONFIG_PATH])
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "ValueError" in captured.err
    assert TEXT.strip() not in captured.err


def record_sidecar(root, *, other_source=False):
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "schemas/prompt_template.md", root / "schemas/prompt_template.md")
    source = SOURCE
    if other_source:
        source = "tests/agent/fixtures/different-mention.md"
        (root / source).write_bytes(TEXT.encode("utf-8"))
    bundle = prepare_sources(root, [source])[0].bundle
    profile = load_profile(root)
    record = synthetic_record(profile)
    record["datasets"][0]["measurement_technique"] = EXPRESSION
    extraction = extract_fixed(root, bundle, profile,
        MockProvider([ProviderResult(status="ok", content=json.dumps(record), finish_reason="stop")]), variant="inherited")
    path = root / "tests/agent/fixtures/extraction-sidecar.json"
    write_json(path, {"source": bundle.model_dump(mode="json"), "extraction": extraction.model_dump(mode="json")})
    return path, extraction


def test_assertion_bound_sidecar_is_hash_and_source_bound_without_mutating_record(scope):
    root, config, _, _ = scope
    path, extraction = record_sidecar(root)
    original = path.read_bytes()
    mention = {**config.mentions[0].model_dump(), "assertion_path": "/datasets/0/measurement_technique"}
    config = amended(config, record_sidecar=path.relative_to(root).as_posix(), record_sidecar_sha256=sha256(original), mentions=(mention,))
    first = runner.inspect_panet(root, config)
    replay = runner.inspect_panet(root, config)
    assert first["result"] == replay["result"]
    assert path.read_bytes() == original
    sidecar = json.loads((root / "runs/agent/panet-unit/sidecar.json").read_text(encoding="utf-8"))
    assert sidecar["record"]["revision_id"] == extraction.snapshot.revision_id
    assert sidecar["record"]["canonical_json"] == extraction.snapshot.canonical_json


@pytest.mark.parametrize("change", ["hash", "source", "assertion"])
def test_record_sidecar_tampering_and_wrong_source_or_assertion_fail(scope, change):
    root, config, _, _ = scope
    path, _ = record_sidecar(root, other_source=change == "source")
    mention = {**config.mentions[0].model_dump(), "assertion_path": "/datasets/0/measurement_technique"}
    if change == "assertion":
        mention["assertion_path"] = "/samples/0/chemical_composition"
    config = amended(config, record_sidecar=path.relative_to(root).as_posix(),
                     record_sidecar_sha256="0" * 64 if change == "hash" else sha256(path.read_bytes()), mentions=(mention,))
    with pytest.raises(ValueError):
        runner.inspect_panet(root, config)
    assert not (root / "runs/agent/panet-unit/result.json").exists()


def test_assertion_path_requires_explicit_record(scope):
    _, config, _, _ = scope
    mention = {**config.mentions[0].model_dump(), "assertion_path": "/datasets/0/measurement_technique"}
    with pytest.raises(ValidationError, match="record"):
        amended(config, mentions=(mention,))


def test_record_bound_run_requires_assertion_path_for_every_mention(scope):
    _, config, _, _ = scope
    with pytest.raises(ValidationError, match="Each mention"):
        amended(config, record_sidecar="tests/agent/fixtures/record.json", record_sidecar_sha256="0" * 64)


def test_input_record_sidecar_is_read_once_for_hashing_and_parsing(scope, monkeypatch):
    root, config, _, _ = scope
    path, _ = record_sidecar(root)
    config = amended(config, record_sidecar=path.relative_to(root).as_posix(),
                     record_sidecar_sha256=sha256(path.read_bytes()),
                     mentions=({**config.mentions[0].model_dump(),
                                "assertion_path": "/datasets/0/measurement_technique"},))
    original_bytes = Path.read_bytes
    original_text = Path.read_text
    reads = []

    def read_bytes(current):
        if current == path:
            reads.append(current)
        return original_bytes(current)

    def read_text(current, *args, **kwargs):
        if current == path:
            pytest.fail("Parsing must use the same bytes that supplied the input hash")
        return original_text(current, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)
    runner.inspect_panet(root, config)
    assert reads == [path]


def rewrite_completion(root, envelope, *, sidecar=None):
    """Simulate an imported tampered artifact with recomputed integrity hashes.

    Storage checksums are not authenticity signatures. Domain replay must still
    compare the entire report and sidecar with its frozen inputs.
    """
    from meta_data_extraction.agent.storage import RunStore, _digest, _json_bytes

    directory = root / "runs/agent/panet-unit"
    if sidecar is not None:
        raw = _json_bytes(sidecar)
        (directory / "sidecar.json").write_bytes(raw)
        envelope["artifacts"]["sidecar.json"] = {"sha256": _digest(raw), "bytes": len(raw)}
    envelope["completion_sha256"] = _digest(_json_bytes({
        key: value for key, value in envelope.items() if key != "completion_sha256"
    }))
    (directory / "result.json").write_bytes(_json_bytes(envelope))
    request = json.loads((directory / "request.json").read_bytes())["request"]
    # Explicitly establish that generic storage integrity accepts these hashes;
    # the following test must therefore exercise the runner's semantic checks.
    assert RunStore.begin(root, "panet-unit", request).completed == envelope["result"]


@pytest.mark.parametrize("key,value", [
    ("version", "m04-forged-version"), ("snapshot_id", "panet:" + "0" * 64),
    ("resource_kind", "forged"), ("model_calls", 1), ("source_incomplete", True),
    ("invented_approval", True),
])
def test_replay_rejects_forged_whole_report_metadata_after_rehashed_storage(scope, key, value):
    root, config, _, _ = scope
    runner.inspect_panet(root, config)
    envelope = json.loads((root / "runs/agent/panet-unit/result.json").read_bytes())
    envelope["result"][key] = value
    rewrite_completion(root, envelope)
    with pytest.raises(ValueError, match="Cached report metadata or sidecar"):
        runner.inspect_panet(root, config)


@pytest.mark.parametrize("change", ["source", "resource", "record", "decisions", "additional_field"])
def test_replay_rejects_forged_sidecar_after_rehashed_storage(scope, change):
    root, config, _, _ = scope
    runner.inspect_panet(root, config)
    directory = root / "runs/agent/panet-unit"
    envelope = json.loads((directory / "result.json").read_bytes())
    sidecar = json.loads((directory / "sidecar.json").read_bytes())
    if change == "source":
        sidecar["source"]["text"] = "Invented publication evidence."
    elif change == "resource":
        sidecar["resource"]["concepts"][0]["label"] = "Invented vocabulary label"
    elif change == "record":
        sidecar["record"] = {"invented": "record"}
    elif change == "decisions":
        sidecar["decisions"] = []
    else:
        sidecar["invented_approval"] = True
    rewrite_completion(root, envelope, sidecar=sidecar)
    with pytest.raises(ValueError, match="Cached report metadata or sidecar"):
        runner.inspect_panet(root, config)


def test_implementation_hashing_uses_only_explicit_files(workspace):
    repository = Path(__file__).resolve().parents[2]
    original = runner._implementation(repository)
    for name in original:
        # Initial M04 runner uses module basenames; support explicit package-relative
        # names when the implementation also pins inherited parser/config helpers.
        relative = name if "/" in name or name.endswith(".py") else f"agent/{name}.py"
        target = workspace / "src/meta_data_extraction" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / "src/meta_data_extraction" / relative, target)
    assert runner._implementation(workspace) == original
    unused = workspace / "src/meta_data_extraction/unused.py"
    unused.write_text("# This is not a source document or runtime import.\n", encoding="utf-8")
    assert runner._implementation(workspace) == original
    target = workspace / "src/meta_data_extraction/agent/panet_resources.py"
    target.write_bytes(target.read_bytes() + b"\n# synthetic implementation change\n")
    assert runner._implementation(workspace) != original
