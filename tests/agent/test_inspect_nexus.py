"""Synthetic runner integration, with an explicitly mocked authentic loader.

The loader test double returns a genuinely parsed synthetic NXDL fixture. Review
artifacts below are invented fixture decisions, not expert judgments or approvals
for production mappings. Authentic resource inspection is recorded separately.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest
from pydantic import ValidationError

from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.agent.nexus_registry import MappingRequest, ReviewRecord, propose_entry
from meta_data_extraction.agent.nexus_resources import NXDLResourceSpec, load_nexus
from meta_data_extraction.agent.nexus_units import QuantityRequest
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.agent.storage import RunConflictError, RunIncompleteError
import meta_data_extraction.agent.inspect_nexus as runner

SPEC_PATH = "configs/agent/nexus-resource.json"
CONFIG_PATH = "configs/agent/nexus-run.json"
REVIEWS_PATH = "runs/agent/synthetic-review-input/reviews.json"
RUN_ID = "nexus-unit"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"))


@pytest.fixture
def scope(workspace, monkeypatch):
    from test_nexus_registry import synthetic_snapshot, temperature_request
    from meta_data_extraction.agent.nexus_resources import NXDLFileSpec

    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "schemas/prompt_template.md", workspace / "schemas/prompt_template.md")
    snapshot = synthetic_snapshot(workspace)
    resource_path = workspace / snapshot.resource.files[0].path
    commit = "a" * 40
    prefix = f"https://raw.githubusercontent.com/nexusformat/definitions/{commit}/"
    # Explicit loader double: no fake authentic resource bytes are manufactured.
    mock_spec = NXDLResourceSpec(
        kind="authentic", commit=commit, release="mocked-authentic-loader-only",
        files=(NXDLFileSpec(path="runs/agent/mock-resource/NXsample.nxdl.xml", sha256="a" * 64,
                            source_url=prefix + "base_classes/NXsample.nxdl.xml"),),
        license_name="LGPL-3.0-or-later", license_url=prefix + "LGPL.txt",
        license_file=NXDLFileSpec(path="runs/agent/mock-resource/LGPL.txt", sha256="b" * 64,
                                  source_url=prefix + "LGPL.txt"),
    )
    write_json(workspace / SPEC_PATH, mock_spec.model_dump(mode="json"))
    config = runner.NeXusRunConfig(
        run_id=RUN_ID, resource_spec=SPEC_PATH, mappings=(temperature_request(),),
        definitions=(runner.DefinitionRequest(class_name="NXsample", field_path="temperature"),),
        quantity_checks=(
            QuantityRequest(raw_expression="Synthetic 25.0 C", value_text="25.0", from_unit="C", to_unit="K",
                            role="acquisition_temperature", unit_category="NX_TEMPERATURE"),
            QuantityRequest(to_unit="K", role="acquisition_temperature", unit_category="NX_TEMPERATURE"),
        ),
    )
    monkeypatch.setattr(runner, "load_nexus", lambda root, requested_spec: snapshot)
    monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic-runtime": "a" * 64})
    return workspace, config, snapshot, resource_path



def amended(config, **changes):
    return runner.NeXusRunConfig.model_validate(config.model_dump() | changes)


def supplied_review(root, config, snapshot, *, decision="approved"):
    entry = propose_entry(load_profile(root), snapshot, config.mappings[0])
    review = ReviewRecord(
        review_id="synthetic-review-not-real-human-approval", entry_id=entry.entry_id,
        entry_sha256=entry.entry_sha256, reviewer="invented fixture reviewer",
        provenance="supplied_human_review", decision=decision,
        reviewed_at="2026-10-07T10:00:00Z", rationale="Synthetic test decision only.",
    )
    path = root / REVIEWS_PATH
    write_json(path, {"reviews": [review.model_dump(mode="json")]})
    return amended(config, reviews_path=REVIEWS_PATH, reviews_sha256=sha256(path.read_bytes())), review


def test_runner_exports_sidecars_without_approval_values_or_canonical(scope):
    root, config, _, _ = scope
    original_schemas = {path: path.read_bytes() for path in (root / "schemas").glob("*.json")}
    report = runner.inspect_nexus(root, config)
    assert report["execution"] == "definition_inspection"
    assert report["model_calls"] == 0
    result = report["result"]
    assert result["approved_guidance"] == []
    assert len(result["inspection_guidance"]) == 1
    assert result["registry"]["entries"][0]["approval_status"] == "unreviewed"
    assert result["canonical_modified"] is False
    assert result["publication_support"] == "not_assessed"
    assert result["nexus_compliance"] == "not_assessed"
    assert result["quantity_checks"][0]["converted_value_text"] == "298.15"
    assert result["quantity_checks"][1]["status"] == "unavailable"
    assert result["quantity_checks"][1]["converted_value_text"] is None
    directory = root / "runs/agent" / RUN_ID
    assert {path.name for path in directory.iterdir()} == {"request.json", "result.json", "sidecar.json"}
    assert all(path.read_bytes() == raw for path, raw in original_schemas.items())
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    replay = runner.inspect_nexus(root, config)
    assert replay["execution"] == "cached_replay"
    assert replay["result"] == result
    assert replay["model_calls"] == 0
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


@pytest.mark.parametrize("decision,approved_count,inspection_count", [
    ("approved", 1, 1), ("rejected", 0, 0), ("needs_changes", 0, 0),
])
def test_explicit_pinned_synthetic_review_controls_guidance(scope, decision, approved_count,
                                                           inspection_count):
    root, config, snapshot, _ = scope
    config, review = supplied_review(root, config, snapshot, decision=decision)
    review_bytes = (root / REVIEWS_PATH).read_bytes()
    report = runner.inspect_nexus(root, config)
    assert len(report["result"]["approved_guidance"]) == approved_count
    assert len(report["result"]["inspection_guidance"]) == inspection_count
    assert report["result"]["registry"]["entries"][0]["review"] == review.model_dump(mode="json")
    assert report["result"]["registry"]["entries"][0]["approval_status"] == decision
    assert (root / REVIEWS_PATH).read_bytes() == review_bytes
    assert runner.inspect_nexus(root, config)["execution"] == "cached_replay"


@pytest.mark.parametrize("changes", [
    {"mode": "live"}, {"unknown": "forbidden"}, {"run_id": "../outputs"},
    {"mappings": []}, {"reviews_path": REVIEWS_PATH}, {"reviews_sha256": "0" * 64},
    {"reviews_sha256": "invalid", "reviews_path": REVIEWS_PATH},
    {"schema_names": [True]},
])
def test_run_config_is_explicit_bounded_and_offline(scope, changes):
    _, config, _, _ = scope
    with pytest.raises(ValueError):
        amended(config, **changes)


def test_duplicate_requests_and_oversized_lists_rejected(scope):
    _, config, _, _ = scope
    for field, values in (
        ("mappings", config.mappings * 2),
        ("definitions", config.definitions * 2),
        ("quantity_checks", config.quantity_checks * 11),
    ):
        with pytest.raises(ValueError):
            amended(config, **{field: values})


def test_strict_configuration_loading_and_input_roots(scope):
    root, config, _, _ = scope
    write_json(root / CONFIG_PATH, config.model_dump(mode="json"))
    assert runner.load_config(root, CONFIG_PATH) == config
    write_json(root / "data/gold/runtime-forbidden.json", config.model_dump(mode="json"))
    with pytest.raises(ValueError):
        runner.load_config(root, "data/gold/runtime-forbidden.json")
    (root / CONFIG_PATH).write_text("run_id: duplicate\nrun_id: duplicate-again\n", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_config(root, CONFIG_PATH)
    (root / CONFIG_PATH).write_bytes(b" " * 1_000_001)
    with pytest.raises(ValueError, match="size limit"):
        runner.load_config(root, CONFIG_PATH)


@pytest.mark.parametrize("field,path", [
    ("resource_spec", "configs/alignment_helper.yaml"),
    ("resource_spec", "data/annotations/nexus.json"),
    ("reviews_path", "data/gold/reviews.json"),
    ("reviews_path", "tests/agent/fixtures/reviews.json"),
    ("reviews_path", "outputs/reviews.json"),
])
def test_forbidden_input_roots_rejected_without_run_outputs(scope, field, path):
    root, config, _, _ = scope
    updates = {field: path}
    if field == "reviews_path":
        updates["reviews_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        runner.inspect_nexus(root, amended(config, **updates))
    assert not (root / "runs/agent" / RUN_ID).exists()


def test_synthetic_configuration_rejected_before_loading_or_claim(scope, monkeypatch):
    root, config, snapshot, _ = scope
    write_json(root / SPEC_PATH, snapshot.resource.model_dump(mode="json"))
    monkeypatch.setattr(runner, "load_nexus", lambda *a: pytest.fail("Synthetic spec must reject before loading"))
    with pytest.raises(ValueError, match="authentic"):
        runner.inspect_nexus(root, config)
    assert not (root / "runs/agent" / RUN_ID).exists()


@pytest.mark.parametrize("change", ["bytes", "stale_entry", "unknown_field", "oversize"])
def test_review_hash_and_structure_are_checked(scope, change):
    root, config, snapshot, _ = scope
    config, _ = supplied_review(root, config, snapshot)
    path = root / REVIEWS_PATH
    if change == "bytes":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        content = json.loads(path.read_bytes())
        if change == "stale_entry":
            content["reviews"][0]["entry_sha256"] = "0" * 64
            content["reviews"][0]["entry_id"] = "nexus-mapping:" + "0" * 64
        elif change == "unknown_field":
            content["reviews"][0]["automatic_approval"] = True
        write_json(path, content)
        if change == "oversize":
            path.write_bytes(b" " * 1_000_001)
        config = amended(config, reviews_sha256=sha256(path.read_bytes()))
    with pytest.raises(ValueError):
        runner.inspect_nexus(root, config)
    assert not (root / "runs/agent" / RUN_ID).exists()


def test_review_bytes_are_read_once_for_hashing_and_parsing(scope, monkeypatch):
    root, config, snapshot, _ = scope
    config, _ = supplied_review(root, config, snapshot)
    path = root / REVIEWS_PATH
    original = Path.read_bytes
    reads = []

    def read_bytes(current):
        if current == path:
            reads.append(current)
        return original(current)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    runner.inspect_nexus(root, config)
    assert reads == [path]


@pytest.mark.parametrize("change", ["implementation", "resource", "dependency", "profile", "quantity", "scope"])
def test_changed_inputs_need_a_fresh_run(scope, monkeypatch, change):
    root, config, _, _ = scope
    runner.inspect_nexus(root, config)
    if change == "implementation":
        monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic-runtime": "b" * 64})
    elif change == "dependency":
        monkeypatch.setattr(runner, "version", lambda name: "synthetic-changed-version")
    elif change == "resource":
        data = json.loads((root / SPEC_PATH).read_bytes())
        data["release"] = "changed-test-resource"
        write_json(root / SPEC_PATH, data)
    elif change == "profile":
        path = root / "schemas/photon_and_neutron.json"
        data = json.loads(path.read_bytes())
        data["samples"]["item_schema"]["sample_name"]["description"] += " Synthetic changed meaning."
        write_json(path, data)
    elif change == "quantity":
        checks = [q.model_dump() for q in config.quantity_checks]
        checks[0].update(value_text="26.0", raw_expression="Synthetic 26.0 C")
        config = amended(config, quantity_checks=checks)
    else:
        mapping = config.mappings[0].model_dump()
        mapping["scope"] += " More explicit synthetic scope."
        config = amended(config, mappings=[mapping])
    with pytest.raises(RunConflictError):
        runner.inspect_nexus(root, config)


@pytest.mark.parametrize("change", ["scope", "profile"])
def test_review_cannot_silently_approve_changed_entry(scope, change):
    root, config, snapshot, _ = scope
    config, _ = supplied_review(root, config, snapshot)
    if change == "scope":
        mapping = config.mappings[0].model_dump()
        mapping["scope"] += " Changed scope after review."
        config = amended(config, mappings=[mapping])
    else:
        path = root / "schemas/photon_and_neutron.json"
        data = json.loads(path.read_bytes())
        data["samples"]["item_schema"]["sample_name"]["description"] += " Changed definition."
        write_json(path, data)
    with pytest.raises(ValueError, match="stale review"):
        runner.inspect_nexus(root, config)
    assert not (root / "runs/agent" / RUN_ID).exists()


def test_profile_changed_during_processing_never_publishes_success(scope, monkeypatch):
    root, config, _, _ = scope
    original = runner.build_registry

    def changed(*args, **kwargs):
        registry = original(*args, **kwargs)
        path = root / "schemas/photon_and_neutron.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return registry

    monkeypatch.setattr(runner, "build_registry", changed)
    with pytest.raises(ValueError):
        runner.inspect_nexus(root, config)
    assert not (root / "runs/agent" / RUN_ID / "result.json").exists()


def test_resource_changed_during_processing_never_publishes_success(scope, monkeypatch):
    root, config, _, resource_path = scope
    original = runner.build_registry

    def changed(*args, **kwargs):
        registry = original(*args, **kwargs)
        resource_path.write_bytes(resource_path.read_bytes() + b"\n")
        return registry

    monkeypatch.setattr(runner, "build_registry", changed)
    with pytest.raises(ValueError):
        runner.inspect_nexus(root, config)
    assert not (root / "runs/agent" / RUN_ID / "result.json").exists()


def test_no_implicit_retry_after_claim_interruption(scope, monkeypatch):
    root, config, _, _ = scope
    original = runner.verify_resources

    def interruption(*args):
        raise RuntimeError("Synthetic interruption after execution claim")

    monkeypatch.setattr(runner, "verify_resources", interruption)
    with pytest.raises(RuntimeError, match="Synthetic interruption"):
        runner.inspect_nexus(root, config)
    monkeypatch.setattr(runner, "verify_resources", original)
    with pytest.raises(RunIncompleteError):
        runner.inspect_nexus(root, config)


def test_output_junction_cannot_redirect_into_released_assets(scope, link_directory):
    root, config, _, _ = scope
    target = root / "outputs"
    target.mkdir()
    sentinel = target / "released.json"
    sentinel.write_bytes(b"released-artifact")
    (root / "runs").mkdir()
    link_directory(root / "runs/agent", target)
    with pytest.raises(ValueError):
        runner.inspect_nexus(root, config)
    assert sentinel.read_bytes() == b"released-artifact"
    assert {p.name for p in target.iterdir()} == {"released.json"}


def rewrite_completion(root, envelope, *, sidecar=None):
    """Generic storage hashes are not authenticity: domain replay must agree."""
    from meta_data_extraction.agent.storage import RunStore, _digest, _json_bytes

    directory = root / "runs/agent" / RUN_ID
    if sidecar is not None:
        raw = _json_bytes(sidecar)
        (directory / "sidecar.json").write_bytes(raw)
        envelope["artifacts"]["sidecar.json"] = {"sha256": _digest(raw), "bytes": len(raw)}
    envelope["completion_sha256"] = _digest(_json_bytes({
        key: value for key, value in envelope.items() if key != "completion_sha256"
    }))
    (directory / "result.json").write_bytes(_json_bytes(envelope))
    request = json.loads((directory / "request.json").read_bytes())["request"]
    assert RunStore.begin(root, RUN_ID, request).completed == envelope["result"]


@pytest.mark.parametrize("key,value", [
    ("version", "forged"), ("snapshot_id", "invented-snapshot"),
    ("profile_id", "invented-profile"), ("model_calls", 1), ("canonical_modified", True),
    ("publication_support", "supported"), ("nexus_compliance", "compliant"),
    ("approved_guidance", [{"invented_approval": True}]), ("definitions", []),
    ("quantity_checks", []), ("extra", "invented-field"),
])
def test_rehashed_report_tampering_rejected_by_domain_replay(scope, key, value):
    root, config, _, _ = scope
    runner.inspect_nexus(root, config)
    envelope = json.loads((root / "runs/agent" / RUN_ID / "result.json").read_bytes())
    envelope["result"][key] = value
    rewrite_completion(root, envelope)
    with pytest.raises(ValueError, match="Cached registry report"):
        runner.inspect_nexus(root, config)


@pytest.mark.parametrize("key", ["profile", "resource", "registry", "definitions", "quantity_checks", "extra"])
def test_rehashed_sidecar_tampering_rejected_by_domain_replay(scope, key):
    root, config, _, _ = scope
    runner.inspect_nexus(root, config)
    directory = root / "runs/agent" / RUN_ID
    envelope = json.loads((directory / "result.json").read_bytes())
    sidecar = json.loads((directory / "sidecar.json").read_bytes())
    sidecar[key] = {"invented": "content"}
    rewrite_completion(root, envelope, sidecar=sidecar)
    with pytest.raises(ValueError, match="Cached registry report"):
        runner.inspect_nexus(root, config)


def test_implementation_fingerprints_only_explicit_runtime_files(workspace):
    repository = Path(__file__).resolve().parents[2]
    baseline = runner._implementation(repository)
    for relative in baseline:
        destination = workspace / "src/meta_data_extraction" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / "src/meta_data_extraction" / relative, destination)
    assert runner._implementation(workspace) == baseline
    (workspace / "src/meta_data_extraction/unrelated.py").write_text("# unindexed\n", encoding="utf-8")
    assert runner._implementation(workspace) == baseline
    path = workspace / "src/meta_data_extraction/agent/nexus_units.py"
    path.write_bytes(path.read_bytes() + b"\n# synthetic change\n")
    assert runner._implementation(workspace) != baseline


def test_cli_uses_safe_generic_errors(scope, monkeypatch, capsys):
    root, _, _, _ = scope
    monkeypatch.setattr(sys, "argv", ["inspect_nexus", "--root", str(root), "--config", CONFIG_PATH])

    def invalid(*args):
        raise ValueError("synthetic-private-value-must-not-appear")

    monkeypatch.setattr(runner, "load_config", invalid)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "ValueError" in output.err
    assert "synthetic-private-value" not in output.err
    assert output.out == ""


def test_cli_summary_labels_synthetic_run_without_model_calls(scope, monkeypatch, capsys):
    root, config, _, _ = scope
    write_json(root / CONFIG_PATH, config.model_dump(mode="json"))
    monkeypatch.setattr(sys, "argv", ["inspect_nexus", "--root", str(root), "--config", CONFIG_PATH])
    assert runner.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["approved_guidance_count"] == 0
    assert summary["inspection_guidance_count"] == 1
    assert summary["definition_count"] == 1
    assert summary["model_calls"] == 0
    assert summary["canonical_modified"] is False
    assert summary["quantity_statuses"] == ["converted", "unavailable"]


@pytest.mark.parametrize("oversized", ["result", "sidecar"])
def test_artifact_bounds_fail_before_any_execution_claim(scope, monkeypatch, oversized):
    root, config, _, _ = scope
    baseline = runner.inspect_nexus(root, config, run_id="synthetic-size-baseline")
    result_bytes = json.dumps(baseline["result"], ensure_ascii=False, sort_keys=True,
                              allow_nan=False, separators=(",", ":")).encode("utf-8")
    sidecar = root / "runs/agent/synthetic-size-baseline/sidecar.json"
    result_size = len(result_bytes) + 1
    sidecar_size = len(sidecar.read_bytes())
    assert sidecar_size > result_size
    cap = result_size - 1 if oversized == "result" else sidecar_size - 1
    monkeypatch.setattr(runner, "MAX_ARTIFACT_BYTES", cap)
    monkeypatch.setattr(runner.RunStore, "begin", lambda *a, **kw: pytest.fail(
        "Oversized artifacts must fail before claiming a run"))
    with pytest.raises(ValueError, match="artifact exceeds the bounded projection size"):
        runner.inspect_nexus(root, config)
    assert not (root / "runs/agent" / RUN_ID).exists()


def test_sidecar_larger_than_configuration_limit_can_replay_with_artifact_bound(scope):
    root, config, _, _ = scope
    prompt = root / "schemas/prompt_template.md"
    # Test-only prompt padding enlarges the retained profile, without pretending
    # that a generated standards definition or model output was observed.
    prompt.write_bytes(prompt.read_bytes() + b"\n" + b"Synthetic profile padding. " * 40_000)
    first = runner.inspect_nexus(root, config)
    relative = f"runs/agent/{RUN_ID}/sidecar.json"
    raw = (root / relative).read_bytes()
    assert 1_000_000 < len(raw) < runner.MAX_ARTIFACT_BYTES
    # Demonstrate why using the default configuration bound for replay was wrong.
    with pytest.raises(ValueError, match="size limit"):
        runner._mapping(root, relative, roots=(f"runs/agent/{RUN_ID}",))
    replay = runner.inspect_nexus(root, config)
    assert replay["execution"] == "cached_replay"
    assert replay["result"] == first["result"]
    assert (root / relative).read_bytes() == raw
