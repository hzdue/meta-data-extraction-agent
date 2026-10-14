"""Source-first packet IO uses invented documents, never natural labels or inference."""
from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from meta_data_extraction.agent.storage import RunIncompleteError
from meta_data_extraction.agent.contracts import at_pointer, digest
from meta_data_extraction.agent.evidence import evidence_ref
from meta_data_extraction.offline.manifest import ArtifactRef
from meta_data_extraction.offline.reference_review import (
    ReferencePacketSpec, load_reference_packet, main, prepare_reference_packet,
    reference_excerpt, submit_reference,
)
from meta_data_extraction.offline.reference_types import ReferenceSubmission, assess_reference, blank_submission
from meta_data_extraction.utils import build_skeleton


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_RELATIVE = "tests/agent/fixtures/invented-review-source.md"
INVENTED_SOURCE = (
    "# Invented source for an IO test\n\n"
    "Two distinct CaCO3 specimens, Sample A and Sample B, were prepared.\n\n"
    "Dataset A measured Sample A at 300 K. Sample B was not measured.\n"
)


@pytest.fixture
def review_workspace(tmp_path):
    # Freeze actual implementation/schema bytes, but all source and review data
    # below are newly invented. The copy is only a temporary test workspace.
    for name in ("pyproject.toml", "uv.lock", "schemas/prompt_template.md",
                 "schemas/generic.json", "schemas/photon_and_neutron.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / name, target)
    for original in (REPOSITORY / "src/meta_data_extraction").rglob("*.py"):
        target = tmp_path / original.relative_to(REPOSITORY)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    source = tmp_path / SOURCE_RELATIVE
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(INVENTED_SOURCE, encoding="utf-8")
    return tmp_path


def pin(root, relative):
    raw = (root / relative).read_bytes()
    return ArtifactRef(path=relative, sha256=hashlib.sha256(raw).hexdigest())


def scope(root, **updates):
    values = dict(run_id="source-review", publication_id="invented-review",
                  split="calibration", data_kind="synthetic",
                  source=pin(root, SOURCE_RELATIVE))
    return ReferencePacketSpec(**(values | updates))


def authored(root, value, name="submission.json"):
    target = root / "tests/offline/fixtures" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return pin(root, target.relative_to(root).as_posix())


def run_files(root):
    return {path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (root / "runs/agent").rglob("*") if path.is_file()}


def test_packet_is_source_only_blank_and_hash_bound(review_workspace, monkeypatch):
    root = review_workspace
    original_open = builtins.open
    original_path_open = Path.open
    forbidden_reads = []

    def inspect(file, mode):
        if isinstance(file, (str, Path)) and "r" in mode:
            normalized = str(file).replace("\\", "/").casefold()
            if (normalized.endswith("/.env") or normalized.endswith("/alignment_helper.yaml")
                    or any(part in normalized for part in (
                        "/data/gold/", "/data/annotations/", "/data/experiment/",
                        "/outputs/", "/predictions/"))):
                forbidden_reads.append(normalized)
                raise AssertionError("Reference preparation read forbidden evaluation or secret data")

    def guarded_open(file, mode="r", *args, **kwargs):
        inspect(file, mode)
        return original_open(file, mode, *args, **kwargs)

    def guarded_path_open(path, mode="r", *args, **kwargs):
        inspect(path, mode)
        return original_path_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    spec = scope(root)
    result = prepare_reference_packet(root, spec)
    assert result["status"] == "awaiting_human_reference"
    assert result["packet_relative"] == "runs/agent/source-review/packet.json"
    assert result["template_relative"] == "runs/agent/source-review/submission-template.json"
    packet = load_reference_packet(root, spec.run_id)
    assert result["packet_id"] == packet.packet_id
    assert packet.bundle.source_sha256 == spec.source.sha256
    assert "Sample B was not measured" in packet.bundle.text
    run = root / "runs/agent/source-review"
    assert (run / "original.md").read_bytes() == (root / spec.source.path).read_bytes()
    assert (run / "source.txt").read_bytes().decode("utf-8") == packet.bundle.text
    assert json.loads((run / "schema.json").read_bytes()) == packet.profile.schema
    for name, metadata in result["files"].items():
        raw = (run / name).read_bytes()
        assert metadata == {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    template = json.loads((root / result["template_relative"]).read_bytes())
    assert template["packet_id"] == packet.packet_id
    assert template["canonical_record"] is None
    assert template["reviewer"] is None
    assert template["evidence"] == []
    assert template["unresolved"] == []
    assert len(template["coverage"]) == 6
    assert all(item["status"] == "unreviewed" for item in template["coverage"])
    assert not (run / "canonical.json").exists()
    assert forbidden_reads == []


def test_prepare_and_load_replay_preserve_every_artifact(review_workspace):
    root = review_workspace
    spec = scope(root)
    result = prepare_reference_packet(root, spec)
    before = run_files(root)
    assert prepare_reference_packet(root, spec) == result
    assert load_reference_packet(root, spec.run_id).packet_id == result["packet_id"]
    assert run_files(root) == before


def test_wrong_source_hash_rejected_before_claim(review_workspace):
    root = review_workspace
    original = scope(root)
    bad = ArtifactRef(path=original.source.path, sha256="0" * 64)
    with pytest.raises(ValueError):
        prepare_reference_packet(root, scope(root, source=bad))
    assert not (root / "runs/agent/source-review").exists()


@pytest.mark.parametrize("relative", ["data/gold/invented.md", "outputs/invented.md", ".env"])
def test_forbidden_source_paths_rejected_before_claim(review_workspace, relative):
    root = review_workspace
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("Invented forbidden input, not a credential or real gold.")
    with pytest.raises(ValueError):
        prepare_reference_packet(root, scope(root, source=pin(root, relative)))
    assert not (root / "runs/agent/source-review").exists()


@pytest.mark.parametrize("updates", [
    {"split": "future_held_out"}, {"unknown_option": True}, {"schemas": ("../gold.json",)},
    {"schemas": ("generic.json",)}, {"schemas": ("photon_and_neutron.json",)},
    {"schemas": ("generic_no_instructions.json", "photon_and_neutron.json")},
])
def test_preparation_scope_does_not_expand_silently(review_workspace, updates):
    with pytest.raises(ValueError):
        prepare_reference_packet(review_workspace, scope(review_workspace, **updates))
    assert not (review_workspace / "runs/agent/source-review").exists()


def test_existing_unclaimed_directory_is_not_overwritten(review_workspace):
    root = review_workspace
    existing = root / "runs/agent/source-review"
    existing.mkdir(parents=True)
    sentinel = existing / "user-notes.txt"
    sentinel.write_bytes(b"Keep this user's work.")
    with pytest.raises((ValueError, RunIncompleteError)):
        prepare_reference_packet(root, scope(root))
    assert sentinel.read_bytes() == b"Keep this user's work."
    assert sorted(path.name for path in existing.iterdir()) == ["user-notes.txt"]


@pytest.mark.parametrize("relative", [
    "runs/agent/source-review/source.txt", "runs/agent/source-review/instructions.md",
    "schemas/generic.json", "src/meta_data_extraction/offline/reference_review.py",
    SOURCE_RELATIVE,
])
def test_packet_rejects_changed_artifacts_resources_or_implementation(review_workspace, relative):
    root = review_workspace
    prepare_reference_packet(root, scope(root))
    target = root / relative
    target.write_bytes(target.read_bytes() + b"\nchanged after preparation\n")
    with pytest.raises((ValueError, RunIncompleteError)):
        load_reference_packet(root, "source-review")


def test_blank_submission_remains_incomplete_with_no_empty_gold_counts(review_workspace):
    root = review_workspace
    prepared = prepare_reference_packet(root, scope(root))
    submitted = pin(root, prepared["template_relative"])
    result = submit_reference(root, "source-review", submitted, run_id="blank-review-receipt")
    assert result["status"] == "incomplete"
    assessment = json.loads((root / result["assessment_relative"]).read_bytes())["assessment"]
    assert assessment["status"] == "incomplete"
    assert len(assessment["declared_entity_counts"]) == 6
    assert set(assessment["declared_entity_counts"].values()) == {None}
    assert assessment["independent_gold_status"] == "not_established"
    assert assessment["scoring_status"] == "blocked_reference_and_alignment_adjudication"
    assert assessment["full_source_coverage"] == "not_established"
    assert not (root / "runs/agent/blank-review-receipt/canonical.json").exists()
    before = run_files(root)
    assert submit_reference(root, "source-review", submitted, run_id="blank-review-receipt") == result
    assert run_files(root) == before


@pytest.mark.parametrize("kind", ["duplicate_key", "unknown_field", "stale_packet", "oversized"])
def test_invalid_submission_is_rejected_before_receipt_claim(review_workspace, kind):
    root = review_workspace
    prepared = prepare_reference_packet(root, scope(root))
    template = json.loads((root / prepared["template_relative"]).read_bytes())
    if kind == "duplicate_key":
        raw = b'{"packet_id":"one","packet_id":"two"}'
    elif kind == "unknown_field":
        raw = json.dumps(template | {"automatic_approval": True}).encode()
    elif kind == "stale_packet":
        raw = json.dumps(template | {"packet_id": "reference-packet:" + "0" * 64}).encode()
    else:
        raw = b" " * (2 * 1024 * 1024 + 1)
    submitted = authored(root, raw)
    with pytest.raises(ValueError):
        submit_reference(root, "source-review", submitted, run_id="rejected-receipt")
    assert not (root / "runs/agent/rejected-receipt").exists()


def test_changed_submission_cannot_replay_a_receipt(review_workspace):
    root = review_workspace
    prepared = prepare_reference_packet(root, scope(root))
    template = json.loads((root / prepared["template_relative"]).read_bytes())
    submitted = authored(root, template)
    submit_reference(root, "source-review", submitted, run_id="review-receipt")
    before = run_files(root)
    (root / submitted.path).write_bytes(json.dumps(template, indent=2).encode())
    with pytest.raises(ValueError):
        submit_reference(root, "source-review", submitted, run_id="review-receipt")
    assert run_files(root) == before


def test_submission_cannot_be_read_from_benchmark_output(review_workspace):
    root = review_workspace
    prepared = prepare_reference_packet(root, scope(root))
    target = root / "outputs/invented-review.json"
    target.parent.mkdir()
    target.write_bytes((root / prepared["template_relative"]).read_bytes())
    with pytest.raises(ValueError):
        submit_reference(root, "source-review", pin(root, "outputs/invented-review.json"),
                         run_id="rejected-receipt")
    assert not (root / "runs/agent/rejected-receipt").exists()


def test_submission_hash_mismatch_is_not_a_recorded_review(review_workspace):
    root = review_workspace
    prepared = prepare_reference_packet(root, scope(root))
    submitted = ArtifactRef(path=prepared["template_relative"], sha256="0" * 64)
    with pytest.raises(ValueError):
        submit_reference(root, "source-review", submitted, run_id="rejected-receipt")
    assert not (root / "runs/agent/rejected-receipt").exists()


def test_tampered_assessment_is_not_accepted_as_receipt_replay(review_workspace):
    root = review_workspace
    prepared = prepare_reference_packet(root, scope(root))
    submitted = pin(root, prepared["template_relative"])
    result = submit_reference(root, "source-review", submitted, run_id="review-receipt")
    target = root / result["assessment_relative"]
    changed = target.read_bytes() + b"\n"
    target.write_bytes(changed)
    with pytest.raises(ValueError):
        submit_reference(root, "source-review", submitted, run_id="review-receipt")
    assert target.read_bytes() == changed


def test_submission_rejects_a_hard_link_to_other_input(review_workspace):
    root = review_workspace
    prepared = prepare_reference_packet(root, scope(root))
    copied = root / "tests/offline/fixtures/linked-submission.json"
    copied.parent.mkdir(parents=True, exist_ok=True)
    original = authored(root, (root / prepared["template_relative"]).read_bytes())
    try:
        os.link(root / original.path, copied)
    except OSError as error:
        pytest.skip(f"Hard links are unavailable in this test workspace: {error.errno}")
    with pytest.raises(ValueError):
        submit_reference(root, "source-review", pin(root, copied.relative_to(root).as_posix()),
                         run_id="rejected-receipt")
    assert not (root / "runs/agent/rejected-receipt").exists()


def test_complete_invented_submission_records_receipt_without_gold_or_runtime_export(review_workspace):
    root = review_workspace
    prepare_reference_packet(root, scope(root))
    packet = load_reference_packet(root, "source-review")
    record = build_skeleton(packet.profile.schema)
    for name in ("has_references", "has_figures", "has_tables"):
        record["document_overview"][name] = False
    for index, name in enumerate(("Sample A", "Sample B"), 1):
        sample = build_skeleton(packet.profile.schema["samples"]["item_schema"])
        sample.update(id=f"sample_{index}", sample_name=name)
        record["samples"].append(sample)
    dataset = build_skeleton(packet.profile.schema["datasets"]["item_schema"])
    dataset.update(id="dataset_1", dataset_name="Dataset A", sample_refs=["sample_1"])
    record["datasets"].append(dataset)
    draft = blank_submission(packet).model_dump(mode="json") | {"canonical_record": record}
    missing = assess_reference(packet, ReferenceSubmission.model_validate(draft))["missing_review_paths"]
    ref = evidence_ref(packet.bundle, packet.bundle.regions[0].region_id).model_dump(mode="json")
    annotations = []
    for path in missing:
        value = at_pointer(record, path)
        absent = value is None or value == [] or value == {}
        annotations.append({"path": path, "status": "not_reported" if absent else "supported",
                            "evidence": [] if absent else [ref],
                            "note": "Invented contract input; semantic support is not verified."})
    # This deliberately invented declaration tests receipt semantics. It is not
    # a record of actual human participation, independent evidence or review time.
    draft.update(evidence=annotations, reviewer={"identity": "Invented unit-test reviewer",
        "provenance": "human", "prior_prediction_access": "not_seen",
        "independence": "declared_independent", "elapsed_seconds": None},
        coverage=[item | {"status": "complete_available_text",
            "note": "Invented test declaration; no person participated."} for item in draft["coverage"]])
    result = submit_reference(root, "source-review", authored(root, draft), run_id="complete-receipt")
    assert result["status"] == "available_text_review_recorded"
    receipt = json.loads((root / result["assessment_relative"]).read_bytes())
    assessment = receipt["assessment"]
    assert result["assessment_sha256"] == digest(assessment)
    assert assessment["declared_entity_counts"]["samples"] == 2
    assert assessment["declared_entity_counts"]["datasets"] == 1
    assert assessment["independent_gold_status"] == "not_established"
    assert assessment["scoring_status"] == "blocked_reference_and_alignment_adjudication"
    assert assessment["reviewer"]["elapsed_seconds"] is None
    assert receipt["submission"]["canonical_record"] == record
    assert not (root / "runs/agent/complete-receipt/canonical.json").exists()
    assert not (root / "data/gold").exists()


def test_excerpt_preserves_exact_offsets_hashes_and_packet_artifacts(review_workspace):
    root = review_workspace
    spec = scope(root)
    prepare_reference_packet(root, spec)
    packet = load_reference_packet(root, spec.run_id)
    region = next(region for region in packet.bundle.regions
                  if "Two distinct" in packet.bundle.text[region.start:region.end])
    before = run_files(root)
    full = reference_excerpt(root, spec.run_id, region.region_id)
    assert full == evidence_ref(packet.bundle, region.region_id).model_dump(mode="json")
    start = packet.bundle.text.index("Sample A", region.start, region.end)
    end = start + len("Sample A")
    selected = reference_excerpt(root, spec.run_id, region.region_id, start=start, end=end)
    assert selected["start"] == start and selected["end"] == end
    assert selected["quotation"] == "Sample A"
    assert selected["source_sha256"] == spec.source.sha256
    assert selected["representation_sha256"] == packet.bundle.representation_sha256
    assert selected["region_id"] == region.region_id and selected["page"] is None
    assert run_files(root) == before


def test_excerpt_rejects_unknown_region_and_out_of_region_offsets(review_workspace):
    root = review_workspace
    prepare_reference_packet(root, scope(root))
    packet = load_reference_packet(root, "source-review")
    region = packet.bundle.regions[-1]
    before = run_files(root)
    with pytest.raises(ValueError):
        reference_excerpt(root, "source-review", "invented-nonexistent-region")
    for start, end in ((region.start - 1, region.end), (region.start, region.end + 1),
                       (region.end, region.start)):
        with pytest.raises(ValueError):
            reference_excerpt(root, "source-review", region.region_id, start=start, end=end)
    assert run_files(root) == before


def test_excerpt_cli_parses_explicit_subspan_and_prints_only_evidence(review_workspace, capsys):
    root = review_workspace
    prepare_reference_packet(root, scope(root))
    packet = load_reference_packet(root, "source-review")
    region = packet.bundle.regions[-1]
    start, end = region.start, min(region.end, region.start + 12)
    before = run_files(root)
    command = ["--root", str(root), "excerpt", "--packet-run", "source-review",
               "--region-id", region.region_id, "--start", str(start), "--end", str(end)]
    assert main(command) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == evidence_ref(
        packet.bundle, region.region_id, start=start, end=end).model_dump(mode="json")
    assert run_files(root) == before
    with pytest.raises(SystemExit) as rejected:
        main(command[:-1] + ["not-an-offset"])
    assert rejected.value.code == 2
    assert "invalid int value" in capsys.readouterr().err
