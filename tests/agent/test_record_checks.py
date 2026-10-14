"""Synthetic M06 structural checks; no model, gold, or file-discovery behavior."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import json
import shutil

import pytest
from pydantic import ValidationError

from extraction_fixtures import synthetic_bundle, synthetic_record
from meta_data_extraction.agent.contracts import canonical_json, digest, snapshot
from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.agent.profiles import load_profile, validate_record
from meta_data_extraction.agent.record_checks import (
    CheckIssue, RecordCheckError, StructureReport, check_structure, verify_structure,
)


@pytest.fixture
def inputs(workspace):
    repository = Path(__file__).resolve().parents[2]
    for name in ("prompt_template.md", "generic_no_instructions.json", "photon_and_neutron_no_instructions.json"):
        shutil.copyfile(repository / "schemas" / name, workspace / "schemas" / name)
    profile = load_profile(workspace)
    return profile, synthetic_bundle(), synthetic_record(profile)


def test_valid_report_reuses_existing_snapshot_and_makes_no_support_or_coverage_claim(inputs):
    profile, bundle, record = inputs
    report = check_structure(profile, bundle, record)
    assert report.status == "valid" and report.issues == ()
    assert report.snapshot == snapshot(record, bundle, profile)
    assert report.candidate == record
    assert report.support_status == "not_assessed"
    assert report.coverage_status == "unsupported"
    assert report.canonical_sha256 == sha256(canonical_json(record).encode("utf-8"))
    assert report.record_id == report.snapshot.record_id
    assert report.revision_id == report.snapshot.revision_id
    verify_structure(report, profile, bundle)
    assert StructureReport.model_validate_json(report.model_dump_json()) == report


def test_unknown_nested_content_is_reported_and_preserved_in_invalid_candidate(inputs):
    profile, bundle, record = inputs
    unexpected = {"raw": ["café", None, False, 7, 2.75, {"more/key~": "kept"}]}
    record["extra/key~"] = unexpected
    record["samples"][0]["unapproved_temperature"] = "80 K"
    before = deepcopy(record)
    report = check_structure(profile, bundle, record)
    assert report.status == "invalid" and report.snapshot is None
    assert {(i.code, i.path) for i in report.issues} == {
        ("unknown_field", "/extra~1key~0"),
        ("unknown_field", "/samples/0/unapproved_temperature"),
    }
    assert all(i.category == "structural" and i.origin == "deterministic" for i in report.issues)
    assert all(i.recommended_action == "review_representation" for i in report.issues)
    assert report.candidate == before == record
    assert report.candidate["extra/key~"] == unexpected
    assert "café" in report.candidate_json
    verify_structure(report, profile, bundle)


def test_dangling_reference_is_preserved_without_entering_snapshot_resolver(inputs, monkeypatch):
    profile, bundle, record = inputs
    record["datasets"][0]["sample_refs"] = ["sample_999"]
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid reference must not enter snapshot resolution")
    monkeypatch.setattr("meta_data_extraction.agent.record_checks.snapshot", forbidden)
    report = check_structure(profile, bundle, record)
    assert report.status == "invalid" and report.snapshot is None
    assert [(i.code, i.path, i.category, i.recommended_action) for i in report.issues] == [
        ("dangling_reference", "/datasets/0/sample_refs/0", "reference", "inspect_source"),
    ]
    assert report.candidate["datasets"][0]["sample_refs"] == ["sample_999"]
    expected_record_id = "record:" + digest([bundle.source_id, bundle.source_sha256, profile.profile_id])
    assert report.record_id == expected_record_id
    assert report.revision_id == "revision:" + digest([expected_record_id, bundle.representation_sha256, record])


def test_duplicate_local_ids_report_ambiguity_and_do_not_merge(inputs):
    profile, bundle, record = inputs
    record["samples"][1]["id"] = "sample_1"
    report = check_structure(profile, bundle, record)
    assert report.snapshot is None
    assert {(i.code, i.path) for i in report.issues} == {
        ("duplicate_local_id", "/samples/1/id"),
        ("ambiguous_reference", "/datasets/0/sample_refs/0"),
        ("dangling_reference", "/datasets/0/sample_refs/1"),
    }
    duplicate = next(i for i in report.issues if i.code == "duplicate_local_id")
    assert duplicate.category == "reference" and duplicate.recommended_action == "human_review"
    assert len(report.candidate["samples"]) == 2


def test_same_composition_different_treatment_specimens_keep_distinct_handles(inputs):
    profile, bundle, record = inputs
    record["samples"][0]["sample_description"] = "heated specimen"
    record["samples"][1]["sample_description"] = "cooled specimen"
    report = check_structure(profile, bundle, record)
    assert report.status == "valid"
    entities = [e for e in report.snapshot.entities if e.collection == "samples"]
    assert len(entities) == 2 and entities[0].handle != entities[1].handle
    assert [s["chemical_composition"] for s in report.candidate["samples"]] == ["CaCO3", "CaCO3"]
    assert len(report.snapshot.references) == 4
    assert report.candidate == record


def test_valid_but_swapped_reference_cannot_be_detected_by_structure_alone(inputs):
    profile, bundle, record = inputs
    record["datasets"][0]["sample_refs"] = ["sample_2", "sample_1"]
    report = check_structure(profile, bundle, record)
    assert report.status == "valid" and not report.issues
    assert report.support_status == "not_assessed"
    assert report.snapshot.canonical["datasets"][0]["sample_refs"] == ["sample_2", "sample_1"]


def test_correctly_missing_optional_value_is_neither_filled_nor_flagged(inputs):
    profile, bundle, record = inputs
    assert record["bibliographic"]["authors"][0]["orcid"] is None
    report = check_structure(profile, bundle, record)
    assert report.status == "valid"
    assert report.candidate["bibliographic"]["authors"][0]["orcid"] is None


def test_reports_all_findings_from_inherited_validator_without_repair(inputs, monkeypatch):
    profile, bundle, record = inputs
    record["document_overview"]["has_figures"] = None
    del record["datasets"][0]["purpose"]
    record["bibliographic"]["keywords"] = ["invented"] * 51
    record["samples"][0]["id"] = "incorrect_prefix"
    expected = validate_record(record, profile)
    calls = []
    def observed(candidate, frozen):
        calls.append((deepcopy(candidate), frozen))
        return validate_record(candidate, frozen)
    monkeypatch.setattr("meta_data_extraction.agent.record_checks.validate_record", observed)
    report = check_structure(profile, bundle, record)
    assert len(calls) == 1 and calls[0] == (record, profile)
    assert [(i.code, i.path, i.message) for i in report.issues] == [
        (i.code, i.path, i.message) for i in expected
    ]
    assert report.candidate == record


def test_empty_object_is_not_successful_empty_extraction(inputs):
    profile, bundle, _ = inputs
    report = check_structure(profile, bundle, {})
    assert report.status == "invalid" and report.snapshot is None
    assert len(report.issues) == 5
    assert {i.code for i in report.issues} == {"missing_field"}
    assert report.candidate == {}


def test_input_and_returned_copies_cannot_mutate_frozen_report(inputs):
    profile, bundle, record = inputs
    report = check_structure(profile, bundle, record)
    before = report.model_dump_json()
    record["samples"].clear()
    copy = report.candidate
    copy["datasets"][0]["sample_refs"].clear()
    assert report.model_dump_json() == before
    with pytest.raises(ValidationError):
        report.status = "invalid"


@pytest.mark.parametrize("bad", [None, [], "{}", 0, True])
def test_nonobject_input_requires_explicit_parse_boundary(inputs, bad):
    profile, bundle, _ = inputs
    with pytest.raises(RecordCheckError, match="parsed JSON object"):
        check_structure(profile, bundle, bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), (1, 2), {1, 2}, Decimal("1"), Path("local")])
def test_nonjson_values_never_disappear_through_coercion(inputs, bad):
    profile, bundle, record = inputs
    record["unexpected"] = bad
    with pytest.raises(RecordCheckError):
        check_structure(profile, bundle, record)


def test_nonstring_keys_never_coerce_into_json_fields(inputs):
    profile, bundle, record = inputs
    record[5] = "would coerce"
    with pytest.raises(RecordCheckError, match="keys must be strings"):
        check_structure(profile, bundle, record)


def test_recursive_candidate_rejected_without_snapshot(inputs):
    profile, bundle, record = inputs
    record["cycle"] = record
    with pytest.raises(RecordCheckError, match="recursive"):
        check_structure(profile, bundle, record)


def test_reordered_object_keys_have_same_record_and_report_identity(inputs):
    profile, bundle, record = inputs
    reordered = json.loads(json.dumps(record, sort_keys=True))
    assert check_structure(profile, bundle, record) == check_structure(profile, bundle, reordered)


@pytest.mark.parametrize("change", ["source_id", "source_bytes", "representation"])
def test_replay_rejects_other_source_or_representation(inputs, change):
    profile, bundle, record = inputs
    report = check_structure(profile, bundle, record)
    if change == "source_id":
        changed = synthetic_bundle(bundle.text, source_id="another-document")
    elif change == "source_bytes":
        changed = synthetic_bundle(bundle.text + " Changed source.")
    else:
        changed = synthetic_bundle(bundle.text, max_chars=len(bundle.text) - 10)
    with pytest.raises(RecordCheckError, match="no longer matches"):
        verify_structure(report, profile, changed)


def test_other_valid_profile_cannot_replay_report(inputs, workspace):
    profile, bundle, record = inputs
    report = check_structure(profile, bundle, record)
    other = load_profile(workspace, ("generic_no_instructions.json", "photon_and_neutron_no_instructions.json"))
    assert validate_record(record, other) == ()
    with pytest.raises(RecordCheckError, match="no longer matches"):
        verify_structure(report, other, bundle)


@pytest.mark.parametrize("change", ["profile_id", "schema_payload_json", "schema_sha256", "schema_names"])
def test_model_copy_cannot_bypass_profile_integrity(inputs, change):
    profile, bundle, record = inputs
    values = {"profile_id": "profile:" + "0" * 64, "schema_payload_json": "{}",
              "schema_sha256": "0" * 64, "schema_names": ("generic.json",)}
    with pytest.raises(RecordCheckError):
        check_structure(profile.model_copy(update={change: values[change]}), bundle, record)


def test_model_copy_cannot_bypass_bundle_integrity(inputs):
    profile, bundle, record = inputs
    with pytest.raises(ValidationError):
        check_structure(profile, bundle.model_copy(update={"text": "Forged source"}), record)


def test_changed_record_changes_revision_and_report_identity(inputs):
    profile, bundle, record = inputs
    first = check_structure(profile, bundle, record)
    record["bibliographic"]["title"] = "Different invented title"
    second = check_structure(profile, bundle, record)
    assert first.record_id == second.record_id
    assert first.revision_id != second.revision_id
    assert first.report_id != second.report_id


def test_report_rejects_snapshot_of_different_revision(inputs):
    profile, bundle, record = inputs
    report = check_structure(profile, bundle, record)
    record["bibliographic"]["title"] = "Changed title"
    payload = report.model_dump(mode="json", exclude={"report_id"})
    payload["snapshot"] = snapshot(record, bundle, profile).model_dump(mode="json")
    with pytest.raises(ValidationError, match="Snapshot does not match"):
        StructureReport(report_id="structure:" + digest(payload), **payload)


def test_forged_message_with_recomputed_hash_rejected_by_semantic_replay(inputs):
    profile, bundle, record = inputs
    record["unapproved"] = "preserve"
    report = check_structure(profile, bundle, record)
    payload = report.model_dump(mode="json", exclude={"report_id"})
    payload["issues"][0]["message"] = "Invented diagnosis"
    forged = StructureReport(report_id="structure:" + digest(payload), **payload)
    with pytest.raises(RecordCheckError, match="findings"):
        verify_structure(forged, profile, bundle)


@pytest.mark.parametrize("field,value", [
    ("coverage_status", "complete"), ("support_status", "supported"),
    ("canonical_sha256", "0" * 64), ("record_id", "record:" + "0" * 64),
    ("revision_id", "revision:" + "0" * 64), ("report_id", "structure:" + "0" * 64),
    ("status", "invalid"), ("snapshot", None),
])
def test_saved_report_cannot_bypass_internal_consistency(inputs, field, value):
    profile, bundle, record = inputs
    report = check_structure(profile, bundle, record)
    with pytest.raises(ValidationError):
        verify_structure(report.model_copy(update={field: value}), profile, bundle)


def test_issue_action_and_category_are_not_model_selected():
    with pytest.raises(ValidationError, match="category/action"):
        CheckIssue(category="structural", code="dangling_reference", path="/datasets/0/sample_refs/0",
                   message="Missing reference", recommended_action="review_representation")
    with pytest.raises(ValidationError):
        CheckIssue(category="reference", code="dangling_reference", path="/datasets/0/sample_refs/0",
                   message="Missing reference", recommended_action="inspect_source", origin="model_judgment")


def test_pure_checks_do_not_read_resources_or_files(inputs, monkeypatch):
    profile, bundle, record = inputs
    def forbidden(*args, **kwargs):
        raise AssertionError("Domain checks must not read files")
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    report = check_structure(profile, bundle, record)
    verify_structure(report, profile, bundle)
