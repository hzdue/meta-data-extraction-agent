"""Invented reference submissions exercise contracts, never human evidence."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from meta_data_extraction.agent.contracts import at_pointer, digest
from meta_data_extraction.agent.evidence import ParserProvenance, bundle_document, evidence_ref
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.data_models import Document
from meta_data_extraction.offline.reference_types import (
    COLLECTIONS, CoverageDeclaration, ReferenceEvidence, ReferencePacket,
    ReferenceSubmission, ReviewerDeclaration, assess_reference, blank_submission,
    build_packet,
)
from meta_data_extraction.utils import build_skeleton


@pytest.fixture
def packet():
    profile = load_profile(Path(__file__).resolve().parents[2])
    text = "Invented reference exercise. Specimen Alpha and Specimen Beta are distinct. " \
        "Diffraction measures both specimens. The fictional source has no tables or figures."
    bundle = bundle_document(Document(id="invented", text=text),
        source_id="tests/offline/fixtures/invented-reference.md", source_bytes=text.encode(),
        source_format="markdown", parser=ParserProvenance(name="synthetic", version="1"))
    return build_packet(publication_id="invented-reference", split="calibration",
                        data_kind="synthetic", bundle=bundle, profile=profile)


@pytest.fixture
def record(packet):
    value = build_skeleton(packet.profile.schema)
    value["bibliographic"]["title"] = "Invented reference exercise"
    value["bibliographic"]["funding"] = [{"funder": "Invented funder", "grant_id": "X-1"}]
    for name in ("has_references", "has_figures", "has_tables"):
        value["document_overview"][name] = False
    for index, name in enumerate(("Alpha", "Beta"), 1):
        sample = build_skeleton(packet.profile.schema["samples"]["item_schema"])
        sample.update(id=f"sample_{index}", sample_name=f"Specimen {name}")
        value["samples"].append(sample)
    dataset = build_skeleton(packet.profile.schema["datasets"]["item_schema"])
    dataset.update(id="dataset_1", dataset_name="Diffraction", measurement_technique="Diffraction",
                   sample_refs=["sample_1", "sample_2"])
    value["datasets"] = [dataset]
    return value


def replace(submission, **changes):
    return ReferenceSubmission.model_validate(submission.model_dump(mode="json") | changes)


def complete_fixture(packet, record):
    """All labels/identity claims below are synthetic contract inputs only."""
    draft = replace(blank_submission(packet), canonical_record=record)
    paths = assess_reference(packet, draft)["missing_review_paths"]
    ref = evidence_ref(packet.bundle, packet.bundle.regions[0].region_id)
    annotations = []
    for path in paths:
        value = at_pointer(record, path)
        absent = value is None or value == [] or value == {}
        annotations.append(ReferenceEvidence(path=path,
            status="not_reported" if absent else "supported",
            evidence=() if absent else (ref,), note="Synthetic annotation for contract testing."))
    return replace(draft, evidence=annotations,
        reviewer=ReviewerDeclaration(identity="Invented reviewer", provenance="human",
            prior_prediction_access="not_seen", independence="declared_independent"),
        coverage=[CoverageDeclaration(collection=name, status="complete_available_text",
            note="Invented coverage declaration; no person participated.") for name in COLLECTIONS])


def test_blank_is_not_zero_entities_or_approval(packet):
    submission = blank_submission(packet)
    result = assess_reference(packet, submission)
    assert submission.reviewer is None and submission.canonical_record is None
    assert {item.collection for item in submission.coverage} == set(COLLECTIONS)
    assert result["status"] == "incomplete"
    assert result["declared_entity_counts"] == {name: None for name in COLLECTIONS}
    assert result["record_revision"] is None and result["references"] == []
    assert not result["declared_source_first_human"]
    assert result["independent_gold_status"] == result["full_source_coverage"] == "not_established"


def test_packet_roundtrip_and_digest_bind_all_fields(packet):
    assert ReferencePacket.model_validate_json(packet.model_dump_json()) == packet
    assert packet.packet_id == "reference-packet:" + digest(packet.model_dump(mode="json", exclude={"packet_id"}))
    for field, value in (("publication_id", "other-paper"), ("split", "historical"), ("data_kind", "natural")):
        with pytest.raises(ValueError, match="packet_digest_mismatch"):
            ReferencePacket.model_validate(packet.model_dump(mode="json") | {field: value})


@pytest.mark.parametrize("names", [
    ("generic.json",), ("photon_and_neutron.json",),
    ("generic_no_instructions.json", "photon_and_neutron_no_instructions.json"),
])
def test_six_collection_packet_rejects_partial_or_ablated_profiles(packet, names):
    profile = load_profile(Path(__file__).resolve().parents[2], names)
    with pytest.raises(ValueError, match="full_canonical_schema_pair_required"):
        build_packet(publication_id="invented", split="calibration", data_kind="synthetic",
                     bundle=packet.bundle, profile=profile)


def test_valid_crlf_prompt_profile_is_accepted_without_changing_raw_hash(packet, workspace):
    from meta_data_extraction.agent.evidence import sha256
    root = Path(__file__).resolve().parents[2]
    (workspace / "schemas").mkdir()
    for name in ("generic.json", "photon_and_neutron.json"):
        shutil.copyfile(root / "schemas" / name, workspace / "schemas" / name)
    raw = (root / "schemas/prompt_template.md").read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8")
    (workspace / "schemas/prompt_template.md").write_bytes(raw)
    profile = load_profile(workspace)
    assert profile.prompt_sha256 == sha256(raw)
    assert profile.prompt_sha256 != sha256(profile.prompt_template.encode("utf-8"))
    made = build_packet(publication_id="invented", split="calibration", data_kind="synthetic",
                        bundle=packet.bundle, profile=profile)
    assert made.profile == profile


def test_mutated_profile_or_bundle_rejected_even_with_recomputed_packet_hash(packet):
    with pytest.raises(ValueError, match="schema or resource identity"):
        build_packet(publication_id="invented", split="calibration", data_kind="synthetic",
            bundle=packet.bundle, profile=packet.profile.model_copy(update={"schema_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="Representation hash"):
        build_packet(publication_id="invented", split="calibration", data_kind="synthetic",
            bundle=packet.bundle.model_copy(update={"text": "Changed source"}), profile=packet.profile)


def test_stale_packet_and_model_construct_are_revalidated(packet):
    blank = blank_submission(packet)
    with pytest.raises(ValueError, match="packet_mismatch"):
        assess_reference(packet, blank.model_copy(update={"packet_id": "reference-packet:" + "0" * 64}))
    with pytest.raises(ValueError, match="duplicate_coverage"):
        assess_reference(packet, blank.model_copy(update={"coverage": blank.coverage[:1] * 2}))


def test_inventory_includes_identity_ids_nulls_and_empty_reference_containers(packet, record):
    result = assess_reference(packet, replace(blank_submission(packet), canonical_record=record))
    assert result["status"] == "incomplete"
    assert result["declared_entity_counts"] == {
        "bibliographic.authors": 0, "bibliographic.affiliations": 0,
        "bibliographic.funding": 1, "facilities": 0, "samples": 2, "datasets": 1,
    }
    assert set(result["missing_review_paths"]) >= {
        "/samples/0", "/samples/1", "/samples/0/id", "/samples/1/id",
        "/samples/0/sample_mass", "/datasets/0", "/datasets/0/id",
        "/datasets/0/sample_refs/0", "/datasets/0/sample_refs/1",
        "/datasets/0/facility_refs", "/datasets/0/measurement_mode",
        "/bibliographic/funding/0", "/bibliographic/subtitle", "/document_overview/section_headings",
    }
    assert "/facilities" not in result["missing_review_paths"]
    assert len(result["references"]) == 2
    assert {r["local_id"] for r in result["references"]} == {"sample_1", "sample_2"}


def test_declared_complete_text_review_still_not_gold_or_verified_human(packet, record):
    submission = complete_fixture(packet, record)
    result = assess_reference(packet, submission)
    assert result["status"] == "available_text_review_recorded"
    assert result["missing_review_paths"] == result["unresolved_paths"] == []
    assert result["review_identity_verification"] == "self_declared"
    assert result["independent_gold_status"] == "not_established"
    assert result["scoring_status"] == "blocked_reference_and_alignment_adjudication"
    assert result["evidence_check"] == "locations_only_semantic_support_not_verified"
    assert result["reviewer"]["elapsed_seconds"] is None


@pytest.mark.parametrize("change", [
    {"provenance": "assistant"}, {"provenance": "synthetic"},
    {"prior_prediction_access": "seen"}, {"prior_prediction_access": "unknown"},
    {"independence": "not_independent"}, {"independence": "not_assessed"},
])
def test_non_independent_or_nonhuman_cannot_complete(packet, record, change):
    submission = complete_fixture(packet, record)
    reviewer = submission.reviewer.model_dump(mode="json") | change
    result = assess_reference(packet, replace(submission, reviewer=reviewer))
    assert result["status"] == "incomplete"
    assert not result["declared_source_first_human"]


@pytest.mark.parametrize("change", ["missing", "uncertain", "unresolved", "partial", "no_reviewer"])
def test_missing_or_unresolved_review_stays_incomplete(packet, record, change):
    submission = complete_fixture(packet, record)
    if change == "missing":
        submission = replace(submission, evidence=submission.evidence[1:])
    elif change == "uncertain":
        evidence = list(submission.evidence)
        evidence[0] = evidence[0].model_copy(update={"status": "uncertain"})
        submission = replace(submission, evidence=evidence)
    elif change == "unresolved":
        submission = replace(submission, unresolved=["Entity granularity remains disputed."])
    elif change == "no_reviewer":
        submission = replace(submission, reviewer=None)
    else:
        coverage = list(submission.coverage)
        coverage[0] = coverage[0].model_copy(update={"status": "partial"})
        submission = replace(submission, coverage=coverage)
    assert assess_reference(packet, submission)["status"] == "incomplete"


@pytest.mark.parametrize("problem", ["empty", "missing_field", "duplicate_id", "dangling", "unknown", "nan"])
def test_invalid_canonical_record_is_not_silently_an_incomplete_draft(packet, record, problem):
    value = deepcopy(record)
    if problem == "empty":
        value = {}
    elif problem == "missing_field":
        del value["samples"]
    elif problem == "duplicate_id":
        value["samples"][1]["id"] = value["samples"][0]["id"]
    elif problem == "dangling":
        value["datasets"][0]["sample_refs"] = ["sample_999"]
    elif problem == "unknown":
        value["private_reasoning"] = "Do not admit undeclared schema fields."
    else:
        value["bibliographic"]["title"] = float("nan")
    with pytest.raises(ValueError):
        assess_reference(packet, replace(blank_submission(packet), canonical_record=value))


def test_nonfinite_mutated_submission_is_rejected_before_json_serialization(packet, record):
    draft = replace(blank_submission(packet), canonical_record=record)
    malformed = deepcopy(record)
    malformed["bibliographic"]["title"] = float("nan")
    forged = draft.model_copy(update={"canonical_record": malformed})
    with pytest.raises(ValueError, match="finite"):
        assess_reference(packet, forged)


@pytest.mark.parametrize("path,status", [
    ("/bibliographic/subtitle", "supported"),
    ("/datasets/0/facility_refs", "supported"),
    ("/samples/0", "not_reported"),
    ("/samples/0/id", "not_reported"),
    ("/samples/0/sample_name", "not_reported"),
    ("/facilities", "uncertain"),
    ("/samples/00/id", "uncertain"),
    ("/not_a_field", "uncertain"),
])
def test_annotation_status_and_exact_path_must_match_record(packet, record, path, status):
    ref = evidence_ref(packet.bundle, packet.bundle.regions[0].region_id)
    annotation = ReferenceEvidence(path=path, status=status, evidence=(ref,), note="Synthetic review.")
    with pytest.raises(ValueError):
        assess_reference(packet, replace(blank_submission(packet), canonical_record=record, evidence=[annotation]))


@pytest.mark.parametrize("field,value", [
    ("quotation", "Not present in source"), ("source_sha256", "0" * 64),
    ("representation_sha256", "0" * 64), ("source_id", "another-source.md"),
    ("region_id", "0" * 64), ("start", 1), ("end", 100000),
])
def test_forged_evidence_rejected(packet, record, field, value):
    ref = evidence_ref(packet.bundle, packet.bundle.regions[0].region_id)
    annotation = ReferenceEvidence(path="/samples/0", status="supported",
        evidence=(ref.model_copy(update={field: value}),), note="Synthetic annotation.")
    with pytest.raises(ValueError, match="evidence_does_not_resolve"):
        assess_reference(packet, replace(blank_submission(packet), canonical_record=record, evidence=[annotation]))


def test_duplicate_paths_and_bad_pointer_syntax_rejected(packet):
    annotation = ReferenceEvidence(path="/samples/0", status="uncertain", note="Needs review.")
    with pytest.raises(ValueError, match="duplicate_evidence"):
        replace(blank_submission(packet), evidence=[annotation, annotation])
    for path in ("samples", "/samples~2", "/samples~"):
        with pytest.raises(ValueError, match="json_pointer"):
            ReferenceEvidence(path=path, status="uncertain", note="Needs review.")
    with pytest.raises(ValueError, match="requires_evidence"):
        ReferenceEvidence(path="/samples/0", status="supported", note="Needs evidence.")
    with pytest.raises(ValueError, match="path_not_reviewable"):
        assess_reference(packet, replace(blank_submission(packet), evidence=[annotation]))


@pytest.mark.parametrize("seconds", [-1, float("inf"), float("nan"), True, "20"])
def test_elapsed_time_does_not_accept_invalid_or_coerced_values(seconds):
    with pytest.raises(ValueError):
        ReviewerDeclaration(identity="Invented reviewer", provenance="synthetic",
            prior_prediction_access="unknown", independence="not_assessed", elapsed_seconds=seconds)


def test_pdf_and_truncation_warnings_survive_declared_text_review(packet, record):
    text = packet.bundle.text
    bundle = bundle_document(Document(id="synthetic-pdf", text=text),
        source_id="tests/offline/fixtures/invented.pdf", source_bytes=b"synthetic-pdf-bytes",
        source_format="pdf", parser=ParserProvenance(name="synthetic", version="1"), max_chars=70)
    pdf_packet = build_packet(publication_id="invented-reference", split="calibration",
        data_kind="synthetic", bundle=bundle, profile=packet.profile)
    result = assess_reference(pdf_packet, complete_fixture(pdf_packet, record))
    assert result["status"] == "available_text_review_recorded"
    assert {w["code"] for w in result["source_warnings"]} >= {"images_unavailable", "truncated", "page_mapping_unavailable"}
    assert result["full_source_coverage"] == "not_established"


def test_inputs_remain_unchanged_and_assessment_is_json_serializable(packet, record):
    import json
    submission = complete_fixture(packet, record)
    original_packet, original_submission = packet.model_dump_json(), submission.model_dump_json()
    result = assess_reference(packet, submission)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert packet.model_dump_json() == original_packet
    assert submission.model_dump_json() == original_submission
    assert assess_reference(packet, submission) == result
