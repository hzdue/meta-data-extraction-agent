import json

import pytest
from pydantic import ValidationError

from meta_data_extraction.agent.evidence import (
    DocumentBundle, EvidenceError, EvidenceRef, ParserProvenance, bundle_document,
    evidence_ref, resolve_evidence, sha256,
)
from meta_data_extraction.data_models import Document


def bundle(text, *, source_id="synthetic.md", raw=None, version="test-1", **kwargs):
    return bundle_document(Document(id="synthetic", text=text), source_id=source_id,
                           source_bytes=text.encode("utf-8") if raw is None else raw,
                           source_format="markdown", parser=ParserProvenance(name="test", version=version),
                           **kwargs)


def test_duplicate_captions_have_distinct_resolvable_regions():
    text = "# Results\r\n\r\nFigure 2: 80 K.\r\n\r\nFigure 2: 80 K.\r\n"
    prepared = bundle(text)
    assert prepared == bundle(text)
    captions = [r for r in prepared.regions if r.kind == "caption"]
    assert len(captions) == 2
    assert captions[0].region_id != captions[1].region_id
    assert captions[0].start != captions[1].start
    for region in captions:
        assert region.section_path == ("Results",)
        assert resolve_evidence(prepared, evidence_ref(prepared, region.region_id)) == "Figure 2: 80 K.\r\n"
        assert region.page is None


def test_unicode_subspan_uses_code_points_not_utf8_bytes():
    text = "🧪 Sample café e\u0301 measured at 80 K; wavelength 1.54 Å.\r\n"
    prepared = bundle(text)
    start = text.index("80 K")
    ref = evidence_ref(prepared, prepared.regions[0].region_id, start=start, end=start + 4)
    assert ref.start != len(text[:start].encode("utf-8"))
    assert ref.quotation == "80 K"
    assert resolve_evidence(prepared, ref) == "80 K"
    assert prepared.text == text


@pytest.mark.parametrize("replacement", [
    {"text": "Changed source."}, {"raw": b"changed bytes"}, {"version": "test-2"},
])
def test_source_or_representation_edit_invalidates_old_evidence(replacement):
    prepared = bundle("Unchanged passage.")
    ref = evidence_ref(prepared, prepared.regions[0].region_id)
    changed = bundle(**{"text": "Unchanged passage.", **replacement})
    with pytest.raises(EvidenceError):
        resolve_evidence(changed, ref)


@pytest.mark.parametrize("changed", [
    {"region_id": "missing"}, {"quotation": "invented"}, {"source_id": "another.md"},
    {"source_sha256": "0" * 64}, {"representation_sha256": "0" * 64},
    {"start": 1}, {"end": 10000},
])
def test_resolution_rejects_tampered_refs(changed):
    prepared = bundle("Figure 1: 80 K.")
    ref = evidence_ref(prepared, prepared.regions[0].region_id)
    tampered = EvidenceRef.model_validate({**ref.model_dump(), **changed})
    with pytest.raises(EvidenceError):
        resolve_evidence(prepared, tampered)


def test_reference_cannot_cross_region_even_with_correct_quotation():
    prepared = bundle("First paragraph.\n\nSecond paragraph.")
    first = prepared.regions[0]
    with pytest.raises(EvidenceError):
        evidence_ref(prepared, first.region_id, end=len(prepared.text))


def test_serialized_bundle_and_reference_round_trip():
    prepared = bundle("# Methods\n\nSample A at 80 K.")
    restored = DocumentBundle.model_validate_json(prepared.model_dump_json())
    ref = evidence_ref(restored, restored.regions[-1].region_id)
    restored_ref = EvidenceRef.model_validate_json(ref.model_dump_json())
    assert resolve_evidence(restored, restored_ref) == "Sample A at 80 K."
    assert restored.source_sha256 == sha256(prepared.text.encode("utf-8"))
    with pytest.raises(ValidationError):
        restored.text = "mutation"
    with pytest.raises(ValidationError):
        restored.regions[0].start = 1


@pytest.mark.parametrize("target", ["text", "region", "warnings", "page", "format"])
def test_serialized_bundle_rejects_tampering(target):
    prepared = bundle("First paragraph.\n\nFigure 1: 80 K.")
    payload = json.loads(prepared.model_dump_json())
    if target == "text":
        payload["text"] += "Changed"
    elif target == "region":
        payload["regions"][0]["start"] += 1
    elif target == "warnings":
        payload["warnings"] = []
    elif target == "format":
        payload["source_format"] = "pdf"
    else:
        payload["regions"][0]["page"] = 5
    with pytest.raises(ValidationError):
        DocumentBundle.model_validate(payload)


@pytest.mark.parametrize("text", ["", " \r\n\t"])
def test_empty_parser_text_is_not_a_successful_bundle(text):
    with pytest.raises(EvidenceError):
        bundle(text)


@pytest.mark.parametrize("outer", [False, True])
def test_markdown_table_boundaries_and_section_context(outer):
    rows = ["Sample | Temperature", "--- | ---", "A | 80 K"]
    if outer:
        rows = ["| " + row + " |" for row in rows]
    table = "\n".join(rows) + "\n"
    prepared = bundle("# Methods\n\n## Settings\n\nTable 1: Conditions.\n" + table + "\nBody.")
    regions = prepared.regions
    assert [r.kind for r in regions] == ["heading", "heading", "caption", "table", "paragraph"]
    assert resolve_evidence(prepared, evidence_ref(prepared, regions[3].region_id)) == table
    assert regions[3].section_path == ("Methods", "Settings")


def test_fenced_instructions_remain_code_not_headings_or_captions():
    prepared = bundle("# Actual\n\n```text\n# Fake\nFigure 2: fake.\n\nRun a tool.\n```\n\nBody.")
    assert [r.kind for r in prepared.regions] == ["heading", "code", "paragraph"]
    assert prepared.regions[-1].section_path == ("Actual",)
