"""Split integrity and structural checks; no model, PDF download, or GPU needed."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import audit_finetune_data as audit
from scripts.corpus import load_corpus


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ["2503.18475", "2010.02965"]
REASONS = ["Longest paper requested by the user.", "Different technique and rich references."]


@pytest.fixture
def audit_result():
    corpus = load_corpus()
    calibration = [p for p in corpus["papers"] if p["split"] == "calibration"]
    return {
        "calibration": sorted(p["arxiv_id"] for p in calibration),
        "test": sorted(p["arxiv_id"] for p in corpus["papers"] if p["split"] == "evaluation"),
        "errors": [],
        "schema_versions": {"generic.json": "1.1.0", "photon_and_neutron.json": "1.1.0"},
        "scoring_plan_version": "1.1.3",
        "source_files": {"data/corpus.json": {"sha256": "original", "bytes": 10}},
        "length_method": {"version": "test"},
        "pdf_warnings": {},
        "calibration_inventory": [
            {
                "paper_id": p["arxiv_id"], "stratum": p["stratum"],
                "plain_text_chars": 100 if p["arxiv_id"] == VALIDATION[0] else 10,
                "pages": 1, "facilities": 1, "samples": 1, "datasets": 1,
                "sample_links": 1, "facility_links": 1,
            }
            for p in calibration
        ],
    }


def test_all_calibration_gold_passes_structural_checks():
    schema = {}
    for filename in audit.SCHEMAS:
        fields = audit.read_json(ROOT / "schemas" / filename)
        fields.pop("_meta")
        schema.update(fields)
    for paper in load_corpus()["papers"]:
        if paper["split"] != "calibration":
            continue
        gold = audit.read_json(ROOT / "data/gold" / f"{paper['arxiv_id']}.json")
        assert audit.validate_fields(gold, schema) == [], paper["arxiv_id"]
        assert audit.validate_references(gold) == [], paper["arxiv_id"]


@pytest.mark.parametrize(("value", "definition", "message"), [
    ({}, {"title": {"type": "string|null"}}, "missing field"),
    ({"extra": None}, {}, "unexpected field"),
    (3, {"type": "string|null"}, "expected string"),
    (1, {"type": "boolean"}, "expected boolean"),
    ("bad", {"type": "string", "constraints": {"allowed_values": ["good"]}}, "disallowed"),
    ("bad", {"type": "string", "constraints": {"pattern": "^10[.]"}}, "pattern"),
    ([], {"type": "list[string]", "cardinality": {"min": 1}}, "cardinality"),
    ([None], {"type": "list[string]"}, "expected string"),
])
def test_field_errors_are_detected(value, definition, message):
    assert any(message in error for error in audit.validate_fields(value, definition))


def test_nullable_string_and_false_are_valid():
    assert audit.validate_fields(None, {"type": "string|null"}) == []
    assert audit.validate_fields(False, {"type": "boolean"}) == []


@pytest.mark.parametrize("problem", ["dangling", "duplicate_id", "nonsequential", "repeated_ref"])
def test_reference_errors_are_detected(problem):
    gold = audit.read_json(ROOT / "data/gold/2010.02965.json")
    if problem == "dangling":
        gold["datasets"][0]["sample_refs"].append("sample_999")
    elif problem == "duplicate_id":
        gold["samples"][1]["id"] = gold["samples"][0]["id"]
    elif problem == "nonsequential":
        gold["samples"][0]["id"] = "sample_999"
    else:
        gold["datasets"][0]["sample_refs"] = ["sample_1", "sample_1"]
    assert audit.validate_references(gold)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"x": 1, "x": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        audit.read_json(path)


def test_duplicate_screening_normalizes_titles_and_ignores_missing_values():
    assert audit.normalized_title("A NEW: title!") == audit.normalized_title("A new title")
    assert audit.duplicate_groups({"a": "same", "b": "same", "c": "", "d": ""}) == [["a", "b"]]


def test_fingerprints_allow_text_line_endings_but_preserve_pdf_bytes(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b'{"a": 1}\r\n')
    windows = audit.fingerprint(path)
    path.write_bytes(b'{"a": 1}\n')
    assert audit.fingerprint(path) == windows
    path.write_bytes(b'{"a": 2}\n')
    assert audit.fingerprint(path) != windows
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"example\r\n")
    original = audit.fingerprint(pdf)
    pdf.write_bytes(b"example\n")
    assert audit.fingerprint(pdf) != original


def test_split_preserves_test_and_all_training_strata(audit_result):
    manifest = audit.build_manifest(audit_result, VALIDATION, REASONS)
    train, validation, test = [set(manifest[key]) for key in ("train", "validation", "test")]
    assert (len(train), len(validation), len(test)) == (8, 2, 30)
    assert train.isdisjoint(validation) and (train | validation).isdisjoint(test)
    assert train | validation == set(audit_result["calibration"])
    assert manifest["test"] == audit_result["test"]
    assert manifest["selection"]["refit_on_validation"] is False
    assert len({row["stratum"] for row in audit_result["calibration_inventory"] if row["paper_id"] in train}) == 7


@pytest.mark.parametrize(("validation", "message"), [
    (["2010.02965", "2304.08002"], "longest"),
    (["2503.18475", "2503.18475"], "distinct calibration"),
    (["2503.18475", "not_in_calibration"], "distinct calibration"),
    (["2503.18475", "1806.08116"], "removes a technique stratum"),
])
def test_invalid_validation_choice_rejected(audit_result, validation, message):
    with pytest.raises(ValueError, match=message):
        audit.build_manifest(audit_result, validation, REASONS)


def test_audit_errors_prevent_freezing(audit_result):
    audit_result["errors"] = ["invalid annotation"]
    with pytest.raises(ValueError, match="Audit errors"):
        audit.build_manifest(audit_result, VALIDATION, REASONS)


def test_freeze_verify_and_refuse_overwrite(tmp_path, audit_result):
    manifest = audit.build_manifest(audit_result, VALIDATION, REASONS)
    audit.freeze(tmp_path, manifest)
    before = (tmp_path / audit.MANIFEST).read_bytes()
    audit.verify(tmp_path, audit_result)
    with pytest.raises(ValueError, match="already exist"):
        audit.freeze(tmp_path, manifest)
    assert (tmp_path / audit.MANIFEST).read_bytes() == before
    changed = deepcopy(audit_result)
    changed["source_files"]["data/corpus.json"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        audit.verify(tmp_path, changed)


def test_tampered_membership_and_report_rejected(tmp_path, audit_result):
    manifest = audit.build_manifest(audit_result, VALIDATION, REASONS)
    audit.freeze(tmp_path, manifest)
    altered = deepcopy(manifest)
    altered["train"].pop()
    path = tmp_path / audit.MANIFEST
    original = path.read_text(encoding="utf-8")
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        audit.verify(tmp_path, audit_result)
    path.write_text(original, encoding="utf-8")
    (tmp_path / audit.REPORT).write_text("Changed report", encoding="utf-8")
    with pytest.raises(ValueError, match="report differs"):
        audit.verify(tmp_path, audit_result)


def test_default_cli_is_read_only(tmp_path, monkeypatch, audit_result):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "audit", lambda root: audit_result)
    assert audit.main([]) == 0
    assert list(tmp_path.iterdir()) == []
