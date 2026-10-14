"""Synthetic-only checks for the isolated, unchanged inherited scorer."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import socket

import pytest

from meta_data_extraction import evaluation
from meta_data_extraction.offline import scoring


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Offline scorer tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


@pytest.fixture
def frozen():
    return scoring.freeze_evaluator(ROOT)


@pytest.fixture
def pair(tmp_path):
    record = {
        "bibliographic": {"title": "A synthetic paper", "authors": [], "affiliations": [], "funding": []},
        "document_overview": {"section_headings": [], "has_references": False, "has_figures": False, "has_tables": False},
        "samples": [{"id": "sample_1", "sample_name": "alpha specimen", "chemical_composition": "TiN"}],
        "facilities": [],
        "datasets": [{"id": "dataset_1", "dataset_name": "alpha diffraction", "measurement_technique": "XRD", "sample_refs": ["sample_1"], "facility_refs": []}],
    }
    gold, pred = tmp_path / "synthetic-paper.json", tmp_path / "prediction.json"
    gold.write_text(json.dumps(record), encoding="utf8")
    predicted = copy.deepcopy(record)
    predicted["samples"][0]["id"] = "pred_sample_7"
    predicted["datasets"][0]["id"] = "pred_dataset_8"
    predicted["datasets"][0]["sample_refs"] = ["pred_sample_7"]
    pred.write_text(json.dumps({"document_id": "synthetic-paper", "output": predicted}), encoding="utf8")
    return gold, pred


def test_freeze_pins_code_helper_schemas_and_installed_dependencies(frozen):
    names = {item.path for item in frozen.files}
    assert names >= {
        "src/meta_data_extraction/evaluation.py", "src/meta_data_extraction/alignment.py",
        "src/meta_data_extraction/normalization.py", "src/meta_data_extraction/config_validation.py",
        "src/meta_data_extraction/utils.py", "configs/evaluation_metrics.yaml",
        "configs/alignment_helper.yaml", "schemas/generic.json", "schemas/photon_and_neutron.json", "uv.lock",
    }
    assert frozen.metadata["plan_version"] == "1.1.3"
    assert frozen.metadata["alignment_helper_version"] == "1.1.3"
    assert frozen.metadata["schema_version_mismatch"] is False
    assert {item.name for item in frozen.dependencies} == {"PyYAML", "numpy", "scipy", "pydantic"}
    assert all(len(item.sha256) == 64 for item in frozen.files)
    assert scoring.EvaluatorSnapshot.model_validate_json(frozen.model_dump_json()) == frozen
    scoring.verify_evaluator(frozen, ROOT)


def test_snapshot_metadata_cannot_be_mutated_through_property(frozen):
    original = frozen.metadata
    original["plan_version"] = "changed"
    assert frozen.metadata["plan_version"] == "1.1.3"


def test_resource_edit_without_version_bump_invalidates_snapshot(frozen, monkeypatch):
    real = scoring._fingerprint

    def changed(root, name):
        value = real(root, name)
        return value.model_copy(update={"sha256": "0" * 64}) if name == "configs/alignment_helper.yaml" else value

    monkeypatch.setattr(scoring, "_fingerprint", changed)
    assert scoring.freeze_evaluator(ROOT).metadata == frozen.metadata
    with pytest.raises(scoring.EvaluatorChangedError, match="evaluator_snapshot_changed"):
        scoring.verify_evaluator(frozen, ROOT)


def test_installed_dependency_change_invalidates_snapshot(frozen, monkeypatch):
    monkeypatch.setattr(scoring, "_dependencies", lambda: (scoring.DependencyVersion(name="scipy", version="changed"),))
    with pytest.raises(scoring.EvaluatorChangedError, match="evaluator_snapshot_changed"):
        scoring.verify_evaluator(frozen, ROOT)


def test_evaluator_imported_from_other_checkout_is_rejected(frozen, monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "__file__", str(tmp_path / "evaluation.py"))
    with pytest.raises(scoring.EvaluatorChangedError, match="evaluator_module_origin_mismatch"):
        scoring.verify_evaluator(frozen, ROOT)


def test_helper_global_repointing_is_rejected(frozen, monkeypatch, tmp_path):
    monkeypatch.setattr(scoring.alignment, "_ALIGNMENT_HELPER_PATH", tmp_path / "other.yaml")
    with pytest.raises(scoring.EvaluatorChangedError, match="alignment_helper_origin_mismatch"):
        scoring.verify_evaluator(frozen, ROOT)


def test_tampered_snapshot_is_rejected(frozen):
    payload = frozen.model_dump(mode="json")
    payload["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        scoring.EvaluatorSnapshot.model_validate_json(json.dumps(payload))


def test_evaluate_pair_is_exact_inherited_report_with_unchanged_inputs(pair, frozen):
    gold, pred = pair
    before = [path.read_bytes() for path in pair]
    actual = scoring.evaluate_pair(ROOT, gold, pred, frozen)
    assert actual == evaluation.evaluate(gold, pred, ROOT / "configs/evaluation_metrics.yaml")
    assert actual["fields"]["datasets.attrs.sample_refs"] == {"tp": 1, "fp": 0, "fn": 0, "f1": 1.0, "n_pairs": 1}
    assert [path.read_bytes() for path in pair] == before


def test_separate_identity_path_preserved(pair, frozen):
    gold, pred = pair
    report = scoring.evaluate_pair(ROOT, gold, pred, frozen, identity_gold=gold)
    assert all(value == {"path": str(gold)} for value in report["identity_sources"].values())


@pytest.mark.parametrize("raw,code", [
    ("invalid JSON", "invalid_json_record"), ("[]", "empty_or_non_object_record"),
    ("{}", "empty_or_non_object_record"), ("null", "empty_or_non_object_record"),
    ('{"document_id":"synthetic","output":null}', "empty_or_non_object_record"),
    ('{"document_id":"synthetic","output":{}}', "empty_or_non_object_record"),
    ('{"a":1,"a":2}', "duplicate_json_key"), ('{"value":NaN}', "non_finite_json_value"),
    ('{"value":1e999}', "non_finite_json_value"), ('{"value":-1e999}', "non_finite_json_value"),
])
def test_invalid_prediction_is_explicit_failure(pair, frozen, raw, code):
    gold, pred = pair
    pred.write_text(raw, encoding="utf8")
    with pytest.raises(scoring.ScoringError, match=code):
        scoring.evaluate_pair(ROOT, gold, pred, frozen)


def test_missing_prediction_is_explicit_failure(pair, frozen):
    gold, pred = pair
    with pytest.raises(scoring.ScoringError, match="invalid_scoring_input_file"):
        scoring.evaluate_pair(ROOT, gold, pred.with_name("absent.json"), frozen)


def test_changed_input_during_evaluation_is_rejected(pair, frozen, monkeypatch):
    real = evaluation.evaluate

    def changed(gold, pred, **kwargs):
        report = real(gold, pred, **kwargs)
        pred.write_text('{"bibliographic":{"title":"changed"}}', encoding="utf8")
        return report

    monkeypatch.setattr(evaluation, "evaluate", changed)
    with pytest.raises(scoring.ScoringError, match="scoring_inputs_changed"):
        scoring.evaluate_pair(ROOT, *pair, frozen)


def test_evaluator_checked_after_scoring(pair, frozen, monkeypatch):
    real = evaluation.evaluate

    def changed(gold, pred, **kwargs):
        report = real(gold, pred, **kwargs)
        monkeypatch.setattr(scoring, "_dependencies", lambda: ())
        return report

    monkeypatch.setattr(evaluation, "evaluate", changed)
    with pytest.raises(scoring.EvaluatorChangedError, match="evaluator_snapshot_changed"):
        scoring.evaluate_pair(ROOT, *pair, frozen)


def test_inherited_exception_is_safe_explicit_failure(pair, frozen, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("private record content must not appear in status")

    monkeypatch.setattr(evaluation, "evaluate", broken)
    with pytest.raises(scoring.ScoringError) as caught:
        scoring.evaluate_pair(ROOT, *pair, frozen)
    assert str(caught.value) == "inherited_evaluation_failed"


def test_parity_ignores_only_observed_path_fields(pair, frozen):
    report = scoring.evaluate_pair(ROOT, *pair, frozen)
    reference = copy.deepcopy(report)
    reference.update(gold="old/gold.json", pred="old/pred.json", plan="old/plan.yaml")
    reference["evaluator"]["plan_file"] = "old/plan.yaml"
    for identity in reference["identity_sources"].values():
        identity["path"] = "old/gold.json"
    before = copy.deepcopy(reference)
    assert scoring.parity(report, reference).status == "matched"
    assert reference == before
    reference["identity_sources"]["samples"]["unobserved_metadata"] = "new"
    assert scoring.parity(report, reference).differences == ("/identity_sources/samples/unobserved_metadata",)


def test_empty_objects_cannot_prove_parity():
    with pytest.raises(scoring.ScoringError, match="invalid_report_shape"):
        scoring.parity({}, {})


@pytest.mark.parametrize("change,pointer", [
    (lambda r: r["fields"]["samples"].update(tp=0), "/fields/samples/tp"),
    (lambda r: r["fields"]["bibliographic.title"].update(score=0.9999999999999999), "/fields/bibliographic.title/score"),
    (lambda r: r["evaluator"].update(alignment_helper_version="old"), "/evaluator/alignment_helper_version"),
])
def test_parity_detects_counts_exact_float_and_version_changes(pair, frozen, change, pointer):
    report = scoring.evaluate_pair(ROOT, *pair, frozen)
    reference = copy.deepcopy(report)
    change(reference)
    result = scoring.parity(report, reference)
    assert result.status == "mismatch" and pointer in result.differences


def test_summary_separates_recovery_attributes_references_and_support(pair, frozen):
    report = scoring.evaluate_pair(ROOT, *pair, frozen)
    rows = {row.field: row for row in scoring.summaries([report], frozen)}
    expected = evaluation.aggregate_reports([report])
    assert {name: row.metrics for name, row in rows.items()} == expected
    assert rows["samples"].category == "entity_recovery"
    assert rows["samples"].support_count == 1
    assert rows["samples.attrs.sample_mass"].category == "conditional_attributes"
    assert rows["samples.attrs.sample_mass"].support_status == "zero"
    assert rows["samples.attrs.sample_mass"].n_pairs == 1
    assert rows["datasets.attrs.sample_refs"].category == "references"
    assert rows["datasets.attrs.sample_refs"].support_count == 1
    assert rows["datasets.attrs.sample_refs"].n_pairs == 1
    assert rows["bibliographic.subtitle"].metrics["mean_score"] == 1.0
    assert rows["bibliographic.subtitle"].support_status == "zero"
    assert rows["bibliographic.title"].support_status == "known"
    assert rows["bibliographic.authors.attrs.full_name"].n_pairs == 0
    assert rows["bibliographic.authors.attrs.full_name"].support_count == 0
    assert "mean_score" not in rows["bibliographic.authors.attrs.full_name"].metrics


def test_conditional_support_sums_pairs_and_preserves_null_agreements(frozen):
    reports = [
        {"fields": {"samples.attrs.sample_mass": {"mean_score": 1.0, "n_pairs": 2, "n_pairs_non_null": 0}}},
        {"fields": {"samples.attrs.sample_mass": {"mean_score": 0.0, "n_pairs": 1, "n_pairs_non_null": 1}}},
    ]
    row, = scoring.summaries(reports, frozen)
    assert row.metrics["mean_score"] == 2 / 3
    assert row.support_count == 1 and row.n_pairs == 3 and row.n_papers == 2


def test_support_unknown_is_not_zero(frozen):
    reports = [{"fields": {"samples.attrs.sample_mass": {"mean_score": 0.5, "n_pairs": 2}}}]
    row, = scoring.summaries(reports, frozen)
    assert row.support_status == "unknown" and row.support_count is None


def test_references_count_gold_links_only_in_matched_parent_pairs(frozen):
    reports = [
        {"fields": {"datasets.attrs.sample_refs": {"tp": 1, "fp": 2, "fn": 3, "f1": 2 / 7, "n_pairs": 2}}},
        {"fields": {"datasets.attrs.sample_refs": {"n_pairs": 0}}},
    ]
    row, = scoring.summaries(reports, frozen)
    assert row.category == "references" and row.support_unit == "gold_references"
    assert row.support_count == 4 and row.n_pairs == 2


def test_empty_report_population_has_no_invented_scores(frozen):
    assert scoring.summaries([], frozen) == ()


def test_unknown_report_field_is_rejected(frozen):
    with pytest.raises(scoring.ScoringError, match="report_field_not_in_frozen_plan"):
        scoring.summaries([{"fields": {"schema_wide_average": {"score": 1}}}], frozen)


def test_metric_classification_comes_from_compiled_plan():
    plan = evaluation._compile_plan({"scoring_plan": {
        "objects": {"metric": "entity", "key_attrs": ["name"], "attrs": {"oddly_named": {"metric": "reference", "target": "objects"}}},
    }})
    by_name = {item.field: item for item in scoring._field_metrics(plan)}
    assert by_name["objects.attrs.oddly_named"].metric == "reference"
    assert by_name["objects.attrs.oddly_named"].parent_entity == "objects"
