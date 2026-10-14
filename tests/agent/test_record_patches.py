"""Invented patch diagnostics; mock judgments are not expert labels or truth."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
from pydantic import ValidationError

from extraction_fixtures import synthetic_bundle, synthetic_record
from meta_data_extraction.agent.contracts import canonical_json, digest, snapshot
from meta_data_extraction.agent.evidence import evidence_ref
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.agent.providers import MockProvider, ProviderResult
from meta_data_extraction.agent.record_patches import (
    PatchPolicy, PatchProposal, PatchReceipt, PreparedPatch, ReplaceScalar,
    apply_supported_patch, build_candidate, verify_candidate,
)
from meta_data_extraction.agent.support_checks import SupportTarget, check_support

DESCRIPTION = "/datasets/0/dataset_description"
SAMPLE_REF = "/datasets/0/sample_refs/0"
FACILITY_REF = "/datasets/0/facility_refs/0"
TECHNIQUE = "/datasets/0/measurement_technique"
ENABLED = PatchPolicy(accept_model_judgments=True)
TEXT = (
    "Invented M06 diagnostic. Sample A and Sample B both contain CaCO3 but are distinct specimens. "
    "Sample A was prepared at 500 K, and Sample B was prepared at 600 K. "
    "Dataset A measured Sample A only, using X-ray diffraction at 300 K at Synthetic Beam Source. "
    "Dataset B measured Sample B only at 250 K. These are fabricated fixture facts.\n"
)


@pytest.fixture
def inputs(workspace):
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "schemas/prompt_template.md", workspace / "schemas/prompt_template.md")
    profile = load_profile(workspace)
    bundle = synthetic_bundle(TEXT, source_id="synthetic-m06-patch.md")
    value = synthetic_record(profile)
    value["datasets"][0].update(dataset_description="Measured at 500 K", sample_refs=["sample_2"])
    record = snapshot(value, bundle, profile)
    return workspace, profile, bundle, record


def op(path=DESCRIPTION, old="Measured at 500 K", new="Measured at 300 K"):
    return ReplaceScalar(path=path, expected_value_json=canonical_json(old), replacement_value_json=canonical_json(new))


def proposal(record, operations=None):
    return PatchProposal(expected_revision=record.revision_id, operations=tuple(operations or [op()]),
                         provenance="synthetic_fixture", summary="Invented correction diagnostic")


def prepare(inputs, operations=None):
    _, profile, bundle, record = inputs
    return build_candidate(profile, bundle, record, proposal(record, operations))


def support(inputs, record, paths, *, verdict="supported", overrides=None, finish_reason="stop", refs=None, diagnoses=None):
    workspace, profile, bundle, _ = inputs
    overrides = overrides or {}
    reference = evidence_ref(bundle, bundle.regions[0].region_id)
    targets = tuple(SupportTarget(path=path, evidence=(reference,) if refs is None else refs) for path in paths)
    judgments = []
    for path in paths:
        state = overrides.get(path, verdict)
        diagnosis = "supported" if state == "supported" else (
            "wrong_sample" if "_refs/" in path else "preparation_acquisition_confusion"
        )
        if state == "insufficient_evidence":
            diagnosis = "missing_evidence"
        diagnosis = (diagnoses or {}).get(path, diagnosis)
        judgments.append(dict(path=path, verdict=state, diagnosis=diagnosis,
                              summary="Synthetic automated judgment, not independent evidence labels.", evidence_indices=[0]))
    provider = MockProvider([ProviderResult(status="ok", content=json.dumps({"judgments": judgments}),
                                           finish_reason=finish_reason)], model_name="mock-patch-diagnostic")
    return check_support(workspace, profile, record, bundle, targets, provider, raw_retention="retain")


def reports(inputs, prepared, *, before_verdict="contradicted", after_verdict="supported"):
    before = support(inputs, prepared.original, prepared.changed_paths, verdict=before_verdict)
    after = support(inputs, prepared.candidate, prepared.required_support_paths, verdict=after_verdict)
    return before, after


def apply(inputs, prepared, before, after, **kwargs):
    _, profile, bundle, current = inputs
    return apply_supported_patch(profile, bundle, current, prepared, before, after, policy=ENABLED, **kwargs)


def test_candidate_is_not_application_and_preserves_entire_original(inputs):
    _, profile, bundle, record = inputs
    original = record.model_dump()
    result = prepare(inputs)
    assert result.status == "ready"
    assert result.original.model_dump() == original and record.model_dump() == original
    assert result.prepared.candidate.canonical["datasets"][0]["dataset_description"] == "Measured at 300 K"
    assert result.prepared.required_support_paths == tuple(sorted((DESCRIPTION, SAMPLE_REF, FACILITY_REF)))
    assert result.prepared.candidate.entities == record.entities
    assert result.prepared.candidate.revision_id != record.revision_id
    verify_candidate(result.prepared, profile, bundle)


def test_atomic_two_operation_correction_rechecks_source_and_target_entities(inputs):
    _, _, _, original = inputs
    result = prepare(inputs, [op(), op(SAMPLE_REF, "sample_2", "sample_1")])
    prepared = result.prepared
    before, after = reports(inputs, prepared)
    outcome = apply(inputs, prepared, before, after)
    assert outcome.status == "applied"
    assert outcome.snapshot.canonical["datasets"][0]["sample_refs"] == ["sample_1"]
    assert original.canonical["datasets"][0]["sample_refs"] == ["sample_2"]
    assert outcome.snapshot.entities == original.entities
    assert len(outcome.snapshot.canonical["samples"]) == 2
    assert outcome.approval_status == "not_expert_approved"
    assert outcome.support_report.judgments[0].provenance == "mock_model_judgment"
    changed_reference = next(r for r in outcome.snapshot.references if r.path == SAMPLE_REF)
    judgment = next(j for j in after.judgments if j.path == SAMPLE_REF)
    assert judgment.entity_handle == changed_reference.source_handle
    assert judgment.target_handles == (changed_reference.target_handle,)


def test_all_old_assertions_invalidated_not_silently_rebound(inputs):
    prepared = prepare(inputs).prepared
    before, after = reports(inputs, prepared)
    outcome = apply(inputs, prepared, before, after)
    assert outcome.invalidated_assertion_revisions == (prepared.original.revision_id,)
    assert all(a.record_revision == outcome.snapshot.revision_id for a in outcome.assertions)
    assert all(a.support_status == "not_assessed" for a in outcome.assertions)
    old_title = next(a for a in outcome.assertions if a.path == "/bibliographic/title")
    assert old_title.evidence == () and old_title.link_status == "unlinked"
    new_description = next(a for a in outcome.assertions if a.path == DESCRIPTION)
    assert new_description.link_status == "location_verified"


@pytest.mark.parametrize("path", [
    "/datasets/0/id", "/datasets/0", "/datasets", "/samples", "/samples/0/sample_name",
    "/samples/0/chemical_composition", "/samples/0/sample_description", "/datasets/0/facility_refs/0",
    "/datasets/0/sample_refs", "/datasets/00/dataset_description", "/datasets/-1/dataset_description",
    "/datasets/0/sample_refs/-", "/bibliographic/title", "/datasets/0/measurement_mode/0",
])
def test_nonallowlisted_edits_never_modify_records(inputs, path):
    result = prepare(inputs, [op(path)])
    assert result.status == "rejected" and result.code == "path_not_allowlisted"
    assert result.original == inputs[3] and result.prepared is None


@pytest.mark.parametrize("bad", ["{}", "[]", '"text" ', "NaN", "Infinity", "null "])
def test_patch_values_require_canonical_scalar_json(bad):
    with pytest.raises(ValueError):
        ReplaceScalar(path=DESCRIPTION, expected_value_json=bad, replacement_value_json='"valid"')


@pytest.mark.parametrize("count", [0, 5])
def test_operation_count_bounded(inputs, count):
    with pytest.raises(ValidationError):
        PatchProposal(expected_revision=inputs[3].revision_id, operations=tuple(op() for _ in range(count)),
                      provenance="synthetic_fixture", summary="Bounded")


def test_missing_target_is_explicit(inputs):
    result = prepare(inputs, [op("/datasets/3/dataset_description")])
    assert result.code == "missing_target" and result.prepared is None


def test_stale_expected_revision_rejected_before_edit(inputs):
    _, profile, bundle, original = inputs
    stale = proposal(original).model_copy(update={"expected_revision": "revision:old"})
    result = build_candidate(profile, bundle, original, stale)
    assert result.status == "stale" and result.original == original


@pytest.mark.parametrize("operations,code", [
    ([op(old="other old value")], "old_value_mismatch"),
    ([op(new="Measured at 500 K")], "no_progress"),
    ([op(), op(new="Measured at 250 K")], "conflicting_operations"),
    ([op(), op(SAMPLE_REF, "sample_2", "sample_999")], "candidate_structure_invalid"),
    ([op(), op(TECHNIQUE, "X-ray diffraction", 3)], "candidate_structure_invalid"),
])
def test_all_operations_rollback_when_any_operation_rejected(inputs, operations, code):
    result = prepare(inputs, operations)
    assert result.status == "rejected" and result.code == code
    assert result.original == inputs[3] and result.prepared is None
    assert inputs[3].canonical["datasets"][0]["dataset_description"] == "Measured at 500 K"


@pytest.mark.parametrize("new", [None, "Measured at 300 K"])
def test_null_values_not_filled_or_deleted_by_correction_policy(inputs, new):
    _, profile, bundle, record = inputs
    value = record.canonical
    value["datasets"][0]["dataset_description"] = None if new is not None else "Measured at 500 K"
    original = snapshot(value, bundle, profile)
    result = build_candidate(profile, bundle, original, proposal(original, [op(old=value["datasets"][0]["dataset_description"], new=new)]))
    assert result.code == "missing_value_change_not_supported"
    assert result.original == original


def test_same_composition_samples_are_not_merged_and_duplicate_attachment_rejected(inputs):
    _, profile, bundle, original = inputs
    record = original.canonical
    record["datasets"][0]["sample_refs"] = ["sample_1", "sample_2"]
    original = snapshot(record, bundle, profile)
    result = build_candidate(profile, bundle, original, proposal(original, [op("/datasets/0/sample_refs/1", "sample_2", "sample_1")]))
    assert result.code == "duplicate_sample_reference"
    assert len(result.original.canonical["samples"]) == 2


def test_default_policy_does_not_authorize_model_judgment_edits(inputs):
    _, profile, bundle, record = inputs
    prepared = prepare(inputs).prepared
    before, after = reports(inputs, prepared)
    outcome = apply_supported_patch(profile, bundle, record, prepared, before, after)
    assert outcome.status == "rejected" and outcome.code == "model_judgment_policy_disabled"
    assert outcome.snapshot == record and outcome.receipt is None


def test_policy_lower_operation_budget_enforced(inputs):
    _, profile, bundle, record = inputs
    prepared = prepare(inputs, [op(), op(SAMPLE_REF, "sample_2", "sample_1")]).prepared
    before, after = reports(inputs, prepared)
    outcome = apply_supported_patch(profile, bundle, record, prepared, before, after,
                                    policy=PatchPolicy(accept_model_judgments=True, max_operations=1))
    assert outcome.code == "operation_budget_exceeded" and outcome.snapshot == record


@pytest.mark.parametrize("before_verdict", ["supported", "insufficient_evidence"])
def test_supported_or_unknown_old_assertion_cannot_be_overwritten(inputs, before_verdict):
    prepared = prepare(inputs).prepared
    before, after = reports(inputs, prepared, before_verdict=before_verdict)
    outcome = apply(inputs, prepared, before, after)
    assert outcome.code == "old_assertion_not_contradicted" and outcome.snapshot == inputs[3]


@pytest.mark.parametrize("missing", [DESCRIPTION, SAMPLE_REF, FACILITY_REF])
def test_every_changed_and_affected_reference_needs_fresh_support(inputs, missing):
    prepared = prepare(inputs).prepared
    before = support(inputs, prepared.original, prepared.changed_paths, verdict="contradicted")
    after = support(inputs, prepared.candidate, tuple(p for p in prepared.required_support_paths if p != missing))
    outcome = apply(inputs, prepared, before, after)
    assert outcome.code == "candidate_support_incomplete" and outcome.snapshot == inputs[3]


@pytest.mark.parametrize("path", [DESCRIPTION, SAMPLE_REF, FACILITY_REF])
def test_harmful_or_unsupported_candidate_blocks_whole_patch(inputs, path):
    prepared = prepare(inputs, [op(), op(SAMPLE_REF, "sample_2", "sample_1")]).prepared
    before = support(inputs, prepared.original, prepared.changed_paths, verdict="contradicted")
    after = support(inputs, prepared.candidate, prepared.required_support_paths, overrides={path: "contradicted"})
    outcome = apply(inputs, prepared, before, after)
    assert outcome.status == "rejected" and outcome.snapshot == inputs[3]
    assert outcome.receipt is None and outcome.assertions == ()


def test_old_report_cannot_authorize_candidate_even_if_values_occur_in_source(inputs):
    prepared = prepare(inputs).prepared
    before, _ = reports(inputs, prepared)
    old_support = support(inputs, prepared.original, prepared.required_support_paths)
    outcome = apply(inputs, prepared, before, old_support)
    assert outcome.code == "support_report_binding_invalid"


@pytest.mark.parametrize("before_side", [True, False])
def test_truncated_or_unknown_finished_check_never_applies(inputs, before_side):
    prepared = prepare(inputs).prepared
    before, after = reports(inputs, prepared)
    if before_side:
        before = support(inputs, prepared.original, prepared.changed_paths, verdict="contradicted", finish_reason="length")
    else:
        after = support(inputs, prepared.candidate, prepared.required_support_paths, finish_reason="length")
    outcome = apply(inputs, prepared, before, after)
    assert outcome.code == "support_check_incomplete" and outcome.snapshot == inputs[3]


def test_stale_current_record_rejected_even_if_patch_field_unchanged(inputs):
    workspace, profile, bundle, original = inputs
    prepared = prepare(inputs).prepared
    before, after = reports(inputs, prepared)
    value = original.canonical
    value["bibliographic"]["title"] = "A changed title"
    current = snapshot(value, bundle, profile)
    outcome = apply((workspace, profile, bundle, current), prepared, before, after)
    assert outcome.status == "stale" and outcome.snapshot == current


def test_application_and_receipt_replay_are_idempotent(inputs):
    _, profile, bundle, original = inputs
    prepared = prepare(inputs).prepared
    before, after = reports(inputs, prepared)
    first = apply(inputs, prepared, before, after)
    second = apply(inputs, prepared, before, after)
    assert second == first
    replay = apply_supported_patch(profile, bundle, first.snapshot, prepared, before, after,
                                   policy=ENABLED, receipt=first.receipt)
    assert replay.status == "replayed" and replay.snapshot == first.snapshot
    assert replay.receipt == first.receipt and replay.assertions == first.assertions
    assert original == prepared.original


def test_matching_replacement_without_receipt_is_not_idempotent_proof(inputs):
    _, profile, bundle, _ = inputs
    prepared = prepare(inputs).prepared
    before, after = reports(inputs, prepared)
    result = apply_supported_patch(profile, bundle, prepared.candidate, prepared, before, after, policy=ENABLED)
    assert result.status == "stale" and result.code == "replay_requires_receipt"


def test_receipt_binds_policy_and_full_reports(inputs):
    _, profile, bundle, original = inputs
    prepared = prepare(inputs).prepared
    before, after = reports(inputs, prepared)
    first = apply(inputs, prepared, before, after)
    result = apply_supported_patch(profile, bundle, first.snapshot, prepared, before, after,
                                   policy=PatchPolicy(accept_model_judgments=True, max_operations=3), receipt=first.receipt)
    assert result.code == "receipt_mismatch"
    altered = first.receipt.model_copy(update={"before_report_digest": "0" * 64})
    result = apply(inputs, prepared, before, after, receipt=altered)
    assert result.code == "receipt_invalid"


def test_recomputed_candidate_digest_cannot_omit_affected_relationships(inputs):
    _, profile, bundle, _ = inputs
    prepared = prepare(inputs).prepared
    payload = prepared.model_dump(mode="json", exclude={"candidate_id"})
    payload["required_support_paths"] = [DESCRIPTION]
    forged = PreparedPatch.model_validate({**payload, "candidate_id": "patch-candidate:" + digest(payload)})
    with pytest.raises(ValueError, match="no longer matches"):
        verify_candidate(forged, profile, bundle)
    before, after = reports(inputs, prepared)
    result = apply(inputs, forged, before, after)
    assert result.code == "invalid_candidate" and result.snapshot == inputs[3]


def test_source_or_profile_change_invalidates_prepared_candidate(inputs):
    _, profile, bundle, _ = inputs
    prepared = prepare(inputs).prepared
    changed = synthetic_bundle(TEXT + "Changed context.", source_id=bundle.source_id)
    with pytest.raises(ValueError):
        verify_candidate(prepared, profile, changed)


def test_approval_cannot_be_fabricated_in_policy_or_receipt(inputs):
    with pytest.raises(ValidationError):
        PatchPolicy(accept_model_judgments=True, approval_status="expert_approved")
    prepared = prepare(inputs).prepared
    before, after = reports(inputs, prepared)
    receipt = apply(inputs, prepared, before, after).receipt
    with pytest.raises(ValidationError):
        PatchReceipt.model_validate({**receipt.model_dump(), "approval_status": "expert_approved"})


def test_conflicting_source_context_requires_review_even_when_contradicted(inputs):
    prepared = prepare(inputs).prepared
    before = support(inputs, prepared.original, prepared.changed_paths, verdict="contradicted",
                     diagnoses={DESCRIPTION: "context_conflict"})
    after = support(inputs, prepared.candidate, prepared.required_support_paths)
    result = apply(inputs, prepared, before, after)
    assert result.code == "source_conflict_requires_review" and result.snapshot == inputs[3]


def test_missing_evidence_never_supports_a_patch(inputs):
    prepared = prepare(inputs).prepared
    before = support(inputs, prepared.original, prepared.changed_paths, verdict="contradicted")
    after = support(inputs, prepared.candidate, prepared.required_support_paths, refs=())
    result = apply(inputs, prepared, before, after)
    assert result.status == "rejected" and result.snapshot == inputs[3]


def test_forged_evidence_and_attachment_handles_rejected(inputs):
    prepared = prepare(inputs, [op(SAMPLE_REF, "sample_2", "sample_1")]).prepared
    before, after = reports(inputs, prepared)
    target = next(j for j in after.judgments if j.path == SAMPLE_REF)
    forged = target.model_copy(update={"target_handles": ("entity:wrong",)})
    altered = after.model_copy(update={"judgments": tuple(forged if j.path == SAMPLE_REF else j for j in after.judgments)})
    result = apply(inputs, prepared, before, altered)
    assert result.code == "support_report_binding_invalid" and result.snapshot == inputs[3]


def test_stale_or_rejected_result_cannot_carry_receipt_or_report(inputs):
    from meta_data_extraction.agent.record_patches import PatchResult
    prepared = prepare(inputs).prepared
    before, after = reports(inputs, prepared)
    success = apply(inputs, prepared, before, after)
    for extra in ({"receipt": success.receipt}, {"support_report": after}):
        with pytest.raises(ValidationError):
            PatchResult(status="rejected", code="example", snapshot=inputs[3], **extra)


def test_reference_array_reordering_is_not_an_attachment_correction(inputs):
    _, profile, bundle, record = inputs
    value = record.canonical
    value["datasets"][0]["sample_refs"] = ["sample_1", "sample_2"]
    original = snapshot(value, bundle, profile)
    patch = proposal(original, [op("/datasets/0/sample_refs/0", "sample_1", "sample_2"),
                                op("/datasets/0/sample_refs/1", "sample_2", "sample_1")])
    result = build_candidate(profile, bundle, original, patch)
    assert result.code == "reference_reorder_not_supported" and result.original == original


def test_derived_relationship_support_targets_cannot_exceed_checker_budget(inputs):
    _, profile, bundle, record = inputs
    value = record.canonical
    base = value["samples"][0]
    value["samples"] = [{**base, "id": f"sample_{i}", "sample_name": f"Specimen {i}"} for i in range(1, 34)]
    value["datasets"][0]["sample_refs"] = [sample["id"] for sample in value["samples"]]
    original = snapshot(value, bundle, profile)
    result = build_candidate(profile, bundle, original, proposal(original))
    assert result.code == "support_target_budget_exceeded" and result.original == original
