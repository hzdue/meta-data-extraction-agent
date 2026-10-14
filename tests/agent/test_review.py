"""Pure M08 review tests with explicitly simulated reviewer decisions only."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from extraction_fixtures import synthetic_bundle, synthetic_record
from meta_data_extraction.agent.contracts import canonical_json, digest, snapshot
from meta_data_extraction.agent.evidence import evidence_ref
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.agent.record_patches import PatchProposal, ReplaceScalar, build_candidate
from meta_data_extraction.agent.review import apply_review, initialize_review, required_review_paths, verify_review_state
from meta_data_extraction.agent.review_types import ReviewAttestation, ReviewError, ReviewResponse, ReviewState
from meta_data_extraction.agent.support_checks import SupportTarget, _prepare
from meta_data_extraction.agent.workflow_types import ReviewCase, WorkflowIssue, WorkflowState

DESCRIPTION = "/datasets/0/dataset_description"
REFERENCE = "/datasets/0/sample_refs/0"
TECHNIQUE = "/datasets/0/measurement_technique"
FACILITY = "/datasets/0/facility_refs/0"
TEXT = ("Explicitly invented M08 engineering fixture. Sample A and Sample B are distinct CaCO3 specimens. "
        "Sample A was prepared at 500 K. Dataset A measured Sample A at 300 K with X-ray diffraction "
        "at Synthetic Beam Source. Sample B belongs to Dataset B.\n")


@pytest.fixture
def review_case(workspace):
    shutil.copyfile(Path(__file__).resolve().parents[2] / "schemas/prompt_template.md",
                    workspace / "schemas/prompt_template.md")
    profile = load_profile(workspace)
    bundle = synthetic_bundle(TEXT)
    canonical = synthetic_record(profile)
    canonical["datasets"][0].update(dataset_description="Measured at 500 K", sample_refs=["sample_2"])
    record = snapshot(canonical, bundle, profile)
    ref = evidence_ref(bundle, bundle.regions[0].region_id)
    issues = tuple(WorkflowIssue(id=f"issue-{i}", path=path, diagnosis="context_conflict",
                                summary="Synthetic question requiring explicit review", query="Dataset A",
                                status="pending_review", evidence=(ref,))
                   for i, path in enumerate((DESCRIPTION, REFERENCE, TECHNIQUE)))
    pending = tuple(ReviewCase(case_id="review:" + digest([record.revision_id, issue.id]),
                              issue_id=issue.id, record_revision=record.revision_id, path=issue.path,
                              summary=issue.summary, evidence=issue.evidence) for issue in issues)
    workflow = WorkflowState(initial_revision=record.revision_id, record=record, issues=issues,
                             pending_reviews=pending, status="pending_review")
    proposal = PatchProposal(expected_revision=record.revision_id, operations=(
        ReplaceScalar(path=DESCRIPTION, expected_value_json=canonical_json("Measured at 500 K"),
                      replacement_value_json=canonical_json("Measured at 300 K")),
        ReplaceScalar(path=REFERENCE, expected_value_json=canonical_json("sample_2"),
                      replacement_value_json=canonical_json("sample_1")),
    ), provenance="synthetic_fixture", summary="Invented paired correction")
    state = initialize_review(workflow, profile, bundle, {"paired": proposal})
    return profile, bundle, state, proposal


def response_for(case, *, state=None, outcome="approve", selected=(0,), key="response-1", proposal=None):
    profile, bundle, original, _ = case
    state = state or original
    return ReviewResponse(idempotency_key=key, expected_state_id=state.state_id,
                          expected_revision=state.current.revision_id,
                          case_ids=tuple(state.cases[i].case_id for i in selected), outcome=outcome,
                          provenance="simulated", reviewer_id="simulated-unit-test-reviewer",
                          rationale="Fabricated reviewer decision to test engineering behavior only.",
                          proposal=proposal)


def completed_response(case, **kwargs):
    profile, bundle, original, _ = case
    state = kwargs.get("state") or original
    response = response_for(case, **kwargs)
    paths = required_review_paths(state, response, profile, bundle)
    candidate = state.current if response.proposal is None else build_candidate(
        profile, bundle, state.current, response.proposal).prepared.candidate
    ref = evidence_ref(bundle, bundle.regions[0].region_id)
    bindings = _prepare(candidate, bundle, tuple(SupportTarget(path=path, evidence=(ref,)) for path in paths))
    attestations = tuple(ReviewAttestation(path=b.path, value_json=b.value_json, role=b.role, evidence=b.evidence)
                         for b in bindings)
    return response.model_copy(update={"attestations": attestations})


def test_initialization_preserves_parent_without_approval(review_case):
    profile, bundle, state, _ = review_case
    assert state.current == state.initial_workflow.record == state.original
    assert state.status == "pending" and not state.decisions and not state.current_assertions
    assert all(case.status == "pending" and case.review_provenance is None for case in state.cases)
    assert state.proposals[0].key == "paired" and state.expert_authentication == "not_authenticated"
    verify_review_state(state, profile, bundle)


def test_scope_requires_exact_dataset_and_relationship_attestations(review_case):
    profile, bundle, state, _ = review_case
    response = response_for(review_case)
    assert required_review_paths(state, response, profile, bundle) == tuple(sorted((DESCRIPTION, REFERENCE, FACILITY)))
    with pytest.raises(ReviewError, match="attestation_scope_mismatch"):
        apply_review(state, response, profile, bundle)


def test_approve_current_resolves_only_selected_case_without_mutation(review_case):
    profile, bundle, state, _ = review_case
    response = completed_response(review_case)
    reviewed = apply_review(state, response, profile, bundle)
    assert reviewed.current == state.current and reviewed.initial_workflow == state.initial_workflow
    assert reviewed.cases[0].status == "approved" and all(c.status == "pending" for c in reviewed.cases[1:])
    assert reviewed.status == "pending" and reviewed.cases[0].review_provenance == "simulated"
    assert reviewed.decisions[0].response == response and reviewed.decisions[0].invalidated_revision is None
    assert all(a.record_revision == reviewed.current.revision_id and a.provenance == "simulated"
               and a.expert_authentication == "not_authenticated" for a in reviewed.current_assertions)
    verify_review_state(reviewed, profile, bundle)


def test_explicit_paired_edit_reuses_narrow_candidate_and_preserves_history(review_case):
    profile, bundle, state, proposal = review_case
    response = completed_response(review_case, outcome="edit", selected=(0, 1), proposal=proposal)
    reviewed = apply_review(state, response, profile, bundle)
    assert reviewed.current.canonical["datasets"][0]["dataset_description"] == "Measured at 300 K"
    assert reviewed.current.canonical["datasets"][0]["sample_refs"] == ["sample_1"]
    assert state.current.canonical["datasets"][0]["sample_refs"] == ["sample_2"]
    assert reviewed.current.entities == state.current.entities
    assert len(reviewed.current.canonical["samples"]) == 2
    assert all(c.status == "approved" for c in reviewed.cases[:2])
    assert reviewed.cases[2].status == "pending"
    assert reviewed.decisions[-1].invalidated_revision == state.current.revision_id
    assert reviewed.decisions[-1].changed_paths == (DESCRIPTION, REFERENCE)
    assert all(a.record_revision == reviewed.current.revision_id for a in reviewed.current_assertions)
    verify_review_state(reviewed, profile, bundle)


def test_edit_invalidates_prior_approval_and_reopens_unselected_cases(review_case):
    profile, bundle, state, proposal = review_case
    first = completed_response(review_case, selected=(2,))
    reviewed = apply_review(state, first, profile, bundle)
    assert reviewed.cases[2].status == "approved"
    second = completed_response(review_case, state=reviewed, outcome="edit", selected=(0, 1), key="response-2", proposal=proposal)
    changed = apply_review(reviewed, second, profile, bundle)
    assert changed.cases[2].status == "pending"
    assert changed.cases[2].last_decision_id is changed.cases[2].review_provenance is None
    assert all(a.path != TECHNIQUE for a in changed.current_assertions)
    assert changed.decisions[0] == reviewed.decisions[0] and len(changed.decisions) == 2
    assert changed.initial_workflow == state.initial_workflow
    verify_review_state(changed, profile, bundle)


def test_edit_reopens_previously_unresolved_other_case(review_case):
    profile, bundle, state, proposal = review_case
    first = response_for(review_case, outcome="unresolved", selected=(2,))
    reviewed = apply_review(state, first, profile, bundle)
    second = completed_response(review_case, state=reviewed, outcome="edit", selected=(0, 1), key="response-2", proposal=proposal)
    changed = apply_review(reviewed, second, profile, bundle)
    assert changed.cases[2].status == "pending" and reviewed.cases[2].status == "unresolved"


def test_unresolved_is_retained_without_approval(review_case):
    profile, bundle, state, _ = review_case
    response = response_for(review_case, outcome="unresolved", selected=(0, 1, 2))
    reviewed = apply_review(state, response, profile, bundle)
    assert reviewed.status == "unresolved" and reviewed.current == state.current
    assert all(c.status == "unresolved" for c in reviewed.cases)
    assert reviewed.current_assertions == () and reviewed.decisions[-1].bindings == ()
    verify_review_state(reviewed, profile, bundle)


def test_approve_all_means_selected_cases_only_not_gold(review_case):
    profile, bundle, state, _ = review_case
    reviewed = apply_review(state, completed_response(review_case, selected=(0, 1, 2)), profile, bundle)
    assert reviewed.status == "completed" and reviewed.coverage_status == "selected_assertions_only"
    assert reviewed.expert_authentication == "not_authenticated"
    assert all(c.review_provenance == "simulated" for c in reviewed.cases)


@pytest.mark.parametrize("field,value,code", [
    ("expected_state_id", "review-state:" + "0" * 64, "stale_state"),
    ("expected_revision", "revision:" + "0" * 64, "stale_revision"),
    ("case_ids", ("unknown",), "unknown_case"),
])
def test_optimistic_binding_rejects_without_effect(review_case, field, value, code):
    profile, bundle, state, _ = review_case
    response = completed_response(review_case).model_copy(update={field: value})
    before = state.model_dump()
    with pytest.raises(ReviewError, match=code): apply_review(state, response, profile, bundle)
    assert state.model_dump() == before


def test_same_revision_changed_state_is_stale(review_case):
    profile, bundle, state, _ = review_case
    first = completed_response(review_case)
    stale = completed_response(review_case, selected=(2,), key="second")
    reviewed = apply_review(state, first, profile, bundle)
    assert reviewed.current.revision_id == state.current.revision_id
    with pytest.raises(ReviewError, match="stale_state"):
        apply_review(reviewed, stale, profile, bundle)


def test_duplicate_and_conflicting_delivery_left_for_transaction_store(review_case):
    profile, bundle, state, _ = review_case
    first = completed_response(review_case)
    reviewed = apply_review(state, first, profile, bundle)
    with pytest.raises(ReviewError, match="duplicate_response"):
        apply_review(reviewed, first, profile, bundle)
    different = first.model_copy(update={"rationale": "Different same-key supplied decision"})
    with pytest.raises(ReviewError, match="idempotency_conflict"):
        apply_review(reviewed, different, profile, bundle)


def test_closed_case_needs_explicit_reopening(review_case):
    profile, bundle, state, _ = review_case
    reviewed = apply_review(state, completed_response(review_case), profile, bundle)
    response = response_for(review_case, state=reviewed, key="second")
    with pytest.raises(ReviewError, match="case_not_pending"):
        apply_review(reviewed, response, profile, bundle)


@pytest.mark.parametrize("mutation,code", [
    ("missing", "attestation_scope_mismatch"), ("extra", "attestation_scope_mismatch"),
    ("value", "attestation_value_mismatch"), ("role", "attestation_role_mismatch"),
    ("quote", "attestation_evidence_invalid"), ("region", "attestation_evidence_invalid"),
    ("source", "attestation_evidence_invalid"), ("offset", "attestation_evidence_invalid"),
    ("duplicate_ref", "attestation_evidence_invalid"),
])
def test_attestations_strict_source_value_role_and_scope(review_case, mutation, code):
    profile, bundle, state, _ = review_case
    response = completed_response(review_case)
    attestations = list(response.attestations)
    first = attestations[0]
    if mutation == "missing": attestations.pop()
    elif mutation == "extra":
        attestations.append(first.model_copy(update={"path": "/bibliographic/title"}))
    elif mutation == "value": attestations[0] = first.model_copy(update={"value_json": canonical_json("Wrong value")})
    elif mutation == "role": attestations[0] = first.model_copy(update={"role": "sample_identity_or_preparation"})
    else:
        ref = first.evidence[0]
        changes = {"quote": {"quotation": "Fabricated"}, "region": {"region_id": "unknown"},
                   "source": {"source_sha256": "0" * 64}, "offset": {"end": 999999}}
        evidence = (ref, ref) if mutation == "duplicate_ref" else (ref.model_copy(update=changes[mutation]),)
        attestations[0] = first.model_copy(update={"evidence": evidence})
    response = response.model_copy(update={"attestations": tuple(attestations)})
    with pytest.raises(ReviewError, match=code): apply_review(state, response, profile, bundle)
    assert state.current.canonical["datasets"][0]["dataset_description"] == "Measured at 500 K"


@pytest.mark.parametrize("value", ["null", "[]", "{}", "NaN", "Infinity", '"x" ', "1.0 "])
def test_attestation_values_are_canonical_nonnull_scalars(review_case, value):
    _, _, state, _ = review_case
    with pytest.raises(ValueError):
        ReviewAttestation(path=DESCRIPTION, value_json=value, role="measurement_acquisition", evidence=state.cases[0].evidence)


@pytest.mark.parametrize("mutation", ["approve_patch", "edit_without_patch", "unresolved_attestation", "empty_case_ids", "duplicate_case", "blank_reviewer", "blank_rationale", "blank_key", "unknown_provenance", "duplicate_attestation"])
def test_response_contract_rejects_ambiguous_approval(review_case, mutation):
    _, _, state, proposal = review_case
    response = completed_response(review_case)
    data = response.model_dump()
    if mutation == "approve_patch": data["proposal"] = proposal
    elif mutation == "edit_without_patch": data["outcome"] = "edit"
    elif mutation == "unresolved_attestation": data["outcome"] = "unresolved"
    elif mutation == "empty_case_ids": data["case_ids"] = ()
    elif mutation == "duplicate_case": data["case_ids"] = (state.cases[0].case_id,) * 2
    elif mutation == "blank_reviewer": data["reviewer_id"] = " "
    elif mutation == "blank_rationale": data["rationale"] = " "
    elif mutation == "blank_key": data["idempotency_key"] = " "
    elif mutation == "unknown_provenance": data["provenance"] = "authenticated_expert"
    else: data["attestations"] = (response.attestations[0],) * 2
    with pytest.raises(ValueError): ReviewResponse.model_validate(data)


def test_edit_cannot_change_unselected_case(review_case):
    profile, bundle, state, proposal = review_case
    response = response_for(review_case, outcome="edit", selected=(0,), proposal=proposal)
    with pytest.raises(ReviewError, match="edit_outside_selected_cases"):
        apply_review(state, response, profile, bundle)


@pytest.mark.parametrize("kind,code", [
    ("stale", "patch_stale_revision"), ("old_value", "patch_old_value_mismatch"),
    ("unknown_sample", "patch_candidate_structure_invalid"), ("null", "patch_missing_value_change_not_supported"),
    ("no_change", "patch_no_progress"), ("identity", "edit_outside_selected_cases"),
])
def test_edit_reuses_m06_shape_identity_and_missingness_guards(review_case, kind, code):
    profile, bundle, state, proposal = review_case
    ops = list(proposal.operations)
    if kind == "stale": proposal = proposal.model_copy(update={"expected_revision": "revision:" + "0" * 64})
    elif kind == "old_value": ops[0] = ops[0].model_copy(update={"expected_value_json": canonical_json("Other old value")})
    elif kind == "unknown_sample": ops[1] = ops[1].model_copy(update={"replacement_value_json": canonical_json("sample_99")})
    elif kind == "null": ops[0] = ops[0].model_copy(update={"replacement_value_json": "null"})
    elif kind == "no_change": ops[0] = ops[0].model_copy(update={"replacement_value_json": ops[0].expected_value_json})
    else: ops[0] = ops[0].model_copy(update={"path": "/samples/0/sample_name"})
    if kind != "stale": proposal = proposal.model_copy(update={"operations": tuple(ops)})
    response = response_for(review_case, outcome="edit", selected=(0, 1), proposal=proposal)
    with pytest.raises(ReviewError, match=code): apply_review(state, response, profile, bundle)


def test_stale_parent_case_rebased_with_no_approval_transfer(review_case):
    profile, bundle, state, _ = review_case
    workflow = state.initial_workflow
    cases = list(workflow.pending_reviews)
    cases[0] = cases[0].model_copy(update={"record_revision": "revision:old"})
    workflow = workflow.model_copy(update={"pending_reviews": tuple(cases)})
    fresh = initialize_review(workflow, profile, bundle)
    assert fresh.cases[0].record_revision == fresh.current.revision_id
    assert fresh.cases[0].status == "pending" and "rebased" in fresh.cases[0].summary
    assert fresh.initial_workflow.pending_reviews[0].record_revision == "revision:old"
    verify_review_state(fresh, profile, bundle)


@pytest.mark.parametrize("kind", ["running", "no_cases", "uncertain", "case_issue_mismatch", "duplicate_issue"])
def test_bad_parent_review_state_rejected(review_case, kind):
    profile, bundle, state, _ = review_case
    workflow = state.initial_workflow
    if kind == "running": workflow = workflow.model_copy(update={"status": "running"})
    elif kind == "no_cases": workflow = workflow.model_copy(update={"status": "unresolved", "pending_reviews": ()})
    elif kind == "uncertain": workflow = workflow.model_copy(update={"uncertain_model_calls": 1, "usage_unknown": True})
    elif kind == "case_issue_mismatch":
        issues = list(workflow.issues)
        issues[0] = issues[0].model_copy(update={"status": "open"})
        workflow = workflow.model_copy(update={"issues": tuple(issues)})
    else:
        cases = list(workflow.pending_reviews)
        cases[1] = cases[1].model_copy(update={"issue_id": cases[0].issue_id})
        workflow = workflow.model_copy(update={"pending_reviews": tuple(cases)})
    with pytest.raises(ValueError): initialize_review(workflow, profile, bundle)


@pytest.mark.parametrize("mutation", ["case_status", "active_assertion", "current_value", "decision_binding", "decision_outcome"])
def test_replay_rejects_rehashed_review_state_forgery(review_case, mutation):
    profile, bundle, state, _ = review_case
    reviewed = apply_review(state, completed_response(review_case), profile, bundle)
    data = reviewed.model_dump(mode="json")
    if mutation == "case_status":
        data["cases"][1].update(status="approved", review_provenance="simulated", last_decision_id=data["decisions"][0]["decision_id"])
    elif mutation == "active_assertion":
        data["current_assertions"][0]["reviewer_id"] = "Another reviewer"
    elif mutation == "current_value":
        canonical = reviewed.current.canonical
        canonical["datasets"][0]["dataset_description"] = "Invented extra edit"
        changed = snapshot(canonical, bundle, profile)
        data["current"] = changed.model_dump(mode="json")
        for case in data["cases"]: case["record_revision"] = changed.revision_id
        for assertion in data["current_assertions"]: assertion["record_revision"] = changed.revision_id
    else:
        decision = data["decisions"][0]
        if mutation == "decision_binding":
            decision["bindings"][0]["entity_handle"] = "wrong-entity"
        else:
            decision["response"]["rationale"] = "Altered supplied response"
        decision["decision_id"] = "review-decision:" + digest({k: v for k, v in decision.items() if k != "decision_id"})
    with pytest.raises(ValueError):
        forged = ReviewState.model_validate(data)
        verify_review_state(forged, profile, bundle)


def test_serialized_round_trip_replays(review_case):
    profile, bundle, state, proposal = review_case
    reviewed = apply_review(state, completed_response(review_case, outcome="edit", selected=(0, 1), proposal=proposal), profile, bundle)
    restored = ReviewState.model_validate_json(reviewed.model_dump_json())
    assert restored == reviewed and restored.state_id == reviewed.state_id
    verify_review_state(restored, profile, bundle)


def test_different_profile_or_source_rejected(review_case):
    profile, bundle, state, _ = review_case
    schema = profile.schema
    schema["datasets"]["description"] = "Changed schema meaning"
    changed_profile = profile.model_copy(update={"schema_payload_json": json.dumps(schema, separators=(",", ":"), ensure_ascii=False)})
    with pytest.raises(ValueError): verify_review_state(state, changed_profile, bundle)
    with pytest.raises(ValueError): verify_review_state(state, profile, synthetic_bundle("Different publication"))


def test_non_dataset_assertion_uses_its_actual_role(review_case):
    profile, bundle, state, _ = review_case
    workflow = state.initial_workflow
    path = "/samples/0/sample_name"
    issue = workflow.issues[0].model_copy(update={"path": path})
    pending = workflow.pending_reviews[0].model_copy(update={"path": path})
    workflow = workflow.model_copy(update={"issues": (issue,), "pending_reviews": (pending,)})
    single = initialize_review(workflow, profile, bundle)
    response = completed_response((profile, bundle, single, None), state=single)
    assert len(response.attestations) == 1 and response.attestations[0].role == "sample_identity_or_preparation"
    reviewed = apply_review(single, response, profile, bundle)
    assert reviewed.status == "completed"


def test_null_case_can_remain_unresolved_but_not_claim_supported(review_case):
    profile, bundle, state, _ = review_case
    workflow = state.initial_workflow
    path = "/samples/0/sample_description"
    issue = workflow.issues[0].model_copy(update={"path": path})
    pending = workflow.pending_reviews[0].model_copy(update={"path": path})
    workflow = workflow.model_copy(update={"issues": (issue,), "pending_reviews": (pending,)})
    single = initialize_review(workflow, profile, bundle)
    with pytest.raises(ReviewError, match="unsupported_case_assertion"):
        required_review_paths(single, response_for((profile, bundle, single, None)), profile, bundle)
    unresolved = apply_review(single, response_for((profile, bundle, single, None), outcome="unresolved"), profile, bundle)
    assert unresolved.status == "unresolved"


def test_latest_unresolved_clears_current_attestation_and_dependent_approval(review_case):
    profile, bundle, state, _ = review_case
    approved = apply_review(state, completed_response(review_case, selected=(0,)), profile, bundle)
    assert any(a.path == REFERENCE for a in approved.current_assertions)
    response = response_for(review_case, state=approved, outcome="unresolved", selected=(1,), key="response-2")
    uncertain = apply_review(approved, response, profile, bundle)
    assert uncertain.current == approved.current
    assert uncertain.cases[1].status == "unresolved" and uncertain.cases[0].status == "pending"
    assert not any(a.path in {REFERENCE, DESCRIPTION} for a in uncertain.current_assertions)
    assert approved.decisions[0] == uncertain.decisions[0]
    verify_review_state(uncertain, profile, bundle)


def test_approval_cannot_silently_overrule_unresolved_dependency(review_case):
    profile, bundle, state, _ = review_case
    uncertain = apply_review(state, response_for(review_case, outcome="unresolved", selected=(1,)), profile, bundle)
    response = response_for(review_case, state=uncertain, selected=(0,), key="response-2")
    with pytest.raises(ReviewError, match="unresolved_dependency"):
        required_review_paths(uncertain, response, profile, bundle)
    with pytest.raises(ReviewError, match="unresolved_dependency"):
        apply_review(uncertain, response, profile, bundle)


def test_explicit_edit_rebases_unresolved_dependency_to_new_revision(review_case):
    profile, bundle, state, proposal = review_case
    uncertain = apply_review(state, response_for(review_case, outcome="unresolved", selected=(1,)), profile, bundle)
    single = proposal.model_copy(update={"operations": (proposal.operations[0],)})
    response = completed_response(review_case, state=uncertain, outcome="edit", selected=(0,), key="response-2", proposal=single)
    revised = apply_review(uncertain, response, profile, bundle)
    assert revised.current.revision_id != state.current.revision_id
    assert revised.cases[1].status == "pending"
    verify_review_state(revised, profile, bundle)


@pytest.mark.parametrize("status", ["cancelled", "budget_exhausted", "no_progress", "timed_out", "failed", "interrupted", "unresolved"])
def test_initializer_only_accepts_explicit_pending_review_workflow(review_case, status):
    profile, bundle, state, _ = review_case
    stopped = state.initial_workflow.model_copy(update={"status": status})
    with pytest.raises(ReviewError, match="pending_review_required"):
        initialize_review(stopped, profile, bundle)


def test_initializer_rejects_remaining_open_issue_even_without_a_case(review_case):
    profile, bundle, state, _ = review_case
    workflow = state.initial_workflow
    open_issue = workflow.issues[0].model_copy(update={"id": "unreviewed-open-issue", "status": "open"})
    unfinished = workflow.model_copy(update={"issues": (*workflow.issues, open_issue)})
    with pytest.raises(ReviewError, match="open_parent_issue"):
        initialize_review(unfinished, profile, bundle)
