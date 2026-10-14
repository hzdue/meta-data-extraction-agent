"""Pure, explicit human/simulated review transitions over immutable records.

M06 automated correction rules are unchanged. A supplied reviewer can affirm
current assertions or explicitly edit the same narrow canonical scope, with
source evidence and relationship attestations. No provider is invoked here.
"""
from __future__ import annotations

from collections.abc import Mapping

from .contracts import RecordSnapshot, assertion_values, canonical_json, digest, owner
from .evidence import DocumentBundle, resolve_evidence
from .profiles import Profile
from .record_checks import _checked_profile, check_structure
from .record_patches import PatchProposal, build_candidate
from .review_types import (
    CatalogProposal, ReviewDecision, ReviewError, ReviewItem, ReviewResponse, ReviewState,
)
from .support_checks import CheckedAssertion, SupportTarget, _prepare
from .workflow_types import WorkflowState


def _inputs(profile: Profile, bundle: DocumentBundle) -> tuple[Profile, DocumentBundle]:
    return _checked_profile(profile), DocumentBundle.model_validate(bundle.model_dump())


def _record(record: RecordSnapshot, profile: Profile, bundle: DocumentBundle) -> RecordSnapshot:
    record = RecordSnapshot.model_validate(record.model_dump())
    result = check_structure(profile, bundle, record.canonical)
    if result.status != "valid" or result.snapshot != record:
        raise ReviewError("record_binding_invalid")
    return record


def initialize_review(
    workflow: WorkflowState, profile: Profile, bundle: DocumentBundle,
    proposals: Mapping[str, PatchProposal] | None = None,
) -> ReviewState:
    """Freeze a review child of a stopped M07 state; never rewrite its parent."""
    profile, bundle = _inputs(profile, bundle)
    workflow = WorkflowState.model_validate(workflow.model_dump())
    if workflow.status != "pending_review" or not workflow.pending_reviews:
        raise ReviewError("pending_review_required")
    if any(issue.status == "open" for issue in workflow.issues):
        raise ReviewError("open_parent_issue")
    current = _record(workflow.record, profile, bundle)
    pending_by_issue = {item.issue_id: item for item in workflow.pending_reviews}
    if len(pending_by_issue) != len(workflow.pending_reviews):
        raise ReviewError("duplicate_issue_case")
    issues = {item.id: item for item in workflow.issues}
    if any(issues[case.issue_id].status != "pending_review" or issues[case.issue_id].path != case.path
           for case in workflow.pending_reviews):
        raise ReviewError("pending_case_issue_mismatch")
    if workflow.uncertain_model_calls:
        raise ReviewError("uncertain_parent_dispatch")
    refs = [ref for item in workflow.issues for ref in item.evidence]
    refs += [ref for item in workflow.pending_reviews for ref in item.evidence]
    refs += [ref for event in workflow.events for ref in event.evidence]
    try:
        for ref in refs:
            resolve_evidence(bundle, ref)
    except ValueError:
        raise ReviewError("parent_evidence_invalid") from None
    raw_proposals = proposals or {}
    if not isinstance(raw_proposals, Mapping) or len(raw_proposals) > 32:
        raise ReviewError("proposal_catalog_invalid")
    catalog = tuple(CatalogProposal(key=key, proposal=PatchProposal.model_validate(value.model_dump()))
                    for key, value in sorted(raw_proposals.items()))
    cases = tuple(ReviewItem(
        case_id=case.case_id, issue_id=case.issue_id, path=case.path,
        summary=case.summary if case.record_revision == current.revision_id else (
            case.summary + " Earlier request rebased to the current revision; no prior approval is transferred."
        ),
        record_revision=current.revision_id, evidence=case.evidence,
    ) for case in workflow.pending_reviews)
    identity = {"workflow_state_id": workflow.state_id,
                "proposals": [item.model_dump(mode="json") for item in catalog]}
    return ReviewState(review_id="review:" + digest(identity), initial_workflow=workflow,
                       current=current, cases=cases, proposals=catalog)


def _prepare_response(
    state: ReviewState, response: ReviewResponse, profile: Profile, bundle: DocumentBundle,
) -> tuple[RecordSnapshot, tuple[str, ...], tuple[str, ...]]:
    if response.expected_state_id != state.state_id:
        raise ReviewError("stale_state")
    if response.expected_revision != state.current.revision_id:
        raise ReviewError("stale_revision")
    by_id = {case.case_id: case for case in state.cases}
    if any(case_id not in by_id for case_id in response.case_ids):
        raise ReviewError("unknown_case")
    selected = tuple(by_id[case_id] for case_id in response.case_ids)
    if any(case.status != "pending" for case in selected):
        raise ReviewError("case_not_pending")
    if response.outcome == "unresolved":
        return state.current, (), ()
    selected_paths = {case.path for case in selected}
    candidate = state.current
    changed: tuple[str, ...] = ()
    required = set(selected_paths)
    if response.outcome == "edit":
        proposal = response.proposal
        assert proposal is not None
        if any(op.path not in selected_paths for op in proposal.operations):
            raise ReviewError("edit_outside_selected_cases")
        built = build_candidate(profile, bundle, state.current, proposal)
        if built.status != "ready" or built.prepared is None:
            raise ReviewError("patch_" + built.code)
        candidate = built.prepared.candidate
        changed = built.prepared.changed_paths
        required.update(built.prepared.required_support_paths)
    values = dict(assertion_values(candidate.canonical))
    if any(path not in values for path in selected_paths):
        raise ReviewError("unsupported_case_assertion")
    # Approving a dataset field also reviews its existing specimen/facility
    # attachments. A correctly occurring scalar does not establish its context.
    touched = {owner(candidate.entities, path).handle for path in selected_paths
               if owner(candidate.entities, path).collection == "datasets"}
    required.update(ref.path for ref in candidate.references if ref.source_handle in touched)
    if response.outcome == "approve" and any(
        case.status == "unresolved" and case.path in required for case in state.cases
    ):
        raise ReviewError("unresolved_dependency")
    if len(required) > 32:
        raise ReviewError("attestation_budget_exceeded")
    return candidate, changed, tuple(sorted(required))


def required_review_paths(
    state: ReviewState, response: ReviewResponse, profile: Profile, bundle: DocumentBundle,
) -> tuple[str, ...]:
    """Return the exact evidence-attestation scope before a response is completed."""
    verify_review_state(state, profile, bundle)
    response = ReviewResponse.model_validate(response.model_dump())
    _, _, required = _prepare_response(state, response, profile, bundle)
    return required


def _attestations(
    response: ReviewResponse, candidate: RecordSnapshot, bundle: DocumentBundle,
    required: tuple[str, ...],
) -> tuple[CheckedAssertion, ...]:
    if {item.path for item in response.attestations} != set(required) or len(response.attestations) != len(required):
        raise ReviewError("attestation_scope_mismatch")
    by_path = {item.path: item for item in response.attestations}
    targets = tuple(SupportTarget(path=path, evidence=by_path[path].evidence) for path in required)
    try:
        checked = _prepare(candidate, bundle, targets)
    except (ValueError, TypeError, KeyError):
        raise ReviewError("attestation_evidence_invalid") from None
    for assertion in checked:
        supplied = by_path[assertion.path]
        if supplied.role != assertion.role:
            raise ReviewError("attestation_role_mismatch")
        if supplied.value_json != assertion.value_json:
            raise ReviewError("attestation_value_mismatch")
    return checked


def _apply(
    state: ReviewState, response: ReviewResponse, profile: Profile, bundle: DocumentBundle,
) -> ReviewState:
    if len(state.decisions) >= 64:
        raise ReviewError("review_response_budget_exceeded")
    if any(item.response.idempotency_key == response.idempotency_key for item in state.decisions):
        raise ReviewError("duplicate_response" if any(item.response == response for item in state.decisions) else "idempotency_conflict")
    candidate, changed, required = _prepare_response(state, response, profile, bundle)
    bindings = () if response.outcome == "unresolved" else _attestations(response, candidate, bundle, required)
    payload = dict(
        version="m08-decision-1", response=response.model_dump(mode="json"),
        previous_revision=state.current.revision_id, resulting_revision=candidate.revision_id,
        changed_paths=changed, required_paths=required,
        bindings=tuple(item.model_dump(mode="json") for item in bindings),
        invalidated_revision=state.current.revision_id if changed else None,
        expert_authentication="not_authenticated", coverage_status="selected_assertions_only",
    )
    decision = ReviewDecision(decision_id="review-decision:" + digest(payload), **payload)
    selected = set(response.case_ids)
    unresolved_paths = {case.path for case in state.cases if case.case_id in selected} if response.outcome == "unresolved" else set()
    reopened = set()
    for case in state.cases:
        if case.status != "approved" or case.case_id in selected:
            continue
        entity = owner(candidate.entities, case.path)
        dependencies = {case.path}
        if entity.collection == "datasets":
            dependencies.update(ref.path for ref in candidate.references if ref.source_handle == entity.handle)
        if dependencies & unresolved_paths:
            reopened.add(case.case_id)
    cases = []
    for case in state.cases:
        values = case.model_dump()
        values["record_revision"] = candidate.revision_id
        if case.case_id in selected:
            values.update(status="unresolved" if response.outcome == "unresolved" else "approved",
                          review_provenance=response.provenance, last_decision_id=decision.decision_id)
        elif changed or case.case_id in reopened:
            # Every old revision approval/closure becomes stale, even when its
            # scalar is unchanged. Historical decisions remain in the append log.
            values.update(status="pending", review_provenance=None, last_decision_id=None)
        cases.append(ReviewItem.model_validate(values))
    active = {} if changed else {item.path: item for item in state.current_assertions}
    for path in unresolved_paths | {case.path for case in state.cases if case.case_id in reopened}:
        active.pop(path, None)
    active.update({item.path: item for item in decision.assertions})
    status = "pending" if any(case.status == "pending" for case in cases) else (
        "unresolved" if any(case.status == "unresolved" for case in cases) else "completed"
    )
    return ReviewState(
        review_id=state.review_id, initial_workflow=state.initial_workflow,
        current=candidate, cases=tuple(cases), proposals=state.proposals,
        decisions=(*state.decisions, decision), current_assertions=tuple(active[path] for path in sorted(active)),
        status=status,
    )


def apply_review(
    state: ReviewState, response: ReviewResponse, profile: Profile, bundle: DocumentBundle,
) -> ReviewState:
    """Validate and apply one supplied review response, atomically and without I/O.

    No model judgment or pending case is converted to human approval. The caller
    persists the exact response and result transactionally and handles duplicate
    delivery before calling this function.
    """
    profile, bundle = _inputs(profile, bundle)
    verify_review_state(state, profile, bundle)
    response = ReviewResponse.model_validate(response.model_dump())
    return _apply(state, response, profile, bundle)


def verify_review_state(state: ReviewState, profile: Profile, bundle: DocumentBundle) -> None:
    """Replay explicit decisions deterministically; do not trust rehashed statuses."""
    profile, bundle = _inputs(profile, bundle)
    state = ReviewState.model_validate(state.model_dump())
    initial = initialize_review(
        state.initial_workflow, profile, bundle,
        {item.key: item.proposal for item in state.proposals},
    )
    rebuilt = initial
    for recorded in state.decisions:
        rebuilt = _apply(rebuilt, recorded.response, profile, bundle)
        if rebuilt.decisions[-1] != recorded:
            raise ReviewError("decision_replay_mismatch")
    if rebuilt != state:
        raise ReviewError("review_state_replay_mismatch")
