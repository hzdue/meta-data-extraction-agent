"""Synthetic M07 action/state contracts; no provider or workflow execution."""
from pathlib import Path
import shutil

import pytest
from pydantic import ValidationError

from extraction_fixtures import synthetic_bundle, synthetic_record
from meta_data_extraction.agent.contracts import ProviderObservation, snapshot
from meta_data_extraction.agent.evidence import evidence_ref
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.agent.workflow_types import (
    ACTION_ADAPTER, ActionEvent, ConsultDefinition, Finish, ProposeUpdate, RequestReview,
    RetrieveSource, ReviewCase, WorkflowBudget, WorkflowIssue, WorkflowState, validate_action,
)


@pytest.fixture
def workflow(workspace):
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "schemas/prompt_template.md", workspace / "schemas/prompt_template.md")
    profile = load_profile(workspace)
    bundle = synthetic_bundle()
    record = snapshot(synthetic_record(profile), bundle, profile)
    issue = WorkflowIssue(id="issue-1", path="/datasets/0/measurement_technique",
                          diagnosis="unsupported_value", summary="Invented technique needs inspection.",
                          query="Sample A Dataset A X-ray diffraction", proposal_ids=("proposal-1",),
                          definition_keys=("definition-1",))
    return profile, bundle, WorkflowState(initial_revision=record.revision_id, record=record, issues=(issue,))


def update_state(state, **changes):
    return WorkflowState.model_validate(state.model_dump() | changes)


def event_for(state, action, **changes):
    values = dict(index=len(state.events), policy="fixed", action=action, status="ok", code="observed",
                  revision_before=state.record.revision_id, revision_after=state.record.revision_id,
                  progress=False, summary="Synthetic observable result.")
    return ActionEvent(**(values | changes))


@pytest.mark.parametrize("action", [
    RetrieveSource(issue_id="issue-1", query="Sample A"),
    ConsultDefinition(issue_id="issue-1", definition_key="definition-1"),
    ProposeUpdate(issue_id="issue-1", proposal_id="proposal-1"),
    RequestReview(issue_id="issue-1", summary="Please inspect the attachment."),
    Finish(outcome="unresolved", summary="Evidence is insufficient."),
])
def test_allowlisted_actions_roundtrip_and_validate_current_catalog(workflow, action):
    _, _, state = workflow
    assert validate_action(ACTION_ADAPTER.validate_json(action.model_dump_json()), state) == action


@pytest.mark.parametrize("action", [
    RetrieveSource(issue_id="unknown", query="Sample A"),
    ProposeUpdate(issue_id="issue-1", proposal_id="unknown"),
    ConsultDefinition(issue_id="issue-1", definition_key="unknown"),
    Finish(outcome="finished", summary="Pretend complete."),
])
def test_action_cannot_invent_issue_proposal_definition_or_success(workflow, action):
    _, _, state = workflow
    with pytest.raises(ValueError):
        validate_action(action, state)


@pytest.mark.parametrize("status", ["resolved", "pending_review"])
def test_actions_require_open_issue(workflow, status):
    _, _, state = workflow
    issue = state.issues[0].model_copy(update={"status": status})
    state = update_state(state, issues=[issue])
    with pytest.raises(ValueError, match="open issue"):
        validate_action(RetrieveSource(issue_id="issue-1", query="source"), state)


def test_finished_outcome_requires_all_issues_resolved_and_no_pending_review(workflow):
    _, _, state = workflow
    state = update_state(state, issues=[state.issues[0].model_copy(update={"status": "resolved"})])
    action = Finish(outcome="finished", summary="All supplied issues resolved.")
    assert validate_action(action, state) == action
    case = ReviewCase(case_id="review-1", issue_id="issue-1", record_revision=state.record.revision_id,
                      path=state.issues[0].path, summary="Still awaiting actual review.")
    with pytest.raises(ValueError):
        validate_action(action, update_state(state, pending_reviews=[case]))
    with pytest.raises(ValidationError, match="Finished state"):
        update_state(state, status="finished", pending_reviews=[case])
    assert case.status == "pending"
    with pytest.raises(ValidationError):
        ReviewCase.model_validate(case.model_dump() | {"status": "approved"})


@pytest.mark.parametrize("status", ["cancelled", "timed_out", "budget_exhausted", "no_progress", "interrupted", "failed"])
def test_stopped_state_cannot_authorize_another_action(workflow, status):
    _, _, state = workflow
    with pytest.raises(ValueError, match="stopped"):
        validate_action(RetrieveSource(issue_id="issue-1", query="source"), update_state(state, status=status))


@pytest.mark.parametrize("changes", [
    {"total_actions": 0}, {"total_actions": 33}, {"per_issue_actions": 9},
    {"max_model_calls": -1}, {"max_model_calls": True}, {"no_progress_limit": 5},
    {"timeout_seconds": 0}, {"timeout_seconds": 301}, {"timeout_seconds": float("nan")},
    {"timeout_seconds": True}, {"unknown_usage": "assume_zero"},
])
def test_budget_contract_rejects_unbounded_or_ambiguous_limits(changes):
    with pytest.raises(ValueError):
        WorkflowBudget(**changes)


def test_unknown_usage_and_uncertain_attempts_are_explicit(workflow):
    _, _, state = workflow
    assert WorkflowBudget().unknown_usage == "stop"
    uncertain = update_state(state, usage_unknown=True, uncertain_model_calls=2, status="interrupted")
    assert uncertain.model_calls == 0 and uncertain.uncertain_model_calls == 2
    assert uncertain.input_tokens == uncertain.output_tokens == 0 and uncertain.usage_unknown
    assert uncertain.state_id != state.state_id


@pytest.mark.parametrize("changes", [
    {"limit": 6}, {"limit": True}, {"max_chars": 8001}, {"query": " "},
    {"query": "x" * 501}, {"shell": "execute arbitrary code"},
])
def test_retrieval_action_arguments_are_bounded_and_cannot_add_tools(changes):
    with pytest.raises(ValueError):
        RetrieveSource(**({"issue_id": "issue-1", "query": "source"} | changes))


def test_model_copy_cannot_bypass_action_validation(workflow):
    _, _, state = workflow
    action = RetrieveSource(issue_id="issue-1", query="source").model_copy(update={"limit": 999})
    with pytest.raises(ValueError):
        validate_action(action, state)


def test_issue_catalog_and_state_ids_are_distinct(workflow):
    _, _, state = workflow
    with pytest.raises(ValidationError, match="issue IDs"):
        update_state(state, issues=[state.issues[0], state.issues[0]])
    for field in ("proposal_ids", "definition_keys"):
        with pytest.raises(ValidationError, match="distinct"):
            WorkflowIssue.model_validate(state.issues[0].model_dump() | {field: ["same", "same"]})
    with pytest.raises(ValidationError, match="catalog"):
        WorkflowIssue.model_validate(state.issues[0].model_dump() | {"consulted_definitions": ["unknown"]})


def test_pending_review_requires_case_and_case_requires_known_issue(workflow):
    _, _, state = workflow
    with pytest.raises(ValidationError, match="actual pending"):
        update_state(state, status="pending_review")
    case = ReviewCase(case_id="case-1", issue_id="unknown", record_revision=state.record.revision_id,
                      path="", summary="Pending inspection.")
    with pytest.raises(ValidationError, match="unknown issue"):
        update_state(state, pending_reviews=[case])


def test_event_history_is_revision_bound_and_preserves_actual_observations(workflow):
    profile, bundle, state = workflow
    action = ProposeUpdate(issue_id="issue-1", proposal_id="proposal-1")
    canonical = state.record.canonical
    canonical["datasets"][0]["measurement_technique"] = "Changed synthetic technique"
    changed = snapshot(canonical, bundle, profile)
    event = event_for(state, action, revision_after=changed.revision_id, progress=True)
    updated = update_state(state, record=changed, events=[event])
    assert updated.initial_revision == state.record.revision_id
    assert updated.record.revision_id == changed.revision_id and updated.state_id != state.state_id
    assert WorkflowState.model_validate_json(updated.model_dump_json()) == updated
    with pytest.raises(ValidationError, match="history"):
        update_state(state, record=changed)
    with pytest.raises(ValidationError, match="continuous"):
        update_state(state, events=[event.model_copy(update={"revision_before": "wrong"})])
    with pytest.raises(ValidationError, match="indices"):
        update_state(state, events=[event_for(state, action), event_for(state, action)])


def test_failed_policy_records_provider_observation_without_fabricated_action(workflow):
    _, _, state = workflow
    observation = ProviderObservation(status="failed", mode="mock", model="synthetic-policy", finish_reason=None,
                                      input_tokens=None, output_tokens=None, latency_ms=None,
                                      error_code="provider_failed", request_id=None, response_sha256=None)
    event = event_for(state, None, status="failed", code="policy_failed", providers=(observation,))
    updated = update_state(state, events=[event], model_calls=1, usage_unknown=True)
    assert updated.events[0].action is None and updated.events[0].providers == (observation,)
    for changes in ({"status": "ok"}, {"progress": True}, {"revision_after": "invented-change"}):
        with pytest.raises(ValidationError, match="unselected"):
            ActionEvent.model_validate(event.model_dump() | changes)


def test_publication_evidence_remains_bound_to_actual_record_source(workflow):
    _, bundle, state = workflow
    ref = evidence_ref(bundle, bundle.regions[0].region_id)
    issue = state.issues[0].model_copy(update={"evidence": (ref,)})
    assert update_state(state, issues=[issue]).issues[0].evidence == (ref,)
    other = synthetic_bundle("Different publication evidence.")
    foreign = evidence_ref(other, other.regions[0].region_id)
    with pytest.raises(ValidationError, match="different source"):
        update_state(state, issues=[issue.model_copy(update={"evidence": (foreign,)})])
    with pytest.raises(ValidationError):
        WorkflowIssue.model_validate(issue.model_dump() | {"evidence": [{"definition_key": "NXsample/temperature"}]})


def test_contracts_do_not_mutate_canonical_or_make_review_approval(workflow):
    _, _, state = workflow
    original = state.record.canonical_json
    case = ReviewCase(case_id="case-1", issue_id="issue-1", record_revision=state.record.revision_id,
                      path=state.issues[0].path, summary="Pending evidence review.")
    issue = state.issues[0].model_copy(update={"status": "pending_review"})
    updated = update_state(state, issues=[issue], status="pending_review", pending_reviews=[case])
    assert updated.record.canonical_json == original
    assert updated.pending_reviews[0].status == "pending"
