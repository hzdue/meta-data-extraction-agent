"""Offline workflow transition diagnostics, with explicitly invented model output."""
from dataclasses import replace
import json
import time

import pytest
from pydantic import ValidationError

from test_workflow_tools import tool_case, response
from meta_data_extraction.agent.providers import MockProvider, ProviderResult
from meta_data_extraction.agent.workflow import WorkflowRuntime, advance, verify_state
from meta_data_extraction.agent.workflow_types import (
    Finish, ProposeUpdate, RetrieveSource, WorkflowBudget, WorkflowState,
)


@pytest.fixture
def workflow_case(tool_case):
    context, state, prepared = tool_case
    state = state.model_copy(update={"issues": tuple(i.model_copy(update={"definition_keys": ()}) for i in state.issues)})
    return context, state, prepared


def selected(action, *, tokens=True, status="ok", finish="stop"):
    return ProviderResult(status=status, content=json.dumps(action), finish_reason=finish,
                          input_tokens=7 if tokens else None, output_tokens=3 if tokens else None)


def retrieval(issue_id="issue-0"):
    return {"kind": "retrieve_source", "issue_id": issue_id, "query": "Dataset A measured"}


def runtime(context, *, policy="issue_aware", provider=None, **budget):
    return WorkflowRuntime(context, WorkflowBudget(**budget), policy, policy_provider=provider)


def until_stopped(state, execution, *, limit=35):
    for _ in range(limit):
        if state.status != "running":
            return state
        state = advance(state, execution)
    raise AssertionError("Bounded controller did not stop")


def test_valid_correction_preserves_history_and_counts_both_checker_calls(workflow_case):
    context, initial, prepared = workflow_case
    result = until_stopped(initial, runtime(context))
    assert result.status == "finished" and result.record == prepared.candidate
    assert [e.action.kind for e in result.events] == ["retrieve_source", "propose_update", "finish"]
    assert result.model_calls == len(context.checker.requests) == 2
    assert result.input_tokens == result.output_tokens == 20
    assert not result.usage_unknown
    assert all(i.status == "resolved" for i in result.issues)
    assert result.events[1].receipt_id
    assert initial.record == prepared.original and initial.events == ()
    assert result.events[1].revision_before == initial.record.revision_id
    assert result.events[1].revision_after == result.record.revision_id
    verify_state(result, runtime(context))


@pytest.mark.parametrize("budget,reason", [
    ({"total_actions": 1}, "total_action_budget"),
    ({"per_issue_actions": 1}, "per_issue_action_budget"),
])
def test_action_budgets_stop_before_another_dispatch(workflow_case, budget, reason):
    context, state, _ = workflow_case
    execution = runtime(context, **budget)
    first = advance(state, execution)
    assert first.status == "running" and len(first.events) == 1
    stopped = advance(first, execution)
    assert stopped.status == "budget_exhausted" and stopped.stop_reason == reason
    assert len(stopped.events) == 1 and not context.checker.requests
    assert stopped.record == state.record


@pytest.mark.parametrize("policy", ["fixed", "issue_aware"])
def test_checker_calls_reserved_before_correction_begins(workflow_case, policy):
    context, state, _ = workflow_case
    result = until_stopped(state, runtime(context, policy=policy, max_model_calls=1))
    assert result.status == "budget_exhausted" and result.stop_reason == "model_call_budget"
    assert not context.checker.requests and result.model_calls == 0
    assert result.record == state.record


def test_generic_call_budget_zero_prevents_policy_dispatch(workflow_case):
    context, state, _ = workflow_case
    provider = MockProvider([selected(retrieval())])
    result = advance(state, runtime(context, policy="generic_llm", provider=provider, max_model_calls=0))
    assert result.status == "budget_exhausted" and result.model_calls == 0
    assert not provider.requests and not context.checker.requests


def test_generic_selection_counts_against_checker_call_budget(workflow_case):
    context, state, _ = workflow_case
    provider = MockProvider([selected({"kind": "propose_update", "issue_id": "issue-0", "proposal_id": "correction"})])
    result = advance(state, runtime(context, policy="generic_llm", provider=provider, max_model_calls=2))
    assert result.status == "budget_exhausted" and result.stop_reason == "model_call_budget"
    assert result.model_calls == 1 and len(provider.requests) == 1 and not context.checker.requests
    assert result.events[0].providers[0].input_tokens == 7
    assert result.record == state.record


def test_generic_and_checker_calls_share_one_total_budget(workflow_case):
    context, state, prepared = workflow_case
    provider = MockProvider([selected({"kind": "propose_update", "issue_id": "issue-0", "proposal_id": "correction"})])
    execution = runtime(context, policy="generic_llm", provider=provider, max_model_calls=3)
    result = advance(state, execution)
    assert result.record == prepared.candidate and result.model_calls == 3
    assert result.input_tokens == 27 and result.output_tokens == 23
    stopped = advance(result, execution)
    assert stopped.status == "budget_exhausted" and stopped.model_calls == 3
    assert len(provider.requests) == 1 and len(context.checker.requests) == 2


def test_repeated_identical_retrieval_stops_when_no_new_evidence_is_added(workflow_case):
    context, state, _ = workflow_case
    state = state.model_copy(update={"issues": tuple(i.model_copy(update={"evidence": ()}) for i in state.issues)})
    provider = MockProvider([selected(retrieval()) for _ in range(8)])
    execution = runtime(context, policy="generic_llm", provider=provider, no_progress_limit=2)
    result = until_stopped(state, execution)
    assert result.status == "no_progress" and result.stop_reason == "no_new_evidence_or_state"
    assert len(result.events) == len(provider.requests) == result.model_calls == 3
    assert [e.progress for e in result.events] == [True, False, False]
    assert result.no_progress_steps == 2 and result.record == state.record
    assert not context.checker.requests


@pytest.mark.parametrize("action", [
    {"kind": "execute_python", "code": "print('bad')"},
    {"kind": "retrieve_source", "issue_id": "unknown", "query": "A"},
    {"kind": "propose_update", "issue_id": "issue-0", "proposal_id": "invented"},
    {"kind": "finish", "outcome": "finished", "summary": "Unjustified completion"},
    {"kind": "request_review", "issue_id": "issue-0", "summary": "Approved", "approval": True},
    {"kind": "retrieve_source", "issue_id": "issue-0", "query": "A", "max_chars": 1000000},
])
def test_invalid_generic_actions_fail_without_tool_effect(workflow_case, action):
    context, state, _ = workflow_case
    provider = MockProvider([selected(action)])
    result = advance(state, runtime(context, policy="generic_llm", provider=provider))
    assert result.status == "failed" and result.stop_reason == "invalid_policy_action"
    assert result.model_calls == 1 and result.record == state.record
    assert result.events[0].action is None and not context.checker.requests


@pytest.mark.parametrize("failure", ["timeout", "failed", "blocked", "rate_limited"])
def test_generic_provider_failure_is_counted_and_never_retried(workflow_case, failure):
    context, state, _ = workflow_case
    provider = MockProvider([ProviderResult(status=failure, error_code="synthetic_failure")])
    execution = runtime(context, policy="generic_llm", provider=provider)
    result = advance(state, execution)
    assert result.status == "failed" and result.stop_reason == "synthetic_failure"
    assert result.model_calls == 1 and result.usage_unknown
    assert advance(result, execution) == result
    assert len(provider.requests) == 1 and not context.checker.requests


@pytest.mark.parametrize("finish", [None, "length", "tool_calls"])
def test_incomplete_generic_completion_does_not_execute_selected_action(workflow_case, finish):
    context, state, _ = workflow_case
    provider = MockProvider([selected(retrieval(), finish=finish)])
    result = advance(state, runtime(context, policy="generic_llm", provider=provider))
    assert result.status == "failed" and result.stop_reason == "policy_completion_incomplete"
    assert result.events[0].evidence == () and result.model_calls == 1


@pytest.mark.parametrize("unknown_policy,expected", [("stop", "unresolved"), ("allow_bounded_calls", "running")])
def test_unknown_policy_usage_is_not_zero_and_dispatch_obeys_explicit_policy(workflow_case, unknown_policy, expected):
    context, state, _ = workflow_case
    provider = MockProvider([selected(retrieval(), tokens=False)])
    result = advance(state, runtime(context, policy="generic_llm", provider=provider, unknown_usage=unknown_policy))
    assert result.status == expected and result.usage_unknown and result.model_calls == 1
    assert result.input_tokens == result.output_tokens == 0
    assert result.events[0].providers[0].input_tokens is None
    assert bool(result.events[0].evidence) == (unknown_policy == "allow_bounded_calls")


@pytest.mark.parametrize("unknown_policy,calls,changed", [("stop", 1, False), ("allow_bounded_calls", 2, True)])
def test_unknown_checker_usage_stops_before_second_call_unless_explicitly_allowed(workflow_case, unknown_policy, calls, changed):
    context, state, prepared = workflow_case
    checker = MockProvider([response(prepared.changed_paths, verdict="contradicted", tokens=False),
                            response(prepared.required_support_paths, tokens=False)])
    context = replace(context, checker=checker)
    provider = MockProvider([selected({"kind": "propose_update", "issue_id": "issue-0", "proposal_id": "correction"})])
    result = advance(state, runtime(context, policy="generic_llm", provider=provider, unknown_usage=unknown_policy))
    assert len(checker.requests) == calls and result.model_calls == calls + 1 and result.usage_unknown
    assert (result.record.revision_id != state.record.revision_id) == changed
    if not changed:
        assert result.status == "unresolved" and result.stop_reason == "unknown_provider_usage"


@pytest.mark.parametrize("kind", ["cancelled", "timeout"])
def test_stop_before_action_produces_no_dispatch_or_patch(workflow_case, kind):
    context, state, _ = workflow_case
    execution = runtime(context)
    if kind == "cancelled": execution.cancelled = lambda: True
    else: execution.deadline_unix = time.time() - 1
    result = advance(state, execution)
    assert result.status == ("cancelled" if kind == "cancelled" else "timed_out")
    assert result.record == state.record and result.events == () and not context.checker.requests


def test_cancellation_after_generic_call_counts_attempt_but_never_dispatches_tool(workflow_case):
    context, state, _ = workflow_case
    provider = MockProvider([selected({"kind": "propose_update", "issue_id": "issue-0", "proposal_id": "correction"})])
    execution = runtime(context, policy="generic_llm", provider=provider)
    execution.cancelled = lambda: bool(provider.requests)
    result = advance(state, execution)
    assert result.status == "cancelled" and result.model_calls == 1
    assert result.record == state.record and not context.checker.requests


@pytest.mark.parametrize("after_call", [1, 2])
def test_cancellation_after_checker_call_never_applies_candidate(workflow_case, after_call):
    context, state, _ = workflow_case
    provider = MockProvider([selected({"kind": "propose_update", "issue_id": "issue-0", "proposal_id": "correction"})])
    execution = runtime(context, policy="generic_llm", provider=provider)
    execution.cancelled = lambda: len(context.checker.requests) >= after_call
    result = advance(state, execution)
    assert result.status == "cancelled" and result.model_calls == after_call + 1
    assert len(context.checker.requests) == after_call and result.record == state.record
    assert result.events[-1].receipt_id is None and all(i.status == "open" for i in result.issues)


def test_identical_initial_state_different_policies_have_transparent_sequences(workflow_case):
    context, state, _ = workflow_case
    issue = state.issues[0].model_copy(update={"diagnosis": "context_conflict", "proposal_ids": (), "evidence": ()})
    state = state.model_copy(update={"issues": (issue,)})
    fixed = until_stopped(state, runtime(context, policy="fixed"))
    aware = until_stopped(state, runtime(context, policy="issue_aware"))
    assert [event.action.kind for event in fixed.events] == ["retrieve_source", "request_review"]
    assert [event.action.kind for event in aware.events] == ["request_review"]
    assert fixed.status == aware.status == "pending_review"
    assert fixed.record == aware.record == state.record and not context.checker.requests
    assert all(case.status == "pending" for case in (*fixed.pending_reviews, *aware.pending_reviews))
    assert all(issue.status == "pending_review" for issue in aware.issues)


def test_request_review_retains_source_evidence_and_never_fabricates_approval(workflow_case):
    context, state, _ = workflow_case
    issue = state.issues[0].model_copy(update={"diagnosis": "context_conflict", "proposal_ids": ()})
    state = state.model_copy(update={"issues": (issue,)})
    execution = runtime(context)
    result = advance(state, execution)
    assert result.pending_reviews[0].record_revision == state.record.revision_id
    assert result.pending_reviews[0].evidence == issue.evidence
    assert result.status == "pending_review" and result.pending_reviews[0].status == "pending"
    with pytest.raises(ValidationError):
        type(result.pending_reviews[0]).model_validate({**result.pending_reviews[0].model_dump(), "status": "approved"})
    assert advance(result, execution) == result


@pytest.mark.parametrize("changes", [{"model_calls": 1}, {"input_tokens": 1}, {"output_tokens": 1}, {"usage_unknown": True}])
def test_saved_usage_tampering_rejected_before_dispatch(workflow_case, changes):
    context, state, _ = workflow_case
    with pytest.raises(ValueError):
        advance(state.model_copy(update=changes), runtime(context))
    assert not context.checker.requests


def test_unavailable_proposal_or_definition_catalog_rejected_before_dispatch(workflow_case):
    context, state, _ = workflow_case
    for changes in ({"proposal_ids": ("unknown",)}, {"definition_keys": ("unknown",)}):
        altered = state.model_copy(update={"issues": (state.issues[0].model_copy(update=changes),)})
        with pytest.raises(ValueError):
            advance(altered, runtime(context))
    assert not context.checker.requests


def test_uncertain_prior_model_reservations_cannot_reset_usage_truth(workflow_case):
    context, state, _ = workflow_case
    state = state.model_copy(update={"uncertain_model_calls": 1, "usage_unknown": True, "status": "interrupted"})
    assert advance(state, runtime(context, max_model_calls=1)) == state
    with pytest.raises(ValueError):
        advance(state, runtime(context, max_model_calls=0))


def test_policy_timeout_exception_is_counted_as_one_unknown_attempt(workflow_case):
    context, state, _ = workflow_case

    class TimeoutPolicy:
        mode = "mock"
        model_name = "synthetic-timeout-policy"
        calls = 0

        def generate(self, prompt):
            self.calls += 1
            raise TimeoutError("private provider detail")

    provider = TimeoutPolicy()
    result = advance(state, runtime(context, policy="generic_llm", provider=provider))
    assert result.status == "failed" and result.stop_reason == "policy_timeout"
    assert result.model_calls == provider.calls == 1 and result.usage_unknown
    assert "private provider detail" not in result.model_dump_json()
    assert not context.checker.requests


def test_deadline_expiring_inside_checker_call_prevents_patch(workflow_case, monkeypatch):
    context, state, _ = workflow_case
    provider = MockProvider([selected({"kind": "propose_update", "issue_id": "issue-0", "proposal_id": "correction"})])
    execution = runtime(context, policy="generic_llm", provider=provider)
    generate = context.checker.generate

    def expire(prompt):
        result = generate(prompt)
        execution.deadline_unix = time.time() - 1
        return result

    monkeypatch.setattr(context.checker, "generate", expire)
    result = advance(state, execution)
    assert result.status == "timed_out" and result.model_calls == 2
    assert len(context.checker.requests) == 1 and result.record == state.record
    assert result.events[0].receipt_id is None


def test_no_match_retrieval_is_bounded_no_progress_not_a_resolved_issue(workflow_case):
    context, state, _ = workflow_case
    action = {"kind": "retrieve_source", "issue_id": "issue-0", "query": "unfindablefixturetoken"}
    provider = MockProvider([selected(action) for _ in range(4)])
    result = until_stopped(state, runtime(context, policy="generic_llm", provider=provider))
    assert result.status == "no_progress" and result.model_calls == 2
    assert all(event.code == "source_no_match" for event in result.events)
    assert result.record == state.record and all(issue.status == "open" for issue in result.issues)


def test_running_uncertain_reservation_stops_before_another_model_request(workflow_case):
    context, state, _ = workflow_case
    provider = MockProvider([selected(retrieval())])
    state = state.model_copy(update={"uncertain_model_calls": 1, "usage_unknown": True})
    result = advance(state, runtime(context, policy="generic_llm", provider=provider,
                                    max_model_calls=1, unknown_usage="allow_bounded_calls"))
    assert result.status == "interrupted" and result.record == state.record
    assert not provider.requests and not context.checker.requests
    assert result.model_calls == 0 and result.uncertain_model_calls == 1


@pytest.mark.parametrize("prior_path,prior_status,fresh_support", [
    ("/datasets/0/measurement_technique", "resolved", False),
    ("/datasets/0/measurement_technique", "open", False),
    ("/datasets/0/facility_refs/0", "resolved", True),
])
def test_revision_change_reopens_stale_resolutions_but_keeps_freshly_rechecked_support(
        workflow_case, prior_path, prior_status, fresh_support):
    context, initial, prepared = workflow_case
    prior_issue = initial.issues[0].model_copy(update={
        "id": "previous-issue", "path": prior_path, "status": prior_status,
        "diagnosis": "unsupported_value", "proposal_ids": (),
        "summary": "Invented prior issue status against the original revision",
    })
    initial = initial.model_copy(update={"issues": (*initial.issues, prior_issue)})
    provider = MockProvider([selected({"kind": "propose_update", "issue_id": "issue-0", "proposal_id": "correction"})])
    result = advance(initial, runtime(context, policy="generic_llm", provider=provider))
    assert result.record == prepared.candidate and result.record.revision_id != initial.record.revision_id
    revised_issue = next(i for i in result.issues if i.id == "previous-issue")
    after = result.events[-1].support_reports[-1]
    assert (prior_path in {j.path for j in after.judgments if j.verdict == "supported"}) == fresh_support
    if fresh_support:
        assert revised_issue.status == "resolved"
    else:
        assert revised_issue.status == "open" and revised_issue.diagnosis == "stale_after_revision"
        assert result.status != "finished"
    assert initial.issues[-1] == prior_issue
