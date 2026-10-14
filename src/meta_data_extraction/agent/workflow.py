"""One bounded transition over ordinary functions; no LangGraph dependency.

The persistence adapter must claim a transition before calling advance. Budgets
and stopping rules belong here, never to the selected policy or document text.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

from .contracts import ProviderObservation, digest
from .evidence import resolve_evidence, sha256
from .providers import ChatAIProvider, MockProvider, Provider, ProviderResult
from .record_checks import check_structure
from .profiles import verify_profile_resources
from .workflow_policies import parse_policy_response, render_policy_prompt, select_action, validate_action
from .workflow_tools import ToolContext, execute_tool
from .workflow_types import ActionEvent, PolicyName, ReviewCase, WorkflowBudget, WorkflowState


class WorkflowMockProvider:
    """Full-script fingerprint with explicit no-dispatch checkpoint positioning."""
    mode = "mock"

    def __init__(self, results: tuple[ProviderResult, ...], *, model_name: str):
        self.results = tuple(ProviderResult.model_validate(r.model_dump()) for r in results)
        self.model_name = model_name
        self._delegate = MockProvider(self.results, model_name=model_name)
        self.configuration_fingerprint = self._delegate.configuration_fingerprint
        self.position = 0
        self._requests: list[str] = []

    @property
    def requests(self) -> tuple[str, ...]:
        return tuple(self._requests)

    def seek(self, completed_calls: int) -> None:
        if type(completed_calls) is not int or completed_calls < 0:
            raise ValueError("Invalid completed mock-call count")
        if completed_calls == self.position:
            return
        if self._requests:
            raise ValueError("Cannot rewind or reposition an already-dispatched mock provider")
        self._delegate = MockProvider(self.results[completed_calls:], model_name=self.model_name)
        self.position = completed_calls

    def generate(self, prompt: str) -> ProviderResult:
        self._requests.append(prompt)
        self.position += 1
        return self._delegate.generate(prompt)


@dataclass
class WorkflowRuntime:
    tools: ToolContext
    budget: WorkflowBudget
    policy: PolicyName
    policy_provider: Provider | None = None
    cancelled: Callable[[], bool] = lambda: False
    deadline_unix: float | None = None

    def __post_init__(self) -> None:
        self.budget = WorkflowBudget.model_validate(self.budget.model_dump())
        if self.policy not in {"fixed", "issue_aware", "generic_llm"}:
            raise ValueError("Unknown workflow policy")
        if (self.policy == "generic_llm") != (self.policy_provider is not None):
            raise ValueError("Only the generic model policy needs its explicit provider")

    def position_mocks(self, state: WorkflowState) -> None:
        checker_calls = sum(report.provider is not None for event in state.events for report in event.support_reports)
        policy_calls = state.model_calls - checker_calls
        for provider, count in ((self.tools.checker, checker_calls), (self.policy_provider, policy_calls)):
            if isinstance(provider, WorkflowMockProvider):
                provider.seek(count)
            elif isinstance(provider, MockProvider) and len(provider.requests) != count:
                raise ValueError("Use WorkflowMockProvider for cross-process mock resumption")

    def should_stop(self) -> str | None:
        if self.cancelled():
            return "cancelled"
        if self.deadline_unix is not None and time.time() >= self.deadline_unix:
            return "timeout"
        return None


def _replace(state: WorkflowState, **changes) -> WorkflowState:
    return WorkflowState.model_validate({**state.model_dump(), **changes})


def verify_state(state: WorkflowState, runtime: WorkflowRuntime) -> None:
    state = WorkflowState.model_validate(state.model_dump())
    context = runtime.tools
    verify_profile_resources(context.profile, context.root)
    structure = check_structure(context.profile, context.bundle, state.record.canonical)
    if structure.snapshot != state.record:
        raise ValueError("Workflow record differs from frozen source/profile")
    for issue in state.issues:
        if not set(issue.proposal_ids) <= set(context.proposals) or not set(issue.definition_keys) <= set(context.definition_requests):
            raise ValueError("Issue refers to an unavailable tool catalog entry")
        for ref in issue.evidence:
            resolve_evidence(context.bundle, ref)
    for event in state.events:
        for ref in event.evidence:
            resolve_evidence(context.bundle, ref)
        checker_observations = tuple(r.provider for r in event.support_reports if r.provider is not None)
        if event.policy != runtime.policy:
            raise ValueError("Saved event used a different policy")
        if runtime.policy != "generic_llm":
            if event.providers != checker_observations:
                raise ValueError("Checker accounting differs from support reports")
        else:
            if event.providers:
                if event.providers[0].model != runtime.policy_provider.model_name or event.providers[1:] != checker_observations:
                    raise ValueError("Policy/checker observations differ from dispatch roles")
            elif checker_observations:
                raise ValueError("A generic policy cannot dispatch checks without a policy call")
    for case in state.pending_reviews:
        for ref in case.evidence:
            resolve_evidence(context.bundle, ref)
    observations = tuple(p for event in state.events for p in event.providers)
    if state.model_calls != len(observations):
        raise ValueError("Workflow model-call accounting differs from recorded observations")
    if state.input_tokens != sum(p.input_tokens or 0 for p in observations) or state.output_tokens != sum(p.output_tokens or 0 for p in observations):
        raise ValueError("Known token accounting differs from recorded observations")
    unknown = bool(state.uncertain_model_calls) or any(p.input_tokens is None or p.output_tokens is None for p in observations)
    if state.usage_unknown != unknown:
        raise ValueError("Unknown usage was silently reset or invented")
    if len(state.events) > runtime.budget.total_actions or state.model_calls + state.uncertain_model_calls > runtime.budget.max_model_calls:
        raise ValueError("Saved state exceeds its frozen run budget")


def _model_selection(state: WorkflowState, runtime: WorkflowRuntime):
    provider = runtime.policy_provider
    prompt = render_policy_prompt(state)
    prompt_hash = sha256(prompt.encode("utf-8"))
    if provider.mode == "live":
        if type(provider) is not ChatAIProvider:
            raise ValueError("Generic live selection requires guarded ChatAI provider")
        if (provider.spec.profile_sha256 != runtime.tools.profile.profile_id.removeprefix("profile:")
            or (runtime.tools.bundle.source_id, runtime.tools.bundle.source_sha256) not in {(s.path, s.sha256) for s in provider.spec.sources}
            or prompt_hash not in provider.spec.approved_prompt_sha256):
            raise ValueError("Generic policy input differs from the approved live scope")
    elif provider.mode != "mock":
        raise ValueError("Unknown model policy mode")
    try:
        response = ProviderResult.model_validate(provider.generate(prompt).model_dump())
        if response.mode != provider.mode:
            response = ProviderResult(status="failed", mode=provider.mode, error_code="provider_mode_mismatch")
    except TimeoutError:
        response = ProviderResult(status="timeout", mode=provider.mode, error_code="policy_timeout")
    except Exception:
        response = ProviderResult(status="failed", mode=provider.mode, error_code="policy_adapter_exception")
    observation = ProviderObservation(**response.model_dump(exclude={"content"}), model=provider.model_name,
        response_sha256=sha256(response.content.encode("utf-8")) if response.content is not None else None)
    if response.status != "ok" or response.content is None:
        return None, observation, prompt_hash, response.error_code or "policy_response_missing"
    if response.finish_reason != "stop":
        return None, observation, prompt_hash, "policy_completion_incomplete"
    try:
        action = validate_action(parse_policy_response(response.content), state)
    except (ValueError, TypeError, KeyError, RecursionError):
        return None, observation, prompt_hash, "invalid_policy_action"
    return action, observation, prompt_hash, None


def advance(state: WorkflowState, runtime: WorkflowRuntime) -> WorkflowState:
    """Run at most one selected action; all stopping states preserve unresolved issues."""
    started = time.perf_counter()
    state = WorkflowState.model_validate(state.model_dump())
    verify_state(state, runtime)
    if state.status != "running":
        return state

    def stop(status: str, code: str) -> WorkflowState:
        return _replace(state, status=status, stop_reason=code,
                        elapsed_ms=state.elapsed_ms + (time.perf_counter() - started) * 1000)

    if state.uncertain_model_calls:
        return stop("interrupted", "uncertain_prior_dispatch")
    stopped = runtime.should_stop()
    if stopped:
        return stop("timed_out" if stopped == "timeout" else "cancelled", stopped)
    if len(state.events) >= runtime.budget.total_actions:
        return stop("budget_exhausted", "total_action_budget")
    if state.no_progress_steps >= runtime.budget.no_progress_limit:
        return stop("no_progress", "no_new_evidence_or_state")
    providers = ()
    prompt_hash = None
    selection_error = None
    action = None
    if runtime.policy == "generic_llm":
        if state.model_calls + state.uncertain_model_calls >= runtime.budget.max_model_calls:
            return stop("budget_exhausted", "model_call_budget")
        try:
            action, observation, prompt_hash, selection_error = _model_selection(state, runtime)
            providers = (observation,)
        except (ValueError, TypeError, KeyError):
            selection_error = "invalid_policy_input"
    else:
        try:
            action = validate_action(select_action(state, runtime.policy), state)
        except (ValueError, TypeError, KeyError):
            selection_error = "invalid_policy_action"

    def emit(*, status="ok", code, summary, current=None, issues=None, reviews=None,
             evidence=(), definition_keys=(), definitions=(), support_reports=(), selection=None,
             receipt_id=None, progress=False, terminal=None, reason=None) -> WorkflowState:
        current = current or state.record
        event = ActionEvent(index=len(state.events), policy=runtime.policy, action=action,
            status=status, code=code, revision_before=state.record.revision_id,
            revision_after=current.revision_id, progress=progress, evidence=evidence,
            definition_keys=definition_keys, definitions=definitions, support_reports=support_reports,
            selection=selection, providers=providers, summary=summary, prompt_sha256=prompt_hash, receipt_id=receipt_id)
        unknown = state.usage_unknown or any(p.input_tokens is None or p.output_tokens is None for p in providers)
        result = _replace(state, record=current, issues=issues if issues is not None else state.issues,
            pending_reviews=reviews if reviews is not None else state.pending_reviews,
            events=(*state.events, event), model_calls=state.model_calls + len(providers),
            input_tokens=state.input_tokens + sum(p.input_tokens or 0 for p in providers),
            output_tokens=state.output_tokens + sum(p.output_tokens or 0 for p in providers),
            usage_unknown=unknown, no_progress_steps=0 if progress else state.no_progress_steps + 1,
            status=terminal or "running", stop_reason=reason or (code if terminal else ""),
            elapsed_ms=state.elapsed_ms + (time.perf_counter() - started) * 1000)
        if result.status == "running" and unknown and runtime.budget.unknown_usage == "stop":
            result = _replace(result, status="unresolved", stop_reason="unknown_provider_usage")
        verify_state(result, runtime)
        return result

    if selection_error:
        return emit(status="failed", code=selection_error, summary="Policy returned no permitted action", terminal="failed")
    stopped = runtime.should_stop()
    if stopped:
        return emit(status="blocked", code=stopped, summary="Stopped after policy selection",
                    terminal="timed_out" if stopped == "timeout" else "cancelled")
    if providers and runtime.budget.unknown_usage == "stop" and any(p.input_tokens is None or p.output_tokens is None for p in providers):
        return emit(status="blocked", code="unknown_provider_usage", summary="Unknown policy usage prevents tool dispatch", terminal="unresolved")
    if action.kind == "finish":
        terminal = "finished" if action.outcome == "finished" else "pending_review" if state.pending_reviews else "unresolved"
        return emit(code="policy_finish", summary=action.summary, progress=True, terminal=terminal)
    issue = next(i for i in state.issues if i.id == action.issue_id)
    issue_actions = sum(e.action is not None and e.action.kind != "finish" and e.action.issue_id == issue.id for e in state.events)
    if issue_actions >= runtime.budget.per_issue_actions:
        if providers:
            return emit(status="blocked", code="per_issue_action_budget", summary="Per-issue budget prevents dispatch", terminal="budget_exhausted")
        return stop("budget_exhausted", "per_issue_action_budget")
    if action.kind == "request_review":
        payload = dict(issue_id=issue.id, record_revision=state.record.revision_id,
                       path=issue.path, summary=action.summary, evidence=issue.evidence)
        case = ReviewCase(case_id="review:" + digest({**payload, "evidence": [r.model_dump(mode="json") for r in issue.evidence]}), **payload)
        updated = tuple(i.model_copy(update={"status": "pending_review"}) if i.id == issue.id else i for i in state.issues)
        reviews = (*state.pending_reviews, case)
        terminal = "pending_review" if not any(i.status == "open" for i in updated) else None
        return emit(code="review_pending", summary=action.summary, issues=updated, reviews=reviews,
                    progress=True, terminal=terminal)
    remaining = runtime.budget.max_model_calls - state.model_calls - state.uncertain_model_calls - len(providers)
    if (action.kind == "propose_update" and runtime.tools.patch_policy.accept_model_judgments
            and remaining < 2):
        if providers:
            return emit(status="blocked", code="model_call_budget", summary="Insufficient checker-call budget", terminal="budget_exhausted")
        return stop("budget_exhausted", "model_call_budget")
    outcome = execute_tool(runtime.tools, action, state, model_budget=remaining, should_stop=runtime.should_stop,
                           unknown_usage=runtime.budget.unknown_usage)
    providers = (*providers, *outcome.providers)
    revision_changed = outcome.current.revision_id != state.record.revision_id
    fresh_supported = {judgment.path for report in outcome.reports
                       if report.record_revision == outcome.current.revision_id and report.status == "succeeded"
                       for judgment in report.judgments if judgment.verdict == "supported"}
    updated = []
    for item in state.issues:
        values = item.model_dump()
        if item.id == issue.id:
            refs = {digest(ref.model_dump(mode="json")): ref for ref in (*item.evidence, *outcome.evidence)}
            # Bound future checker evidence to the most recent eight inspected regions.
            values["evidence"] = tuple(refs.values())[-8:]
            values["consulted_definitions"] = tuple(dict.fromkeys((*item.consulted_definitions, *outcome.definition_keys)))
        if item.status == "open" and item.path in outcome.resolved_paths:
            values["status"] = "resolved"
        elif revision_changed and item.status != "pending_review" and not (
            item.status == "resolved" and item.path in fresh_supported
        ):
            values["status"] = "open"
            values["diagnosis"] = "stale_after_revision"
            values["summary"] = "Earlier issue state requires a fresh check or review after the record revision changed"
        updated.append(type(item).model_validate(values))
    seen = {digest(ref.model_dump(mode="json")) for ref in issue.evidence}
    seen.update(digest(ref.model_dump(mode="json")) for event in state.events
                if event.action is not None and event.action.kind != "finish" and event.action.issue_id == issue.id
                for ref in event.evidence)
    progress = (outcome.current.revision_id != state.record.revision_id
                or any(digest(ref.model_dump(mode="json")) not in seen for ref in outcome.evidence)
                or bool(set(outcome.definition_keys) - set(issue.consulted_definitions))
                or any(a.status != b.status for a, b in zip(updated, state.issues, strict=True)))
    terminal = "timed_out" if outcome.code == "timeout" else "cancelled" if outcome.code == "cancelled" else None
    return emit(status=outcome.status, code=outcome.code, summary=outcome.summary,
                current=outcome.current, issues=tuple(updated), evidence=outcome.evidence,
                definition_keys=outcome.definition_keys, definitions=outcome.definitions,
                support_reports=outcome.reports, selection=outcome.selection,
                receipt_id=outcome.receipt_id, progress=progress, terminal=terminal)
