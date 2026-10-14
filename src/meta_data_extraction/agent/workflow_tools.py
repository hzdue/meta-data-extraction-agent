"""Allowlisted M07 tool adapters over source, standards and M06 correction APIs.

This layer never chooses an action or treats standards definitions as source
evidence. Provider calls occur only in the bounded M06 correction checks.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import TypeAdapter

from meta_data_extraction.agent.contracts import ProviderObservation, RecordSnapshot, digest
from meta_data_extraction.agent.evidence import DocumentBundle, EvidenceRef, FrozenModel, resolve_evidence
from meta_data_extraction.agent.nexus_registry import Guidance, ProfileRegistry, SemanticRole, lookup_guidance
from meta_data_extraction.agent.nexus_resources import NXDLSnapshot, verify_resources
from meta_data_extraction.agent.profiles import Profile, verify_profile_resources
from meta_data_extraction.agent.providers import ChatAIProvider, Provider
from meta_data_extraction.agent.record_checks import _checked_profile, check_structure
from meta_data_extraction.agent.record_patches import (
    PatchPolicy, PatchProposal, apply_supported_patch, build_candidate,
)
from meta_data_extraction.agent.retrieval import Selection, SourceCorpus
from meta_data_extraction.agent.support_checks import SupportReport, SupportTarget, check_support
from meta_data_extraction.agent.workflow_types import Action, WorkflowState, validate_action


class ToolOutcome(FrozenModel):
    status: Literal["ok", "blocked", "failed"]
    code: str
    summary: str
    current: RecordSnapshot
    evidence: tuple[EvidenceRef, ...] = ()
    selection: Selection | None = None
    definition_keys: tuple[str, ...] = ()
    definitions: tuple[Guidance, ...] = ()
    resolved_paths: tuple[str, ...] = ()
    providers: tuple[ProviderObservation, ...] = ()
    reports: tuple[SupportReport, ...] = ()
    receipt_id: str | None = None


def _key(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError("Tool keys must be explicit bounded strings")
    return value


def _provider_identity(provider: Provider) -> dict:
    if provider.mode not in {"mock", "live"} or not provider.model_name:
        raise ValueError("Unknown checker provider mode or model")
    if provider.mode == "live" and type(provider) is not ChatAIProvider:
        raise ValueError("Live tools require the guarded ChatAI provider")
    return dict(
        model=provider.model_name, mode=provider.mode,
        configuration=getattr(provider, "configuration_fingerprint", None),
        scope=provider.spec.model_dump(mode="json") if type(provider) is ChatAIProvider else None,
    )


def _scope(context: ToolContext) -> str:
    return digest({
        "root": str(context.root), "profile": context.profile.model_dump(mode="json"),
        "bundle": context.bundle.model_dump(mode="json"),
        "proposals": {key: value.model_dump(mode="json") for key, value in context.proposals.items()},
        "provider": _provider_identity(context.checker),
        "patch_policy": context.patch_policy.model_dump(mode="json"),
        "nxdl": context.nxdl.model_dump(mode="json") if context.nxdl is not None else None,
        "registry": context.registry.model_dump(mode="json") if context.registry is not None else None,
        "definition_requests": dict(context.definition_requests),
    })


@dataclass(frozen=True)
class ToolContext:
    root: Path | str
    profile: Profile
    bundle: DocumentBundle
    proposals: Mapping[str, PatchProposal]
    checker: Provider
    patch_policy: PatchPolicy
    nxdl: NXDLSnapshot | None = None
    registry: ProfileRegistry | None = None
    definition_requests: Mapping[str, tuple[str, SemanticRole]] = field(default_factory=dict)
    _scope_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "profile", _checked_profile(self.profile))
        object.__setattr__(self, "bundle", DocumentBundle.model_validate(self.bundle.model_dump()))
        object.__setattr__(self, "patch_policy", PatchPolicy.model_validate(self.patch_policy.model_dump()))
        if len(self.proposals) > 32 or len(self.definition_requests) > 32:
            raise ValueError("Tool context permits at most 32 frozen proposals/definition requests")
        proposals = {_key(key): PatchProposal.model_validate(value.model_dump()) for key, value in self.proposals.items()}
        definitions = {}
        for key, value in self.definition_requests.items():
            if not isinstance(value, tuple) or len(value) != 2 or not isinstance(value[0], str):
                raise ValueError("Definition requests require a canonical field and semantic role")
            definitions[_key(key)] = (value[0], TypeAdapter(SemanticRole).validate_python(value[1]))
        object.__setattr__(self, "proposals", MappingProxyType(proposals))
        object.__setattr__(self, "definition_requests", MappingProxyType(definitions))
        if (self.nxdl is None) != (self.registry is None):
            raise ValueError("NXDL snapshot and reviewed registry must be supplied together")
        if self.nxdl is not None:
            object.__setattr__(self, "nxdl", NXDLSnapshot.model_validate(self.nxdl.model_dump()))
            object.__setattr__(self, "registry", ProfileRegistry.model_validate(self.registry.model_dump()))
            if (self.registry.profile_id, self.registry.resource_snapshot_id) != (
                self.profile.profile_id, self.nxdl.snapshot_id,
            ):
                raise ValueError("Definition registry differs from frozen profile/resource")
        verify_profile_resources(self.profile, self.root)
        object.__setattr__(self, "_scope_id", _scope(self))


def _validate_context(context: ToolContext, state: WorkflowState) -> WorkflowState:
    if type(context) is not ToolContext or _scope(context) != context._scope_id:
        raise ValueError("Frozen tool context changed")
    _checked_profile(context.profile)
    DocumentBundle.model_validate(context.bundle.model_dump())
    verify_profile_resources(context.profile, context.root)
    state = WorkflowState.model_validate(state.model_dump())
    checked = check_structure(context.profile, context.bundle, state.record.canonical)
    if checked.status != "valid" or checked.snapshot != state.record:
        raise ValueError("Workflow record differs from the frozen profile/source")
    return state


def _stop(should_stop: Callable[[], str | None]) -> str | None:
    try:
        value = should_stop()
    except Exception:
        return "stop_check_failed"
    if value is None:
        return None
    return value if isinstance(value, str) and value in {"timeout", "cancelled"} else "stopped"


def execute_tool(
    context: ToolContext, action: Action, state: WorkflowState, *,
    model_budget: int, should_stop: Callable[[], str | None] = lambda: None,
    unknown_usage: Literal["stop", "allow_bounded_calls"] = "stop",
) -> ToolOutcome:
    """Execute one selected tool; persist/account its outcome in the coordinator.

    Two checker-call slots must be available before a correction starts. Calls
    are never retried; cancellation/timeout after either check prevents mutation.
    Definitions remain separate typed guidance even when their text has numbers.
    """
    current = RecordSnapshot.model_validate(state.record.model_dump())
    reports: list[SupportReport] = []

    def outcome(status: str, code: str, summary: str, **values) -> ToolOutcome:
        return ToolOutcome(
            status=status, code=code, summary=summary, current=values.pop("current", current),
            providers=tuple(report.provider for report in reports if report.provider is not None),
            reports=tuple(reports), **values,
        )

    if not isinstance(unknown_usage, str) or unknown_usage not in {"stop", "allow_bounded_calls"}:
        return outcome("blocked", "invalid_usage_policy", "Choose an explicit unknown-usage policy")
    if type(model_budget) is not int or model_budget < 0:
        return outcome("blocked", "invalid_model_budget", "A nonnegative integer model-call budget is required")
    try:
        state = _validate_context(context, state)
        action = validate_action(action, state)
    except (ValueError, TypeError, KeyError, AttributeError):
        return outcome("blocked", "invalid_tool_request", "Tool request or frozen input binding is invalid")
    stopped = _stop(should_stop)
    if stopped:
        return outcome("blocked", stopped, "Tool execution stopped before dispatch")
    if action.kind not in {"retrieve_source", "consult_definition", "propose_update"}:
        return outcome("blocked", "coordinator_action", "This action is handled by the workflow coordinator")
    issue = next(issue for issue in state.issues if issue.id == action.issue_id)

    if action.kind == "retrieve_source":
        try:
            selection = SourceCorpus(bundles=(context.bundle,)).search(
                action.query, max_chars=action.max_chars, limit=action.limit,
                source_ids=(context.bundle.source_id,),
            )
        except (ValueError, TypeError, KeyError):
            return outcome("failed", "source_retrieval_failed", "Requested source evidence could not be selected")
        stopped = _stop(should_stop)
        if stopped:
            return outcome("blocked", stopped, "Source inspection stopped before accepting its result")
        evidence = tuple(passage.evidence for passage in selection.passages)
        if not evidence:
            return outcome("blocked", "source_no_match", "No supplied source region matches the query", selection=selection)
        return outcome(
            "ok", "source_inspected",
            "Source excerpts inspected; lexical matches alone do not establish scientific support",
            evidence=evidence, selection=selection,
        )

    if action.kind == "consult_definition":
        if context.nxdl is None or action.definition_key not in context.definition_requests:
            return outcome("blocked", "approved_definition_unavailable", "No approved definition is configured for this key")
        try:
            verify_resources(context.nxdl, context.root)
            canonical_field, role = context.definition_requests[action.definition_key]
            definitions = lookup_guidance(
                context.profile, context.nxdl, context.registry, canonical_field, role,
                include_proposals=False,
            )
            if any(item.approval_status != "approved" or item.resolution_status != "resolved" or item.unresolved for item in definitions):
                raise ValueError("Unapproved or unresolved guidance")
        except (ValueError, TypeError, KeyError, OSError):
            return outcome("blocked", "definition_binding_invalid", "Approved definition resource or review bindings are unavailable")
        stopped = _stop(should_stop)
        if stopped:
            return outcome("blocked", stopped, "Definition inspection stopped before accepting its result")
        if not definitions:
            return outcome("blocked", "approved_definition_unavailable", "Proposed or unresolved mappings do not supply runtime guidance")
        return outcome(
            "ok", "definition_inspected", "Approved representation guidance inspected; publication evidence is still required",
            definition_keys=(action.definition_key,), definitions=definitions,
        )

    if action.proposal_id not in context.proposals:
        return outcome("blocked", "proposal_unavailable", "No frozen correction proposal exists for this key")
    proposal = context.proposals[action.proposal_id]
    open_paths = {item.path for item in state.issues if item.status == "open"}
    if any(op.path not in open_paths for op in proposal.operations):
        return outcome("blocked", "proposal_outside_open_issues", "Every correction must target an explicitly open issue")
    if not context.patch_policy.accept_model_judgments:
        return outcome("blocked", "model_judgment_policy_disabled", "Automated correction policy is disabled")
    if model_budget < 2:
        return outcome("blocked", "model_budget_exhausted", "Two checker-call slots are required before correction")
    if not issue.evidence:
        return outcome("blocked", "source_evidence_required", "Inspect publication source evidence before proposing a correction")
    if len(issue.evidence) > 8:
        return outcome("blocked", "evidence_budget_exceeded", "Selected correction evidence exceeds the eight-link bound")
    try:
        for ref in issue.evidence:
            resolve_evidence(context.bundle, ref)
        built = build_candidate(context.profile, context.bundle, current, proposal)
    except (ValueError, TypeError, KeyError):
        return outcome("blocked", "candidate_binding_invalid", "Correction proposal or inspected publication evidence is invalid")
    if built.status != "ready":
        return outcome("blocked", built.code, "Correction candidate was not ready; the record is unchanged")
    prepared = built.prepared
    assert prepared is not None
    before_targets = tuple(SupportTarget(path=p, evidence=issue.evidence) for p in prepared.changed_paths)
    after_targets = tuple(SupportTarget(path=p, evidence=issue.evidence) for p in prepared.required_support_paths)
    for candidate, targets, stage in (
        (prepared.original, before_targets, "before"),
        (prepared.candidate, after_targets, "after"),
    ):
        stopped = _stop(should_stop)
        if stopped:
            return outcome("blocked", stopped, "Correction stopped before checker dispatch")
        try:
            report = check_support(
                context.root, context.profile, candidate, context.bundle, targets,
                context.checker, raw_retention="hash_only",
            )
        except (ValueError, TypeError, KeyError, OSError):
            return outcome("blocked", "support_checker_rejected", "Checker scope or frozen input was rejected before a usable result")
        reports.append(report)
        stopped = _stop(should_stop)
        if stopped:
            return outcome("blocked", stopped, "Correction stopped after checking; no patch was applied")
        if report.status != "succeeded":
            return outcome("blocked", "support_check_incomplete", "Checker did not produce a complete usable support judgment")
        if unknown_usage == "stop" and report.provider is not None and (
            report.provider.input_tokens is None or report.provider.output_tokens is None
        ):
            return outcome("blocked", "unknown_provider_usage", "Provider usage is unknown; no further check or patch is allowed")
        if stage == "before" and any(
            judgment.verdict != "contradicted" or not judgment.evidence or judgment.diagnosis == "context_conflict"
            for judgment in report.judgments
        ):
            return outcome("blocked", "original_not_contradicted", "Original assertions require clear source contradictions before correction")
    patched = apply_supported_patch(
        context.profile, context.bundle, current, prepared, reports[0], reports[1],
        policy=context.patch_policy,
    )
    if patched.status not in {"applied", "replayed"}:
        return outcome("blocked", patched.code, "Source-supported correction gate rejected the candidate")
    return outcome(
        "ok", "patch_applied", "Bounded correction applied with fresh automated support checks; expert approval remains absent",
        current=patched.snapshot, resolved_paths=prepared.changed_paths, receipt_id=patched.receipt.receipt_id,
    )
