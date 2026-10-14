"""Typed, bounded M07 workflow state and allowlisted action arguments.

These contracts grant no tool permission or human approval. The executor owns
resource budgets and verifies effects against source and record snapshots.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, TypeAdapter, field_validator, model_validator

from .contracts import ProviderObservation, RecordSnapshot, digest
from .evidence import EvidenceRef, FrozenModel
from .nexus_registry import Guidance
from .support_checks import SupportReport
from .retrieval import Selection

PolicyName = Literal["fixed", "issue_aware", "generic_llm"]
Nonempty = Annotated[str, Field(min_length=1, max_length=500)]


class _Action(FrozenModel):
    @field_validator("*", mode="after")
    @classmethod
    def _nonblank(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Action text must not be blank")
        return value


class RetrieveSource(_Action):
    kind: Literal["retrieve_source"] = "retrieve_source"
    issue_id: Nonempty
    query: Nonempty
    max_chars: int = Field(default=4000, ge=1, le=8000, strict=True)
    limit: int = Field(default=3, ge=1, le=5, strict=True)


class ConsultDefinition(_Action):
    kind: Literal["consult_definition"] = "consult_definition"
    issue_id: Nonempty
    definition_key: Nonempty


class ProposeUpdate(_Action):
    kind: Literal["propose_update"] = "propose_update"
    issue_id: Nonempty
    proposal_id: Nonempty


class RequestReview(_Action):
    kind: Literal["request_review"] = "request_review"
    issue_id: Nonempty
    summary: Nonempty


class Finish(_Action):
    kind: Literal["finish"] = "finish"
    outcome: Literal["finished", "unresolved"]
    summary: Nonempty


Action = Annotated[Union[RetrieveSource, ConsultDefinition, ProposeUpdate, RequestReview, Finish],
                   Field(discriminator="kind")]
WorkflowAction = Action
ACTION_ADAPTER = TypeAdapter(Action)


class WorkflowIssue(FrozenModel):
    id: Nonempty
    path: str = Field(max_length=500)
    diagnosis: str = Field(min_length=1, max_length=100)
    summary: Nonempty
    query: Nonempty
    proposal_ids: tuple[Nonempty, ...] = Field(default=(), max_length=32)
    definition_keys: tuple[Nonempty, ...] = Field(default=(), max_length=32)
    status: Literal["open", "resolved", "pending_review"] = "open"
    evidence: tuple[EvidenceRef, ...] = Field(default=(), max_length=128)
    consulted_definitions: tuple[Nonempty, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def _scope(self) -> WorkflowIssue:
        if any(not value.strip() for value in (self.id, self.diagnosis, self.summary, self.query)):
            raise ValueError("Issue descriptions and identifiers must not be blank")
        if self.path and not self.path.startswith("/"):
            raise ValueError("Issue path must be a JSON Pointer")
        for values in (self.proposal_ids, self.definition_keys, self.consulted_definitions):
            if len(set(values)) != len(values) or any(not value.strip() for value in values):
                raise ValueError("Issue catalog entries must be distinct nonblank identifiers")
        if not set(self.consulted_definitions).issubset(self.definition_keys):
            raise ValueError("Consulted definitions must belong to the issue catalog")
        if len({digest(ref.model_dump(mode="json")) for ref in self.evidence}) != len(self.evidence):
            raise ValueError("Inspected evidence must be distinct")
        return self


class WorkflowBudget(FrozenModel):
    total_actions: int = Field(default=12, ge=1, le=32, strict=True)
    per_issue_actions: int = Field(default=5, ge=1, le=8, strict=True)
    max_model_calls: int = Field(default=8, ge=0, le=32, strict=True)
    no_progress_limit: int = Field(default=2, ge=1, le=4, strict=True)
    timeout_seconds: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)
    unknown_usage: Literal["stop", "allow_bounded_calls"] = "stop"

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _numeric_timeout(cls, value):
        if type(value) not in {int, float}:
            raise ValueError("Timeout must be an explicit numeric duration")
        return value


class ReviewCase(FrozenModel):
    case_id: Nonempty
    issue_id: Nonempty
    record_revision: Nonempty
    path: str = Field(max_length=500)
    summary: Nonempty
    evidence: tuple[EvidenceRef, ...] = Field(default=(), max_length=128)
    status: Literal["pending"] = "pending"


class ActionEvent(FrozenModel):
    index: int = Field(ge=0, strict=True)
    policy: PolicyName
    action: Action | None = None
    status: Literal["ok", "blocked", "failed"]
    code: Nonempty
    revision_before: Nonempty
    revision_after: Nonempty
    progress: bool = Field(strict=True)
    evidence: tuple[EvidenceRef, ...] = Field(default=(), max_length=128)
    definition_keys: tuple[Nonempty, ...] = Field(default=(), max_length=32)
    providers: tuple[ProviderObservation, ...] = Field(default=(), max_length=32)
    definitions: tuple[Guidance, ...] = Field(default=(), max_length=32)
    selection: Selection | None = None
    support_reports: tuple[SupportReport, ...] = Field(default=(), max_length=32)
    summary: Nonempty
    prompt_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    receipt_id: Nonempty | None = None

    @model_validator(mode="after")
    def _unselected(self) -> ActionEvent:
        if self.action is None and (self.status == "ok" or self.progress or self.receipt_id is not None
                                    or self.evidence or self.definition_keys or self.definitions or self.support_reports or self.selection is not None
                                    or self.revision_before != self.revision_after):
            raise ValueError("An unselected action can only record a blocked or failed policy observation")
        if any(report.raw_output is not None or report.raw_retention != "hash_only"
               for report in self.support_reports):
            raise ValueError("Persisted workflow support reports require hash-only retention")
        return self


class WorkflowState(FrozenModel):
    version: Literal["m07-state-1"] = "m07-state-1"
    initial_revision: Nonempty
    record: RecordSnapshot
    issues: tuple[WorkflowIssue, ...] = Field(max_length=32)
    events: tuple[ActionEvent, ...] = Field(default=(), max_length=32)
    status: Literal["running", "finished", "unresolved", "pending_review", "budget_exhausted",
                    "no_progress", "cancelled", "timed_out", "failed", "interrupted"] = "running"
    stop_reason: str = Field(default="", max_length=500)
    model_calls: int = Field(default=0, ge=0, strict=True)
    uncertain_model_calls: int = Field(default=0, ge=0, strict=True)
    input_tokens: int = Field(default=0, ge=0, strict=True)
    output_tokens: int = Field(default=0, ge=0, strict=True)
    usage_unknown: bool = Field(default=False, strict=True)
    no_progress_steps: int = Field(default=0, ge=0, strict=True)
    elapsed_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    pending_reviews: tuple[ReviewCase, ...] = Field(default=(), max_length=32)

    @property
    def state_id(self) -> str:
        return "workflow-state:" + digest(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def _bindings(self) -> WorkflowState:
        issue_ids = {issue.id for issue in self.issues}
        if len(issue_ids) != len(self.issues):
            raise ValueError("Workflow issue IDs must be distinct")
        if len({case.case_id for case in self.pending_reviews}) != len(self.pending_reviews):
            raise ValueError("Pending review case IDs must be distinct")
        if any(case.issue_id not in issue_ids for case in self.pending_reviews):
            raise ValueError("Review case targets an unknown issue")
        if self.status == "finished" and (self.pending_reviews or any(i.status != "resolved" for i in self.issues)):
            raise ValueError("Finished state cannot contain unresolved issues or pending reviews")
        if self.status == "pending_review" and not self.pending_reviews:
            raise ValueError("Pending review state requires an actual pending case")
        indices = [event.index for event in self.events]
        if indices != sorted(set(indices)):
            raise ValueError("Action events need ordered distinct indices")
        revision = self.initial_revision
        source_binding = (self.record.source_id, self.record.source_sha256, self.record.representation_sha256)
        for event in self.events:
            if event.revision_before != revision:
                raise ValueError("Action events do not form a continuous record revision history")
            revision = event.revision_after
            if event.action is not None and event.action.kind != "finish" and event.action.issue_id not in issue_ids:
                raise ValueError("Action event targets an unknown issue")
        if revision != self.record.revision_id:
            raise ValueError("Current record revision differs from workflow history")
        refs = [ref for issue in self.issues for ref in issue.evidence]
        refs += [ref for event in self.events for ref in event.evidence]
        refs += [ref for case in self.pending_reviews for ref in case.evidence]
        if any((ref.source_id, ref.source_sha256, ref.representation_sha256) != source_binding for ref in refs):
            raise ValueError("Workflow evidence belongs to a different source or representation")
        return self


def validate_action(action: Action, state: WorkflowState) -> Action:
    """Validate one selection against the current explicitly supplied catalog."""
    state = WorkflowState.model_validate(state.model_dump())
    action = ACTION_ADAPTER.validate_python(action)
    if state.status != "running":
        raise ValueError("A stopped workflow cannot select another action")
    if action.kind == "finish":
        if action.outcome == "finished" and (state.pending_reviews or any(i.status != "resolved" for i in state.issues)):
            raise ValueError("Cannot finish successfully while issues remain unresolved")
        return action
    issue = next((item for item in state.issues if item.id == action.issue_id), None)
    if issue is None or issue.status != "open":
        raise ValueError("Action must target an open issue in this workflow")
    if action.kind == "propose_update" and action.proposal_id not in issue.proposal_ids:
        raise ValueError("Proposal is absent from the issue allowlist")
    if action.kind == "consult_definition" and action.definition_key not in issue.definition_keys:
        raise ValueError("Definition is absent from the issue allowlist")
    return action
