"""Immutable M08 review artifacts; reviewer identity is explicitly self-declared."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .contracts import RecordSnapshot, canonical_json, digest
from .evidence import EvidenceRef, FrozenModel
from .record_patches import PatchProposal
from .support_checks import CheckedAssertion, Role
from .workflow_types import WorkflowState

ReviewProvenance = Literal["simulated", "user_supplied"]


class ReviewError(ValueError):
    """Safe machine-readable rejection; a rejected response has no side effect."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ReviewAttestation(FrozenModel):
    path: str = Field(min_length=1, max_length=500)
    value_json: str = Field(min_length=1, max_length=20000)
    role: Role
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1, max_length=8)

    @field_validator("value_json")
    @classmethod
    def _scalar(cls, value: str) -> str:
        parsed = json.loads(value)
        if isinstance(parsed, (dict, list)) or parsed is None or canonical_json(parsed) != value:
            raise ValueError("Attestation requires one non-null canonical JSON scalar")
        return value


class ReviewResponse(FrozenModel):
    version: Literal["m08-response-1"] = "m08-response-1"
    idempotency_key: str = Field(min_length=1, max_length=160)
    expected_state_id: str = Field(pattern=r"^review-state:[0-9a-f]{64}$")
    expected_revision: str = Field(pattern=r"^revision:[0-9a-f]{64}$")
    case_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    outcome: Literal["approve", "edit", "unresolved"]
    provenance: ReviewProvenance
    reviewer_id: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=2000)
    proposal: PatchProposal | None = None
    attestations: tuple[ReviewAttestation, ...] = Field(default=(), max_length=32)

    @field_validator("idempotency_key", "reviewer_id", "rationale")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Review metadata cannot be blank")
        return value

    @model_validator(mode="after")
    def _shape(self) -> ReviewResponse:
        if len(set(self.case_ids)) != len(self.case_ids) or any(
            not case_id.strip() or len(case_id) > 500 for case_id in self.case_ids
        ):
            raise ValueError("Selected cases must be distinct bounded identifiers")
        if len({item.path for item in self.attestations}) != len(self.attestations):
            raise ValueError("One attestation per assertion path is required")
        if (self.outcome == "edit") != (self.proposal is not None):
            raise ValueError("Only an explicit edit response carries a patch")
        if self.outcome == "unresolved" and self.attestations:
            raise ValueError("An unresolved response cannot approve assertions")
        return self

    @property
    def response_id(self) -> str:
        return "review-response:" + digest(self.model_dump(mode="json"))


class CatalogProposal(FrozenModel):
    key: str = Field(min_length=1, max_length=500)
    proposal: PatchProposal

    @field_validator("key")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A proposal key cannot be blank")
        return value


class ReviewItem(FrozenModel):
    case_id: str = Field(min_length=1, max_length=500)
    issue_id: str = Field(min_length=1, max_length=500)
    path: str = Field(max_length=500)
    summary: str = Field(min_length=1, max_length=2000)
    record_revision: str = Field(pattern=r"^revision:[0-9a-f]{64}$")
    status: Literal["pending", "approved", "unresolved"] = "pending"
    evidence: tuple[EvidenceRef, ...] = Field(default=(), max_length=128)
    review_provenance: ReviewProvenance | None = None
    last_decision_id: str | None = None

    @model_validator(mode="after")
    def _status(self) -> ReviewItem:
        if self.status == "pending":
            if self.review_provenance is not None or self.last_decision_id is not None:
                raise ValueError("A pending case has no current review decision")
        elif self.review_provenance is None or self.last_decision_id is None:
            raise ValueError("A closed case needs explicit review provenance and decision")
        return self


class ReviewedAssertion(CheckedAssertion):
    decision_id: str
    reviewer_id: str
    provenance: ReviewProvenance
    support_status: Literal["reviewer_attested"] = "reviewer_attested"
    expert_authentication: Literal["not_authenticated"] = "not_authenticated"


class ReviewDecision(FrozenModel):
    version: Literal["m08-decision-1"] = "m08-decision-1"
    decision_id: str
    response: ReviewResponse
    previous_revision: str
    resulting_revision: str
    changed_paths: tuple[str, ...] = ()
    required_paths: tuple[str, ...] = ()
    bindings: tuple[CheckedAssertion, ...] = ()
    invalidated_revision: str | None = None
    expert_authentication: Literal["not_authenticated"] = "not_authenticated"
    coverage_status: Literal["selected_assertions_only"] = "selected_assertions_only"

    @model_validator(mode="after")
    def _identity(self) -> ReviewDecision:
        content = self.model_dump(mode="json", exclude={"decision_id"})
        if self.decision_id != "review-decision:" + digest(content):
            raise ValueError("Review decision content fingerprint differs")
        if self.response.expected_revision != self.previous_revision:
            raise ValueError("Review response and prior revision differ")
        if self.response.outcome == "edit":
            if not self.changed_paths or self.invalidated_revision != self.previous_revision:
                raise ValueError("An edit must invalidate its prior revision")
        elif self.changed_paths or self.invalidated_revision is not None or self.previous_revision != self.resulting_revision:
            raise ValueError("A non-edit response cannot change the record")
        return self

    @property
    def assertions(self) -> tuple[ReviewedAssertion, ...]:
        return tuple(ReviewedAssertion(
            **item.model_dump(), decision_id=self.decision_id, reviewer_id=self.response.reviewer_id,
            provenance=self.response.provenance,
        ) for item in self.bindings)


class ReviewState(FrozenModel):
    version: Literal["m08-review-state-1"] = "m08-review-state-1"
    review_id: str
    initial_workflow: WorkflowState
    current: RecordSnapshot
    cases: tuple[ReviewItem, ...] = Field(min_length=1, max_length=32)
    proposals: tuple[CatalogProposal, ...] = Field(default=(), max_length=32)
    decisions: tuple[ReviewDecision, ...] = Field(default=(), max_length=64)
    current_assertions: tuple[ReviewedAssertion, ...] = Field(default=(), max_length=1024)
    status: Literal["pending", "completed", "unresolved"] = "pending"
    expert_authentication: Literal["not_authenticated"] = "not_authenticated"
    coverage_status: Literal["selected_assertions_only"] = "selected_assertions_only"

    @property
    def state_id(self) -> str:
        return "review-state:" + digest(self.model_dump(mode="json"))

    @property
    def original(self) -> RecordSnapshot:
        return self.initial_workflow.record

    @model_validator(mode="after")
    def _bindings(self) -> ReviewState:
        identity = {"workflow_state_id": self.initial_workflow.state_id,
                    "proposals": [item.model_dump(mode="json") for item in self.proposals]}
        if self.review_id != "review:" + digest(identity):
            raise ValueError("Review session identity differs from its frozen parent")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Review case IDs must be distinct")
        keys = tuple(item.key for item in self.proposals)
        if tuple(sorted(set(keys))) != keys:
            raise ValueError("Proposal catalog must have sorted distinct keys")
        idempotency = tuple(item.response.idempotency_key for item in self.decisions)
        if len(set(idempotency)) != len(idempotency):
            raise ValueError("Review decisions must have distinct idempotency keys")
        if any(case.record_revision != self.current.revision_id for case in self.cases):
            raise ValueError("Current review cases must bind the current revision")
        if self.current.record_id != self.initial_workflow.record.record_id:
            raise ValueError("A review cannot change document identity")
        expected = "pending" if any(case.status == "pending" for case in self.cases) else (
            "unresolved" if any(case.status == "unresolved" for case in self.cases) else "completed"
        )
        if self.status != expected:
            raise ValueError("Review status differs from its cases")
        if len({item.path for item in self.current_assertions}) != len(self.current_assertions):
            raise ValueError("Current reviewed assertions require distinct paths")
        if any(item.record_revision != self.current.revision_id for item in self.current_assertions):
            raise ValueError("A past revision cannot supply current approval")
        return self
