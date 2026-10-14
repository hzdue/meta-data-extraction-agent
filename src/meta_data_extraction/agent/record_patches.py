"""Atomic, bounded corrections gated by fresh source-support judgments.

This is a pure executor, not an expert review system. Model judgments can be
wrong; accepting them requires an explicit policy and never marks expert approval.
Only existing dataset scalar leaves and individual sample-reference items are
replaceable. All prior assertion bindings are invalidated after an applied edit.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import Field, model_validator

from meta_data_extraction.agent.contracts import (
    AssertionEvidence, RecordSnapshot, assertion_values, at_pointer, canonical_json, digest, owner,
)
from meta_data_extraction.agent.evidence import DocumentBundle, FrozenModel
from meta_data_extraction.agent.profiles import Profile
from meta_data_extraction.agent.record_checks import check_structure
from meta_data_extraction.agent.support_checks import MAX_TARGETS, SupportReport, verify_report

PATCH_VERSION = "m06-atomic-patch-1"
_ALLOWED = re.compile(r"^/datasets/(0|[1-9][0-9]*)/(?:dataset_description|measurement_technique|sample_refs/(0|[1-9][0-9]*))$")


class PatchError(ValueError):
    """A serialized patch or receipt does not match its content bindings."""


def _scalar(value_json: str):
    value = json.loads(value_json)
    if isinstance(value, (dict, list)) or canonical_json(value) != value_json:
        raise ValueError("Patch values must be canonical JSON scalars")
    return value


class ReplaceScalar(FrozenModel):
    op: Literal["replace_scalar"] = "replace_scalar"
    path: str = Field(min_length=1, max_length=200)
    expected_value_json: str = Field(min_length=1, max_length=20000)
    replacement_value_json: str = Field(min_length=1, max_length=20000)

    @model_validator(mode="after")
    def _values(self):
        _scalar(self.expected_value_json)
        _scalar(self.replacement_value_json)
        return self


class PatchProposal(FrozenModel):
    expected_revision: str = Field(min_length=1)
    operations: tuple[ReplaceScalar, ...] = Field(min_length=1, max_length=4)
    provenance: Literal["model_proposal", "supplied_proposal", "synthetic_fixture"]
    summary: str = Field(min_length=1, max_length=500)

    @property
    def proposal_id(self) -> str:
        return "patch-proposal:" + digest(self.model_dump(mode="json"))


class PatchPolicy(FrozenModel):
    version: Literal["m06-contradiction-correction-1"] = "m06-contradiction-correction-1"
    accept_model_judgments: bool = Field(default=False, strict=True)
    max_operations: int = Field(default=4, ge=1, le=4, strict=True)
    # This policy is deliberately a correction gate, not inferred permission to
    # fill missing values, overrule supported facts, or perform human review.
    approval_status: Literal["not_expert_approved"] = "not_expert_approved"


class PreparedPatch(FrozenModel):
    version: Literal["m06-atomic-patch-1"] = PATCH_VERSION
    original: RecordSnapshot
    candidate: RecordSnapshot
    proposal: PatchProposal
    changed_paths: tuple[str, ...]
    required_support_paths: tuple[str, ...]
    candidate_id: str

    @model_validator(mode="after")
    def _identity(self):
        payload = self.model_dump(mode="json", exclude={"candidate_id"})
        if self.candidate_id != "patch-candidate:" + digest(payload):
            raise ValueError("Candidate content fingerprint differs")
        return self


class CandidateResult(FrozenModel):
    status: Literal["ready", "stale", "rejected"]
    code: str
    original: RecordSnapshot
    prepared: PreparedPatch | None = None

    @model_validator(mode="after")
    def _state(self):
        if (self.status == "ready") != (self.prepared is not None):
            raise ValueError("Only ready candidates may carry a prepared patch")
        if self.prepared is not None and self.prepared.original != self.original:
            raise ValueError("Prepared original differs")
        return self


class PatchReceipt(FrozenModel):
    version: Literal["m06-atomic-patch-1"] = PATCH_VERSION
    candidate_id: str
    proposal_id: str
    previous_revision: str
    resulting_revision: str
    before_report_digest: str
    after_report_digest: str
    policy: PatchPolicy
    changed_paths: tuple[str, ...]
    required_support_paths: tuple[str, ...]
    approval_status: Literal["not_expert_approved"] = "not_expert_approved"
    receipt_id: str

    @model_validator(mode="after")
    def _identity(self):
        payload = self.model_dump(mode="json", exclude={"receipt_id"})
        if self.receipt_id != "patch-receipt:" + digest(payload):
            raise ValueError("Receipt content fingerprint differs")
        return self


class PatchResult(FrozenModel):
    status: Literal["applied", "replayed", "rejected", "stale"]
    code: str
    snapshot: RecordSnapshot
    receipt: PatchReceipt | None = None
    # Fresh M02 evidence bindings retain their honest not_assessed status;
    # the separately supplied M06 report contains automated support judgments.
    assertions: tuple[AssertionEvidence, ...] = ()
    invalidated_assertion_revisions: tuple[str, ...] = ()
    support_report: SupportReport | None = None
    approval_status: Literal["not_expert_approved"] = "not_expert_approved"

    @model_validator(mode="after")
    def _state(self):
        applied = self.status in {"applied", "replayed"}
        if applied != (self.receipt is not None and self.support_report is not None):
            raise ValueError("Applied patches need a receipt and new support report")
        if not applied and (self.receipt is not None or self.support_report is not None
                            or self.assertions or self.invalidated_assertion_revisions):
            raise ValueError("Rejected patches cannot replace assertion evidence")
        if applied and self.snapshot.revision_id != self.receipt.resulting_revision:
            raise ValueError("Receipt differs from resulting snapshot")
        return self


def build_candidate(profile: Profile, bundle: DocumentBundle, record: RecordSnapshot,
                    proposal: PatchProposal) -> CandidateResult:
    """Build an immutable candidate; this does not authorize or apply any edit."""
    record = RecordSnapshot.model_validate(record.model_dump())
    bundle = DocumentBundle.model_validate(bundle.model_dump())
    proposal = PatchProposal.model_validate(proposal.model_dump())

    def reject(code: str, status="rejected"):
        return CandidateResult(status=status, code=code, original=record)

    if proposal.expected_revision != record.revision_id:
        return reject("stale_revision", "stale")
    original_check = check_structure(profile, bundle, record.canonical)
    if original_check.status != "valid" or original_check.snapshot != record:
        return reject("original_binding_or_structure_invalid")
    paths = tuple(op.path for op in proposal.operations)
    if len(set(paths)) != len(paths):
        return reject("conflicting_operations")
    candidate = record.canonical
    touched_datasets = set()
    for op in proposal.operations:
        if not _ALLOWED.fullmatch(op.path):
            return reject("path_not_allowlisted")
        try:
            old_value = at_pointer(candidate, op.path)
        except (KeyError, IndexError, TypeError, ValueError):
            return reject("missing_target")
        if isinstance(old_value, (dict, list)) or canonical_json(old_value) != op.expected_value_json:
            return reject("old_value_mismatch")
        new_value = _scalar(op.replacement_value_json)
        if old_value is None or new_value is None:
            return reject("missing_value_change_not_supported")
        if op.expected_value_json == op.replacement_value_json:
            return reject("no_progress")
        parts = op.path[1:].split("/")
        parent = candidate
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        leaf = int(parts[-1]) if isinstance(parent, list) else parts[-1]
        parent[leaf] = new_value
        touched_datasets.add("/datasets/" + parts[1])
    # Revalidate the entire candidate, including every canonical relationship.
    checked = check_structure(profile, bundle, candidate)
    if checked.status != "valid" or checked.snapshot is None:
        return reject("candidate_structure_invalid")
    revised = checked.snapshot
    if revised.entities != record.entities:
        return reject("entity_identity_changed")
    # A sample reassignment must not collapse two slots into the same sample.
    for dataset in candidate["datasets"]:
        if len(dataset["sample_refs"]) != len(set(dataset["sample_refs"])):
            return reject("duplicate_sample_reference")
    affected = {ref.path for ref in revised.references
                if owner(revised.entities, ref.path).path in touched_datasets}
    required = tuple(sorted(set(paths) | affected))
    if len(required) > MAX_TARGETS:
        return reject("support_target_budget_exceeded")
    for dataset_path in touched_datasets:
        old_refs = at_pointer(record.canonical, dataset_path)["sample_refs"]
        new_refs = at_pointer(candidate, dataset_path)["sample_refs"]
        if old_refs != new_refs and sorted(old_refs) == sorted(new_refs):
            return reject("reference_reorder_not_supported")
    payload = dict(version=PATCH_VERSION, original=record.model_dump(mode="json"),
                   candidate=revised.model_dump(mode="json"), proposal=proposal.model_dump(mode="json"),
                   changed_paths=paths, required_support_paths=required)
    prepared = PreparedPatch.model_validate({**payload, "candidate_id": "patch-candidate:" + digest(payload)})
    return CandidateResult(status="ready", code="candidate_requires_support", original=record, prepared=prepared)


def verify_candidate(prepared: PreparedPatch, profile: Profile, bundle: DocumentBundle) -> None:
    prepared = PreparedPatch.model_validate(prepared.model_dump())
    rebuilt = build_candidate(profile, bundle, prepared.original, prepared.proposal)
    if rebuilt.status != "ready" or rebuilt.prepared != prepared:
        raise PatchError("Candidate no longer matches its original, proposal, or affected relationships")


def _fresh_assertions(record: RecordSnapshot, report: SupportReport) -> tuple[AssertionEvidence, ...]:
    judgments = {judgment.path: judgment for judgment in report.judgments}
    values = []
    for path, value in assertion_values(record.canonical):
        entity = owner(record.entities, path)
        refs = judgments[path].evidence if path in judgments else ()
        values.append(AssertionEvidence(record_revision=record.revision_id, entity_handle=entity.handle,
                                       path=path, field_path=path[len(entity.path):], value_json=canonical_json(value),
                                       link_status="location_verified" if refs else "unlinked", evidence=refs))
    return tuple(values)


def apply_supported_patch(profile: Profile, bundle: DocumentBundle, current: RecordSnapshot,
                          prepared: PreparedPatch, before: SupportReport, after: SupportReport, *,
                          policy: PatchPolicy | None = None, receipt: PatchReceipt | None = None) -> PatchResult:
    """Apply all operations or none, using whole-revision-bound automated reports.

    Replaying against the candidate requires the exact receipt; merely matching
    replacement values is insufficient. Neither replay nor application writes files.
    """
    current = RecordSnapshot.model_validate(current.model_dump())
    policy = PatchPolicy.model_validate((policy or PatchPolicy()).model_dump())

    def reject(code: str, status="rejected"):
        return PatchResult(status=status, code=code, snapshot=current)

    try:
        verify_candidate(prepared, profile, bundle)
    except (ValueError, KeyError, IndexError, TypeError):
        return reject("invalid_candidate")
    if current not in (prepared.original, prepared.candidate):
        return reject("stale_revision", "stale")
    if not policy.accept_model_judgments:
        return reject("model_judgment_policy_disabled")
    if len(prepared.proposal.operations) > policy.max_operations:
        return reject("operation_budget_exceeded")
    try:
        verify_report(before, profile, prepared.original, bundle)
        verify_report(after, profile, prepared.candidate, bundle)
    except (ValueError, KeyError, IndexError, TypeError):
        return reject("support_report_binding_invalid")
    if before.status != "succeeded" or after.status != "succeeded":
        return reject("support_check_incomplete")
    old = {judgment.path: judgment for judgment in before.judgments}
    new = {judgment.path: judgment for judgment in after.judgments}
    for path in prepared.changed_paths:
        if path not in old or old[path].verdict != "contradicted" or not old[path].evidence:
            return reject("old_assertion_not_contradicted")
        if old[path].diagnosis == "context_conflict":
            return reject("source_conflict_requires_review")
    for path in prepared.required_support_paths:
        if path not in new or new[path].verdict != "supported" or not new[path].evidence:
            return reject("candidate_support_incomplete")
        judgment = new[path]
        expected_role = "attachment" if any(ref.path == path for ref in prepared.candidate.references) else "measurement_acquisition"
        if judgment.role != expected_role:
            return reject("candidate_semantic_role_mismatch")
        if judgment.provenance not in {"mock_model_judgment", "live_model_judgment"}:
            return reject("unsupported_judgment_provenance")
    payload = dict(version=PATCH_VERSION, candidate_id=prepared.candidate_id,
                   proposal_id=prepared.proposal.proposal_id,
                   previous_revision=prepared.original.revision_id,
                   resulting_revision=prepared.candidate.revision_id,
                   before_report_digest=digest(before.model_dump(mode="json")),
                   after_report_digest=digest(after.model_dump(mode="json")),
                   policy=policy.model_dump(mode="json"), changed_paths=prepared.changed_paths,
                   required_support_paths=prepared.required_support_paths,
                   approval_status="not_expert_approved")
    expected_receipt = PatchReceipt.model_validate({**payload, "receipt_id": "patch-receipt:" + digest(payload)})
    if receipt is not None:
        try:
            receipt = PatchReceipt.model_validate(receipt.model_dump())
        except ValueError:
            return reject("receipt_invalid")
        if receipt != expected_receipt:
            return reject("receipt_mismatch")
    if current == prepared.candidate and receipt is None:
        return reject("replay_requires_receipt", "stale")
    return PatchResult(status="replayed" if current == prepared.candidate else "applied",
                       code="atomic_correction_applied", snapshot=prepared.candidate, receipt=expected_receipt,
                       assertions=_fresh_assertions(prepared.candidate, after),
                       invalidated_assertion_revisions=(prepared.original.revision_id,), support_report=after)
