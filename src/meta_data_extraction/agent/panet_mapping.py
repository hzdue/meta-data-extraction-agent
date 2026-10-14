"""Unreviewed PaNET mapping proposals anchored to exact publication mentions.

Lexical identity, current-study context supplied by a caller, and an ontology ID
are insufficient to establish scientific support. This module has no approval
state and never edits the canonical extraction.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from .contracts import RecordSnapshot, at_pointer, canonical_json, digest, owner
from .evidence import DocumentBundle, EvidenceRef, FrozenModel, resolve_evidence
from .panet_lookup import CandidateSet, verify_candidates
from .panet_resources import OntologySnapshot, PANET_NAMESPACE

MAPPING_VERSION = "m04-panet-mapping-v1"
Reason = Literal[
    "exact_lexical_candidate", "related_work", "usage_uncertain", "no_lexical_match",
    "ambiguous_exact_match", "candidate_set_truncated", "lexical_only",
    "malformed_id", "unknown_id", "obsolete_id", "not_retrieved", "selection_not_exact",
]


class TechniqueMention(FrozenModel):
    original_expression: str = Field(min_length=1, max_length=4096, strict=True)
    mention: EvidenceRef
    context: EvidenceRef
    context_use: Literal["current_study", "related_work", "uncertain"]
    context_provenance: Literal["supplied_unreviewed"] = "supplied_unreviewed"

    @model_validator(mode="after")
    def _binding(self) -> TechniqueMention:
        if not self.original_expression.strip() or self.original_expression != self.mention.quotation:
            raise ValueError("The original expression must be the exact nonempty mention quotation")
        identity = lambda ref: (ref.source_id, ref.source_sha256, ref.representation_sha256)
        if (identity(self.mention) != identity(self.context)
                or not self.context.start <= self.mention.start < self.mention.end <= self.context.end):
            raise ValueError("The supplied context must contain this mention in the same source snapshot")
        return self


class AssertionBinding(FrozenModel):
    record_id: str
    revision_id: str
    profile_id: str
    entity_handle: str
    path: str
    value_json: str


class MappingDecision(FrozenModel):
    policy_version: Literal["m04-panet-mapping-v1"] = MAPPING_VERSION
    snapshot_id: str
    source_id: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    representation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mention: TechniqueMention
    candidates: CandidateSet
    selected_id: str | None
    binding: AssertionBinding | None
    status: Literal["proposed", "unresolved", "no_match"]
    reason: Reason
    proposed_id: str | None
    approval_status: Literal["unreviewed"] = "unreviewed"
    support_status: Literal["not_assessed"] = "not_assessed"
    cache_key: str = Field(pattern=r"^mapping:[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _integrity(self) -> MappingDecision:
        if (self.status == "proposed") != (self.proposed_id is not None):
            raise ValueError("Only a proposal may carry a proposed concept ID")
        if (self.status == "proposed") != (self.reason == "exact_lexical_candidate"):
            raise ValueError("Only an exact lexical candidate may be proposed")
        if (self.status == "no_match") != (self.reason == "no_lexical_match"):
            raise ValueError("No-match status requires the explicit no-match reason")
        if self.snapshot_id != self.candidates.snapshot_id:
            raise ValueError("Mapping and candidates must refer to the same ontology snapshot")
        ref = self.mention.mention
        if (self.source_id, self.source_sha256, self.representation_sha256) != (
                ref.source_id, ref.source_sha256, ref.representation_sha256):
            raise ValueError("Mapping and mention source identities differ")
        if self.candidates.query != self.mention.original_expression:
            raise ValueError("Candidate query must preserve the exact original expression")
        payload = self.model_dump(mode="json", exclude={"cache_key"})
        if self.cache_key != "mapping:" + digest(payload):
            raise ValueError("Mapping cache identity does not match its full provenance and decision")
        return self


def _record_binding(record: RecordSnapshot | None, bundle: DocumentBundle,
                    mention: TechniqueMention, path: str | None) -> AssertionBinding | None:
    if (record is None) != (path is None):
        raise ValueError("A canonical binding requires both a record snapshot and assertion path")
    if record is None:
        return None
    record = RecordSnapshot.model_validate(record.model_dump())
    if (record.source_id, record.source_sha256, record.representation_sha256) != (
            bundle.source_id, bundle.source_sha256, bundle.representation_sha256):
        raise ValueError("Record and mention belong to different source snapshots")
    if not isinstance(path, str) or not re.fullmatch(r"/datasets/(?:0|[1-9][0-9]*)/measurement_technique", path):
        raise ValueError("Only a dataset measurement_technique assertion may receive a canonical binding")
    try:
        value = at_pointer(record.canonical, path)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("The measurement-technique assertion does not exist") from exc
    if value != mention.original_expression:
        raise ValueError("The canonical technique must retain the exact original expression")
    return AssertionBinding(record_id=record.record_id, revision_id=record.revision_id,
                            profile_id=record.profile_id, entity_handle=owner(record.entities, path).handle,
                            path=path, value_json=canonical_json(value))


def _selected_problem(snapshot: OntologySnapshot, candidates: CandidateSet,
                      selected_id: str | None) -> Reason | None:
    if selected_id is None:
        return None
    if not isinstance(selected_id, str):
        raise ValueError("A selected concept ID must be text or None")
    spec = snapshot.resource
    valid = (re.fullmatch(re.escape(PANET_NAMESPACE) + r"PaNET[0-9]+", selected_id) is not None
             if spec.kind == "authentic" else
             selected_id.startswith(spec.namespace) and len(selected_id) > len(spec.namespace))
    if not valid:
        return "malformed_id"
    concept = next((item for item in snapshot.concepts if item.concept_id == selected_id), None)
    if concept is None:
        return "unknown_id"
    if concept.obsolete:
        return "obsolete_id"
    if selected_id not in {candidate.concept_id for candidate in candidates.candidates}:
        return "not_retrieved"
    return None


def propose_mapping(snapshot: OntologySnapshot, candidates: CandidateSet,
                    bundle: DocumentBundle, mention: TechniqueMention, *,
                    record: RecordSnapshot | None = None, assertion_path: str | None = None,
                    selected_id: str | None = None) -> MappingDecision:
    """Return a conservative proposal or explicit unresolved/no-match result.

    A caller-supplied context annotation remains unreviewed. Selection cannot
    override ambiguity, a related-work mention, an obsolete ID, or a lexical
    mismatch. If a bounded result ends in an exact match, omitted candidates may
    contain another exact match. An overlap suffix proves all exact candidates
    have been returned because the verified lookup ranks every exact match first.
    """
    snapshot = OntologySnapshot.model_validate(snapshot.model_dump())
    bundle = DocumentBundle.model_validate(bundle.model_dump())
    mention = TechniqueMention.model_validate(mention.model_dump())
    candidates = verify_candidates(snapshot, candidates)
    resolve_evidence(bundle, mention.context)
    resolve_evidence(bundle, mention.mention)
    if candidates.query != mention.original_expression:
        raise ValueError("Candidates were retrieved for a different source expression")
    binding = _record_binding(record, bundle, mention, assertion_path)
    reason = _selected_problem(snapshot, candidates, selected_id)
    proposed_id = None
    status = "unresolved"
    exact = [candidate for candidate in candidates.candidates if candidate.match_type == "exact"]
    eligible = [candidate for candidate in exact if candidate.matched_kind in {"label", "exact", "alternative"}]
    if reason is not None:
        pass
    elif mention.context_use == "related_work":
        reason = "related_work"
    elif mention.context_use != "current_study":
        reason = "usage_uncertain"
    elif not candidates.candidates:
        status, reason = "no_match", "no_lexical_match"
    elif candidates.omitted_candidates and candidates.candidates[-1].match_type == "exact":
        reason = "candidate_set_truncated"
    elif len(exact) > 1:
        reason = "ambiguous_exact_match"
    elif selected_id is not None and selected_id not in {candidate.concept_id for candidate in eligible}:
        reason = "selection_not_exact"
    elif len(eligible) == 1:
        status, reason, proposed_id = "proposed", "exact_lexical_candidate", eligible[0].concept_id
    else:
        reason = "lexical_only"
    payload = {
        "policy_version": MAPPING_VERSION, "snapshot_id": snapshot.snapshot_id,
        "source_id": bundle.source_id, "source_sha256": bundle.source_sha256,
        "representation_sha256": bundle.representation_sha256,
        "mention": mention.model_dump(mode="json"), "candidates": candidates.model_dump(mode="json"),
        "selected_id": selected_id, "binding": binding.model_dump(mode="json") if binding else None,
        "status": status, "reason": reason, "proposed_id": proposed_id,
        "approval_status": "unreviewed", "support_status": "not_assessed",
    }
    return MappingDecision.model_validate({**payload, "cache_key": "mapping:" + digest(payload)})


def verify_mapping(snapshot: OntologySnapshot, bundle: DocumentBundle,
                   decision: MappingDecision, record: RecordSnapshot | None = None) -> None:
    """Resolve publication evidence and recompute all persisted mapping semantics."""
    decision = MappingDecision.model_validate(decision.model_dump())
    expected = propose_mapping(snapshot, decision.candidates, bundle, decision.mention,
                               record=record, assertion_path=decision.binding.path if decision.binding else None,
                               selected_id=decision.selected_id)
    if decision != expected:
        raise ValueError("Mapping no longer matches the supplied source, record, resource, and policy")
