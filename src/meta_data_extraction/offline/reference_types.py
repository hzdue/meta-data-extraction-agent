"""Source-first reference drafts, never automatic gold or runtime approvals.

This module performs no I/O. An adapter must verify the actual source and schema
files before constructing a packet. Evidence resolution verifies locations, not
whether a person's scientific interpretation is correct.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from meta_data_extraction.agent.contracts import digest
from meta_data_extraction.agent.evidence import DocumentBundle, EvidenceRef, resolve_evidence
from meta_data_extraction.agent.profiles import Profile
from meta_data_extraction.agent.record_checks import _checked_profile, _json_value, check_structure
from meta_data_extraction.offline.manifest import Collection, OfflineModel


COLLECTIONS: tuple[Collection, ...] = (
    "bibliographic.authors", "bibliographic.affiliations", "bibliographic.funding",
    "facilities", "samples", "datasets",
)


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("reference_nonblank_text_required")
    return value


class ReferencePacket(OfflineModel):
    version: Literal["m10-reference-packet-1"] = "m10-reference-packet-1"
    packet_id: str = Field(pattern=r"^reference-packet:[0-9a-f]{64}$")
    publication_id: str = Field(min_length=1, max_length=500)
    split: Literal["calibration", "historical"]
    data_kind: Literal["natural", "synthetic"]
    bundle: DocumentBundle
    profile: Profile

    _publication = field_validator("publication_id")(_nonblank)

    @model_validator(mode="after")
    def _integrity(self) -> ReferencePacket:
        DocumentBundle.model_validate(self.bundle.model_dump(mode="json"))
        _checked_profile(self.profile)
        if (len(self.profile.schema_names) != 2 or set(self.profile.schema_names)
                != {"generic.json", "photon_and_neutron.json"}):
            raise ValueError("reference_full_canonical_schema_pair_required")
        # The inherited profile pins raw prompt bytes while its text has normal
        # newline decoding. Original-file verification belongs to the I/O layer.
        if self.packet_id != "reference-packet:" + digest(self.model_dump(mode="json", exclude={"packet_id"})):
            raise ValueError("reference_packet_digest_mismatch")
        return self


def build_packet(*, publication_id: str, split: str, data_kind: str,
                 bundle: DocumentBundle, profile: Profile) -> ReferencePacket:
    payload = dict(version="m10-reference-packet-1", publication_id=publication_id,
                   split=split, data_kind=data_kind, bundle=bundle.model_dump(mode="json"),
                   profile=profile.model_dump(mode="json"))
    return ReferencePacket(packet_id="reference-packet:" + digest(payload), **payload)


class ReviewerDeclaration(OfflineModel):
    identity: str = Field(min_length=1, max_length=500)
    provenance: Literal["human", "synthetic", "assistant"]
    prior_prediction_access: Literal["not_seen", "seen", "unknown"]
    independence: Literal["declared_independent", "not_independent", "not_assessed"]
    elapsed_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)

    _identity = field_validator("identity")(_nonblank)


class ReferenceEvidence(OfflineModel):
    path: str = Field(min_length=1, max_length=1000)
    status: Literal["supported", "not_reported", "uncertain"]
    evidence: tuple[EvidenceRef, ...] = Field(default=(), max_length=8)
    note: str = Field(min_length=1, max_length=4000)

    _note = field_validator("note")(_nonblank)

    @field_validator("path")
    @classmethod
    def _pointer(cls, value: str) -> str:
        if not value.startswith("/") or re.search(r"~(?![01])", value):
            raise ValueError("reference_invalid_json_pointer")
        return value

    @model_validator(mode="after")
    def _support(self) -> ReferenceEvidence:
        if self.status == "supported" and not self.evidence:
            raise ValueError("reference_supported_requires_evidence")
        return self


class CoverageDeclaration(OfflineModel):
    collection: Collection
    status: Literal["unreviewed", "partial", "complete_available_text"]
    note: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def _reviewed_note(self) -> CoverageDeclaration:
        if self.status != "unreviewed" and not self.note.strip():
            raise ValueError("reference_reviewed_coverage_requires_note")
        return self


class ReferenceSubmission(OfflineModel):
    version: Literal["m10-reference-submission-1"] = "m10-reference-submission-1"
    packet_id: str = Field(pattern=r"^reference-packet:[0-9a-f]{64}$")
    reviewer: ReviewerDeclaration | None = None
    canonical_record: dict[str, Any] | None = None
    evidence: tuple[ReferenceEvidence, ...] = Field(default=(), max_length=10000)
    coverage: tuple[CoverageDeclaration, ...] = Field(default=(), max_length=6)
    unresolved: tuple[str, ...] = Field(default=(), max_length=1000)

    @field_validator("canonical_record", mode="before")
    @classmethod
    def _uncoerced_record(cls, value: Any) -> Any:
        if value is not None:
            _json_value(value)
        return value

    @field_validator("unresolved")
    @classmethod
    def _unresolved(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() or len(value) > 4000 for value in values):
            raise ValueError("reference_invalid_unresolved_note")
        return values

    @model_validator(mode="after")
    def _distinct(self) -> ReferenceSubmission:
        if len({item.path for item in self.evidence}) != len(self.evidence):
            raise ValueError("reference_duplicate_evidence_path")
        if len({item.collection for item in self.coverage}) != len(self.coverage):
            raise ValueError("reference_duplicate_coverage_collection")
        return self


def blank_submission(packet: ReferencePacket) -> ReferenceSubmission:
    packet = ReferencePacket.model_validate(packet.model_dump(mode="json"))
    return ReferenceSubmission(packet_id=packet.packet_id, coverage=tuple(
        CoverageDeclaration(collection=name, status="unreviewed") for name in COLLECTIONS))


def _review_values(value: Any, path: str = "") -> dict[str, Any]:
    """Include nulls, local IDs and empty non-entity containers explicitly."""
    if isinstance(value, dict) and value:
        return {pointer: item for key, child in value.items()
                for pointer, item in _review_values(child, path + "/" + key.replace("~", "~0").replace("/", "~1")).items()}
    if isinstance(value, list) and value:
        return {pointer: item for index, child in enumerate(value)
                for pointer, item in _review_values(child, f"{path}/{index}").items()}
    collection_paths = {"/" + name.replace(".", "/") for name in COLLECTIONS}
    return {} if path in collection_paths else {path: value}


def assess_reference(packet: ReferencePacket, submission: ReferenceSubmission) -> dict[str, Any]:
    """Validate draft locations/shape without certifying interpretation or gold.

An unresolved or incomplete draft is a valid outcome. Malformed records, stale
packet bindings and forged evidence fail explicitly instead of becoming drafts.
"""
    packet = ReferencePacket.model_validate(packet.model_dump(mode="json"))
    # Python-mode validation must precede JSON serialization: Pydantic's JSON
    # serializer converts nonfinite floats to null, which would mask bad input.
    submission = ReferenceSubmission.model_validate(submission.model_dump())
    if packet.packet_id != submission.packet_id:
        raise ValueError("reference_submission_packet_mismatch")
    declared_coverage = {item.collection: item.status for item in submission.coverage}
    coverage = {name: declared_coverage.get(name, "unreviewed") for name in COLLECTIONS}
    counts: dict[str, int | None] = {name: None for name in COLLECTIONS}
    values: dict[str, Any] = {}
    revision = None
    entities, references = [], []
    if submission.canonical_record is not None:
        structure = check_structure(packet.profile, packet.bundle, submission.canonical_record)
        if structure.status != "valid" or structure.snapshot is None:
            raise ValueError("reference_invalid_canonical_structure")
        record = structure.snapshot
        revision = record.revision_id
        entities = [entity.model_dump(mode="json") for entity in record.entities if entity.collection]
        references = [ref.model_dump(mode="json") for ref in record.references]
        counter = Counter(entity["collection"].replace("/", ".") for entity in entities)
        counts = {name: counter[name] for name in COLLECTIONS}
        values = _review_values(record.canonical)
        # Entity object annotations concern identity/granularity. A local ID is
        # assigned notation; its evidence supports the referenced entity, not a
        # claim that the publication literally printed "sample_1".
        values.update({entity["path"]: {"entity_identity": True} for entity in entities})
    reviewed = {}
    for item in submission.evidence:
        if item.path not in values:
            raise ValueError("reference_annotation_path_not_reviewable")
        value = values[item.path]
        absent = value is None or value == [] or value == {}
        if (absent and item.status == "supported") or (not absent and item.status == "not_reported"):
            raise ValueError("reference_annotation_status_value_mismatch")
        for ref in item.evidence:
            try:
                resolve_evidence(packet.bundle, ref)
            except ValueError as exc:
                raise ValueError("reference_evidence_does_not_resolve") from exc
        reviewed[item.path] = item.status
    missing = sorted(set(values) - set(reviewed))
    uncertain = sorted(path for path, status in reviewed.items() if status == "uncertain")
    reviewer = submission.reviewer
    declared_source_first_human = bool(reviewer and reviewer.provenance == "human"
        and reviewer.prior_prediction_access == "not_seen"
        and reviewer.independence == "declared_independent")
    complete = (submission.canonical_record is not None and declared_source_first_human
        and all(status == "complete_available_text" for status in coverage.values())
        and not missing and not uncertain and not submission.unresolved)
    return {
        "version": "m10-reference-assessment-1", "packet_id": packet.packet_id,
        "submission_id": "reference-submission:" + digest(submission.model_dump(mode="json")),
        "status": "available_text_review_recorded" if complete else "incomplete",
        "record_revision": revision, "declared_entity_counts": counts,
        "entities": entities, "references": references, "missing_review_paths": missing,
        "unresolved_paths": uncertain, "unresolved": list(submission.unresolved),
        "coverage_status": coverage, "source_warnings": [w.model_dump(mode="json") for w in packet.bundle.warnings],
        "declared_source_first_human": declared_source_first_human,
        "reviewer": reviewer.model_dump(mode="json") if reviewer else None,
        "independent_gold_status": "not_established",
        "scoring_status": "blocked_reference_and_alignment_adjudication",
        "review_identity_verification": "self_declared", "full_source_coverage": "not_established",
        "evidence_check": "locations_only_semantic_support_not_verified",
    }
