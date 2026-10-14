"""Small, review-gated NeXus guidance registry over the inherited profile.

Mappings are sidecars, never canonical schema edits or publication evidence.
A proposed entry cannot approve itself. Supplied human review artifacts must bind
its complete content, current profile and current NXDL snapshot. This module
checks those bindings; it does not authenticate a reviewer or implement M08's
human-review workflow. Class/field contexts are not universal HDF5 paths.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from meta_data_extraction.agent.contracts import canonical_json, digest
from meta_data_extraction.agent.evidence import FrozenModel
from meta_data_extraction.agent.nexus_resources import (
    NXDLField, NXDLSnapshot, NexusResourceError, retrieve_definition,
)
from meta_data_extraction.agent.profiles import Profile

REGISTRY_VERSION = "m05-registry-1"
REVIEW_VERSION = "m05-review-1"
SemanticRole = Literal[
    "sample_identity", "composition", "sample_form", "specimen_mass",
    "preparation_temperature", "acquisition_temperature", "sample_origin",
]
Relation = Literal["exact", "equivalent", "partial", "no_match"]

# This small allowlist expresses the current inherited schema's scope. It does
# not infer arbitrary roles from field names or introduce new canonical fields.
_FIELD_ROLES = {
    "/samples/*/sample_name": frozenset({"sample_identity"}),
    "/samples/*/chemical_composition": frozenset({"composition"}),
    "/samples/*/sample_state": frozenset({"sample_form"}),
    "/samples/*/sample_mass": frozenset({"specimen_mass"}),
    "/samples/*/sample_description": frozenset({"preparation_temperature"}),
    "/samples/*/supplier_or_creator": frozenset({"sample_origin"}),
    "/datasets/*/dataset_description": frozenset({"acquisition_temperature"}),
}
_ROLE_UNITS = {
    "specimen_mass": "NX_MASS",
    "preparation_temperature": "NX_TEMPERATURE",
    "acquisition_temperature": "NX_TEMPERATURE",
}
_PARTIAL_ROLES = frozenset({"preparation_temperature", "acquisition_temperature"})


class RegistryError(ValueError):
    """A registry entry or review violates its frozen scope or bindings."""


class MappingRequest(FrozenModel):
    mapping_key: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_-]*$")
    canonical_field: str
    semantic_role: SemanticRole
    relation: Relation
    target_class: str | None = None
    target_path: str | None = None
    target_role: SemanticRole | None = None
    scope: str = Field(min_length=1, max_length=2000)
    expected_units: str | None = None
    conversion_policy: Literal["preserve_raw"] = "preserve_raw"

    @model_validator(mode="after")
    def _scope(self) -> MappingRequest:
        if self.semantic_role not in _FIELD_ROLES.get(self.canonical_field, ()):
            raise ValueError("Field and semantic role are outside the selected inherited scope")
        if not self.scope.strip():
            raise ValueError("Mapping scope must be explicit")
        targets = (self.target_class, self.target_path, self.target_role)
        if self.relation == "no_match":
            if any(value is not None for value in targets):
                raise ValueError("No-match entries must not assign a target")
        elif any(value is None or not str(value).strip() for value in targets):
            raise ValueError("A mapping needs an explicit class, field context and target role")
        elif self.target_role != self.semantic_role:
            raise ValueError("Equal quantities do not authorize a semantic-role transfer")
        if self.semantic_role in _PARTIAL_ROLES and self.relation not in {"partial", "no_match"}:
            raise ValueError("A temperature aspect only partially maps a free-text description")
        expected = _ROLE_UNITS.get(self.semantic_role)
        if self.expected_units is not None and self.expected_units != expected:
            raise ValueError("Expected units category is incompatible with the selected role")
        return self


class MappingEntry(FrozenModel):
    registry_version: Literal["m05-registry-1"] = REGISTRY_VERSION
    entry_id: str
    entry_sha256: str
    request: MappingRequest
    profile_id: str
    field_definition_json: str
    field_definition_sha256: str
    resource_snapshot_id: str
    target: NXDLField | None
    resolution_status: Literal["resolved", "unresolved", "no_match"]
    unresolved: tuple[str, ...] = ()
    approval_status: Literal["unreviewed"] = "unreviewed"
    support_status: Literal["not_assessed"] = "not_assessed"
    scientific_value_policy: Literal["source_evidence_required"] = "source_evidence_required"
    application_compliance: Literal["not_assessed"] = "not_assessed"

    @model_validator(mode="after")
    def _identity(self) -> MappingEntry:
        field = json.loads(self.field_definition_json)
        if (not isinstance(field, dict) or "type" not in field
                or canonical_json(field) != self.field_definition_json
                or digest(field) != self.field_definition_sha256):
            raise ValueError("Field definition fingerprint differs")
        content = self.model_dump(mode="json", exclude={"entry_id", "entry_sha256"})
        if self.entry_sha256 != digest(content) or self.entry_id != "nexus-mapping:" + self.entry_sha256:
            raise ValueError("Mapping content fingerprint differs")
        if self.resolution_status == "resolved" and (self.target is None or self.unresolved):
            raise ValueError("Resolved mappings need a resolved target")
        if self.resolution_status == "unresolved" and not self.unresolved:
            raise ValueError("Unresolved mappings need diagnostics")
        if (self.resolution_status == "no_match") != (self.request.relation == "no_match"):
            raise ValueError("No-match status and relation differ")
        if self.request.relation == "no_match" and (self.target is not None or self.unresolved):
            raise ValueError("No-match entries cannot supply target guidance")
        return self


class ReviewRecord(FrozenModel):
    """An externally supplied content-bound decision, not proof of identity.

    The caller must preserve the supplied artifact and approve its origin. The
    runner additionally pins its exact file bytes. No production helper creates
    a review record or infers human approval from a proposed mapping.
    """
    review_version: Literal["m05-review-1"] = REVIEW_VERSION
    review_id: str = Field(min_length=1, max_length=160)
    entry_id: str = Field(pattern=r"^nexus-mapping:[0-9a-f]{64}$")
    entry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1, max_length=200)
    provenance: Literal["supplied_human_review"]
    decision: Literal["approved", "rejected", "needs_changes"]
    reviewed_at: str
    rationale: str = Field(min_length=1, max_length=4000)

    @field_validator("review_id", "reviewer", "rationale")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Review metadata must be nonblank")
        return value

    @field_validator("reviewed_at")
    @classmethod
    def _timestamp(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Review timestamp needs an explicit timezone")
        return value

    @model_validator(mode="after")
    def _entry(self) -> ReviewRecord:
        if self.entry_id != "nexus-mapping:" + self.entry_sha256:
            raise ValueError("Review entry ID and content hash differ")
        return self


class ReviewedMapping(FrozenModel):
    entry: MappingEntry
    review: ReviewRecord | None = None
    approval_status: Literal["unreviewed", "approved", "rejected", "needs_changes"]

    @model_validator(mode="after")
    def _review_binding(self) -> ReviewedMapping:
        if self.review is None:
            if self.approval_status != "unreviewed":
                raise ValueError("Approval requires a supplied review record")
        elif (self.approval_status != self.review.decision
              or self.review.entry_id != self.entry.entry_id
              or self.review.entry_sha256 != self.entry.entry_sha256):
            raise ValueError("Review no longer binds the exact entry content")
        if self.approval_status == "approved" and self.entry.resolution_status == "unresolved":
            raise ValueError("Unresolved resource references cannot become approved guidance")
        return self


class ProfileRegistry(FrozenModel):
    registry_version: Literal["m05-registry-1"] = REGISTRY_VERSION
    registry_id: str
    profile_id: str
    resource_snapshot_id: str
    entries: tuple[ReviewedMapping, ...]
    scientific_value_policy: Literal["source_evidence_required"] = "source_evidence_required"
    application_compliance: Literal["not_assessed"] = "not_assessed"

    @model_validator(mode="after")
    def _identity(self) -> ProfileRegistry:
        content = self.model_dump(mode="json", exclude={"registry_id"})
        if self.registry_id != "nexus-registry:" + digest(content):
            raise ValueError("Registry fingerprint differs")
        keys = [item.entry.request.mapping_key for item in self.entries]
        review_ids = [item.review.review_id for item in self.entries if item.review is not None]
        if len(keys) != len(set(keys)) or len(review_ids) != len(set(review_ids)):
            raise ValueError("Duplicate mapping keys or review IDs")
        for item in self.entries:
            if (item.entry.profile_id != self.profile_id
                    or item.entry.resource_snapshot_id != self.resource_snapshot_id):
                raise ValueError("Registry entries belong to a different profile or resource")
        return self


class Guidance(FrozenModel):
    entry_id: str
    field_definition_json: str
    semantic_role: SemanticRole
    relation: Relation
    scope: str
    target: NXDLField
    approval_status: Literal["unreviewed", "approved"]
    review: ReviewRecord | None
    resolution_status: Literal["resolved", "unresolved"]
    unresolved: tuple[str, ...]
    conversion_policy: Literal["preserve_raw"] = "preserve_raw"
    support_status: Literal["not_assessed"] = "not_assessed"
    scientific_value_policy: Literal["source_evidence_required"] = "source_evidence_required"
    application_compliance: Literal["not_assessed"] = "not_assessed"


def _field_definition(profile: Profile, path: str) -> dict:
    current = profile.schema
    try:
        for token in path[1:].split("/"):
            if token == "*":
                current = current["item_schema"]
            else:
                current = current[token]
    except (KeyError, TypeError):
        raise RegistryError("Canonical field is absent from the frozen profile") from None
    if not isinstance(current, dict) or "type" not in current or "item_schema" in current:
        raise RegistryError("Expected a declared leaf field in the inherited profile")
    return current


def propose_entry(profile: Profile, snapshot: NXDLSnapshot, request: MappingRequest) -> MappingEntry:
    """Freeze one proposal without adding a scientific value or any approval.

    Role annotations are supplied assertions requiring human review. Matching
    units alone never establishes matching roles. Necessary unresolved NXDL
    dependencies remain explicit and block later approval.
    """
    request = MappingRequest.model_validate(request)
    snapshot = NXDLSnapshot.model_validate(snapshot)
    field = _field_definition(profile, request.canonical_field)
    target = None
    unresolved: tuple[str, ...] = ()
    if request.relation == "no_match":
        resolution = "no_match"
    else:
        try:
            target = retrieve_definition(snapshot, request.target_class, request.target_path)
        except NexusResourceError:
            unresolved = ("target_definition_unavailable",)
        if target is not None:
            # The loader propagates relevant class-level dependencies into
            # each declaration. Unrelated nested group limitations do not
            # change this field's own context.
            unresolved = tuple(sorted(set(target.unresolved)))
            expected_units = _ROLE_UNITS.get(request.semantic_role)
            if expected_units is not None:
                if target.units is None:
                    unresolved += ("target_units_not_stated",)
                elif target.units != expected_units:
                    raise RegistryError("NXDL units category is incompatible with the selected role")
            elif target.units not in {None, "NX_UNITLESS"}:
                raise RegistryError("A dimensional target cannot represent this non-quantity role")
        if target is not None and request.target_class == "NXsample" and request.target_path == "temperature":
            if request.semantic_role != "acquisition_temperature":
                raise RegistryError("NXsample temperature guidance does not describe sample preparation")
        if target is not None and request.relation in {"exact", "equivalent"}:
            # Canonical values preserve source strings; non-text NXDL fields
            # instead declare different typed representations.
            if target.nx_type != "NX_CHAR":
                raise RegistryError("A string-to-nontext or unknown representation is only a partial mapping")
        resolution = "unresolved" if unresolved else "resolved"
    content = dict(
        registry_version=REGISTRY_VERSION, request=request.model_dump(mode="json"),
        profile_id=profile.profile_id, field_definition_json=canonical_json(field),
        field_definition_sha256=digest(field), resource_snapshot_id=snapshot.snapshot_id,
        target=None if target is None else target.model_dump(mode="json"),
        resolution_status=resolution, unresolved=unresolved,
        approval_status="unreviewed", support_status="not_assessed",
        scientific_value_policy="source_evidence_required", application_compliance="not_assessed",
    )
    entry_hash = digest(content)
    return MappingEntry(entry_id="nexus-mapping:" + entry_hash, entry_sha256=entry_hash, **content)


def build_registry(
    profile: Profile, snapshot: NXDLSnapshot, entries: tuple[MappingEntry, ...],
    reviews: tuple[ReviewRecord, ...] = (),
) -> ProfileRegistry:
    """Validate current bindings and attach only explicitly supplied reviews."""
    if not entries or len(entries) > 32 or len(reviews) > len(entries):
        raise RegistryError("Expected 1 to 32 entries and at most one review per entry")
    entries = tuple(MappingEntry.model_validate(item) for item in entries)
    reviews = tuple(ReviewRecord.model_validate(item) for item in reviews)
    entry_ids = {item.entry_id for item in entries}
    if len(entry_ids) != len(entries):
        raise RegistryError("Duplicate mapping entry")
    by_entry = {review.entry_id: review for review in reviews}
    if len(by_entry) != len(reviews) or any(key not in entry_ids for key in by_entry):
        raise RegistryError("Duplicate, unknown or stale review entry")
    result = []
    for entry in entries:
        if propose_entry(profile, snapshot, entry.request) != entry:
            raise RegistryError("Entry differs from the current profile or NXDL definition")
        review = by_entry.get(entry.entry_id)
        result.append(ReviewedMapping(entry=entry, review=review,
                                      approval_status="unreviewed" if review is None else review.decision))
    content = dict(registry_version=REGISTRY_VERSION, profile_id=profile.profile_id,
                   resource_snapshot_id=snapshot.snapshot_id,
                   entries=[item.model_dump(mode="json") for item in result],
                   scientific_value_policy="source_evidence_required",
                   application_compliance="not_assessed")
    return ProfileRegistry(registry_id="nexus-registry:" + digest(content), **content)


def verify_registry(profile: Profile, snapshot: NXDLSnapshot, registry: ProfileRegistry) -> None:
    registry = ProfileRegistry.model_validate(registry)
    rebuilt = build_registry(profile, snapshot, tuple(item.entry for item in registry.entries),
                             tuple(item.review for item in registry.entries if item.review is not None))
    if rebuilt != registry:
        raise RegistryError("Registry differs from the current inputs")


def lookup_guidance(
    profile: Profile, snapshot: NXDLSnapshot, registry: ProfileRegistry,
    canonical_field: str, semantic_role: SemanticRole, *, include_proposals: bool = False,
) -> tuple[Guidance, ...]:
    """Retrieve approved guidance by default; proposals require explicit opt-in.

    Neither approved definitions nor required application fields assert that a
    publication reports a value. This API has no record/value input or export.
    """
    verify_registry(profile, snapshot, registry)
    if type(include_proposals) is not bool:
        raise RegistryError("Proposal opt-in must be an explicit boolean")
    if semantic_role not in _FIELD_ROLES.get(canonical_field, ()):
        raise RegistryError("Requested field and role are outside the selected profile scope")
    results = []
    for item in registry.entries:
        entry = item.entry
        if (entry.request.canonical_field != canonical_field
                or entry.request.semantic_role != semantic_role or entry.target is None):
            continue
        if item.approval_status != "approved" and not (
            include_proposals and item.approval_status == "unreviewed"
        ):
            continue
        results.append(Guidance(
            entry_id=entry.entry_id, field_definition_json=entry.field_definition_json,
            semantic_role=semantic_role, relation=entry.request.relation, scope=entry.request.scope,
            target=entry.target, approval_status=item.approval_status, review=item.review,
            resolution_status=entry.resolution_status, unresolved=entry.unresolved,
        ))
    return tuple(results)
