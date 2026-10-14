"""Read-only standards adapters for freshly extracted publication records.

PaNET results remain revision-bound, unreviewed sidecars. Exact occurrence is
only location evidence: this adapter never infers that a technique was used in
the current study. NeXus runtime selection accepts only the existing registry's
approved, resolved guidance and never manufactures review records.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from meta_data_extraction.config_validation import load_yaml_mapping

from .contracts import AssertionEvidence, RecordSnapshot, at_pointer, canonical_json, digest, owner
from .evidence import DocumentBundle, EvidenceRef, FrozenModel, evidence_ref, resolve_evidence, sha256
from .nexus_registry import ProfileRegistry, SemanticRole, lookup_guidance, verify_registry
from .nexus_resources import NXDLResourceSpec, NXDLSnapshot, load_nexus
from .panet_lookup import CandidateSet, lexical_candidates, normalize_query
from .panet_mapping import MappingDecision, TechniqueMention, propose_mapping
from .panet_resources import ResourceSpec, load_panet
from .paths import input_file
from .profiles import Profile, verify_profile_resources

STANDARDS_VERSION = "m10-publication-standards-1"
MAX_REGISTRY_BYTES = 4_000_000


class PaNETSettings(FrozenModel):
    resource: ResourceSpec
    families: tuple[str, ...] = Field(min_length=1, max_length=2)
    candidate_limit: int = Field(default=5, ge=1, le=50, strict=True)
    max_mentions_per_assertion: int = Field(default=4, ge=1, le=8, strict=True)

    @model_validator(mode="after")
    def _scope(self) -> PaNETSettings:
        if any(not value.strip() for value in self.families) or len(set(self.families)) != len(self.families):
            raise ValueError("Technique families must be distinct, nonblank resource IDs")
        return self


class NeXusSettings(FrozenModel):
    resource: NXDLResourceSpec
    registry_path: str = Field(min_length=1, max_length=500)
    registry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class TechniqueInspection(FrozenModel):
    path: str
    entity_handle: str
    original_expression: str
    status: Literal["inspected", "not_located", "not_searchable"]
    evidence_origin: Literal["assertion_evidence", "source_literal_search"]
    candidates: CandidateSet | None
    decisions: tuple[MappingDecision, ...]
    mentions_truncated: bool = Field(strict=True)
    approval_status: Literal["unreviewed"] = "unreviewed"
    support_status: Literal["not_assessed"] = "not_assessed"

    @model_validator(mode="after")
    def _status(self) -> TechniqueInspection:
        if (self.status == "inspected") != bool(self.decisions):
            raise ValueError("Inspected techniques require actual publication mentions")
        if (self.status == "not_searchable") != (self.candidates is None):
            raise ValueError("Only unsearchable expressions lack candidates")
        if self.mentions_truncated and not self.decisions:
            raise ValueError("Truncated mention selection requires retained mentions")
        for decision in self.decisions:
            if (decision.binding is None or decision.binding.path != self.path
                    or decision.binding.entity_handle != self.entity_handle
                    or decision.mention.original_expression != self.original_expression
                    or decision.candidates != self.candidates
                    or decision.mention.context_use != "uncertain"
                    or decision.status != "unresolved" or decision.reason != "usage_uncertain"):
                raise ValueError("Inspection must retain unreviewed, usage-uncertain record bindings")
        return self


class PaNETRecordReport(FrozenModel):
    version: Literal["m10-publication-standards-1"] = STANDARDS_VERSION
    report_id: str
    record_id: str
    revision_id: str
    profile_id: str
    source_id: str
    source_sha256: str
    representation_sha256: str
    settings: PaNETSettings
    snapshot_id: str
    techniques: tuple[TechniqueInspection, ...]
    canonical_modified: Literal[False] = False
    model_calls: Literal[0] = 0
    publication_support: Literal["not_assessed"] = "not_assessed"

    @model_validator(mode="after")
    def _binding(self) -> PaNETRecordReport:
        if self.report_id != "publication-standards:" + digest(self.model_dump(mode="json", exclude={"report_id"})):
            raise ValueError("Standards report identity differs from its full content")
        paths = [item.path for item in self.techniques]
        if len(paths) != len(set(paths)):
            raise ValueError("Technique inspection paths must be unique")
        for item in self.techniques:
            if item.candidates is not None and item.candidates.snapshot_id != self.snapshot_id:
                raise ValueError("Technique candidates use another standards snapshot")
            for decision in item.decisions:
                binding = decision.binding
                if (binding.record_id, binding.revision_id, binding.profile_id) != (
                        self.record_id, self.revision_id, self.profile_id):
                    raise ValueError("Technique decision belongs to a different record revision")
                if (decision.source_id, decision.source_sha256, decision.representation_sha256) != (
                        self.source_id, self.source_sha256, self.representation_sha256):
                    raise ValueError("Technique decision belongs to another source")
        return self


class NeXusRuntimeSelection(FrozenModel):
    settings: NeXusSettings
    snapshot: NXDLSnapshot
    registry: ProfileRegistry
    status: Literal["ready", "blocked_no_approved_guidance"]
    definition_requests: dict[str, tuple[str, SemanticRole]]
    publication_support: Literal["not_assessed"] = "not_assessed"
    application_compliance: Literal["not_assessed"] = "not_assessed"

    @model_validator(mode="after")
    def _scope(self) -> NeXusRuntimeSelection:
        if (self.snapshot.resource != self.settings.resource
                or self.registry.resource_snapshot_id != self.snapshot.snapshot_id):
            raise ValueError("Runtime guidance resource bindings differ")
        expected = {item.entry.request.mapping_key: (
            item.entry.request.canonical_field, item.entry.request.semantic_role)
            for item in self.registry.entries
            if item.approval_status == "approved" and item.entry.resolution_status == "resolved"
            and item.entry.target is not None and not item.entry.unresolved}
        if self.definition_requests != expected or (self.status == "ready") != bool(expected):
            raise ValueError("Definition requests must contain only approved, resolved registry entries")
        return self


def _checked_assertions(record: RecordSnapshot, bundle: DocumentBundle,
                        assertions: tuple[AssertionEvidence, ...]) -> dict[str, AssertionEvidence]:
    result = {}
    for supplied in assertions:
        assertion = AssertionEvidence.model_validate(supplied.model_dump())
        if assertion.path in result:
            raise ValueError("Duplicate assertion evidence path")
        entity = owner(record.entities, assertion.path)
        if (assertion.record_revision != record.revision_id
                or assertion.entity_handle != entity.handle
                or assertion.field_path != assertion.path[len(entity.path):]
                or assertion.value_json != canonical_json(at_pointer(record.canonical, assertion.path))
                or bool(assertion.evidence) != (assertion.link_status == "location_verified")):
            raise ValueError("Assertion evidence no longer binds the current record revision")
        for reference in assertion.evidence:
            resolve_evidence(bundle, reference)
        result[assertion.path] = assertion
    return result


def _mentions(bundle: DocumentBundle, expression: str, contexts: tuple[EvidenceRef, ...],
              limit: int) -> tuple[tuple[TechniqueMention, ...], bool]:
    retained = []
    seen = set()
    for context in contexts:
        start = context.start
        while True:
            start = bundle.text.find(expression, start, context.end)
            if start < 0:
                break
            end = start + len(expression)
            identity = (context.region_id, start, end)
            if identity not in seen:
                if len(retained) == limit:
                    return tuple(retained), True
                seen.add(identity)
                retained.append(TechniqueMention(
                    original_expression=expression,
                    mention=evidence_ref(bundle, context.region_id, start=start, end=end),
                    context=context, context_use="uncertain",
                ))
            start = end
    return tuple(retained), False


def map_record_techniques(
    root: Path | str, bundle: DocumentBundle, record: RecordSnapshot,
    settings: PaNETSettings, *, assertions: tuple[AssertionEvidence, ...] = (),
) -> PaNETRecordReport:
    """Inspect each emitted technique, preserving original expressions and IDs.

Verified assertion locations restrict mention selection when present. Otherwise
only exact source occurrences are used, explicitly labelled as a literal search.
Neither path establishes usage or attachment support. No normalization or model
call occurs. Call again after an edit to obtain sidecars for the new revision.
"""
    bundle = DocumentBundle.model_validate(bundle.model_dump())
    record = RecordSnapshot.model_validate(record.model_dump())
    settings = PaNETSettings.model_validate(settings.model_dump())
    if (record.source_id, record.source_sha256, record.representation_sha256) != (
            bundle.source_id, bundle.source_sha256, bundle.representation_sha256):
        raise ValueError("Record and source snapshot differ")
    evidence = _checked_assertions(record, bundle, assertions)
    ontology = load_panet(root, settings.resource)
    # Validate family IDs even when the extracted record contains no techniques.
    lexical_candidates(ontology, "scope validation", families=settings.families, limit=1)
    datasets = record.canonical.get("datasets", [])
    if not isinstance(datasets, list) or len(datasets) > 128:
        raise ValueError("Standards inspection permits at most 128 dataset assertions")
    techniques = []
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError("Dataset must be an object")
        expression = dataset.get("measurement_technique")
        if expression is None:
            continue
        if not isinstance(expression, str):
            raise ValueError("Technique assertion must retain its source string")
        path = f"/datasets/{index}/measurement_technique"
        assertion = evidence.get(path)
        linked = assertion is not None and bool(assertion.evidence)
        origin = "assertion_evidence" if linked else "source_literal_search"
        candidates = None
        decisions = ()
        truncated = False
        status = "not_searchable"
        if 1 <= len(expression) <= 4096 and normalize_query(expression):
            candidates = lexical_candidates(ontology, expression, families=settings.families,
                                            limit=settings.candidate_limit)
            contexts = assertion.evidence if linked else tuple(
                evidence_ref(bundle, region.region_id) for region in bundle.regions)
            mentions, truncated = _mentions(bundle, expression, contexts, settings.max_mentions_per_assertion)
            decisions = tuple(propose_mapping(ontology, candidates, bundle, mention,
                                              record=record, assertion_path=path) for mention in mentions)
            status = "inspected" if decisions else "not_located"
        techniques.append(TechniqueInspection(
            path=path, entity_handle=owner(record.entities, path).handle,
            original_expression=expression, status=status, evidence_origin=origin,
            candidates=candidates, decisions=decisions, mentions_truncated=truncated,
        ))
    content = dict(
        version=STANDARDS_VERSION, record_id=record.record_id, revision_id=record.revision_id,
        profile_id=record.profile_id, source_id=bundle.source_id, source_sha256=bundle.source_sha256,
        representation_sha256=bundle.representation_sha256, settings=settings.model_dump(mode="json"),
        snapshot_id=ontology.snapshot_id, techniques=[item.model_dump(mode="json") for item in techniques],
        canonical_modified=False, model_calls=0, publication_support="not_assessed",
    )
    return PaNETRecordReport.model_validate({**content, "report_id": "publication-standards:" + digest(content)})


def _registry_bytes(root: Path | str, settings: NeXusSettings) -> bytes:
    roots = ("runs/agent",) if settings.resource.kind == "authentic" else ("tests/agent/fixtures", "runs/agent")
    path = input_file(root, settings.registry_path, allowed_roots=roots)
    if path.stat().st_size > MAX_REGISTRY_BYTES:
        raise ValueError("Registry artifact exceeds its size limit")
    raw = path.read_bytes()
    if len(raw) > MAX_REGISTRY_BYTES or sha256(raw) != settings.registry_sha256:
        raise ValueError("Registry artifact differs from its pinned content")
    return raw


def load_reviewed_nexus(root: Path | str, profile: Profile,
                       settings: NeXusSettings) -> NeXusRuntimeSelection:
    """Load an explicit pinned registry; unreviewed entries never enable tools.

The existing registry verifies review content binding, not reviewer identity.
The operator remains responsible for the origin of supplied review artifacts.
Synthetic resources and fabricated unit-test reviews retain synthetic provenance.
"""
    settings = NeXusSettings.model_validate(settings.model_dump())
    profile = Profile.model_validate(profile.model_dump())
    verify_profile_resources(profile, root)
    resource = load_nexus(root, settings.resource)
    registry = ProfileRegistry.model_validate(load_yaml_mapping(
        _registry_bytes(root, settings).decode("utf-8"), source="pinned standards registry"))
    verify_registry(profile, resource, registry)
    requests = {}
    for item in registry.entries:
        entry = item.entry
        if (item.approval_status != "approved" or entry.resolution_status != "resolved"
                or entry.target is None or entry.unresolved):
            continue
        request = entry.request
        guidance = lookup_guidance(profile, resource, registry, request.canonical_field,
                                   request.semantic_role, include_proposals=False)
        if not guidance or any(item.approval_status != "approved" or item.resolution_status != "resolved"
                               or item.unresolved for item in guidance):
            raise ValueError("Runtime guidance failed its approval/resolution gate")
        requests[request.mapping_key] = (request.canonical_field, request.semantic_role)
    return NeXusRuntimeSelection(
        settings=settings, snapshot=resource, registry=registry, definition_requests=requests,
        status="ready" if requests else "blocked_no_approved_guidance",
    )


def standards_input_pins(root: Path | str, *, panet: PaNETSettings | None = None,
                        nexus: NeXusSettings | None = None) -> dict[str, str]:
    """Recheck resources and expose every immutable input path/hash for a run pin.

No file discovery, download, artifact write or directory hashing takes place.
The caller separately freezes these settings and its profile/implementation.
"""
    pins: dict[str, str] = {}

    def add(path: str, fingerprint: str) -> None:
        key = path.replace("\\", "/")
        if key in pins and pins[key] != fingerprint:
            raise ValueError("Conflicting standards pins for one path")
        pins[key] = fingerprint

    if panet is not None:
        panet = PaNETSettings.model_validate(panet.model_dump())
        load_panet(root, panet.resource)
        add(panet.resource.path, panet.resource.sha256)
    if nexus is not None:
        nexus = NeXusSettings.model_validate(nexus.model_dump())
        load_nexus(root, nexus.resource)
        _registry_bytes(root, nexus)
        for item in (*nexus.resource.files, *((nexus.resource.license_file,) if nexus.resource.license_file else ())):
            add(item.path, item.sha256)
        add(nexus.registry_path, nexus.registry_sha256)
    return dict(sorted(pins.items()))
