"""Versioned immutable record/evidence envelopes; canonical JSON stays unchanged."""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, model_validator

from meta_data_extraction.agent.evidence import DocumentBundle, EvidenceRef, FrozenModel, sha256
from meta_data_extraction.agent.profiles import Profile, ValidationIssue, entity_addresses, reference_addresses


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8"))


def at_pointer(record: Any, pointer: str) -> Any:
    if pointer == "":
        return record
    if not pointer.startswith("/"):
        raise ValueError("Expected a JSON Pointer")
    current = record
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not token.isdecimal() or (len(token) > 1 and token.startswith("0")):
                raise ValueError("Invalid array address")
            current = current[int(token)]
        else:
            current = current[token]
    return current


class EntityHandle(FrozenModel):
    handle: str
    collection: str
    local_id: str | None
    path: str


class ReferenceLink(FrozenModel):
    path: str
    source_handle: str
    target_handle: str
    local_id: str


def _entities(record: dict, record_id: str) -> tuple[EntityHandle, ...]:
    entities = [EntityHandle(handle="entity:" + digest([record_id, "document"]),
                             collection="", local_id=None, path="")]
    occurrences: dict[tuple[str, str], int] = {}
    for address in entity_addresses(record):
        if address.local_id is not None:
            identity: Any = address.local_id
        else:
            content_hash = digest(at_pointer(record, address.path))
            key = (address.collection, content_hash)
            occurrences[key] = occurrences.get(key, 0) + 1
            identity = [content_hash, occurrences[key]]
        entities.append(EntityHandle(handle="entity:" + digest([record_id, address.collection, identity]),
                                     collection=address.collection, local_id=address.local_id, path=address.path))
    return tuple(entities)


def owner(entities: tuple[EntityHandle, ...], path: str) -> EntityHandle:
    return max((entity for entity in entities if not entity.path or path.startswith(entity.path + "/")),
               key=lambda entity: len(entity.path))


def _references(record: dict, entities: tuple[EntityHandle, ...]) -> tuple[ReferenceLink, ...]:
    by_local = {(e.collection, e.local_id): e for e in entities if e.local_id is not None}
    return tuple(ReferenceLink(path=address.path, source_handle=owner(entities, address.path).handle,
                               target_handle=by_local[(address.target_collection, address.local_id)].handle,
                               local_id=address.local_id)
                 for address in reference_addresses(record))


class RecordSnapshot(FrozenModel):
    state_version: Literal["m02-record-1"] = "m02-record-1"
    record_id: str
    revision_id: str
    profile_id: str
    source_id: str
    source_sha256: str
    representation_sha256: str
    canonical_json: str
    entities: tuple[EntityHandle, ...]
    references: tuple[ReferenceLink, ...]

    @property
    def canonical(self) -> dict:
        return json.loads(self.canonical_json)

    @model_validator(mode="after")
    def _integrity(self) -> RecordSnapshot:
        record = self.canonical
        if not isinstance(record, dict) or canonical_json(record) != self.canonical_json:
            raise ValueError("Snapshot requires canonical serialized object data")
        record_id = "record:" + digest([self.source_id, self.source_sha256, self.profile_id])
        revision = "revision:" + digest([record_id, self.representation_sha256, record])
        if (self.record_id, self.revision_id) != (record_id, revision):
            raise ValueError("Record or revision fingerprint does not match")
        if self.entities != _entities(record, record_id) or self.references != _references(record, self.entities):
            raise ValueError("Entity/reference sidecar does not match the record")
        return self


def snapshot(record: dict, bundle: DocumentBundle, profile: Profile) -> RecordSnapshot:
    record_id = "record:" + digest([bundle.source_id, bundle.source_sha256, profile.profile_id])
    entities = _entities(record, record_id)
    return RecordSnapshot(record_id=record_id,
                          revision_id="revision:" + digest([record_id, bundle.representation_sha256, record]),
                          profile_id=profile.profile_id, source_id=bundle.source_id,
                          source_sha256=bundle.source_sha256, representation_sha256=bundle.representation_sha256,
                          canonical_json=canonical_json(record), entities=entities,
                          references=_references(record, entities))


class AssertionEvidence(FrozenModel):
    record_revision: str
    entity_handle: str
    path: str
    field_path: str
    value_json: str
    link_status: Literal["location_verified", "unlinked", "not_requested"]
    support_status: Literal["not_assessed"] = "not_assessed"
    evidence: tuple[EvidenceRef, ...] = ()


def assertion_values(record: dict, parent: str = "") -> tuple[tuple[str, Any], ...]:
    values = []
    if isinstance(record, dict):
        for key, value in record.items():
            if key == "id":
                continue  # generated local IDs are technical handles, not publication assertions
            path = parent + "/" + key.replace("~", "~0").replace("/", "~1")
            values.extend(assertion_values(value, path))
    elif isinstance(record, list):
        for index, value in enumerate(record):
            values.extend(assertion_values(value, parent + f"/{index}"))
    elif record is not None:
        values.append((parent, record))
    return tuple(values)


class ExtractionIssue(FrozenModel):
    code: str
    path: str = ""
    message: str


class ProviderObservation(FrozenModel):
    status: Literal["ok", "timeout", "rate_limited", "failed", "blocked"]
    mode: Literal["mock", "live"]
    model: str
    finish_reason: str | None
    input_tokens: int | None = Field(ge=0, strict=True)
    output_tokens: int | None = Field(ge=0, strict=True)
    latency_ms: float | None = Field(ge=0, allow_inf_nan=False)
    error_code: str | None
    request_id: str | None
    response_sha256: str | None


class FixedExtraction(FrozenModel):
    state_version: Literal["m02-extraction-1"] = "m02-extraction-1"
    variant: Literal["inherited", "evidence"]
    status: Literal["succeeded", "needs_review", "failed"]
    parse_status: Literal["not_attempted", "valid", "recovered", "failed"]
    schema_status: Literal["not_checked", "valid", "invalid"]
    evidence_status: Literal["not_checked", "not_requested", "linked", "incomplete", "invalid"]
    response_truncated: bool = False
    source_incomplete: bool = False
    provider: ProviderObservation
    snapshot: RecordSnapshot | None = None
    assertions: tuple[AssertionEvidence, ...] = ()
    schema_issues: tuple[ValidationIssue, ...] = ()
    issues: tuple[ExtractionIssue, ...] = ()
    repairs: tuple[str, ...] = ()
    raw_output: str | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _bindings(self) -> FixedExtraction:
        if self.status != "failed" and (self.snapshot is None or self.schema_status != "valid"
                                         or self.provider.status != "ok" or self.response_truncated):
            raise ValueError("Non-failed extraction requires a complete, structurally valid response")
        if self.status != "failed" and (self.parse_status not in {"valid", "recovered"} or self.schema_issues):
            raise ValueError("Non-failed extraction requires successful parsing and no schema issues")
        if self.status == "succeeded" and (self.issues or self.repairs or self.source_incomplete
                                            or self.provider.finish_reason != "stop"):
            raise ValueError("Unresolved diagnostics require review")
        if self.snapshot is not None:
            expected = "not_requested" if self.variant == "inherited" else (
                "invalid" if any(i.code == "invalid_evidence" for i in self.issues) else
                "incomplete" if any(a.link_status == "unlinked" for a in self.assertions) else "linked"
            )
            if self.evidence_status != expected or (self.status == "succeeded" and expected in {"invalid", "incomplete"}):
                raise ValueError("Evidence completeness/status differs from assertion diagnostics")
            if any((a.link_status == "not_requested") != (self.variant == "inherited") for a in self.assertions):
                raise ValueError("Assertion status must match the requested evidence variant")
        if self.snapshot is None:
            if self.assertions:
                raise ValueError("Assertions require a record snapshot")
            return self
        values = dict(assertion_values(self.snapshot.canonical))
        if {a.path for a in self.assertions} != set(values) or len(self.assertions) != len(values):
            raise ValueError("Every emitted assertion needs an explicit evidence status")
        for assertion in self.assertions:
            entity = owner(self.snapshot.entities, assertion.path)
            if (assertion.record_revision != self.snapshot.revision_id
                    or assertion.entity_handle != entity.handle
                    or assertion.field_path != assertion.path[len(entity.path):]
                    or assertion.value_json != canonical_json(values[assertion.path])):
                raise ValueError("Assertion no longer addresses this entity/revision/value")
            if bool(assertion.evidence) != (assertion.link_status == "location_verified"):
                raise ValueError("Evidence link status is inconsistent")
            if any((ref.source_id, ref.source_sha256, ref.representation_sha256) != (
                self.snapshot.source_id, self.snapshot.source_sha256, self.snapshot.representation_sha256,
            ) for ref in assertion.evidence):
                raise ValueError("Assertion evidence belongs to another source snapshot")
        return self
