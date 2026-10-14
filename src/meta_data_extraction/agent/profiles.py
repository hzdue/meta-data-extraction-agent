"""Frozen inherited schemas and structural checks, without evaluator dependencies.

The inherited files are field-definition schemas, not JSON Schema. This adapter
checks their declared shapes and local references; it does not reinterpret their
scientific instructions or infer evidence support from structural validity.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from meta_data_extraction import utils
from meta_data_extraction.agent.paths import input_file


PROFILE_VERSION = "m02-profile-v1"
SCHEMA_NAMES = frozenset({
    "generic.json", "photon_and_neutron.json",
    "generic_no_instructions.json", "photon_and_neutron_no_instructions.json",
})
DEFAULT_SCHEMAS = ("generic.json", "photon_and_neutron.json")
PROMPT_NAME = "prompt_template.md"


class ProfileError(ValueError):
    """A profile could not be frozen or its resources have changed."""


class ResourceFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str
    version: str | None


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_version: Literal["m02-profile-v1"] = PROFILE_VERSION
    profile_id: str
    schema_names: tuple[str, ...]
    schema_payload_json: str
    schema_sha256: str
    schema_files: tuple[ResourceFingerprint, ...]
    prompt_template: str
    prompt_version: str | None
    prompt_sha256: str
    validation_notes: tuple[str, ...]

    @property
    def schema_json(self) -> str:
        return self.schema_payload_json

    @property
    def schema(self) -> dict[str, Any]:
        """Return a new object so callers cannot mutate the frozen snapshot."""
        return json.loads(self.schema_json)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: Literal[
        "missing_field", "unknown_field", "type_mismatch", "cardinality",
        "invalid_value", "invalid_local_id", "duplicate_local_id",
        "dangling_reference", "ambiguous_reference", "schema_definition",
    ]
    path: str
    message: str


class EntityAddress(BaseModel):
    """An address inside one revision, not an evidence or cross-revision ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    collection: str
    local_id: str | None
    path: str


class ReferenceAddress(BaseModel):
    """An emitted canonical local reference, before internal handle resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    target_collection: str
    local_id: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _schema_paths(root: Path | str, names: tuple[str, ...]) -> tuple[Path, ...]:
    if not names or len(set(names)) != len(names) or any(n not in SCHEMA_NAMES for n in names):
        raise ProfileError("Expected distinct allowlisted schema filenames")
    # Avoid silent replacement of the same namespace when full and ablated
    # variants are accidentally selected together.
    families = tuple(n.replace("_no_instructions", "") for n in names)
    if len(set(families)) != len(families):
        raise ProfileError("Choose only one variant of each schema family")
    return tuple(input_file(root, f"schemas/{n}", allowed_roots=("schemas",)) for n in names)


def load_profile(
    root: Path | str, schema_names: tuple[str, ...] = DEFAULT_SCHEMAS,
) -> Profile:
    """Freeze allowlisted resources using the inherited schema-loading helpers.

    Absolute, already checked paths bind inherited loaders to this checkout,
    without changing their module globals or reading few-shot/evaluation data.
    Rechecking bytes detects resources changing during snapshot construction.
    """
    names = tuple(schema_names)
    paths = _schema_paths(root, names)
    prompt_path = input_file(root, f"schemas/{PROMPT_NAME}", allowed_roots=("schemas",))
    resource_bytes = tuple(p.read_bytes() for p in paths)
    prompt_bytes = prompt_path.read_bytes()
    absolute_names = [str(p) for p in paths]
    merged = utils.load_merged_schema(absolute_names)
    versions = utils.get_schema_refs(absolute_names)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt_version = utils.get_prompt_version(prompt_path)
    if any(p.read_bytes() != raw for p, raw in zip(paths, resource_bytes, strict=True)) or (
        prompt_path.read_bytes() != prompt_bytes
    ):
        raise ProfileError("Profile resources changed while loading")
    schema_json = _canonical(merged)
    fingerprints = tuple(
        ResourceFingerprint(name=name, sha256=_sha256(raw), version=versions[str(path)])
        for name, path, raw in zip(names, paths, resource_bytes, strict=True)
    )
    prompt_hash = _sha256(prompt_bytes)
    schema_hash = _sha256(schema_json.encode("utf-8"))
    identity = _canonical({
        "profile_version": PROFILE_VERSION,
        "schemas": [f.model_dump() for f in fingerprints],
        "schema_sha256": schema_hash,
        "prompt_sha256": prompt_hash,
        "prompt_version": prompt_version,
    })
    return Profile(
        profile_id="profile:" + _sha256(identity.encode("utf-8")),
        schema_names=names, schema_payload_json=schema_json, schema_sha256=schema_hash,
        schema_files=fingerprints, prompt_template=prompt_template,
        prompt_version=prompt_version, prompt_sha256=prompt_hash,
        validation_notes=(
            "Declared boolean fields require true/false; the inherited skeleton uses null "
            "placeholders for them. Unknown-boolean missingness is unresolved and is not coerced.",
            "Regex patterns and allowed values are checked verbatim; prose format constraints "
            "and scientific extraction instructions require evidence review and are not inferred.",
            "Structural validity and resolved local references do not establish publication support, "
            "entity recovery, granularity, or correctness of attachments.",
        ),
    )


def verify_profile_resources(profile: Profile, root: Path | str) -> None:
    """Fail before dispatch if any frozen resource differs from this checkout."""
    current = load_profile(root, profile.schema_names)
    if current != profile:
        raise ProfileError("Frozen profile differs from current schema or prompt resources")


def render_inherited_prompt(profile: Profile, document_text: str) -> str:
    """Match utils.build_prompt using only the immutable snapshot.

    Keep the inherited ordering and escaping. Insert document text last so text
    resembling template placeholders is not interpreted as prompt formatting.
    """
    template = profile.prompt_template
    if template.startswith("---"):
        _, _, template = template.split("---", 2)
    schema = profile.schema
    rendered = (
        template.strip()
        .replace("{{SCHEMA_JSON}}", json.dumps(schema, indent=2))
        .replace("{{OUTPUT_SKELETON}}", json.dumps(utils.build_skeleton(schema), indent=2))
    )
    return rendered.replace("{{DOCUMENT_TEXT}}", document_text)


def _pointer(parent: str, key: str | int) -> str:
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


# These relations are stated explicitly in the canonical field descriptions and
# instructions. They do not use alignment aliases, gold data, or evaluator code.
_COLLECTIONS = (
    ("bibliographic/affiliations", "aff"),
    ("bibliographic/authors", "author"),
    ("facilities", "facility"),
    ("samples", "sample"),
    ("datasets", "dataset"),
)
_REFERENCES = (
    ("bibliographic/authors", "affiliation_refs", "bibliographic/affiliations"),
    ("datasets", "sample_refs", "samples"),
    ("datasets", "facility_refs", "facilities"),
)


def _collection(record: Any, path: str) -> list[Any]:
    current = record
    for part in path.split("/"):
        if not isinstance(current, dict):
            return []
        current = current.get(part)
    return current if isinstance(current, list) else []


def entity_addresses(record: dict[str, Any]) -> tuple[EntityAddress, ...]:
    """Enumerate entities without repairing, adding canonical IDs, or merging.

    Funding entries deliberately retain ``local_id=None``: the inherited schema
    defines no ID for them. Cross-revision handles belong in an internal envelope.
    """
    identified = tuple(
        EntityAddress(collection=collection, local_id=item["id"], path=f"/{collection}/{index}")
        for collection, _ in _COLLECTIONS
        for index, item in enumerate(_collection(record, collection))
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )
    return identified + tuple(
        EntityAddress(collection="bibliographic/funding", local_id=None,
                      path=f"/bibliographic/funding/{index}")
        for index, item in enumerate(_collection(record, "bibliographic/funding"))
        if isinstance(item, dict)
    )


def reference_addresses(record: dict[str, Any]) -> tuple[ReferenceAddress, ...]:
    """Expose declared reference fields without guessing or resolving targets.

    Duplicate and dangling references remain distinct, allowing an envelope to
    preserve invalid output while structural diagnostics report its problems.
    Non-string items are covered by ``validate_record`` and have no local ID.
    """
    return tuple(
        ReferenceAddress(path=f"/{source}/{index}/{field}/{ref_index}",
                         target_collection=target, local_id=ref)
        for source, field, target in _REFERENCES
        for index, item in enumerate(_collection(record, source))
        if isinstance(item, dict) and isinstance(item.get(field), list)
        for ref_index, ref in enumerate(item[field])
        if isinstance(ref, str)
    )


def validate_record(record: dict[str, Any], profile: Profile) -> tuple[ValidationIssue, ...]:
    """Check the frozen canonical shape and local references, without mutation.

    Missing fields, unknown fields and invalid values remain visible. An empty
    object cannot masquerade as an empty extraction. Nullable fields and empty
    lists retain their distinct canonical meanings.
    """
    issues: list[ValidationIssue] = []

    def issue(code: str, path: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, path=path, message=message))

    def object_shape(value: Any, definitions: dict[str, Any], path: str) -> None:
        if not isinstance(value, dict):
            issue("type_mismatch", path, "Expected an object")
            return
        for key in value:
            if key not in definitions:
                issue("unknown_field", _pointer(path, key), "Field is absent from the frozen schema")
        for key, definition in definitions.items():
            child = _pointer(path, key)
            if key not in value:
                issue("missing_field", child, "Canonical output requires this field")
            elif not isinstance(definition, dict):
                issue("schema_definition", child, "Unsupported field definition")
            elif "type" not in definition:
                object_shape(value[key], definition, child)
            else:
                field_shape(value[key], definition, child)

    def field_shape(value: Any, definition: dict[str, Any], path: str) -> None:
        kind = definition.get("type")
        if not isinstance(kind, str):
            issue("schema_definition", path, "Field type must be a string")
            return
        options = kind.split("|")
        if value is None and "null" in options:
            return
        if kind.startswith("list[") and kind.endswith("]"):
            if not isinstance(value, list):
                issue("type_mismatch", path, f"Expected {kind}")
                return
            cardinality = definition.get("cardinality", {})
            if len(value) < cardinality.get("min", 0) or len(value) > cardinality.get("max", float("inf")):
                issue("cardinality", path, "List length is outside the declared cardinality")
            item_schema = definition.get("item_schema")
            for index, item in enumerate(value):
                child = _pointer(path, index)
                if isinstance(item_schema, dict):
                    object_shape(item, item_schema, child)
                elif kind == "list[string]":
                    field_shape(item, {"type": "string", "constraints": definition.get("constraints", {})}, child)
                else:
                    issue("schema_definition", child, "List item has no supported item schema")
            return
        if "string" in options:
            valid = isinstance(value, str)
        elif kind == "boolean":
            valid = type(value) is bool
        else:
            issue("schema_definition", path, f"Unsupported declared type: {kind}")
            return
        if not valid:
            issue("type_mismatch", path, f"Expected {kind}")
            return
        allowed = definition.get("constraints", {}).get("allowed_values")
        if allowed is not None and value not in allowed:
            issue("invalid_value", path, "Value is absent from the declared allowed values")
        pattern = definition.get("constraints", {}).get("pattern")
        if pattern is not None and isinstance(value, str):
            try:
                if re.fullmatch(pattern, value) is None:
                    issue("invalid_value", path, "Value does not match the declared pattern")
            except re.error:
                issue("schema_definition", path, "Declared regular expression is invalid")

    object_shape(record, profile.schema, "")
    counts: dict[str, dict[str, int]] = {}
    for collection, prefix in _COLLECTIONS:
        counts[collection] = {}
        for index, item in enumerate(_collection(record, collection)):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            local_id = item["id"]
            path = f"/{collection}/{index}/id"
            if re.fullmatch(rf"{prefix}_[1-9][0-9]*", local_id) is None:
                issue("invalid_local_id", path, f"Expected local ID of the form {prefix}_1, {prefix}_2, ...")
            counts[collection][local_id] = counts[collection].get(local_id, 0) + 1
            if counts[collection][local_id] > 1:
                issue("duplicate_local_id", path, "Local ID is duplicated within its collection")
    for ref in reference_addresses(record):
        count = counts[ref.target_collection].get(ref.local_id, 0)
        if count == 0:
            issue("dangling_reference", ref.path, f"Reference does not resolve in {ref.target_collection}")
        elif count > 1:
            issue("ambiguous_reference", ref.path, f"Reference resolves to multiple entries in {ref.target_collection}")
    return tuple(issues)
