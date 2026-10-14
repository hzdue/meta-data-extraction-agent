"""Deterministic record checks, preserving invalid candidate data without repair.

This layer checks the frozen canonical shape and local references. It makes no
claim about publication support, correct attachments, or missing gold entities.
"""
from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import Field, model_validator

from meta_data_extraction.agent.contracts import RecordSnapshot, canonical_json, digest, snapshot
from meta_data_extraction.agent.evidence import DocumentBundle, FrozenModel, sha256
from meta_data_extraction.agent.profiles import Profile, ValidationIssue, validate_record

CHECK_VERSION = "m06-structure-1"


class RecordCheckError(ValueError):
    """Inputs or a saved structural report do not match their bindings."""


class CheckIssue(FrozenModel):
    category: Literal["structural", "reference"]
    code: Literal[
        "missing_field", "unknown_field", "type_mismatch", "cardinality",
        "invalid_value", "invalid_local_id", "duplicate_local_id",
        "dangling_reference", "ambiguous_reference", "schema_definition",
    ]
    path: str
    message: str = Field(min_length=1)
    origin: Literal["deterministic"] = "deterministic"
    recommended_action: Literal["review_representation", "inspect_source", "human_review"]

    @model_validator(mode="after")
    def _semantics(self) -> CheckIssue:
        category, action = _classification(self.code)
        if (self.category, self.recommended_action) != (category, action):
            raise ValueError("Issue category/action differs from the structural check code")
        if self.path and not self.path.startswith("/"):
            raise ValueError("Issue path must be a JSON Pointer")
        return self


def _classification(code: str) -> tuple[str, str]:
    if code in {"dangling_reference", "ambiguous_reference"}:
        return "reference", "inspect_source"
    if code in {"invalid_local_id", "duplicate_local_id"}:
        return "reference", "human_review"
    if code == "schema_definition":
        return "structural", "human_review"
    return "structural", "review_representation"


def _issue(issue: ValidationIssue) -> CheckIssue:
    category, action = _classification(issue.code)
    return CheckIssue(category=category, code=issue.code, path=issue.path,
                      message=issue.message, recommended_action=action)


def _json_value(value: Any) -> None:
    """Reject Python coercions that would silently change supplied JSON data."""
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise RecordCheckError("Candidate object keys must be strings")
        for child in value.values():
            _json_value(child)
    elif type(value) is list:
        for child in value:
            _json_value(child)
    elif type(value) is float:
        if not math.isfinite(value):
            raise RecordCheckError("Candidate numbers must be finite")
    elif value is not None and type(value) not in {str, int, bool}:
        raise RecordCheckError("Candidate requires uncoerced JSON values")


def _checked_profile(profile: Profile) -> Profile:
    """Check schema/identity consistency; resource files are not read here."""
    profile = Profile.model_validate(profile.model_dump())
    schema_json = json.dumps(profile.schema, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if (profile.schema_json != schema_json or sha256(schema_json.encode("utf-8")) != profile.schema_sha256
            or tuple(resource.name for resource in profile.schema_files) != profile.schema_names):
        raise RecordCheckError("Frozen profile schema or resource identity is inconsistent")
    identity = json.dumps({
        "profile_version": profile.profile_version,
        "schemas": [resource.model_dump() for resource in profile.schema_files],
        "schema_sha256": profile.schema_sha256,
        "prompt_sha256": profile.prompt_sha256,
        "prompt_version": profile.prompt_version,
    }, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if profile.profile_id != "profile:" + sha256(identity.encode("utf-8")):
        raise RecordCheckError("Frozen profile identity is inconsistent")
    return profile


class StructureReport(FrozenModel):
    check_version: Literal["m06-structure-1"] = CHECK_VERSION
    report_id: str
    profile_id: str
    source_id: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    representation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    record_id: str
    revision_id: str
    candidate_json: str
    canonical_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: Literal["valid", "invalid"]
    issues: tuple[CheckIssue, ...]
    snapshot: RecordSnapshot | None
    support_status: Literal["not_assessed"] = "not_assessed"
    coverage_status: Literal["unsupported"] = "unsupported"

    @property
    def candidate(self) -> dict[str, Any]:
        """Return an independent object, including every unexpected field."""
        return json.loads(self.candidate_json)

    @model_validator(mode="after")
    def _integrity(self) -> StructureReport:
        candidate = self.candidate
        if type(candidate) is not dict or canonical_json(candidate) != self.candidate_json:
            raise ValueError("Candidate must be a canonical serialized object")
        if sha256(self.candidate_json.encode("utf-8")) != self.canonical_sha256:
            raise ValueError("Candidate hash does not match")
        record_id = "record:" + digest([self.source_id, self.source_sha256, self.profile_id])
        revision_id = "revision:" + digest([record_id, self.representation_sha256, candidate])
        if (self.record_id, self.revision_id) != (record_id, revision_id):
            raise ValueError("Candidate record/revision binding does not match")
        if (self.status == "valid") != (not self.issues):
            raise ValueError("Structural status must reflect every structural issue")
        if (self.snapshot is not None) != (self.status == "valid"):
            raise ValueError("Only structurally valid data can have a record snapshot")
        if self.snapshot is not None:
            actual = (self.snapshot.record_id, self.snapshot.revision_id, self.snapshot.profile_id,
                      self.snapshot.source_id, self.snapshot.source_sha256,
                      self.snapshot.representation_sha256, self.snapshot.canonical_json)
            expected = (self.record_id, self.revision_id, self.profile_id, self.source_id,
                        self.source_sha256, self.representation_sha256, self.candidate_json)
            if actual != expected:
                raise ValueError("Snapshot does not match the structural candidate")
        payload = self.model_dump(mode="json", exclude={"report_id"})
        if self.report_id != "structure:" + digest(payload):
            raise ValueError("Structural report identity does not match")
        return self


def check_structure(profile: Profile, bundle: DocumentBundle, record: dict[str, Any]) -> StructureReport:
    """Preserve a JSON object and report the inherited validator's full findings.

    Invalid records remain addressable by the same record/revision hash rules as
    valid snapshots. They never enter RecordSnapshot's resolved-reference path.
    Source bytes and on-disk profile resources are the calling adapter's concern;
    this pure function revalidates the supplied immutable inputs only.
    """
    profile = _checked_profile(profile)
    bundle = DocumentBundle.model_validate(bundle)
    if type(record) is not dict:
        raise RecordCheckError("Structural checks require a parsed JSON object")
    try:
        _json_value(record)
        candidate_json = canonical_json(record)
    except RecursionError as exc:
        raise RecordCheckError("Candidate contains recursive or excessively deep data") from exc
    candidate = json.loads(candidate_json)
    issues = tuple(_issue(issue) for issue in validate_record(candidate, profile))
    record_id = "record:" + digest([bundle.source_id, bundle.source_sha256, profile.profile_id])
    payload = {
        "check_version": CHECK_VERSION, "profile_id": profile.profile_id,
        "source_id": bundle.source_id, "source_sha256": bundle.source_sha256,
        "representation_sha256": bundle.representation_sha256,
        "record_id": record_id,
        "revision_id": "revision:" + digest([record_id, bundle.representation_sha256, candidate]),
        "candidate_json": candidate_json, "canonical_sha256": sha256(candidate_json.encode("utf-8")),
        "status": "invalid" if issues else "valid", "issues": [issue.model_dump(mode="json") for issue in issues],
        "snapshot": None if issues else snapshot(candidate, bundle, profile).model_dump(mode="json"),
        "support_status": "not_assessed", "coverage_status": "unsupported",
    }
    return StructureReport(report_id="structure:" + digest(payload), **payload)


def verify_structure(report: StructureReport, profile: Profile, bundle: DocumentBundle) -> None:
    """Recompute findings, including after a caller recomputes a forged digest."""
    report = StructureReport.model_validate(report)
    if check_structure(profile, bundle, report.candidate) != report:
        raise RecordCheckError("Structural report no longer matches its profile, source, or findings")
