"""Explicit publication splits and hash-pinned offline inputs.

Declarations do not establish undisclosed publication identities: a curator must
list preprints, versions, chunks and synthetic derivatives. No directory scan or
gold read occurs during manifest validation. File access is a separate operation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from meta_data_extraction.agent.paths import _parts, input_file, validate_run_id

Split = Literal["calibration", "historical", "future_held_out"]
Role = Literal["source", "gold", "prediction", "sidecar", "report"]
Collection = Literal[
    "bibliographic.authors", "bibliographic.affiliations", "bibliographic.funding",
    "facilities", "samples", "datasets",
]


class ManifestError(ValueError):
    """An offline declaration or hash-pinned input is invalid."""


class OfflineModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class ArtifactRef(OfflineModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        # Reuse the inherited agent boundary's portable lexical path rules.
        return "/".join(_parts(value))


class PredictionRef(OfflineModel):
    variant: str = Field(min_length=1)
    status: Literal["ready", "failed", "unprocessed"]
    artifact: ArtifactRef | None = None
    sidecar: ArtifactRef | None = None
    reference_report: ArtifactRef | None = None
    failure_code: str | None = Field(default=None, min_length=1)
    execution_mode: Literal["live", "mock", "cached_replay", "unknown"] = "unknown"
    request_count: int | None = Field(default=None, ge=0, strict=True)
    input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)

    @model_validator(mode="after")
    def _state(self) -> PredictionRef:
        if (self.status == "ready") != (self.artifact is not None):
            raise ValueError("Only a ready prediction must have a canonical artifact")
        if self.status == "ready" and self.failure_code is not None:
            raise ValueError("A ready prediction cannot declare a failure")
        if self.status == "failed" and self.failure_code is None:
            raise ValueError("A failed prediction needs an explicit failure code")
        if self.reference_report is not None and self.status != "ready":
            raise ValueError("Reference parity requires a ready prediction")
        return self


def _identity(value: str) -> str:
    value = unicodedata.normalize("NFC", value).strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/",
                   "https://dx.doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
            break
    if re.fullmatch(r"10\.\d{4,9}/\S+", value):
        return "doi:" + value
    preprint = re.fullmatch(
        r"(?:arxiv:|https?://arxiv\.org/(?:abs|pdf)/)?"
        r"(\d{4}\.\d{4,5}|[a-z.-]+/\d{7})(?:v\d+)?(?:\.pdf)?", value,
    )
    if preprint:
        return "arxiv:" + preprint[1]
    return value


class Publication(OfflineModel):
    publication_id: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    derived_from: str | None = Field(default=None, min_length=1)
    split: Split
    data_kind: Literal["natural", "synthetic"]
    sources: tuple[ArtifactRef, ...] = ()
    gold: ArtifactRef | None = None
    identity_gold: ArtifactRef | None = None
    predictions: tuple[PredictionRef, ...] = ()
    alignment_status: Literal[
        "frozen_historical", "synthetic", "unreviewed", "adjudication_required",
    ]
    blocked_collections: tuple[Collection, ...] = ()

    @field_validator("publication_id", "derived_from")
    @classmethod
    def _identifier(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("Publication identifiers must be nonblank with no edge whitespace")
        return value

    @model_validator(mode="after")
    def _declarations(self) -> Publication:
        identities = [_identity(item) for item in (self.publication_id, *self.aliases)]
        if any(not item for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("Publication aliases must be nonblank and distinct identities")
        variants = [item.variant for item in self.predictions]
        if len(set(variants)) != len(variants):
            raise ValueError("A publication may declare only one prediction per variant")
        if len(set(self.blocked_collections)) != len(self.blocked_collections):
            raise ValueError("Blocked collections must be distinct")
        if self.alignment_status == "adjudication_required" and not self.blocked_collections:
            raise ValueError("Required adjudication must identify blocked collections")
        if self.alignment_status == "synthetic" and self.data_kind != "synthetic":
            raise ValueError("Synthetic alignment applies only to synthetic data")
        if self.alignment_status == "frozen_historical" and self.split == "future_held_out":
            raise ValueError("Future held-out papers cannot claim frozen historical alignment")
        if self.identity_gold is not None and self.gold is None:
            raise ValueError("Identity gold requires a separately supplied gold record")
        paths = [item.path.casefold() for item in self.sources]
        if len(set(paths)) != len(paths):
            raise ValueError("Source paths must be distinct within a publication")
        return self


class SplitManifest(OfflineModel):
    version: Literal["m03-split-1"] = "m03-split-1"
    experiment_id: str
    description: str = Field(min_length=1)
    primary_outcome: str = Field(min_length=1)
    variants: tuple[str, ...] = Field(min_length=1)
    publications: tuple[Publication, ...] = ()
    split_notes: str = Field(min_length=1)
    held_out_policy: str = Field(min_length=1)

    @field_validator("experiment_id")
    @classmethod
    def _experiment_id(cls, value: str) -> str:
        return validate_run_id(value)

    def publication_group(self, publication_id: str) -> str:
        """Return the declared original publication, never a chunk/variant unit."""
        publications = {item.publication_id: item for item in self.publications}
        seen: set[str] = set()
        current = publication_id
        while True:
            if current in seen:
                raise ManifestError("Publication derivation contains a cycle")
            if current not in publications:
                raise ManifestError("Publication derivation points outside this manifest")
            seen.add(current)
            parent = publications[current].derived_from
            if parent is None:
                return current
            current = parent

    def matches_publication(self, publication_id: str, identifier: str) -> bool:
        """Match a declared identity only; never infer identity from record values."""
        group = self.publication_group(publication_id)
        allowed = {
            _identity(identity)
            for publication in self.publications
            if publication.publication_id in {publication_id, group}
            for identity in (publication.publication_id, *publication.aliases)
        }
        return _identity(identifier) in allowed

    @model_validator(mode="after")
    def _split_integrity(self) -> SplitManifest:
        if any(not item.strip() for item in self.variants) or len(set(self.variants)) != len(self.variants):
            raise ValueError("Variants must be nonblank and distinct")
        ids = [item.publication_id for item in self.publications]
        if len(set(ids)) != len(ids):
            raise ValueError("Publication IDs must be unique")
        publications = {item.publication_id: item for item in self.publications}
        claims: dict[tuple[str, str], tuple[str, str]] = {}
        file_hashes: dict[str, str] = {}
        for item in self.publications:
            group = self.publication_group(item.publication_id)
            if item.split != publications[group].split:
                raise ValueError("Every derivative must remain in its publication's split")
            if any(pred.variant not in self.variants for pred in item.predictions):
                raise ValueError("Prediction references an undeclared variant")
            identities = [("identity", _identity(value)) for value in (item.publication_id, *item.aliases)]
            sources = [key for source in item.sources for key in (
                ("source_path", source.path.casefold()), ("source_hash", source.sha256),
            )]
            for key in identities + sources:
                if key in claims and claims[key] != (group, item.split):
                    raise ValueError("Shared publication identity or source requires common lineage and split")
                claims[key] = (group, item.split)
            artifacts = [*item.sources, item.gold, item.identity_gold]
            for prediction in item.predictions:
                artifacts.extend((prediction.artifact, prediction.sidecar, prediction.reference_report))
            for artifact in artifacts:
                if artifact is None:
                    continue
                path = artifact.path.casefold()
                if path in file_hashes and file_hashes[path] != artifact.sha256:
                    raise ValueError("An input path cannot declare conflicting frozen hashes")
                file_hashes[path] = artifact.sha256
                key = ("artifact_path", path)
                if key in claims and claims[key] != (group, item.split):
                    raise ValueError("An artifact cannot belong to unrelated publication groups or splits")
                claims[key] = (group, item.split)
        return self


def manifest_digest(manifest: SplitManifest) -> str:
    manifest = SplitManifest.model_validate(manifest)
    payload = json.dumps(manifest.model_dump(mode="json"), sort_keys=True,
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strict_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ManifestError("Duplicate JSON keys are not permitted")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ManifestError("Nonfinite JSON numbers are not permitted")

    def number(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ManifestError("Nonfinite JSON numbers are not permitted")
        return parsed

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def load_manifest(root: Path | str, relative: str) -> SplitManifest:
    path = input_file(root, relative, allowed_roots=(
        "configs/agent", "configs/offline", "tests/offline/fixtures", "runs/agent",
    ))
    if path.suffix.lower() != ".json":
        raise ManifestError("The offline split manifest must be an explicit JSON file")
    return SplitManifest.model_validate(strict_json(path.read_bytes()))


_ROOTS: dict[str, tuple[str, ...]] = {
    "source": ("data/raw", "tests/offline/fixtures", "tests/agent/fixtures"),
    "gold": ("data/gold", "tests/offline/fixtures"),
    "prediction": ("data/experiment", "runs/agent", "tests/offline/fixtures"),
    "sidecar": ("data/experiment", "runs/agent", "tests/offline/fixtures"),
    "report": ("outputs/evaluation", "runs/agent", "tests/offline/fixtures"),
}


def read_artifact(root: Path | str, ref: ArtifactRef, *, role: Role,
                  split: Split | None = None, allow_final_gold: bool = False) -> bytes:
    """Read one named, hash-pinned offline input; never infer final-test approval.

    ``allow_final_gold`` is a caller's explicit offline scoring authorization,
    not a property stored in the manifest and not permission for runtime access.
    The hashes detect accidental changes, not hostile filesystem replacement.
    """
    ref = ArtifactRef.model_validate(ref)
    if role not in _ROOTS:
        raise ManifestError("Unrecognized offline artifact role")
    if role == "gold":
        if split not in ("calibration", "historical", "future_held_out"):
            raise ManifestError("Gold access requires an explicit publication split")
        if split == "future_held_out" and allow_final_gold is not True:
            raise ManifestError("Final-test gold requires explicit offline scoring authorization")
    path = input_file(root, ref.path, allowed_roots=_ROOTS[role])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref.sha256:
        raise ManifestError("Offline artifact content does not match its frozen hash")
    return raw


def artifact_json(root: Path | str, ref: ArtifactRef, *, role: Role,
                  split: Split | None = None, allow_final_gold: bool = False) -> dict[str, Any]:
    value = strict_json(read_artifact(root, ref, role=role, split=split,
                                     allow_final_gold=allow_final_gold))
    if not isinstance(value, dict):
        raise ManifestError("Expected an offline JSON object")
    return value
