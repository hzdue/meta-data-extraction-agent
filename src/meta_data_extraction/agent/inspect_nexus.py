"""Offline inspection of pinned NXDL definitions and proposed profile mappings.

This command publishes definition/registry sidecars only. It does not call an
extractor, supply experimental facts, create reviews, or export a NeXus file.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
from typing import Literal

from pydantic import Field, model_validator

from meta_data_extraction.config_validation import load_yaml_mapping
from .evidence import FrozenModel, sha256
from .nexus_registry import MappingRequest, ReviewRecord, build_registry, lookup_guidance, propose_entry
from .nexus_resources import NXDLResourceSpec, load_nexus, retrieve_definition, verify_resources
from .nexus_units import QuantityRequest, convert_quantity
from .paths import input_file, validate_run_id, workspace_root
from .profiles import DEFAULT_SCHEMAS, load_profile, verify_profile_resources
from .storage import RunIncompleteError, RunStore


MAX_ARTIFACT_BYTES = 8_000_000


class DefinitionRequest(FrozenModel):
    class_name: str = Field(min_length=1, max_length=100)
    field_path: str = Field(min_length=1, max_length=1000)


class ReviewInput(FrozenModel):
    reviews: tuple[ReviewRecord, ...] = Field(max_length=50)


class NeXusRunConfig(FrozenModel):
    mode: Literal["offline"] = "offline"
    run_id: str
    resource_spec: str
    schema_names: tuple[str, ...] = DEFAULT_SCHEMAS
    mappings: tuple[MappingRequest, ...] = Field(min_length=1, max_length=20)
    definitions: tuple[DefinitionRequest, ...] = Field(default=(), max_length=20)
    quantity_checks: tuple[QuantityRequest, ...] = Field(default=(), max_length=20)
    reviews_path: str | None = None
    reviews_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _scope(self) -> NeXusRunConfig:
        validate_run_id(self.run_id)
        if (self.reviews_path is None) != (self.reviews_sha256 is None):
            raise ValueError("A supplied review artifact requires its exact byte hash")
        if len({m.mapping_key for m in self.mappings}) != len(self.mappings):
            raise ValueError("Mapping keys must be distinct")
        if len({(d.class_name, d.field_path) for d in self.definitions}) != len(self.definitions):
            raise ValueError("Definition requests must be distinct")
        return self


def _mapping(root: Path, relative: str, *, roots: tuple[str, ...], max_bytes: int = 1_000_000) -> dict:
    path = input_file(root, relative, allowed_roots=roots)
    if path.stat().st_size > max_bytes:
        raise ValueError("Inspection input exceeds the size limit")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError("Inspection input exceeds the size limit")
    return load_yaml_mapping(raw.decode("utf-8"), source=relative)


def load_config(root: Path | str, relative: str) -> NeXusRunConfig:
    return NeXusRunConfig.model_validate(_mapping(workspace_root(root), relative, roots=("configs/agent",)))


def _implementation(root: Path) -> dict[str, str]:
    names = ("paths", "config", "evidence", "contracts", "profiles", "storage",
             "nexus_resources", "nexus_registry", "nexus_units", "inspect_nexus")
    paths = [f"agent/{name}.py" for name in names] + ["config_validation.py", "data_models.py", "utils.py"]
    return {relative: sha256(input_file(root, f"src/meta_data_extraction/{relative}",
                                        allowed_roots=("src/meta_data_extraction",)).read_bytes())
            for relative in paths}


def inspect_nexus(root: Path | str, config: NeXusRunConfig, *, run_id: str | None = None) -> dict:
    """Inspect authentic definitions; default guidance contains approved entries only.

    Review files are explicit operator-supplied local artifacts. Hashes detect
    stale content, not the real-world identity or authority of a reviewer.
    """
    started = time.perf_counter()
    root = workspace_root(root)
    config = NeXusRunConfig.model_validate(config.model_dump())
    run_id = validate_run_id(run_id or config.run_id)
    spec = NXDLResourceSpec.model_validate(_mapping(root, config.resource_spec, roots=("configs/agent",)))
    if spec.kind != "authentic":
        raise ValueError("Inspection requires authentic NXDL; synthetic resources are test-only")
    snapshot = load_nexus(root, spec)
    profile = load_profile(root, config.schema_names)
    reviews: tuple[ReviewRecord, ...] = ()
    if config.reviews_path is not None:
        path = input_file(root, config.reviews_path, allowed_roots=("runs/agent",))
        if path.stat().st_size > 1_000_000:
            raise ValueError("Review artifact exceeds the size limit")
        raw = path.read_bytes()
        if len(raw) > 1_000_000 or sha256(raw) != config.reviews_sha256:
            raise ValueError("Review artifact content differs from its declared pin")
        reviews = ReviewInput.model_validate(load_yaml_mapping(raw.decode("utf-8"), source="supplied reviews")).reviews
    entries = tuple(propose_entry(profile, snapshot, mapping) for mapping in config.mappings)
    registry = build_registry(profile, snapshot, entries, reviews=reviews)
    definitions = tuple(retrieve_definition(snapshot, d.class_name, d.field_path) for d in config.definitions)
    conversions = tuple(convert_quantity(**request.model_dump()) for request in config.quantity_checks)
    keys = tuple(dict.fromkeys((m.canonical_field, m.semantic_role) for m in config.mappings))
    approved = tuple(g for field, role in keys for g in lookup_guidance(profile, snapshot, registry, field, role))
    proposals = tuple(g for field, role in keys for g in lookup_guidance(
        profile, snapshot, registry, field, role, include_proposals=True))
    request = {"version": "m05-run-1", "config": config.model_dump(mode="json"),
               "resource": spec.model_dump(mode="json"), "snapshot_id": snapshot.snapshot_id,
               "profile_id": profile.profile_id, "implementation": _implementation(root),
               "python": platform.python_version(),
               "dependencies": {name: version(name) for name in ("pydantic", "PyYAML")}, "model_calls": 0}
    result = {"version": "m05-result-1", "snapshot_id": snapshot.snapshot_id,
              "profile_id": profile.profile_id, "model_calls": 0,
              "registry": registry.model_dump(mode="json"),
              "approved_guidance": [g.model_dump(mode="json") for g in approved],
              "inspection_guidance": [g.model_dump(mode="json") for g in proposals],
              "definitions": [d.model_dump(mode="json") for d in definitions],
              "quantity_checks": [c.model_dump(mode="json") for c in conversions],
              "publication_support": "not_assessed", "nexus_compliance": "not_assessed",
              "canonical_modified": False}
    sidecar = {"profile": profile.model_dump(mode="json"), "resource": snapshot.model_dump(mode="json"),
               "registry": result["registry"], "definitions": result["definitions"],
               "quantity_checks": result["quantity_checks"]}
    # Keep every successful artifact within the bound accepted on replay.
    for artifact in (result, sidecar):
        encoded = json.dumps(artifact, ensure_ascii=False, sort_keys=True,
                             allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) + 1 > MAX_ARTIFACT_BYTES:
            raise ValueError("Inspection artifact exceeds the bounded projection size")
    store = RunStore.begin(root, run_id, request)
    verify_resources(snapshot, root)
    verify_profile_resources(profile, root)
    if store.completed is not None:
        saved = _mapping(root, f"runs/agent/{run_id}/sidecar.json", roots=(f"runs/agent/{run_id}",),
                         max_bytes=MAX_ARTIFACT_BYTES)
        if store.completed != result or saved != sidecar:
            raise ValueError("Cached registry report differs from the freshly verified inputs")
        execution = "cached_replay"
    else:
        store.complete(result, canonical=None, sidecar=sidecar, raw_response=None, retain_raw=False)
        execution = "definition_inspection"
    return {"execution": execution, "run_id": run_id, "model_calls": 0,
            "elapsed_ms": (time.perf_counter() - started) * 1000, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        report = inspect_nexus(args.root, load_config(args.root, args.config), run_id=args.run_id)
    except (ValueError, OSError, KeyError, TypeError, RunIncompleteError) as error:
        parser.exit(2, f"NeXus inspection failed ({type(error).__name__}); no successful registry report.\n")
    result = report["result"]
    print(json.dumps({"run_id": report["run_id"], "execution": report["execution"], "model_calls": 0,
                      "approved_guidance_count": len(result["approved_guidance"]),
                      "inspection_guidance_count": len(result["inspection_guidance"]),
                      "definition_count": len(result["definitions"]),
                      "quantity_statuses": [q["status"] for q in result["quantity_checks"]],
                      "canonical_modified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
