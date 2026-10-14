"""Synthetic offline M06 checks and bounded patch inspection with immutable replay.

Only explicitly supplied test fixtures enter this CLI. The checker and patch
functions remain separately usable; no live service or standards are selected here.
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
from .contracts import canonical_json, digest
from .evidence import DocumentBundle, FrozenModel, evidence_ref, sha256
from .paths import input_file, validate_run_id, workspace_root
from .profiles import DEFAULT_SCHEMAS, load_profile, verify_profile_resources
from .providers import MockProvider, ProviderResult
from .record_checks import check_structure, verify_structure
from .record_patches import PatchPolicy, PatchProposal, apply_supported_patch, build_candidate
from .sources import prepare_sources
from .storage import RunIncompleteError, RunStore
from .support_checks import SupportTarget, check_support, diagnose_reports, interpret_support_response, verify_report

MAX_INPUT_BYTES = 1_000_000
MAX_ARTIFACT_BYTES = 8_000_000
MODEL_NAME = "m06-synthetic-checker"


class InputPin(FrozenModel):
    path: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Span(FrozenModel):
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def _range(self) -> Span:
        if self.start >= self.end:
            raise ValueError("Evidence span must be nonempty")
        return self


class TargetRequest(FrozenModel):
    path: str = Field(min_length=1, max_length=500)
    spans: tuple[Span, ...] = Field(min_length=1, max_length=8)


class RecordRunConfig(FrozenModel):
    mode: Literal["synthetic_offline"] = "synthetic_offline"
    run_id: str
    source: InputPin
    record: InputPin
    schema_names: tuple[str, ...] = DEFAULT_SCHEMAS
    targets: tuple[TargetRequest, ...] = Field(default=(), max_length=32)
    before_response: InputPin | None = None
    patch: PatchProposal | None = None
    patch_policy: PatchPolicy = Field(default_factory=PatchPolicy)
    after_targets: tuple[TargetRequest, ...] = Field(default=(), max_length=32)
    after_response: InputPin | None = None
    raw_retention: Literal["retain", "hash_only"]

    @model_validator(mode="after")
    def _scope(self) -> RecordRunConfig:
        validate_run_id(self.run_id)
        if bool(self.targets) != (self.before_response is not None):
            raise ValueError("A checker fixture and its target scope must be supplied together")
        for targets in (self.targets, self.after_targets):
            if len({target.path for target in targets}) != len(targets):
                raise ValueError("Check targets must be distinct")
        if self.patch is None:
            if self.after_response is not None or self.after_targets or self.patch_policy.accept_model_judgments:
                raise ValueError("Candidate checks and enabled patch policy require a patch")
        elif not self.targets or not self.after_targets or self.after_response is None:
            raise ValueError("A patch requires explicit before/candidate checker fixtures")
        elif self.patch.provenance != "synthetic_fixture":
            raise ValueError("This diagnostic CLI requires a synthetic patch declaration")
        return self


def _bytes(root: Path, relative: str, *, roots: tuple[str, ...]) -> bytes:
    path = input_file(root, relative, allowed_roots=roots)
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Inspection input exceeds the size bound")
    raw = path.read_bytes()
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Inspection input exceeds the size bound")
    return raw


def _fixture(root: Path, pin: InputPin) -> bytes:
    raw = _bytes(root, pin.path, roots=("tests/agent/fixtures",))
    if sha256(raw) != pin.sha256:
        raise ValueError("Fixture content differs from its declared hash")
    return raw


def load_config(root: Path | str, relative: str) -> RecordRunConfig:
    raw = _bytes(workspace_root(root), relative, roots=("configs/agent",))
    return RecordRunConfig.model_validate(load_yaml_mapping(raw.decode("utf-8"), source=relative))


def _targets(bundle: DocumentBundle, requests: tuple[TargetRequest, ...]) -> tuple[SupportTarget, ...]:
    targets = []
    for request in requests:
        refs = []
        for span in request.spans:
            regions = [r for r in bundle.regions if r.start <= span.start < span.end <= r.end]
            if not regions:
                raise ValueError("The declared evidence span does not resolve in an actual source region")
            region = min(regions, key=lambda r: (r.end - r.start, r.region_id))
            refs.append(evidence_ref(bundle, region.region_id, start=span.start, end=span.end))
        targets.append(SupportTarget(path=request.path, evidence=tuple(refs)))
    return tuple(targets)


def _implementation(root: Path) -> dict[str, str]:
    names = ["agent/" + name + ".py" for name in (
        "config", "paths", "evidence", "sources", "profiles", "providers", "contracts",
        "storage", "record_checks", "support_checks", "record_patches", "inspect_record",
    )] + ["config_validation.py", "components/llm.py", "components/parser.py", "base/llm.py", "utils.py", "data_models.py"]
    return {name: sha256(input_file(root, "src/meta_data_extraction/" + name,
                                   allowed_roots=("src/meta_data_extraction",)).read_bytes()) for name in names}


def inspect_record(root: Path | str, config: RecordRunConfig, *, run_id: str | None = None) -> dict:
    """Run declared synthetic diagnostics, or reinterpret fixtures without dispatch on replay."""
    started = time.perf_counter()
    root = workspace_root(root)
    config = RecordRunConfig.model_validate(config.model_dump())
    run_id = validate_run_id(run_id or config.run_id)
    _fixture(root, config.source)
    raw_record = _fixture(root, config.record)
    raw_before = _fixture(root, config.before_response).decode("utf-8") if config.before_response else None
    raw_after = _fixture(root, config.after_response).decode("utf-8") if config.after_response else None
    prepared_source = prepare_sources(root, [config.source.path])[0]
    if prepared_source.bundle is None:
        raise ValueError("Source preparation failed")
    bundle = prepared_source.bundle
    if bundle.source_sha256 != config.source.sha256:
        raise ValueError("Source changed while preparing its evidence bundle")
    profile = load_profile(root, config.schema_names)
    record = load_yaml_mapping(raw_record.decode("utf-8"), source="synthetic record fixture")
    structural = check_structure(profile, bundle, record)
    targets = _targets(bundle, config.targets)
    after_targets = _targets(bundle, config.after_targets)
    candidate = build_candidate(profile, bundle, structural.snapshot, config.patch) if (
        structural.snapshot is not None and config.patch is not None) else None
    request = {"version": "m06-run-1", "config": config.model_dump(mode="json"),
               "profile_id": profile.profile_id, "source_snapshot_sha256": digest(bundle.model_dump(mode="json")),
               "record_revision": structural.revision_id, "model": MODEL_NAME,
               "implementation": _implementation(root), "python": platform.python_version(),
               "dependencies": {name: version(name) for name in ("pydantic", "PyYAML", "openai", "httpx", "pymupdf4llm", "pymupdf")},
               "standards_resources": [], "live_requests": 0}
    store = RunStore.begin(root, run_id, request)
    cached = store.completed is not None
    new_mock_requests = 0

    def inspect(snapshot, selected, raw):
        nonlocal new_mock_requests
        response = ProviderResult(status="ok", mode="mock", content=raw, finish_reason="stop")
        if cached:
            report = interpret_support_response(profile, snapshot, bundle, selected, response,
                                                model_name=MODEL_NAME, raw_retention=config.raw_retention)
        else:
            provider = MockProvider((response,), model_name=MODEL_NAME)
            report = check_support(root, profile, snapshot, bundle, selected, provider,
                                   raw_retention=config.raw_retention)
            new_mock_requests += len(provider.requests)
        verify_report(report, profile, snapshot, bundle, root=root)
        return report

    before = inspect(structural.snapshot, targets, raw_before) if structural.snapshot is not None and targets else None
    after = None
    patched = None
    patch_state = {"status": "not_requested", "code": "no_patch_requested"}
    if config.patch is not None:
        if candidate is None:
            patch_state = {"status": "rejected", "code": "invalid_original_structure"}
        elif candidate.prepared is None:
            patch_state = {"status": candidate.status, "code": candidate.code}
        elif not config.patch_policy.accept_model_judgments:
            patch_state = {"status": "rejected", "code": "model_judgment_policy_disabled"}
        elif before is None or before.status != "succeeded":
            patch_state = {"status": "rejected", "code": "before_support_incomplete"}
        elif any(j.path in candidate.prepared.changed_paths and j.diagnosis == "context_conflict"
                 for j in before.judgments):
            patch_state = {"status": "rejected", "code": "source_conflict_requires_review"}
        elif any(not any(j.path == path and j.verdict == "contradicted" and j.evidence for j in before.judgments)
                 for path in candidate.prepared.changed_paths):
            patch_state = {"status": "rejected", "code": "old_assertion_not_contradicted"}
        else:
            after = inspect(candidate.prepared.candidate, after_targets, raw_after)
            patched = apply_supported_patch(profile, bundle, structural.snapshot, candidate.prepared,
                                             before, after, policy=config.patch_policy)
            patch_state = {"status": patched.status, "code": patched.code}
    final_snapshot = patched.snapshot if patched is not None else structural.snapshot
    before_diagnosis = diagnose_reports((before,), profile, structural.snapshot, bundle) if before else None
    after_diagnosis = diagnose_reports((after,), profile, candidate.prepared.candidate, bundle) if after else None
    result = {"version": "m06-result-1", "mode": "synthetic_offline", "structure": structural.model_dump(mode="json"),
              "before": before.model_dump(mode="json") if before else None,
              "candidate": candidate.model_dump(mode="json") if candidate else None,
              "after": after.model_dump(mode="json") if after else None,
              "before_diagnosis": before_diagnosis.model_dump(mode="json") if before_diagnosis else None,
              "after_diagnosis": after_diagnosis.model_dump(mode="json") if after_diagnosis else None,
              "patch": patched.model_dump(mode="json") if patched else None, "patch_state": patch_state,
              "support_execution": "blocked_by_structure" if structural.snapshot is None else
                  "checked" if before else "not_requested",
              "coverage_status": "unsupported", "support_scope": "selected_assertions_only",
              "expert_approval": "not_supplied", "live_requests": 0,
              "mock_requests": int(before is not None and before.provider is not None) + int(after is not None and after.provider is not None),
              "cost": None, "cost_status": "not_applicable_mock",
              "final_revision": final_snapshot.revision_id if final_snapshot else None}
    sidecar = {"source": bundle.model_dump(mode="json"), "profile": profile.model_dump(mode="json"), "analysis": result}
    retained = {name: report.raw_output for name, report in (("before", before), ("after", after))
                if report is not None and report.raw_output is not None}
    raw_output = canonical_json(retained) if retained else None
    for artifact in (result, sidecar):
        if len(canonical_json(artifact).encode("utf-8")) + 1 > MAX_ARTIFACT_BYTES:
            raise ValueError("Inspection artifact exceeds the size bound")
    verify_structure(structural, profile, bundle)
    verify_profile_resources(profile, root)
    # RunStore.complete compares the entire expected envelope on replay, including
    # canonical/sidecar/raw bytes. Rehashed but semantically forged caches still fail.
    store.complete(result, canonical=final_snapshot.canonical if final_snapshot else None,
                   sidecar=sidecar, raw_response=raw_output, retain_raw=config.raw_retention == "retain")
    return {"run_id": run_id, "execution": "cached_replay" if cached else "synthetic_inspection",
            "new_mock_requests": new_mock_requests, "new_live_requests": 0,
            "elapsed_ms": (time.perf_counter() - started) * 1000, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        report = inspect_record(args.root, load_config(args.root, args.config), run_id=args.run_id)
    except (ValueError, OSError, KeyError, TypeError, RunIncompleteError) as error:
        parser.exit(2, f"Record inspection failed ({type(error).__name__}); no successful inspection report.\n")
    result = report["result"]
    print(json.dumps({"run_id": report["run_id"], "execution": report["execution"],
                      "new_mock_requests": report["new_mock_requests"], "new_live_requests": 0,
                      "structure": result["structure"]["status"], "patch": result["patch_state"],
                      "coverage_status": "unsupported"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
