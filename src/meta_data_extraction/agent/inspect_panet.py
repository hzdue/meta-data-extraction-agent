"""Offline PaNET proposal inspection over explicit source and resource snapshots."""
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
from .contracts import FixedExtraction, RecordSnapshot, digest
from .evidence import DocumentBundle, EvidenceRef, FrozenModel, sha256
from .panet_lookup import lexical_candidates
from .panet_mapping import MappingDecision, TechniqueMention, propose_mapping, verify_mapping
from .panet_resources import ResourceSpec, load_panet, verify_resources
from .paths import input_file, validate_run_id, workspace_root
from .sources import prepare_sources
from .storage import RunIncompleteError, RunStore


class MentionRequest(FrozenModel):
    original_expression: str = Field(min_length=1, max_length=4096)
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    context_start: int = Field(ge=0, strict=True)
    context_end: int = Field(gt=0, strict=True)
    context_use: Literal["current_study", "related_work", "uncertain"]
    assertion_path: str | None = None
    selected_id: str | None = None

    @model_validator(mode="after")
    def _range(self) -> MentionRequest:
        if not self.context_start <= self.start < self.end <= self.context_end:
            raise ValueError("Mention must be contained in the declared context")
        return self


class PaNETRunConfig(FrozenModel):
    mode: Literal["offline"] = "offline"
    run_id: str
    resource_spec: str
    source: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    families: tuple[str, ...] = Field(min_length=1, max_length=2)
    candidate_limit: int = Field(default=5, ge=1, le=50, strict=True)
    mentions: tuple[MentionRequest, ...] = Field(min_length=1, max_length=50)
    record_sidecar: str | None = None
    record_sidecar_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _scope(self) -> PaNETRunConfig:
        validate_run_id(self.run_id)
        if len(set(self.families)) != len(self.families):
            raise ValueError("Technique families must be distinct")
        if (self.record_sidecar is None) != (self.record_sidecar_sha256 is None):
            raise ValueError("A record sidecar requires its exact content hash")
        if any(m.assertion_path is not None for m in self.mentions) and self.record_sidecar is None:
            raise ValueError("Assertion-bound mentions require a supplied record")
        if self.record_sidecar is not None and any(m.assertion_path is None for m in self.mentions):
            raise ValueError("Each mention in a record-bound run needs an explicit assertion path")
        return self


def _mapping(root: Path, relative: str, *, roots: tuple[str, ...]) -> dict:
    path = input_file(root, relative, allowed_roots=roots)
    return load_yaml_mapping(path.read_text(encoding="utf-8"), source=relative)


def load_config(root: Path | str, relative: str) -> PaNETRunConfig:
    root = workspace_root(root)
    return PaNETRunConfig.model_validate(_mapping(root, relative, roots=("configs/agent",)))


def _ref(bundle: DocumentBundle, start: int, end: int) -> EvidenceRef:
    regions = [region for region in bundle.regions if region.start <= start < end <= region.end]
    if not regions:
        raise ValueError("The declared span must resolve inside an actual source region")
    region = min(regions, key=lambda r: (r.end - r.start, r.region_id))
    return EvidenceRef(source_id=bundle.source_id, source_sha256=bundle.source_sha256,
                       representation_sha256=bundle.representation_sha256, region_id=region.region_id,
                       start=start, end=end, quotation=bundle.text[start:end])


def _implementation(root: Path) -> dict[str, str]:
    names = ("config", "paths", "evidence", "sources", "contracts", "profiles", "storage",
             "panet_resources", "panet_lookup", "panet_mapping", "inspect_panet")
    paths = [f"agent/{name}.py" for name in names]
    paths.extend(("config_validation.py", "data_models.py", "utils.py", "components/parser.py"))
    return {relative: sha256(input_file(root, f"src/meta_data_extraction/{relative}",
                                        allowed_roots=("src/meta_data_extraction",)).read_bytes())
            for relative in paths}


def inspect_panet(root: Path | str, config: PaNETRunConfig, *, run_id: str | None = None) -> dict:
    """Create only unapproved mapping sidecars, or verify a completed local replay."""
    started = time.perf_counter()
    root = workspace_root(root)
    config = PaNETRunConfig.model_validate(config.model_dump())
    run_id = validate_run_id(run_id or config.run_id)
    spec = ResourceSpec.model_validate(_mapping(root, config.resource_spec, roots=("configs/agent",)))
    if spec.kind != "authentic":
        raise ValueError("Inspection runs require authentic PaNET; synthetic ontologies are test-only")
    snapshot = load_panet(root, spec)
    prepared = prepare_sources(root, [config.source])[0]
    if prepared.bundle is None:
        raise ValueError("Source preparation failed: " + prepared.error.code)
    bundle = prepared.bundle
    if bundle.source_sha256 != config.source_sha256:
        raise ValueError("Source content differs from the declared mention scope")
    record: RecordSnapshot | None = None
    if config.record_sidecar:
        path = input_file(root, config.record_sidecar, allowed_roots=("runs/agent", "tests/agent/fixtures"))
        raw_sidecar = path.read_bytes()
        if sha256(raw_sidecar) != config.record_sidecar_sha256:
            raise ValueError("Record sidecar changed")
        sidecar = load_yaml_mapping(raw_sidecar.decode("utf-8"), source=config.record_sidecar)
        extraction = FixedExtraction.model_validate(sidecar["extraction"])
        if extraction.snapshot is None or extraction.status == "failed":
            raise ValueError("A failed extraction cannot supply a mapping target")
        if DocumentBundle.model_validate(sidecar["source"]) != bundle:
            raise ValueError("Record sidecar belongs to a different prepared source")
        record = extraction.snapshot
    mentions = tuple(TechniqueMention(
        original_expression=m.original_expression, mention=_ref(bundle, m.start, m.end),
        context=_ref(bundle, m.context_start, m.context_end), context_use=m.context_use,
    ) for m in config.mentions)
    request = {"version": "m04-run-1", "config": config.model_dump(mode="json"),
               "source_snapshot_sha256": digest(bundle.model_dump(mode="json")),
               "resource": spec.model_dump(mode="json"), "snapshot_id": snapshot.snapshot_id,
               "record_revision": record.revision_id if record else None,
               "implementation": _implementation(root), "python": platform.python_version(),
               "dependencies": {name: version(name) for name in ("pydantic", "PyYAML", "pymupdf4llm", "pymupdf")},
               "model_calls": 0}
    store = RunStore.begin(root, run_id, request)

    def result_for(decisions: tuple[MappingDecision, ...]) -> dict:
        return {"version": "m04-result-1", "snapshot_id": snapshot.snapshot_id,
                "resource_kind": spec.kind, "model_calls": 0,
                "source_incomplete": prepared.status == "partial",
                "decisions": [decision.model_dump(mode="json") for decision in decisions]}

    def sidecar_for(decisions: tuple[MappingDecision, ...]) -> dict:
        return {"source": bundle.model_dump(mode="json"), "resource": snapshot.model_dump(mode="json"),
                "record": record.model_dump(mode="json") if record else None,
                "decisions": [decision.model_dump(mode="json") for decision in decisions]}

    if store.completed is not None:
        decisions = tuple(MappingDecision.model_validate(item) for item in store.completed["decisions"])
        if len(decisions) != len(mentions):
            raise ValueError("Cached mapping count differs from requested mentions")
        for decision, mention, request_mention in zip(decisions, mentions, config.mentions, strict=True):
            if decision.mention != mention or decision.selected_id != request_mention.selected_id:
                raise ValueError("Cached mapping differs from the requested source mention")
            expected = propose_mapping(snapshot, lexical_candidates(snapshot, mention.original_expression,
                families=config.families, limit=config.candidate_limit), bundle, mention,
                record=record, assertion_path=request_mention.assertion_path, selected_id=request_mention.selected_id)
            verify_mapping(snapshot, bundle, decision, record=record)
            if decision != expected:
                raise ValueError("Cached mapping differs from the current frozen policy")
        if store.completed != result_for(decisions) or _mapping(root,
                f"runs/agent/{run_id}/sidecar.json", roots=(f"runs/agent/{run_id}",)) != sidecar_for(decisions):
            raise ValueError("Cached report metadata or sidecar differs from its verified inputs")
        verify_resources(snapshot, root)
        return {"execution": "cached_replay", "run_id": run_id, "model_calls": 0,
                "elapsed_ms": (time.perf_counter() - started) * 1000, "result": store.completed}
    decisions = tuple(propose_mapping(snapshot,
        lexical_candidates(snapshot, mention.original_expression, families=config.families, limit=config.candidate_limit),
        bundle, mention, record=record, assertion_path=request_mention.assertion_path,
        selected_id=request_mention.selected_id,
    ) for mention, request_mention in zip(mentions, config.mentions, strict=True))
    verify_resources(snapshot, root)
    for decision in decisions:
        verify_mapping(snapshot, bundle, decision, record=record)
    result = result_for(decisions)
    store.complete(result, canonical=None, sidecar=sidecar_for(decisions), raw_response=None, retain_raw=False)
    return {"execution": "lexical_lookup", "run_id": run_id, "model_calls": 0,
            "elapsed_ms": (time.perf_counter() - started) * 1000, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        report = inspect_panet(args.root, load_config(args.root, args.config), run_id=args.run_id)
    except (ValueError, OSError, KeyError, TypeError, RunIncompleteError) as error:
        # Exception values can contain supplied document/config text; log only the type.
        parser.exit(2, f"PaNET inspection failed ({type(error).__name__}); no successful mapping report.\n")
    print(json.dumps({"run_id": report["run_id"], "execution": report["execution"], "model_calls": 0,
                      "decisions": [{key: decision[key] for key in ("status", "reason", "proposed_id")}
                                    for decision in report["result"]["decisions"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
