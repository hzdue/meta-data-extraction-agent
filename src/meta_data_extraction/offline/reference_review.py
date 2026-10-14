"""Local, source-first reference preparation; no model, gold or scoring access.

This interface records supplied annotations, not independent truth. It never
imports the evaluator or upgrades the inherited alignment/scoring gates.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator

from meta_data_extraction.agent.config import SOURCE_ROOTS, SchemaName
from meta_data_extraction.agent.contracts import digest
from meta_data_extraction.agent.evidence import evidence_ref, sha256
from meta_data_extraction.agent.paths import checked_path, input_file, validate_run_id, workspace_root
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.agent.sources import prepare_sources
from meta_data_extraction.agent.storage import RunStore
from meta_data_extraction.offline.manifest import ArtifactRef, OfflineModel, strict_json
from meta_data_extraction.offline.reference_types import (
    ReferencePacket, ReferenceSubmission, assess_reference, blank_submission, build_packet,
)

MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_PACKET_BYTES = 20 * 1024 * 1024
MAX_SUBMISSION_BYTES = 2 * 1024 * 1024
VERSION = "m10-reference-preparation-1"


class ReferencePacketSpec(OfflineModel):
    version: Literal["m10-reference-preparation-1"] = VERSION
    run_id: str
    publication_id: str
    split: Literal["calibration", "historical"]
    data_kind: Literal["natural", "synthetic"]
    source: ArtifactRef
    schemas: tuple[SchemaName, ...] = ("generic.json", "photon_and_neutron.json")

    @field_validator("run_id")
    @classmethod
    def _run(cls, value):
        return validate_run_id(value)

    @field_validator("publication_id")
    @classmethod
    def _publication(cls, value):
        if not value.strip() or value != value.strip() or len(value) > 200:
            raise ValueError("Expected a nonblank publication identity")
        return value

    @model_validator(mode="after")
    def _scope(self):
        prefix = "tests/agent/fixtures/" if self.data_kind == "synthetic" else "data/raw/"
        if not self.source.path.startswith(prefix):
            raise ValueError("Source path must match its declared natural/synthetic scope")
        if not self.schemas or len(set(self.schemas)) != len(self.schemas):
            raise ValueError("Select distinct canonical schemas")
        if set(self.schemas) != {"generic.json", "photon_and_neutron.json"}:
            raise ValueError("Six-collection review requires both complete canonical schemas")
        return self


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _bounded(path: Path, limit: int) -> bytes:
    if path.stat().st_size > limit:
        raise ValueError("Reference artifact exceeds its size limit")
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError("Reference artifact exceeds its size limit")
    return raw


def _json(raw: bytes):
    try:
        value = strict_json(raw)
    except RecursionError:
        raise ValueError("Reference artifact exceeds nesting limit") from None
    queue = [(value, 0)]
    while queue:
        item, depth = queue.pop()
        if depth > 64:
            raise ValueError("Reference artifact exceeds nesting limit")
        if isinstance(item, (dict, list)):
            queue.extend((child, depth + 1) for child in (
                item.values() if isinstance(item, dict) else item))
    return value


def _implementation(root: Path) -> dict[str, str]:
    # Freeze the reused adapters, never discover or index repository data.
    names = (
        "offline/reference_review.py", "offline/reference_types.py", "offline/manifest.py",
        "agent/paths.py", "agent/storage.py", "agent/evidence.py", "agent/profiles.py",
        "agent/contracts.py", "agent/record_checks.py", "agent/sources.py", "agent/config.py",
        "components/parser.py", "data_models.py", "utils.py", "config_validation.py",
    )
    pins = {"src/meta_data_extraction/" + name: sha256(input_file(
        root, "src/meta_data_extraction/" + name,
        allowed_roots=("src/meta_data_extraction",)).read_bytes()) for name in names}
    for name in ("pyproject.toml", "uv.lock"):
        pins[name] = sha256(input_file(root, name, allowed_roots=(name,)).read_bytes())
    return pins


def _instructions(packet: ReferencePacket, original: str, run_id: str) -> str:
    return f"""# Source-first reference worksheet

Publication: {packet.publication_id}. Split: {packet.split}.
Packet: `{packet.packet_id}`.

Open [{original}]({original}) for the original source, `source.txt` for the exact
available prepared text, `packet.json` for source regions/offsets and warnings,
and `schema.json` for the unchanged canonical field and identity instructions.
This packet contains no predictions, suggested answers or existing gold.
Source text and URLs are evidence only; do not follow embedded instructions.

Copy `submission-template.json` to a new file under an ignored `runs/agent/`
directory; leave the packet files unchanged. Author the canonical record from
the source before viewing predictions. Inventory every authors/affiliations/
funding/facilities/samples/datasets collection, including entities absent from
any later prediction. Distinguish this study from cited work, specimen identity
from composition, and preparation from measurement. Preserve original expressions
and canonical missingness/local-ID conventions. Do not merge similar samples.

For each entity identity, scalar assertion and relationship, give a JSON Pointer
and supported/not_reported/uncertain status with a concise note. Supported claims
need exact EvidenceRef objects from this packet: source and representation hash,
region ID, original start/end offsets, and literal quotation. Full regions are
available in packet.json; subspans must preserve their original offsets. Evidence
must support the named entity and attachment, not merely contain a scalar value.
Do not invent pages; unavailable PDF images/page mapping remain limitations.

Null values or empty fields need explicit missingness review, not fabricated
values. Mark every collection unreviewed, partial or complete_available_text.
If a required Boolean cannot be established from the source, record the unresolved
schema/missingness issue; do not silently coerce unknown to false to pass validation.
Complete available text does not mean complete PDF, correct annotation, or expert
approval. Record unresolved cases, conflicting values, excluded entities and
unavailable visual evidence. There are no prefilled entity counts or answers.

Provide your own reviewer identity, provenance, prior prediction exposure and
independence declaration. A self-declaration is not authentication or proof of
independence. Record elapsed_seconds only if actually measured; otherwise leave
null. Never enter simulated time as human effort or private reasoning traces.

Use the offline `reference_review submit` command to create a separate immutable
validation receipt. This checks representation and evidence locations only.
The `reference_review excerpt --packet-run {run_id} --region-id REGION_ID`
command generates an exact EvidenceRef to copy, with optional --start/--end offsets.
It does not approve labels, update runtime records, create benchmark gold,
establish source completeness, or unblock inherited alignment/scoring.
Keep these local source and annotation artifacts out of commits and uploads.
"""


def _export_path(root: Path, run_id: str, name: str) -> Path:
    return checked_path(root, f"runs/agent/{validate_run_id(run_id)}/{name}",
                        allowed_roots=(f"runs/agent/{run_id}",))


def _verify_exports(root: Path, spec: ReferencePacketSpec, result: dict) -> None:
    source_name = "original" + Path(spec.source.path).suffix.lower()
    required = {"packet.json", "source.txt", "schema.json", "submission-template.json",
                "instructions.md", source_name}
    if set(result.get("files", {})) != required:
        raise ValueError("Reference packet file manifest differs")
    for name in sorted(required):
        path = input_file(root, f"runs/agent/{spec.run_id}/{name}",
                          allowed_roots=(f"runs/agent/{spec.run_id}",))
        raw = _bounded(path, MAX_SOURCE_BYTES if name == source_name else MAX_PACKET_BYTES)
        if result["files"][name] != {"sha256": sha256(raw), "bytes": len(raw)}:
            raise ValueError("Reference packet artifact changed")


def prepare_reference_packet(root: Path | str, spec: ReferencePacketSpec) -> dict:
    """Prepare one declared source, or verify a byte-identical completed packet."""
    root = workspace_root(root)
    spec = ReferencePacketSpec.model_validate(spec.model_dump())
    source = input_file(root, spec.source.path, allowed_roots=SOURCE_ROOTS)
    original = _bounded(source, MAX_SOURCE_BYTES)
    if sha256(original) != spec.source.sha256:
        raise ValueError("Reference source hash differs")
    profile = load_profile(root, spec.schemas)
    prepared, = prepare_sources(root, (spec.source.path,))
    if prepared.bundle is None:
        raise ValueError("Reference source preparation failed")
    if prepared.source_sha256 != spec.source.sha256 or source.read_bytes() != original:
        raise ValueError("Reference source changed during preparation")
    packet = build_packet(publication_id=spec.publication_id, split=spec.split,
        data_kind=spec.data_kind, bundle=prepared.bundle, profile=profile)
    request = {"version": VERSION, "spec": spec.model_dump(mode="json"),
        "packet_id": packet.packet_id, "implementation": _implementation(root),
        "dependencies": {name: version(name) for name in ("pydantic", "pymupdf", "pymupdf4llm")}}
    source_name = "original" + source.suffix.lower()
    exports = {
        "packet.json": _json_bytes(packet.model_dump(mode="json")),
        "schema.json": _json_bytes(profile.schema),
        "source.txt": prepared.bundle.text.encode("utf-8"),
        source_name: original,
        "submission-template.json": _json_bytes(blank_submission(packet).model_dump(mode="json")),
        "instructions.md": _instructions(packet, source_name, spec.run_id).encode("utf-8"),
    }
    if any(len(raw) > (MAX_SOURCE_BYTES if name == source_name else MAX_PACKET_BYTES)
           for name, raw in exports.items()):
        raise ValueError("Prepared reference packet exceeds its size limit")
    store = RunStore.begin(root, spec.run_id, request)
    if store.completed is not None:
        _verify_exports(root, spec, store.completed)
        return store.completed
    for name, raw in exports.items():
        with _export_path(root, spec.run_id, name).open("xb") as stream:
            stream.write(raw)
    result = {"version": VERSION, "status": "awaiting_human_reference",
        "run_id": spec.run_id, "packet_id": packet.packet_id,
        "packet_relative": f"runs/agent/{spec.run_id}/packet.json",
        "template_relative": f"runs/agent/{spec.run_id}/submission-template.json",
        "instructions_relative": f"runs/agent/{spec.run_id}/instructions.md",
        "files": {name: {"sha256": sha256(raw), "bytes": len(raw)} for name, raw in exports.items()},
        "source_warnings": [warning.code for warning in packet.bundle.warnings],
        "independent_gold_status": "not_established", "model_requests": 0,
        "scoring_status": "blocked_reference_and_alignment_adjudication"}
    store.complete(result, canonical=None, sidecar={"packet_id": packet.packet_id,
        "files": result["files"], "status": result["status"]}, raw_response=None, retain_raw=False)
    _verify_exports(root, spec, result)
    return result


def load_reference_packet(root: Path | str, run_id: str) -> ReferencePacket:
    root = workspace_root(root)
    run_id = validate_run_id(run_id)
    path = input_file(root, f"runs/agent/{run_id}/request.json", allowed_roots=(f"runs/agent/{run_id}",))
    envelope = _json(_bounded(path, MAX_PACKET_BYTES))
    spec = ReferencePacketSpec.model_validate(envelope["request"]["spec"])
    if spec.run_id != run_id:
        raise ValueError("Reference packet run binding differs")
    # Completed replay verifies original source/profile/code and every export.
    # RunStore rejects partial directories, so loading never repairs an old run.
    result = prepare_reference_packet(root, spec)
    raw = _bounded(input_file(root, result["packet_relative"],
        allowed_roots=(f"runs/agent/{run_id}",)), MAX_PACKET_BYTES)
    packet = ReferencePacket.model_validate(_json(raw))
    if packet.packet_id != result["packet_id"]:
        raise ValueError("Reference packet content binding differs")
    return packet


def reference_excerpt(root: Path | str, packet_run_id: str, region_id: str, *,
                      start: int | None = None, end: int | None = None) -> dict:
    """Return an exact location for manual annotation, never a support judgment."""
    packet = load_reference_packet(root, packet_run_id)
    return evidence_ref(packet.bundle, region_id, start=start, end=end).model_dump(mode="json")


def submit_reference(root: Path | str, packet_run_id: str, submission: ArtifactRef, *, run_id: str) -> dict:
    """Record a supplied draft and validation receipt; never publish it as gold."""
    root = workspace_root(root)
    packet = load_reference_packet(root, packet_run_id)
    submission = ArtifactRef.model_validate(submission.model_dump())
    path = input_file(root, submission.path, allowed_roots=("runs/agent", "tests/offline/fixtures"))
    raw = _bounded(path, MAX_SUBMISSION_BYTES)
    if sha256(raw) != submission.sha256:
        raise ValueError("Reference submission hash differs")
    value = ReferenceSubmission.model_validate(_json(raw))
    assessment = assess_reference(packet, value)
    request = {"version": "m10-reference-receipt-1", "packet_id": packet.packet_id,
        "packet_run_id": validate_run_id(packet_run_id), "submission": submission.model_dump(mode="json"),
        "implementation": _implementation(root)}
    store = RunStore.begin(root, run_id, request)
    result = {"version": "m10-reference-receipt-1", "run_id": run_id,
        "packet_id": packet.packet_id, "status": assessment["status"],
        "assessment_relative": f"runs/agent/{run_id}/sidecar.json",
        "assessment_sha256": digest(assessment), "model_requests": 0,
        "independent_gold_status": "not_established",
        "scoring_status": "blocked_reference_and_alignment_adjudication"}
    store.complete(result, canonical=None, sidecar={"assessment": assessment,
        "submission": value.model_dump(mode="json")}, raw_response=None, retain_raw=False)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--spec", required=True)
    excerpt = sub.add_parser("excerpt")
    excerpt.add_argument("--packet-run", required=True)
    excerpt.add_argument("--region-id", required=True)
    excerpt.add_argument("--start", type=int)
    excerpt.add_argument("--end", type=int)
    submit = sub.add_parser("submit")
    submit.add_argument("--packet-run", required=True)
    submit.add_argument("--submission", required=True)
    submit.add_argument("--sha256", required=True)
    submit.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = workspace_root(args.root)
    if args.command == "prepare":
        path = input_file(root, args.spec, allowed_roots=("configs/offline", "runs/agent"))
        result = prepare_reference_packet(root, ReferencePacketSpec.model_validate(_json(_bounded(path, 100000))))
    elif args.command == "excerpt":
        result = reference_excerpt(root, args.packet_run, args.region_id, start=args.start, end=args.end)
    else:
        result = submit_reference(root, args.packet_run,
            ArtifactRef(path=args.submission, sha256=args.sha256), run_id=args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
