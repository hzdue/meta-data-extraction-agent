"""Exact, local-only packaging for the M10 synthetic review pilot.

Verification detects drift and unwanted files, not a hostile owner rewriting code
and its manifest together. Run a container with a read-only root filesystem for
an OS-enforced boundary; the plain local process retains the operator's rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from .paths import RunDirectory, _parts, input_file, workspace_root

MARKER = "pilot_bundle_manifest.json"
CONFIG = "configs/agent/pilot_bundle.json"
VERSION = "m10-runtime-bundle-1"
MAX_FILE_BYTES = 2_000_000

# These are individual reviewed files, never a repository or package-directory
# glob. Parser/provider modules remain unchanged behind the mock-only API.
_IDENTITY_FILES = (
    "pyproject.toml", "uv.lock", "README.md", "LICENSE", CONFIG,
    "src/meta_data_extraction/__init__.py",
    "src/meta_data_extraction/app_config.py",
    "src/meta_data_extraction/config_validation.py",
    "src/meta_data_extraction/data_models.py", "src/meta_data_extraction/utils.py",
    "src/meta_data_extraction/base/__init__.py",
    "src/meta_data_extraction/base/llm.py", "src/meta_data_extraction/base/parser.py",
    "src/meta_data_extraction/components/__init__.py",
    "src/meta_data_extraction/components/llm.py",
    "src/meta_data_extraction/components/parser.py",
    "src/meta_data_extraction/agent/__init__.py",
    "src/meta_data_extraction/agent/config.py",
    "src/meta_data_extraction/agent/contracts.py",
    "src/meta_data_extraction/agent/evidence.py",
    "src/meta_data_extraction/agent/inspect_record.py",
    "src/meta_data_extraction/agent/inspect_workflow.py",
    "src/meta_data_extraction/agent/nexus_registry.py",
    "src/meta_data_extraction/agent/nexus_resources.py",
    "src/meta_data_extraction/agent/paths.py",
    "src/meta_data_extraction/agent/pilot_bundle.py",
    "src/meta_data_extraction/agent/pilot_core.py",
    "src/meta_data_extraction/agent/pilot_http.py",
    "src/meta_data_extraction/agent/pilot_ui.html",
    "src/meta_data_extraction/agent/profiles.py",
    "src/meta_data_extraction/agent/providers.py",
    "src/meta_data_extraction/agent/record_checks.py",
    "src/meta_data_extraction/agent/record_patches.py",
    "src/meta_data_extraction/agent/retrieval.py",
    "src/meta_data_extraction/agent/review.py",
    "src/meta_data_extraction/agent/review_cli.py",
    "src/meta_data_extraction/agent/review_store.py",
    "src/meta_data_extraction/agent/review_types.py",
    "src/meta_data_extraction/agent/sources.py",
    "src/meta_data_extraction/agent/storage.py",
    "src/meta_data_extraction/agent/support_checks.py",
    "src/meta_data_extraction/agent/workflow.py",
    "src/meta_data_extraction/agent/workflow_graph.py",
    "src/meta_data_extraction/agent/workflow_policies.py",
    "src/meta_data_extraction/agent/workflow_tools.py",
    "src/meta_data_extraction/agent/workflow_types.py",
    "schemas/generic.json", "schemas/photon_and_neutron.json",
    "schemas/prompt_template.md",
    "configs/agent/workflow_review.json", "configs/agent/record_inspection.json",
    "tests/agent/fixtures/m06_source.md",
    "tests/agent/fixtures/m06_record.json",
    "tests/agent/fixtures/m06_before_response.json",
    "tests/agent/fixtures/m06_after_response.json",
    "tests/agent/fixtures/m07_checker_before.json",
    "tests/agent/fixtures/m07_checker_after.json",
    "tests/agent/fixtures/m08_policy_retrieve_description.json",
    "tests/agent/fixtures/m08_policy_review_description.json",
    "tests/agent/fixtures/m08_policy_retrieve_attachment.json",
    "tests/agent/fixtures/m08_policy_review_attachment.json",
)
BUNDLE_FILES = dict((name, name) for name in _IDENTITY_FILES) | {
    "deploy/agent/Dockerfile": "Dockerfile",
    "deploy/agent/.dockerignore": ".dockerignore",
    "docs/agent/PILOT.md": "PILOT.md",
}


class BundleError(ValueError):
    """The local runtime bundle is incomplete, changed, or outside its scope."""


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(raw: bytes) -> dict:
    if len(raw) > MAX_FILE_BYTES:
        raise BundleError("Bundle metadata exceeds its size bound")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise BundleError("Duplicate bundle metadata key")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(BundleError("Nonfinite metadata")))
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise BundleError("Invalid bundle metadata") from error
    if not isinstance(value, dict):
        raise BundleError("Expected bundle metadata object")
    return value


def _root(value: Path | str) -> Path:
    supplied = Path(value).absolute()
    for path in (supplied, *supplied.parents):
        if path.is_symlink() or path.is_junction():
            raise BundleError("Linked bundle roots are not permitted")
    return workspace_root(supplied)


def _read(path: Path) -> bytes:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise BundleError("Bundle inputs must be ordinary unlinked files")
    if info.st_size > MAX_FILE_BYTES:
        raise BundleError("Bundle input exceeds its size bound")
    raw = path.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raise BundleError("Bundle input exceeds its size bound")
    return raw


def _configuration(raw: bytes) -> dict:
    value = _json(raw)
    expected = {"version": VERSION, "scope": "single_user_synthetic_mock_review",
                "files": BUNDLE_FILES, "mutable_root": "runs/agent",
                "live_model_requests": 0, "external_tracing": False,
                "license_policy": "preserve_existing_unresolved_no_redistribution_approval"}
    if (value != expected or type(value.get("live_model_requests")) is not int
            or type(value.get("external_tracing")) is not bool):
        raise BundleError("Bundle configuration differs from the reviewed allowlist")
    return value


def build_bundle(root: Path | str, run_id: str) -> dict:
    """Copy only the reviewed assets into a fresh ignored run directory."""
    root = _root(root)
    raw_files = {}
    for source in BUNDLE_FILES:
        path = input_file(root, source, allowed_roots=(source,))
        raw_files[source] = _read(path)
    _configuration(raw_files[CONFIG])
    # Fail before creating anything if an input changed during preflight.
    for source, raw in raw_files.items():
        if _read(input_file(root, source, allowed_roots=(source,))) != raw:
            raise BundleError("Bundle source changed while preparing the snapshot")
    run = RunDirectory.create(root, run_id)
    runtime = run.path / "runtime"
    runtime.mkdir()
    pins = {}
    for source, target in BUNDLE_FILES.items():
        destination = runtime / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw_files[source])
        pins[target] = {"source": source, "sha256": _digest(raw_files[source]),
                        "size_bytes": len(raw_files[source])}
    manifest = {"version": VERSION, "scope": "single_user_synthetic_mock_review",
                "bundle_id": run_id, "files": pins, "mutable_root": "runs/agent",
                "license_status": "unresolved_empty" if not raw_files["LICENSE"].strip() else "preserved_unreviewed",
                "new_live_requests": 0, "external_tracing": "disabled"}
    # Marker is written last. An interrupted build is never a valid bundle and
    # is never overwritten or automatically rebuilt.
    with (runtime / MARKER).open("xb") as stream:
        stream.write((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    result = verify_bundle(runtime)
    return result | {"runtime_root": str(runtime),
                     "runtime_relative": runtime.relative_to(root).as_posix()}


def verify_bundle(root: Path | str) -> dict:
    """Read-only inventory and hash verification, allowing only runtime runs."""
    root = _root(root)
    marker = input_file(root, MARKER, allowed_roots=(MARKER,))
    raw_manifest = _read(marker)
    manifest = _json(raw_manifest)
    if set(manifest) != {"version", "scope", "bundle_id", "files", "mutable_root",
                         "license_status", "new_live_requests", "external_tracing"}:
        raise BundleError("Unexpected bundle manifest fields")
    if (manifest["version"] != VERSION or manifest["scope"] != "single_user_synthetic_mock_review"
            or manifest["mutable_root"] != "runs/agent" or manifest["new_live_requests"] != 0
            or type(manifest["new_live_requests"]) is not int or manifest["external_tracing"] != "disabled"):
        raise BundleError("Unsupported bundle scope")
    from .paths import validate_run_id
    validate_run_id(manifest["bundle_id"])
    pins = manifest["files"]
    if not isinstance(pins, dict) or set(pins) != set(BUNDLE_FILES.values()):
        raise BundleError("Bundle inventory differs from the reviewed allowlist")
    for source, target in BUNDLE_FILES.items():
        raw = _read(input_file(root, target, allowed_roots=(target,)))
        if (not isinstance(pins[target], dict) or type(pins[target].get("size_bytes")) is not int
                or pins[target] != {"source": source, "sha256": _digest(raw), "size_bytes": len(raw)}):
            raise BundleError("Bundle file differs from its pinned bytes")
    _configuration(_read(root / CONFIG))
    license_status = "unresolved_empty" if not _read(root / "LICENSE").strip() else "preserved_unreviewed"
    if manifest["license_status"] != license_status:
        raise BundleError("Bundle license status differs from its preserved input")
    allowed_files = set(pins) | {MARKER}
    allowed_dirs = {"runs", "runs/agent"}
    for name in allowed_files:
        allowed_dirs.update(parent.as_posix() for parent in Path(name).parents if parent != Path("."))
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            relative = path.relative_to(root).as_posix()
            _parts(relative)
            if path.is_symlink() or path.is_junction():
                raise BundleError("Linked runtime paths are not permitted")
            is_dir = path.is_dir()
            info = path.stat()
            if not is_dir and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
                raise BundleError("Runtime files must be ordinary unlinked files")
            mutable = relative.startswith("runs/agent/")
            lowered = name.casefold()
            if lowered == ".env" or lowered.startswith(".env.") or lowered in {"credentials", "id_rsa", "id_ed25519"} or path.suffix.casefold() in {".pem", ".key"}:
                raise BundleError("Credential files do not belong in the runtime bundle")
            if not mutable and relative not in (allowed_dirs if is_dir else allowed_files):
                raise BundleError("Unexpected file or directory outside runtime runs")
    return {"version": VERSION, "bundle_id": manifest["bundle_id"], "verified": True,
            "file_count": len(pins), "manifest_sha256": _digest(raw_manifest),
            "license_status": license_status, "new_live_requests": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", default=".")
    build.add_argument("--run-id", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", required=True)
    args = parser.parse_args()
    try:
        result = build_bundle(args.root, args.run_id) if args.command == "build" else verify_bundle(args.root)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Pilot bundle operation failed ({type(error).__name__}); no verified runtime.\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
