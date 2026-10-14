"""Local mock-only runtime packaging; every subprocess explicitly denies egress."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from meta_data_extraction.agent.pilot_bundle import (
    BUNDLE_FILES, CONFIG, MARKER, BundleError, build_bundle, verify_bundle,
)

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def bundle_source(tmp_path):
    root = tmp_path / "source"
    for name in BUNDLE_FILES:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / name, destination)
    return root


@pytest.fixture
def bundle(bundle_source):
    return Path(build_bundle(bundle_source, "m10-package-test")["runtime_root"])


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_exact_allowlist_and_preserved_license(bundle):
    assert set(hashes(bundle)) == set(BUNDLE_FILES.values()) | {MARKER}
    info = verify_bundle(bundle)
    assert info["verified"] and info["file_count"] == len(BUNDLE_FILES)
    assert info["new_live_requests"] == 0
    assert info["license_status"] == "unresolved_empty"
    assert (bundle / "LICENSE").read_bytes() == (REPOSITORY / "LICENSE").read_bytes()
    for path in hashes(bundle):
        assert not any(part in {"offline", "evaluation", "alignment", "gold", "outputs", "paper", ".env"}
                       for part in Path(path).parts)
        assert not (path.startswith("tests/") and path.endswith(".py"))


def test_verification_is_read_only_and_run_files_are_mutable(bundle):
    mutable = bundle / "runs/agent/m10-pilot-test"
    mutable.mkdir(parents=True)
    (mutable / "result.json").write_text('{"status":"pending"}')
    before = hashes(bundle)
    assert verify_bundle(bundle) == verify_bundle(bundle)
    assert hashes(bundle) == before


def test_build_refuses_reuse_without_overwriting(bundle_source, bundle):
    before = hashes(bundle)
    with pytest.raises(FileExistsError):
        build_bundle(bundle_source, "m10-package-test")
    assert hashes(bundle) == before


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "C:\\escape", "CON", "", "a" * 81])
def test_invalid_build_locations(bundle_source, run_id):
    with pytest.raises(ValueError):
        build_bundle(bundle_source, run_id)
    assert not (bundle_source / "runs").exists()


@pytest.mark.parametrize("name", ["LICENSE", "schemas/generic.json", "configs/agent/workflow_review.json",
                                  "src/meta_data_extraction/agent/review.py", "uv.lock", MARKER])
def test_pin_changes_are_rejected(bundle, name):
    path = bundle / name
    path.write_bytes(path.read_bytes() + b" ")
    if name == MARKER:  # Whitespace is harmless JSON; corrupt a semantic field too.
        data = json.loads(path.read_bytes())
        data["new_live_requests"] = 1
        write_json(path, data)
    with pytest.raises(ValueError):
        verify_bundle(bundle)


@pytest.mark.parametrize("name", [".env", "data/gold/example.json", "outputs/result.json",
                                  "src/meta_data_extraction/offline/scoring.py", "configs/alignment_helper.yaml",
                                  "tests/test_hidden.py", "runs/other/test.json", ".venv/pyvenv.cfg"])
def test_unlisted_files_rejected(bundle, name):
    path = bundle / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("UNREAD_PRIVATE_VALUE")
    with pytest.raises(ValueError) as error:
        verify_bundle(bundle)
    assert "UNREAD_PRIVATE_VALUE" not in str(error.value)


def test_unlisted_empty_directory_rejected(bundle):
    (bundle / "data").mkdir()
    with pytest.raises(BundleError, match="Unexpected"):
        verify_bundle(bundle)


@pytest.mark.parametrize("name", [".env", ".env.local", ".env.private", "credentials", "secret.pem", "secret.key"])
def test_credentials_rejected_even_in_mutable_runs(bundle, name):
    path = bundle / "runs/agent/m10-run" / name
    path.parent.mkdir(parents=True)
    path.write_text("NOT_A_REAL_KEY")
    with pytest.raises(BundleError, match="Credential"):
        verify_bundle(bundle)


@pytest.mark.parametrize("source", [".env", "data/gold/record.json", "configs/alignment_helper.yaml",
                                    "src/meta_data_extraction/offline/scoring.py", "../escape"])
def test_configuration_cannot_expand_allowlist(bundle_source, source):
    path = bundle_source / CONFIG
    data = json.loads(path.read_bytes())
    data["files"][source] = source
    write_json(path, data)
    with pytest.raises(BundleError, match="allowlist"):
        build_bundle(bundle_source, "m10-invalid")
    assert not (bundle_source / "runs").exists()


def test_missing_asset_is_preflight_failure(bundle_source):
    # Only a synthetic temporary copy is removed; no repository asset is touched.
    (bundle_source / "schemas/generic.json").unlink()
    with pytest.raises(ValueError):
        build_bundle(bundle_source, "m10-incomplete")
    assert not (bundle_source / "runs").exists()


def test_hard_linked_input_rejected(bundle_source, tmp_path):
    path = bundle_source / "README.md"
    os.link(path, tmp_path / "readme-alias")
    with pytest.raises(ValueError):
        build_bundle(bundle_source, "m10-hardlink")


def test_linked_bundle_root_rejected(bundle, tmp_path, link_directory):
    alias = tmp_path / "runtime-alias"
    link_directory(alias, bundle)
    with pytest.raises(BundleError, match="Linked"):
        verify_bundle(alias)


def test_linked_mutable_directory_rejected(bundle, tmp_path, link_directory):
    outer = tmp_path / "outer"
    outer.mkdir()
    alias = bundle / "runs/agent/linked"
    alias.parent.mkdir(parents=True)
    link_directory(alias, outer)
    with pytest.raises(ValueError):
        verify_bundle(bundle)


def test_metadata_duplicate_and_nonfinite_rejected(bundle):
    marker = bundle / MARKER
    marker.write_text('{"version":"a","version":"b"}')
    with pytest.raises(BundleError, match="Duplicate"):
        verify_bundle(bundle)
    marker.write_text('{"value":NaN}')
    with pytest.raises(BundleError, match="Nonfinite"):
        verify_bundle(bundle)


def test_clean_subprocess_runtime_imports_and_mock_recovery(bundle):
    # -I ignores checkout PYTHONPATH. Explicitly deny all socket dispatch before
    # imports; verify every loaded project module resolves within the bundle.
    code = r'''
import importlib.util, json, pathlib, socket, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))
def denied(*args, **kwargs): raise AssertionError("Unexpected network access")
socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
socket.getaddrinfo = denied
from meta_data_extraction.agent.pilot_bundle import verify_bundle
from meta_data_extraction.agent.inspect_workflow import inspect_workflow, load_config
from meta_data_extraction.agent.review_store import ReviewStore
import meta_data_extraction.agent.pilot_core
import meta_data_extraction.agent.pilot_http
assert importlib.util.find_spec("meta_data_extraction.offline") is None
assert importlib.util.find_spec("meta_data_extraction.evaluation") is None
info = verify_bundle(root)
config = load_config(root, "configs/agent/workflow_review.json")
first = inspect_workflow(root, config, run_id="m10-relocated-test")
assert first["state"].status == "pending_review"
assert first["new_live_requests"] == 0
replay = inspect_workflow(root, config, run_id="m10-relocated-test")
assert replay["new_model_calls"] == 0 and replay["new_live_requests"] == 0
with ReviewStore.open(root, "m10-relocated-test") as store:
    assert store.state.cases
verify_bundle(root)
for name, module in tuple(sys.modules.items()):
    if name.startswith("meta_data_extraction") and getattr(module, "__file__", None):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(root)
print(json.dumps({"status":first["state"].status,"new_live_requests":0,"file_count":info["file_count"]}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(bundle)],
                            cwd=bundle, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "pending_review"


def test_docker_context_allowlist_matches_bundle():
    lines = (REPOSITORY / "deploy/agent/.dockerignore").read_text().splitlines()
    files = {line[1:] for line in lines if line.startswith("!") and not line.endswith("/")}
    assert files == set(BUNDLE_FILES.values()) | {MARKER}
    assert "**" in lines
    docker = (REPOSITORY / "deploy/agent/Dockerfile").read_text()
    assert "USER 10001:10001" in docker
    assert "PYTHONDONTWRITEBYTECODE=1" in docker
    assert '"127.0.0.1"' in docker and '"0.0.0.0"' not in docker
