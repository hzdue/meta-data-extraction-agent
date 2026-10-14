import json
import os

import pytest

from meta_data_extraction.agent.paths import PathPolicyError, RunDirectory


@pytest.mark.parametrize("run_id", [
    "", ".", "..", "../outputs", "../../paper", "data/gold", "data/annotations",
    "data/experiment",
    "/absolute", "C:\\outputs", "C:outputs", "\\\\server\\share", "a/b", "a\\b",
    "name:stream", "NUL", "con", "LPT1", "a.", "a ", "-a", "x" * 81,
])
def test_invalid_run_ids_cannot_write(workspace, run_id):
    with pytest.raises(PathPolicyError):
        RunDirectory.create(workspace, run_id)
    assert not (workspace / "runs").exists()


def test_run_named_outputs_stays_inside_runs(workspace):
    assert RunDirectory.create(workspace, "outputs").path == workspace / "runs/agent/outputs"
    assert not (workspace / "outputs").exists()


def test_fresh_run_and_exclusive_artifact(workspace):
    run = RunDirectory.create(workspace, "synthetic-01")
    path = run.write_json("check.json", {"synthetic": True})
    assert json.loads(path.read_text()) == {"synthetic": True}
    with pytest.raises(FileExistsError):
        RunDirectory.create(workspace, "synthetic-01")
    with pytest.raises(FileExistsError):
        run.write_json("check.json", {"synthetic": False})
    assert json.loads(path.read_text()) == {"synthetic": True}


@pytest.mark.parametrize("filename", [
    "../outputs/result.json", "/result.json", "C:\\result.json", "a/b.json",
    "a\\b.json", "a.json:stream", "CON.json", "result.txt", "result.json.",
])
def test_artifact_paths_cannot_escape(workspace, filename):
    run = RunDirectory.create(workspace, "safe")
    with pytest.raises(PathPolicyError):
        run.write_json(filename, {})
    assert list(run.path.iterdir()) == []


def test_bad_payload_does_not_create_partial_artifact(workspace):
    run = RunDirectory.create(workspace, "safe")
    with pytest.raises(ValueError):
        run.write_json("bad.json", {"number": float("nan")})
    assert list(run.path.iterdir()) == []


@pytest.mark.parametrize("relative", ["runs", "runs/agent", "runs/agent/linked"])
def test_linked_output_ancestor_is_rejected(workspace, relative, link_directory):
    release = workspace / "outputs"
    release.mkdir()
    sentinel = release / "sentinel.json"
    sentinel.write_text("released")
    alias = workspace / relative
    alias.parent.mkdir(parents=True, exist_ok=True)
    link_directory(alias, release)
    try:
        with pytest.raises(PathPolicyError):
            RunDirectory.create(workspace, "linked")
        assert sentinel.read_text() == "released"
        assert list(release.iterdir()) == [sentinel]
    finally:
        alias.rmdir() if os.name == "nt" else alias.unlink()


def test_run_is_revalidated_before_writing(workspace, link_directory):
    run = RunDirectory.create(workspace, "swapped")
    path = run.path
    path.rmdir()
    release = workspace / "paper"
    release.mkdir()
    link_directory(path, release)
    try:
        with pytest.raises(PathPolicyError):
            run.write_json("result.json", {})
        assert list(release.iterdir()) == []
    finally:
        path.rmdir() if os.name == "nt" else path.unlink()


def test_hard_linked_artifact_is_rejected(workspace):
    release = workspace / "released.json"
    release.write_text("released")
    run = RunDirectory.create(workspace, "safe")
    os.link(release, run.path / "result.json")
    with pytest.raises(PathPolicyError):
        run.write_json("result.json", {})
    assert release.read_text() == "released"
