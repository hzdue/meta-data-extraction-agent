"""Synthetic workspace fixtures; agent tests must never access the network."""
from pathlib import Path
import os
import shutil
import socket
import subprocess

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Network access is forbidden in agent tests")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


@pytest.fixture
def link_directory():
    def create(alias: Path, target: Path):
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
                check=True, capture_output=True,
            )
            assert alias.is_junction()
        else:
            alias.symlink_to(target, target_is_directory=True)
    return create


@pytest.fixture
def workspace(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    (tmp_path / "src/meta_data_extraction").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    for relative in (
        "configs/agent/offline.yaml", "tests/agent/fixtures/synthetic_source.md",
        "schemas/generic.json", "schemas/photon_and_neutron.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
    return tmp_path
