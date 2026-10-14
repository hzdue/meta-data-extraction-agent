"""Offline evaluator fixtures never use the network or released gold."""
from pathlib import Path
import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline evaluation tests")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src/meta_data_extraction").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    (tmp_path / "tests/offline/fixtures").mkdir(parents=True)
    return tmp_path
