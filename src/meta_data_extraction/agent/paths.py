"""Local path policy for agent inputs and fresh, non-overwriting run artifacts.

This is an application boundary, not an OS sandbox against concurrent filesystem
replacement or a caller using unrelated filesystem APIs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any


class PathPolicyError(ValueError):
    """A path is outside the agent's permitted local layout."""


def _parts(value: str) -> tuple[str, ...]:
    # Apply portable rules on Windows and POSIX, including Windows ADS/devices.
    parts = tuple(value.replace("\\", "/").split("/"))
    if PureWindowsPath(value).drive or any(
        not part or part in {".", ".."} or part.endswith((".", " "))
        or re.search(r'[<>:"|?*\x00-\x1f]', part)
        or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])", part.split(".")[0])
        for part in parts
    ):
        raise PathPolicyError("Expected a portable relative path without traversal or devices")
    return parts


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise PathPolicyError("run_id must contain 1-80 ASCII letters, digits, '-' or '_'")
    _parts(value)
    return value


def workspace_root(root: Path | str) -> Path:
    root = Path(root).resolve(strict=True)
    if not (root / "pyproject.toml").is_file() or not (
        root / "src" / "meta_data_extraction"
    ).is_dir():
        raise PathPolicyError("Expected the meta_data_extraction checkout root")
    return root


def checked_path(root: Path | str, relative: str, *, allowed_roots: tuple[str, ...]) -> Path:
    """Resolve a relative allowlisted path, refusing links at every component."""
    root = workspace_root(root)
    parts = _parts(relative)
    prefixes = tuple(_parts(prefix) for prefix in allowed_roots)
    if not any(parts[:len(prefix)] == prefix for prefix in prefixes):
        raise PathPolicyError("Path is outside the permitted roots")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink() or path.is_junction():
            raise PathPolicyError("Symlinks and junctions are not permitted")
    if not path.resolve().is_relative_to(root):
        raise PathPolicyError("Path escapes the checkout")
    if path.is_file() and path.stat().st_nlink > 1:
        raise PathPolicyError("Hard-linked files are not permitted")
    return path


def input_file(root: Path | str, relative: str, *, allowed_roots: tuple[str, ...]) -> Path:
    path = checked_path(root, relative, allowed_roots=allowed_roots)
    if not path.is_file():
        raise PathPolicyError("Expected an existing individual input file")
    return path


@dataclass(frozen=True)
class RunDirectory:
    """Only creates fresh runs/agent/<run_id> directories; never resumes implicitly."""

    root: Path
    run_id: str

    @property
    def path(self) -> Path:
        validate_run_id(self.run_id)
        return checked_path(
            self.root, f"runs/agent/{self.run_id}", allowed_roots=("runs/agent",)
        )

    @classmethod
    def create(cls, root: Path | str, run_id: str) -> RunDirectory:
        run = cls(workspace_root(root), validate_run_id(run_id))
        path = run.path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Recheck after parent creation; an existing final directory is an error.
        run.path.mkdir(exist_ok=False)
        return run

    def write_json(self, filename: str, payload: Any) -> Path:
        """Write one flat JSON artifact exclusively, without replacing any file."""
        parts = _parts(filename)
        if len(parts) != 1 or not filename.endswith(".json"):
            raise PathPolicyError("Expected a single .json artifact filename")
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        # Revalidate ancestors at write time, including a swapped run directory.
        self.path
        target = checked_path(
            self.root, f"runs/agent/{self.run_id}/{filename}", allowed_roots=("runs/agent",)
        )
        with target.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        return target
