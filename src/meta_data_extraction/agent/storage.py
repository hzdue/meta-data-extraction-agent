"""Exclusive, verifiable M02 exports within the existing local run boundary.

A run directory is an execution claim. Only a validated completion marker permits
replay; a partial run never authorizes another provider call. This is an
application boundary, not protection against an adversary replacing files during
filesystem operations (the same limitation as :mod:`.paths`).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import RunDirectory, checked_path, validate_run_id, workspace_root


class RunConflictError(ValueError):
    """An existing run has different inputs or invalid/changed completed output."""


class RunIncompleteError(RuntimeError):
    """An existing execution claim is incomplete; automatic dispatch is forbidden."""


def _json_bytes(value: Any) -> bytes:
    """Canonical local JSON; reject values that cannot round-trip faithfully."""
    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise TypeError("Expected JSON-compatible values")

    check(value)
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _object_copy(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    return json.loads(_json_bytes(value))


def _artifact_path(run: RunDirectory, name: str) -> Path:
    # Names originate only from constants in this module or a checked manifest.
    return checked_path(run.root, f"runs/agent/{run.run_id}/{name}",
                        allowed_roots=("runs/agent",))


def _read_json(run: RunDirectory, name: str) -> dict[str, Any]:
    path = _artifact_path(run, name)
    try:
        payload = json.loads(path.read_bytes())
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        return payload
    except (OSError, ValueError, UnicodeError) as exc:
        raise RunConflictError(f"Run artifact {name} is unavailable or invalid") from exc


def _write_exclusive(run: RunDirectory, name: str, payload: bytes) -> None:
    # Recheck run ancestors and hard links immediately before the exclusive open.
    path = _artifact_path(run, name)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


_ARTIFACTS = frozenset({"canonical.json", "sidecar.json", "raw-response.json"})
_VERSION = 1


def _read_completion(run: RunDirectory, request_hash: str) -> dict[str, Any]:
    envelope = _read_json(run, "result.json")
    if set(envelope) != {"version", "request_sha256", "result", "artifacts",
                         "completion_sha256"}:
        raise RunConflictError("Unrecognized run completion structure")
    signed = {key: value for key, value in envelope.items() if key != "completion_sha256"}
    if (envelope["version"] != _VERSION or envelope["request_sha256"] != request_hash
            or envelope["completion_sha256"] != _digest(_json_bytes(signed))
            or not isinstance(envelope["result"], dict)
            or not isinstance(envelope["artifacts"], dict)):
        raise RunConflictError("Run completion metadata does not match its request")
    artifacts = envelope["artifacts"]
    if "sidecar.json" not in artifacts or not set(artifacts).issubset(_ARTIFACTS):
        raise RunConflictError("Run completion has an invalid artifact manifest")
    # Reject undeclared canonical/raw artifacts as well as changed declared ones.
    for name in _ARTIFACTS:
        path = _artifact_path(run, name)
        if name not in artifacts:
            if path.exists():
                raise RunConflictError("Run has an undeclared artifact")
            continue
        metadata = artifacts[name]
        if (not isinstance(metadata, dict) or set(metadata) != {"sha256", "bytes"}
                or type(metadata["bytes"]) is not int or metadata["bytes"] < 0):
            raise RunConflictError("Run artifact metadata is invalid")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise RunConflictError("Run artifact is unavailable") from exc
        if len(content) != metadata["bytes"] or _digest(content) != metadata["sha256"]:
            raise RunConflictError("Run artifact content has changed")
    return envelope


def _check_request(run: RunDirectory, expected: dict[str, Any], expected_hash: str) -> None:
    saved = _read_json(run, "request.json")
    if saved != {"version": _VERSION, "request_sha256": expected_hash, "request": expected}:
        raise RunConflictError("Run ID is already associated with different inputs")


@dataclass
class RunStore:
    """Idempotent completed-run replay with exclusive, fail-closed new execution.

    ``completed`` and ``complete`` expose the original result object. ``result.json``
    wraps it in a versioned manifest with SHA256 hashes of separately stored
    canonical data, sidecars, and optionally raw output. Hashes detect accidental
    changes; they are not signatures authenticating an untrusted filesystem.
    """

    run: RunDirectory
    completed: dict[str, Any] | None
    _request: dict[str, Any]
    _request_hash: str

    @classmethod
    def begin(cls, root: Path | str, run_id: str, request: dict[str, Any]) -> RunStore:
        request = _object_copy(request)
        request_hash = _digest(_json_bytes(request))
        run = RunDirectory(workspace_root(root), validate_run_id(run_id))
        try:
            run = RunDirectory.create(root, run_id)
        except FileExistsError:
            # Refuse unsafe links even when another process owns the directory.
            if not run.path.is_dir():
                raise RunConflictError("Run ID exists but is not a directory")
            request_path = _artifact_path(run, "request.json")
            result_path = _artifact_path(run, "result.json")
            if not request_path.exists():
                raise RunIncompleteError("Run execution was claimed but no request was saved")
            _check_request(run, request, request_hash)
            if not result_path.exists():
                raise RunIncompleteError("Run has no completion marker; no automatic retry")
            completed = _read_completion(run, request_hash)["result"]
            return cls(run, completed, request, request_hash)
        _write_exclusive(run, "request.json", _json_bytes({
            "version": _VERSION, "request_sha256": request_hash, "request": request,
        }))
        return cls(run, None, request, request_hash)

    def complete(self, result: dict[str, Any], *, canonical: dict[str, Any] | None,
                 sidecar: dict[str, Any], raw_response: str | None,
                 retain_raw: bool) -> dict[str, Any]:
        """Publish immutable exports and the completion marker last.

        A failed/partial export remains incomplete. The caller must choose a new
        run or perform explicit operator recovery; rerunning never repeats model
        dispatch. When retention is disabled no raw text is stored, even in the
        completion fingerprint. Repetition compares the persisted output only.
        """
        result = _object_copy(result)
        if type(retain_raw) is not bool:
            raise TypeError("retain_raw must be a boolean")
        if raw_response is not None and not isinstance(raw_response, str):
            raise TypeError("raw_response must be text or None")
        exports = {"sidecar.json": _json_bytes(_object_copy(sidecar))}
        if canonical is not None:
            exports["canonical.json"] = _json_bytes(_object_copy(canonical))
        if retain_raw and raw_response is not None:
            exports["raw-response.json"] = _json_bytes({"raw_response": raw_response})
        signed = {"version": _VERSION, "request_sha256": self._request_hash,
                  "result": result, "artifacts": {
                      name: {"sha256": _digest(payload), "bytes": len(payload)}
                      for name, payload in exports.items()
                  }}
        envelope = {**signed, "completion_sha256": _digest(_json_bytes(signed))}
        _check_request(self.run, self._request, self._request_hash)
        if _artifact_path(self.run, "result.json").exists():
            existing = _read_completion(self.run, self._request_hash)
            if existing != envelope:
                raise RunConflictError("Completed run cannot be replaced by different output")
            self.completed = _object_copy(result)
            return _object_copy(result)
        if self.completed is not None:
            raise RunConflictError("The completed run marker has disappeared")
        # Any previous publication attempt is an incomplete execution. Check every
        # name before writing, including files excluded by this retention policy.
        for name in (*_ARTIFACTS, "result.pending.json"):
            if _artifact_path(self.run, name).exists():
                raise RunIncompleteError("Run export was interrupted; no automatic rewrite")
        for name, payload in exports.items():
            _write_exclusive(self.run, name, payload)
        _write_exclusive(self.run, "result.pending.json", _json_bytes(envelope))
        pending = _artifact_path(self.run, "result.pending.json")
        target = _artifact_path(self.run, "result.json")
        # Windows rename refuses an existing target; POSIX rename would replace it.
        # On POSIX publish via an exclusive hard link then remove only our own
        # temporary name. Both branches make a fully flushed marker visible at once.
        if os.name == "nt":
            pending.rename(target)
        else:
            os.link(pending, target)
            pending.unlink()
        _read_completion(self.run, self._request_hash)
        self.completed = _object_copy(result)
        return _object_copy(result)
