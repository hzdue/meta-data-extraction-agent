"""Single-user local facade over the existing synthetic workflow and review store.

The bundle is the only source/configuration scope. There is no model-service,
upload, standards-enrichment, evaluator, or external-export entry point here.
OS-held leases serialize local writers; this is not an authenticated service or
an operating-system sandbox against a hostile local filesystem owner.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import re
from typing import Any

from pydantic import ValidationError

from .evidence import sha256
from .inspect_workflow import inspect_workflow, load_config
from .paths import PathPolicyError, RunDirectory, checked_path, validate_run_id, workspace_root
from .pilot_bundle import BundleError, verify_bundle
from .review_store import ReviewStore
from .review_types import ReviewError, ReviewResponse
from .storage import RunConflictError, RunIncompleteError, _digest, _json_bytes
from .workflow_graph import WorkflowBusyError, _lease

VERSION = "m10-local-pilot-1"
CONFIG_PATH = "configs/agent/workflow_review.json"
LOCK_ID = "m10-service-lock"
MAX_RUNS = 8
MAX_STORAGE_BYTES = 128 * 1024 * 1024
MAX_STORAGE_FILES = 4096
MAX_RESPONSE_BYTES = 1_000_000
MAX_STEPS = 12
MAX_JSON_BYTES = 8_000_000
MUTATION_RESERVE_BYTES = 32 * 1024 * 1024
WINDOWS_PATH_LIMIT = 260 if os.name == "nt" else None


class PilotError(Exception):
    """Transport-safe error; neither the message nor code contains input text."""

    def __init__(self, code: str, status_code: int = 409):
        self.code = code if re.fullmatch(r"[a-z][a-z0-9_]{0,99}", code) else "operation_failed"
        self.status_code = status_code if status_code in {400, 404, 409, 413, 500, 507} else 500
        super().__init__(self.code)


def _safe(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except PilotError:
            raise
        except WorkflowBusyError:
            raise PilotError("pilot_busy") from None
        except BundleError:
            raise PilotError("bundle_invalid") from None
        except ReviewError as error:
            raise PilotError(error.code) from None
        except ValidationError:
            raise PilotError("invalid_response", 400) from None
        except PathPolicyError:
            raise PilotError("path_rejected", 400) from None
        except RunIncompleteError:
            raise PilotError("incomplete_run") from None
        except RunConflictError:
            raise PilotError("run_conflict") from None
        except ValueError:
            raise PilotError("state_or_response_rejected") from None
        except Exception:
            # Never forward arbitrary parser, SQLite, filesystem, or model errors.
            raise PilotError("operation_failed", 500) from None
    return wrapped


def _bounded_json(value: Any, *, maximum: int) -> Any:
    if not isinstance(value, dict):
        raise PilotError("invalid_response", 400)
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if depth > 32:
            raise PilotError("json_too_deep", 413)
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise PilotError("invalid_response", 400)
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend((child, depth + 1) for child in item)
        elif item is not None and type(item) not in {str, int, float, bool}:
            raise PilotError("invalid_response", 400)
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError):
        raise PilotError("invalid_response", 400) from None
    if len(raw) > maximum:
        raise PilotError("response_too_large", 413)
    return json.loads(raw)


def _read_json(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise PilotError("artifact_too_large", 413)
    raw = path.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise PilotError("artifact_too_large", 413)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PilotError("artifact_invalid")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(PilotError("artifact_invalid")))
    except (ValueError, RecursionError, UnicodeError):
        raise PilotError("artifact_invalid") from None
    if not isinstance(value, dict):
        raise PilotError("artifact_invalid")
    return value


class PilotService:
    """A fixed synthetic workflow, restartable M08 review, and local JSON export."""

    @_safe
    def __init__(self, root: Path | str):
        verify_bundle(root)
        self.root = workspace_root(root)

    def _verify(self) -> dict:
        return verify_bundle(self.root)

    def _path(self, relative: str) -> Path:
        return checked_path(self.root, relative, allowed_roots=("runs/agent",))

    def _run(self, run_id: str) -> RunDirectory:
        if (not isinstance(run_id, str) or not re.fullmatch(r"m10-[a-z0-9][a-z0-9_-]{0,59}", run_id)
                or run_id == LOCK_ID):
            raise PilotError("invalid_run_id", 400)
        run = RunDirectory(self.root, validate_run_id(run_id))
        longest = str(run.path / "review/exports" / ("0" * 64) / "canonical.json.pending")
        if WINDOWS_PATH_LIMIT is not None and len(longest.encode("utf-16-le")) // 2 >= WINDOWS_PATH_LIMIT:
            raise PilotError("run_path_too_long", 400)
        return run

    def _usage(self) -> dict:
        directory = self._path("runs/agent")
        if not directory.exists():
            return {"runs": 0, "files": 0, "bytes": 0}
        run_count = 0
        files = 0
        size = 0
        for child in directory.iterdir():
            if child.name != LOCK_ID:
                self._run(child.name)
                if not child.is_dir():
                    raise PilotError("unexpected_run_artifact")
                run_count += 1
        pending = [directory]
        while pending:
            current = pending.pop()
            for child in current.iterdir():
                path = self._path(child.relative_to(self.root).as_posix())
                if path.is_dir():
                    pending.append(path)
                elif path.is_file():
                    files += 1
                    size += path.stat().st_size
                else:
                    raise PilotError("unexpected_run_artifact")
                if files > MAX_STORAGE_FILES or size > MAX_STORAGE_BYTES:
                    raise PilotError("storage_limit", 507)
        if run_count > MAX_RUNS:
            raise PilotError("run_limit", 507)
        return {"runs": run_count, "files": files, "bytes": size}

    @contextmanager
    def _operation(self, *, reserve: bool = False):
        self._verify()
        # Reuse the inherited OS lease, with a separate name so ReviewStore and
        # workflow execution can take their own run leases inside this guard.
        lock = RunDirectory(self.root, LOCK_ID)
        lock.path.mkdir(parents=True, exist_ok=True)
        with _lease(lock):
            self._verify()
            usage = self._usage()
            if reserve and usage["bytes"] + MUTATION_RESERVE_BYTES > MAX_STORAGE_BYTES:
                raise PilotError("storage_limit", 507)
            yield

    def _existing(self, run: RunDirectory) -> None:
        if not run.path.is_dir():
            raise PilotError("run_not_found", 404)
        request_path = self._path(f"runs/agent/{run.run_id}/request.json")
        if not request_path.is_file():
            raise PilotError("incomplete_run")
        envelope = _read_json(request_path)
        request = envelope.get("request")
        config = load_config(self.root, CONFIG_PATH).model_dump(mode="json")
        if (set(envelope) != {"version", "request_sha256", "request"}
                or envelope["version"] != 1 or not isinstance(request, dict)
                or envelope["request_sha256"] != _digest(_json_bytes(request))
                or request.get("version") != "m07-run-1"
                or request.get("declared", {}).get("config") != config
                or request.get("declared", {}).get("mode") != "synthetic_offline"
                or request.get("policy_provider", {}).get("mode") != "mock"):
            raise PilotError("run_scope_mismatch")

    @staticmethod
    def _review_summary(view: dict) -> dict:
        return {"status": view["status"], "state_id": view["state_id"],
                "record_revision": view["record_revision"], "decisions": view["decisions"],
                "pending_cases": sum(case["status"] == "pending" for case in view["cases"]),
                "unresolved_cases": sum(case["status"] == "unresolved" for case in view["cases"]),
                "approved_cases": sum(case["status"] == "approved" for case in view["cases"])}

    def _summary(self, run: RunDirectory, result: dict) -> dict:
        state = result["state"]
        summary = {"version": VERSION, "run_id": run.run_id, "execution": result["execution"],
                   "workflow_status": state.status, "status": state.status,
                   "state_id": None, "record_revision": state.record.revision_id,
                   "total_model_calls": state.model_calls, "uncertain_model_calls": state.uncertain_model_calls,
                   "new_model_calls": result["new_model_calls"], "new_live_requests": result["new_live_requests"],
                   "usage_unknown": state.usage_unknown, "pending_cases": len(state.pending_reviews),
                   "actions": len(state.events), "coverage_status": "unsupported",
                   "expert_authentication": "not_authenticated", "synthetic_only": True}
        if state.status == "pending_review":
            with ReviewStore.open(self.root, run.run_id) as store:
                summary.update(self._review_summary(store.view()))
        return summary

    @_safe
    def health(self) -> dict:
        manifest = self._verify()
        usage = self._usage()
        return {"version": VERSION, "status": "ready", "synthetic_only": True,
                "bundle_id": manifest.get("bundle_id"),
                "capabilities": ["fixed_mock_workflow", "restartable_local_review", "state_bound_json_export"],
                "model_execution": "deterministic_mock_only", "live_requests": 0,
                "allowed_config": CONFIG_PATH,
                "limits": {"runs": MAX_RUNS, "storage_bytes": MAX_STORAGE_BYTES,
                           "storage_files": MAX_STORAGE_FILES, "response_bytes": MAX_RESPONSE_BYTES,
                           "max_steps_per_request": MAX_STEPS},
                "usage": usage, "reviewer_identity": "self_declared_not_authenticated",
                "coverage_status": "unsupported", "empirical_status": "not_run",
                "migration_policy": "exact_bundle_and_runtime_required_no_automatic_migration",
                "concurrency": "single_user_os_lease_serialized",
                "network_scope": "loopback_only_no_upload_or_external_export",
                "limitations": ["Invented source and mock responses; no natural-paper performance evidence.",
                                "Pending review is not approval; selected assertions do not establish completeness.",
                                "Interrupted dispatch stays uncertain and is never retried automatically."]}

    @_safe
    def start(self, run_id: str, *, max_steps: int | None = None) -> dict:
        run = self._run(run_id)
        if max_steps is not None and (type(max_steps) is not int or not 1 <= max_steps <= MAX_STEPS):
            raise PilotError("invalid_max_steps", 400)
        with self._operation(reserve=True):
            if run.path.exists():
                self._existing(run)
            elif self._usage()["runs"] >= MAX_RUNS:
                raise PilotError("run_limit", 507)
            config = load_config(self.root, CONFIG_PATH)
            result = inspect_workflow(self.root, config, run_id=run.run_id, max_steps=max_steps)
            return self._summary(run, result)

    @_safe
    def status(self, run_id: str) -> dict:
        run = self._run(run_id)
        with self._operation():
            self._existing(run)
            if not self._path(f"runs/agent/{run.run_id}/result.json").is_file():
                return {"version": VERSION, "run_id": run.run_id, "execution": "inspection",
                        "workflow_status": "incomplete", "status": "incomplete",
                        "state_id": None, "record_revision": None, "pending_cases": None,
                        "new_model_calls": 0, "new_live_requests": 0, "synthetic_only": True,
                        "expert_authentication": "not_authenticated",
                        "resume": "explicit_start_same_id_validates_checkpoints_no_budget_reset"}
            # The existing completed replay verifies the full frozen request,
            # canonical exports and journal. With the service lease and marker
            # check this cannot continue a partial run or dispatch a tool.
            result = inspect_workflow(self.root, load_config(self.root, CONFIG_PATH), run_id=run.run_id)
            if result["execution"] != "cached_replay" or result["new_model_calls"] or result["new_live_requests"]:
                raise PilotError("unexpected_dispatch")
            return self._summary(run, result)

    def _review_open(self, run: RunDirectory):
        self._existing(run)
        if not self._path(f"runs/agent/{run.run_id}/result.json").is_file():
            raise PilotError("review_not_ready")
        return ReviewStore.open(self.root, run.run_id)

    @_safe
    def review(self, run_id: str) -> dict:
        run = self._run(run_id)
        with self._operation(), self._review_open(run) as store:
            return store.view()

    @_safe
    def submit(self, run_id: str, response: dict) -> dict:
        run = self._run(run_id)
        validated = ReviewResponse.model_validate(_bounded_json(response, maximum=MAX_RESPONSE_BYTES))
        with self._operation(reserve=True), self._review_open(run) as store:
            result = store.submit(validated)
            view = store.view()
            try:
                manifest = store.export()
            except Exception:
                # The durable response may already be committed. Retrying this
                # exact idempotency key or explicit export safely recovers it.
                raise PilotError("export_pending_after_commit") from None
            return {"version": VERSION, "run_id": run.run_id, "execution": result["execution"],
                    **self._review_summary(view), "response_provenance": validated.provenance,
                    "response_outcome": validated.outcome, "reviewer_identity": "self_declared_not_authenticated",
                    "export": manifest, "new_model_calls": 0, "new_live_requests": 0}

    @_safe
    def export(self, run_id: str, expected_state_id: str) -> dict:
        run = self._run(run_id)
        if not isinstance(expected_state_id, str) or not re.fullmatch(r"review-state:[0-9a-f]{64}", expected_state_id):
            raise PilotError("invalid_state_id", 400)
        with self._operation(reserve=True), self._review_open(run) as store:
            if store.state.state_id != expected_state_id:
                raise PilotError("stale_state")
            manifest = store.export()
            exported = {}
            for name in ("canonical.json", "sidecar.json"):
                path = self._path(manifest["directory"] + "/" + name)
                raw = path.read_bytes()
                pin = manifest["artifacts"][name]
                if len(raw) != pin["bytes"] or sha256(raw) != pin["sha256"]:
                    raise PilotError("export_integrity_failure")
                exported[name.removesuffix(".json")] = _read_json(path)
            return {"manifest": manifest, **exported}
