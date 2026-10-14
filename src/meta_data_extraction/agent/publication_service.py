"""One frozen publication scope behind the existing authenticated local transport.

Status and review never dispatch models. The HTML landing page deliberately has
no interactive review interface: the inherited pilot page describes a synthetic
fixture and must not be presented as a live-publication interface.
"""
from __future__ import annotations

from functools import wraps
import re
from pathlib import Path

from pydantic import ValidationError

from .contracts import digest
from .evidence import sha256
from .paths import PathPolicyError, checked_path, validate_run_id, workspace_root
from .pilot_core import PilotError, _bounded_json, _read_json
from .pilot_http import make_server
from .publication_types import PublicationSpec
from .review_store import ReviewStore
from .review_types import ReviewError, ReviewResponse
from .storage import RunConflictError, RunIncompleteError
from .workflow_graph import WorkflowBusyError

VERSION = "publication-service-1"
MAX_RESPONSE_BYTES = 131072
_STATE = re.compile(r"review-state:[0-9a-f]{64}")
_HTTP_RUN = re.compile(r"m10-[a-z0-9][a-z0-9_-]{0,27}")
_PAGE = b'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Publication API</title><h1>Local publication API</h1>
<p>This service exposes one operator-configured publication run.</p>
<p>Authenticated JSON endpoints provide run status, review, submission and export.
Use the publication CLI to inspect review cases and submit explicit decisions.</p>
<p>This page does not start a run or approve a review. There is no document upload
or browser review interface for this service.</p></html>'''


def _safe(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except PilotError:
            raise
        except ReviewError as error:
            raise PilotError(error.code) from None
        except ValidationError:
            raise PilotError("invalid_request", 400) from None
        except WorkflowBusyError:
            raise PilotError("publication_busy") from None
        except PathPolicyError:
            raise PilotError("path_rejected", 400) from None
        except RunConflictError:
            raise PilotError("run_conflict") from None
        except RunIncompleteError:
            raise PilotError("incomplete_run") from None
        except FileNotFoundError:
            raise PilotError("run_not_found", 404) from None
        except ValueError:
            raise PilotError("state_or_scope_rejected") from None
        except Exception:
            raise PilotError("operation_failed", 500) from None
    return wrapped


def _runner():
    # Import lazily: constructing a service and health checks never start a run.
    from . import publication_run
    return publication_run


class PublicationService:
    def __init__(self, root: Path | str, spec: PublicationSpec, *,
                 configured_chatai_endpoint: str | None = None,
                 load_dotenv_file: bool = False):
        self.root = workspace_root(root)
        self.spec = PublicationSpec.model_validate(spec.model_dump(mode="json"))
        self.spec_sha256 = digest(self.spec.model_dump(mode="json"))
        if type(load_dotenv_file) is not bool:
            raise ValueError("An explicit credential-loading choice is required")
        if configured_chatai_endpoint is not None and (
            self.spec.mode != "live" or self.spec.live.endpoint != configured_chatai_endpoint
        ):
            raise ValueError("Configured endpoint differs from publication scope")
        self.configured_chatai_endpoint = configured_chatai_endpoint
        self.load_dotenv_file = load_dotenv_file

    def _scope(self, run_id: str) -> None:
        if not isinstance(run_id, str) or run_id != self.spec.run_id:
            raise PilotError("run_outside_scope", 404)
        if digest(self.spec.model_dump(mode="json")) != self.spec_sha256:
            raise PilotError("publication_scope_changed")

    def _dispatch_scope(self) -> None:
        if self.spec.mode == "live" and self.configured_chatai_endpoint != self.spec.live.endpoint:
            raise PilotError("trusted_endpoint_required", 400)

    def _read(self, run_id: str) -> dict:
        self._scope(run_id)
        result = _runner().read_publication(self.root, run_id)
        if result.get("run_id") != run_id or result.get("spec_sha256") != self.spec_sha256:
            raise PilotError("publication_scope_changed")
        return result

    def _review_open(self, run_id: str):
        result = self._read(run_id)
        if result.get("workflow_status", result.get("status")) != "pending_review":
            raise PilotError("review_not_ready")
        child = result.get("workflow_run_id")
        if not isinstance(child, str) or child == run_id:
            raise PilotError("workflow_binding_invalid")
        validate_run_id(child)
        return ReviewStore.open(self.root, child)

    @_safe
    def health(self) -> dict:
        self._scope(self.spec.run_id)
        return {"version": VERSION, "status": "ready", "run_id": self.spec.run_id,
                "mode": self.spec.mode, "split": self.spec.split,
                "spec_sha256": self.spec_sha256, "interface": "authenticated_json_api",
                "new_model_calls": 0, "new_live_requests": 0}

    @_safe
    def start(self, run_id: str, *, max_steps: int | None = None) -> dict:
        self._scope(run_id)
        self._dispatch_scope()
        if max_steps is not None and (type(max_steps) is not int or not 1 <= max_steps <= 12):
            raise PilotError("max_steps_invalid", 400)
        return _runner().run_publication(
            self.root, self.spec, max_steps=max_steps,
            configured_chatai_endpoint=self.configured_chatai_endpoint,
            load_dotenv_file=self.load_dotenv_file,
        )

    @_safe
    def status(self, run_id: str) -> dict:
        return self._read(run_id)

    @_safe
    def review(self, run_id: str) -> dict:
        with self._review_open(run_id) as store:
            return {**store.view(), "publication_run_id": run_id}

    @_safe
    def submit(self, run_id: str, response: dict) -> dict:
        self._scope(run_id)
        validated = ReviewResponse.model_validate(_bounded_json(response, maximum=MAX_RESPONSE_BYTES))
        with self._review_open(run_id) as store:
            result = store.submit(validated)
            view = store.view()
            try:
                manifest = store.export()
            except Exception:
                raise PilotError("export_pending_after_commit") from None
            return {"version": VERSION, "run_id": store.run.run_id,
                    "publication_run_id": run_id, "execution": result["execution"],
                    "state_id": view["state_id"], "record_revision": view["record_revision"],
                    "status": view["status"], "decisions": view["decisions"],
                    "response_provenance": validated.provenance,
                    "response_outcome": validated.outcome,
                    "reviewer_identity": "self_declared_not_authenticated",
                    "export": manifest, "new_model_calls": 0, "new_live_requests": 0}

    @_safe
    def export(self, run_id: str, expected_state_id: str) -> dict:
        self._scope(run_id)
        if not isinstance(expected_state_id, str) or not _STATE.fullmatch(expected_state_id):
            raise PilotError("invalid_state_id", 400)
        with self._review_open(run_id) as store:
            if store.state.state_id != expected_state_id:
                raise PilotError("stale_state")
            manifest = store.export()
            expected_root = f"runs/agent/{store.run.run_id}/review/exports"
            exported = {}
            for name in ("canonical.json", "sidecar.json"):
                path = checked_path(self.root, manifest["directory"] + "/" + name,
                                    allowed_roots=(expected_root,))
                raw = path.read_bytes()
                pin = manifest["artifacts"][name]
                if len(raw) != pin["bytes"] or sha256(raw) != pin["sha256"]:
                    raise PilotError("export_integrity_failure")
                exported[name.removesuffix(".json")] = _read_json(path)
            return {"publication_run_id": run_id, "manifest": manifest, **exported}


def make_publication_server(service: PublicationService, token: str, *,
                            host="127.0.0.1", port=8765, log_stream=None):
    """Reuse HTTP authentication/limits; only this server instance gets a new page."""
    if not _HTTP_RUN.fullmatch(service.spec.run_id):
        raise ValueError("HTTP publication run IDs must begin with m10-")
    service._scope(service.spec.run_id)
    service._dispatch_scope()
    server = make_server(service, token, host=host, port=port, log_stream=log_stream)
    server.ui = _PAGE
    server.csp = "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    return server
