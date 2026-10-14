"""Durable local review transactions extending an immutable pending M07 run.

No provider is constructed or called. Reviewer labels are self-declared, and
content hashes provide local integrity checks rather than identity authentication.
"""
from __future__ import annotations

from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sqlite3
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .contracts import at_pointer, canonical_json, digest, owner
from .evidence import DocumentBundle, resolve_evidence, sha256
from .paths import RunDirectory, checked_path, validate_run_id, workspace_root
from .profiles import Profile, verify_profile_resources
from .record_checks import check_structure
from .record_patches import PatchProposal, build_candidate
from .retrieval import SourceCorpus
from .review import apply_review, initialize_review
from .review_types import ReviewResponse, ReviewState
from .storage import RunIncompleteError, _digest, _json_bytes, _read_completion
from .workflow_graph import _implementation as workflow_implementation, _lease, _transition
from .workflow_types import WorkflowState

VERSION = "m08-review-store-1"
MAX_JSON_BYTES = 8_000_000
PARENT_FILES = ("request.json", "result.json", "canonical.json", "sidecar.json",
                "dispatch.sqlite", "checkpoints.sqlite")


def _json(raw: bytes) -> Any:
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Review input exceeds the persistence bound")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _serialized(value) -> str:
    raw = canonical_json(value.model_dump(mode="json"))
    if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("Review state exceeds the persistence bound")
    return raw


def _implementation() -> dict:
    base = Path(__file__).parent
    names = ("review_types.py", "review.py", "review_store.py", "review_cli.py")
    return {"python": platform.python_version(),
            "files": {name: sha256((base / name).read_bytes()) for name in names},
            "workflow": workflow_implementation(),
            "dependencies": {name: version(name) for name in
                             ("pydantic", "langgraph-checkpoint", "langgraph-checkpoint-sqlite")}}


class ReviewStore:
    """Use as a context manager; one OS-held run lease covers local review I/O."""

    @classmethod
    def open(cls, root: Path | str, run_id: str) -> ReviewStore:
        self = cls()
        self.run = RunDirectory(workspace_root(root), validate_run_id(run_id))
        if not self.run.path.is_dir():
            raise ValueError("Review requires an existing completed pending workflow")
        self._guard = _lease(self.run)
        self._guard.__enter__()
        self.conn = None
        try:
            self._load_parent()
            self._open_database()
            self._read_state()
            return self
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> ReviewStore:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self._guard is not None:
            self._guard.__exit__(None, None, None)
            self._guard = None

    def _path(self, name: str) -> Path:
        return checked_path(self.run.root, f"runs/agent/{self.run.run_id}/{name}",
                            allowed_roots=("runs/agent",))

    def _pins(self) -> dict[str, str]:
        pins = {}
        for name in PARENT_FILES:
            path = self._path(name)
            if not path.is_file():
                raise ValueError("Pending workflow artifact is missing")
            if name.endswith(".sqlite"):
                for suffix in ("-wal", "-shm", "-journal"):
                    if self._path(name + suffix).exists():
                        raise ValueError("Completed parent has an outstanding SQLite side file")
            pins[name] = sha256(path.read_bytes())
        return pins

    def _load_parent(self) -> None:
        self.parent_pins = self._pins()
        envelope = _json(self._path("request.json").read_bytes())
        if (set(envelope) != {"version", "request_sha256", "request"}
                or envelope["version"] != 1
                or envelope["request_sha256"] != _digest(_json_bytes(envelope["request"]))):
            raise ValueError("Workflow request fingerprint differs")
        request = envelope["request"]
        if request.get("version") != "m07-run-1" or request.get("implementation") != workflow_implementation():
            raise ValueError("Incompatible workflow version or implementation; explicit migration required")
        expected_dependencies = {name: version(name) for name in ("langgraph", "langgraph-checkpoint",
            "langgraph-checkpoint-sqlite", "langchain-core", "langsmith", "pydantic")}
        if request.get("dependencies") != expected_dependencies or request.get("python") != platform.python_version():
            raise ValueError("Workflow runtime changed; explicit compatibility migration required")
        completion = _read_completion(self.run, envelope["request_sha256"])
        result = completion["result"]
        if result.get("version") != "m07-result-1":
            raise ValueError("Incompatible workflow result version")
        workflow = WorkflowState.model_validate(result["state"])
        if workflow.status != "pending_review":
            raise ValueError("Only an explicitly pending workflow can enter review")
        sidecar = _json(self._path("sidecar.json").read_bytes())
        if set(sidecar) != {"workflow", "source", "profile"} or sidecar["workflow"] != workflow.model_dump(mode="json"):
            raise ValueError("Workflow sidecar differs from its completion")
        self.bundle = DocumentBundle.model_validate(sidecar["source"])
        self.profile = Profile.model_validate(sidecar["profile"])
        verify_profile_resources(self.profile, self.run.root)
        structure = check_structure(self.profile, self.bundle, workflow.record.canonical)
        if structure.snapshot != workflow.record:
            raise ValueError("Pending record structure or source binding differs")
        if _json(self._path("canonical.json").read_bytes()) != workflow.record.canonical:
            raise ValueError("Pending canonical export differs")
        refs = [r for item in workflow.issues for r in item.evidence]
        refs += [r for item in workflow.pending_reviews for r in item.evidence]
        refs += [r for item in workflow.events for r in item.evidence]
        for ref in refs:
            resolve_evidence(self.bundle, ref)
        self._verify_terminal(request, workflow, envelope["request_sha256"])
        self._verify_accounting(request, result, workflow)
        proposals = {}
        declared = request.get("declared", {})
        # M07's general graph records catalog IDs only. Its fixture inspector also
        # freezes this explicit proposal body; unavailable bodies stay unavailable.
        if declared.get("version") == "m07-inspection-1":
            proposal = declared.get("seed_config", {}).get("patch")
            if proposal is not None:
                proposals["correct_dataset_a"] = PatchProposal.model_validate(proposal)
        elif declared.get("version") == "publication-workflow-1":
            if set(declared) != {"version", "proposals"} or type(declared["proposals"]) is not dict:
                raise ValueError("Invalid publication workflow catalog")
            if len(declared["proposals"]) > 32:
                raise ValueError("Publication proposal catalog exceeds its bound")
            initial = WorkflowState.model_validate(request["initial_state"])
            allowed = {key for issue in initial.issues for key in issue.proposal_ids}
            for key, raw in declared["proposals"].items():
                proposal = PatchProposal.model_validate(raw)
                if key != proposal.proposal_id or key not in allowed:
                    raise ValueError("Publication proposal identity differs from the frozen catalog")
                proposals[key] = proposal
        self.initial = initialize_review(workflow, self.profile, self.bundle, proposals=proposals)
        self.workflow = workflow
        self.accounting = {name: result.get(name) for name in (
            "model_calls", "uncertain_model_calls", "live_requests", "live_provider_invocations",
            "live_dispatch_unknown", "usage_unknown", "cost", "cost_status")}

    @staticmethod
    def _verify_accounting(request: dict, result: dict, workflow: WorkflowState) -> None:
        observations = tuple(item for event in workflow.events for item in event.providers)
        expected = {
            "model_calls": len(observations), "uncertain_model_calls": workflow.uncertain_model_calls,
            "live_requests": sum(item.mode == "live" and item.request_id is not None for item in observations),
            "live_provider_invocations": sum(item.mode == "live" for item in observations),
            "usage_unknown": workflow.usage_unknown,
            "live_dispatch_unknown": any(item.mode == "live" and item.request_id is None
                                         and item.status != "blocked" for item in observations),
        }
        if workflow.uncertain_model_calls:
            raise ValueError("Uncertain parent dispatch requires explicit recovery before review")
        if any(type(result.get(key)) is not type(value) or result.get(key) != value for key, value in expected.items()):
            raise ValueError("Workflow completion accounting differs from its observations")
        if (workflow.model_calls != len(observations)
                or workflow.input_tokens != sum(item.input_tokens or 0 for item in observations)
                or workflow.output_tokens != sum(item.output_tokens or 0 for item in observations)
                or workflow.usage_unknown != any(item.input_tokens is None or item.output_tokens is None for item in observations)):
            raise ValueError("Workflow usage accounting differs from its observations")
        budget = request["budget"]
        if len(workflow.events) > budget["total_actions"] or workflow.model_calls > budget["max_model_calls"]:
            raise ValueError("Workflow exceeds its frozen budget")
        if (result.get("policy") != request.get("policy") or result.get("cost") is not None
                or result.get("raw_retention") != "hash_only" or result.get("coverage_status") != "unsupported"
                or result.get("expert_approval") != "not_supplied"
                or result.get("cost_status") not in {"unknown", "not_applicable_mock"}):
            raise ValueError("Workflow completion scope or cost metadata differs")
        policy_provider = request.get("policy_provider") or {}
        if (expected["live_provider_invocations"] or policy_provider.get("mode") == "live") and result["cost_status"] != "unknown":
            raise ValueError("Live provider cost cannot become zero or mock cost")

    def _verify_terminal(self, request: dict, workflow: WorkflowState, request_hash: str) -> None:
        initial = WorkflowState.model_validate(request["initial_state"])
        current = initial
        conn = sqlite3.connect(self._path("dispatch.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            conn.execute("PRAGMA trusted_schema=OFF")
            meta = dict(conn.execute("SELECT key,value FROM metadata"))
            if (meta.get("version"), meta.get("request_hash"), meta.get("initial_state")) != (
                "m07-dispatch-1", request_hash, _serialized(initial)):
                raise ValueError("Workflow journal bindings differ")
            seen = set()
            while current.status == "running":
                if current.state_id in seen or len(seen) > 33:
                    raise ValueError("Workflow journal has a cycle or exceeds its bound")
                seen.add(current.state_id)
                row = conn.execute("SELECT before_json,after_json,after_id FROM steps WHERE state_id=?",
                                   (current.state_id,)).fetchone()
                if row is None or row[0] != _serialized(current) or row[1] is None:
                    raise ValueError("Workflow journal has no committed terminal state")
                following = WorkflowState.model_validate_json(row[1])
                if row[1] != _serialized(following) or row[2] != following.state_id:
                    raise ValueError("Workflow journal state differs")
                _transition(current, following)
                current = following
            if current != workflow:
                raise ValueError("Pending workflow differs from journal terminal state")
        finally:
            conn.close()
        conn = sqlite3.connect(self._path("checkpoints.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            conn.execute("PRAGMA trusted_schema=OFF")
            row = conn.execute("SELECT type,checkpoint FROM checkpoints WHERE thread_id=? AND checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1",
                               (self.run.run_id,)).fetchone()
            if row is None:
                raise ValueError("Pending workflow checkpoint is missing")
            serde = JsonPlusSerializer(pickle_fallback=False, allowed_json_modules=[], allowed_msgpack_modules=[])
            checkpoint = serde.loads_typed((row[0], row[1]))
            if checkpoint.get("channel_values", {}).get("payload") != workflow.model_dump(mode="json"):
                raise ValueError("Pending workflow checkpoint differs from committed state")
        finally:
            conn.close()

    def _open_database(self) -> None:
        directory = self._path("review")
        fresh = not directory.exists()
        if fresh:
            directory.mkdir()
        for suffix in ("", "-wal", "-shm", "-journal"):
            self._path("review/review.sqlite" + suffix)
        path = self._path("review/review.sqlite")
        if not fresh and not path.is_file():
            raise RunIncompleteError("Review database is missing; no implicit reinitialization")
        if fresh:
            with path.open("xb"):
                pass
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA trusted_schema=OFF")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.expected_metadata = {"version": VERSION, "parent_pins": canonical_json(self.parent_pins),
            "implementation": canonical_json(_implementation()), "initial_state": _serialized(self.initial)}
        if fresh:
            with self.conn:
                self.conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                self.conn.execute("CREATE TABLE responses (seq INTEGER PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL, response_json TEXT NOT NULL, before_id TEXT NOT NULL, after_json TEXT NOT NULL, after_id TEXT NOT NULL)")
                self.conn.executemany("INSERT INTO metadata VALUES (?, ?)", self.expected_metadata.items())

    def _check_database_paths(self) -> None:
        if self.conn is None:
            raise ValueError("Review store is closed")
        for suffix in ("", "-wal", "-shm", "-journal"):
            self._path("review/review.sqlite" + suffix)

    def _read_state(self) -> ReviewState:
        self._check_database_paths()
        if self._pins() != self.parent_pins:
            raise ValueError("Frozen parent artifacts changed")
        verify_profile_resources(self.profile, self.run.root)
        if _implementation() != _json(self.expected_metadata["implementation"].encode()):
            raise ValueError("Review implementation changed; explicit migration required")
        metadata = dict(self.conn.execute("SELECT key,value FROM metadata"))
        if metadata != self.expected_metadata:
            raise ValueError("Review state version, implementation or inputs differ; explicit migration required")
        current = self.initial
        for index, row in enumerate(self.conn.execute("SELECT seq,idempotency_key,response_json,before_id,after_json,after_id FROM responses ORDER BY seq"), 1):
            if index > 64 or row[0] != index or row[3] != current.state_id:
                raise ValueError("Review response chain is invalid")
            response = ReviewResponse.model_validate(_json(row[2].encode()))
            if row[1] != response.idempotency_key or row[2] != _serialized(response):
                raise ValueError("Stored response differs from its idempotency key")
            following = apply_review(current, response, self.profile, self.bundle)
            if row[4] != _serialized(following) or row[5] != following.state_id:
                raise ValueError("Review transition differs from validated response")
            current = following
        return current

    @property
    def state(self) -> ReviewState:
        return self._read_state()

    def submit(self, response: ReviewResponse) -> dict:
        response = ReviewResponse.model_validate(response.model_dump(mode="json"))
        raw = _serialized(response)
        self._check_database_paths()
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            current = self._read_state()
            old = self.conn.execute("SELECT response_json,after_json FROM responses WHERE idempotency_key=?",
                                    (response.idempotency_key,)).fetchone()
            if old is not None:
                if old[0] != raw:
                    raise ValueError("Idempotency key is already bound to a different response")
                # Return the current head, not an old revision that could overwrite
                # a later export. The original decision remains in history.
                return {"execution": "replayed", "state": current}
            following = apply_review(current, response, self.profile, self.bundle)
            seq = self.conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] + 1
            self.conn.execute("INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?)",
                              (seq, response.idempotency_key, raw, current.state_id,
                               _serialized(following), following.state_id))
            self._before_commit(following)
        self._after_commit(following)
        return {"execution": "applied", "state": following}

    def _before_commit(self, state: ReviewState) -> None:
        """Fault-injection seam; all review effects are still in one transaction."""

    def _after_commit(self, state: ReviewState) -> None:
        """Fault-injection seam after durable commit and before any derived export."""

    def _publish_file(self, relative: str, payload: bytes) -> None:
        target = self._path(relative)
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError("An immutable review export cannot be replaced")
            return
        pending = self._path(relative + ".pending")
        # Only a private deterministic staging file for this exact committed state
        # may be rewritten after an interrupted write; final files never change.
        with pending.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        target = self._path(relative)
        if target.exists():
            raise ValueError("Export target appeared during publication")
        # The OS-held run lease serializes all application writers. A rename
        # publishes atomically without the crash window of a temporary hard link.
        pending.rename(target)

    def export(self) -> dict:
        state = self.state
        state_hash = digest(state.model_dump(mode="json"))
        relative = "review/exports/" + state_hash
        self._path(relative).mkdir(parents=True, exist_ok=True)
        exports = {"canonical.json": _json_bytes(state.current.canonical),
                   "sidecar.json": _json_bytes({"version": "m08-review-export-1",
                       "review": state.model_dump(mode="json"), "parent_sha256": self.parent_pins,
                       "original_accounting": self.accounting, "new_model_calls": 0,
                       "new_live_requests": 0, "coverage_status": "unsupported",
                       "reviewer_identity": "self_declared_not_authenticated",
                       "expert_approval": "not_verified"})}
        manifest = {"version": "m08-review-export-1", "state_id": state.state_id,
                    "record_revision": state.current.revision_id, "status": state.status,
                    "directory": f"runs/agent/{self.run.run_id}/{relative}",
                    "artifacts": {name: {"sha256": sha256(raw), "bytes": len(raw)}
                                  for name, raw in exports.items()}}
        marker = _json_bytes(manifest)
        result = self._path(relative + "/result.json")
        if result.exists():
            if result.read_bytes() != marker or any(
                not self._path(relative + "/" + name).is_file()
                or self._path(relative + "/" + name).read_bytes() != raw
                for name, raw in exports.items()):
                raise ValueError("Completed review export differs from committed state")
            return manifest
        for name, raw in exports.items():
            self._publish_file(relative + "/" + name, raw)
        self._publish_file(relative + "/result.json", marker)
        return manifest

    def view(self) -> dict:
        state = self.state
        cases = []
        for case in state.cases:
            issue = next(item for item in self.workflow.issues if item.id == case.issue_id)
            entity = owner(state.current.entities, case.path)
            refs = case.evidence
            availability = "recorded_review_evidence"
            selection_summary = None
            if not refs:
                selection = SourceCorpus(bundles=(self.bundle,)).search(issue.query, max_chars=4000, limit=3)
                refs = tuple(item.evidence for item in selection.passages)
                availability = "supplied_source_search_for_display"
                selection_summary = selection.model_dump(mode="json", exclude={"text", "passages"})
            passages = [{"evidence": ref.model_dump(mode="json"), "text": resolve_evidence(self.bundle, ref)} for ref in refs]
            definitions = [definition.model_dump(mode="json") for event in self.workflow.events
                           if event.action is not None and getattr(event.action, "issue_id", None) == issue.id
                           for definition in event.definitions]
            proposals = [item.model_dump(mode="json") for item in state.proposals
                         if item.key in issue.proposal_ids]
            linked = {link.target_handle for link in state.current.references if link.source_handle == entity.handle}
            linked_entities = [{"entity": item.model_dump(mode="json"), "record": at_pointer(state.current.canonical, item.path)}
                               for item in state.current.entities if item.handle in linked]
            candidate_entities = []
            for item in state.proposals:
                if item.key not in issue.proposal_ids:
                    continue
                built = build_candidate(self.profile, self.bundle, state.current, item.proposal)
                if built.prepared is None:
                    continue
                candidate = built.prepared.candidate
                targets = {link.target_handle for link in candidate.references if link.source_handle == entity.handle}
                candidate_entities.append({"proposal_key": item.key, "entities": [
                    {"entity": target.model_dump(mode="json"), "record": at_pointer(candidate.canonical, target.path)}
                    for target in candidate.entities if target.handle in targets]})
            cases.append({"case_id": case.case_id, "issue_id": case.issue_id, "status": case.status,
                "review_provenance": case.review_provenance, "last_decision_id": case.last_decision_id,
                "record_revision": state.current.revision_id, "path": case.path,
                "value": at_pointer(state.current.canonical, case.path), "entity": entity.model_dump(mode="json"),
                "entity_record": at_pointer(state.current.canonical, entity.path) if entity.path else None,
                "linked_entities": linked_entities, "candidate_linked_entities": candidate_entities,
                "uncertainty": {"diagnosis": issue.diagnosis, "summary": case.summary,
                                "numerical_confidence": "not_available"},
                "passages": passages, "evidence_display": availability,
                "evidence_coverage": "displayed_passages_only", "display_selection": selection_summary,
                "definitions": definitions, "definitions_role": "representation_guidance_not_publication_evidence",
                "candidate_corrections": proposals, "candidate_availability": "recorded" if proposals else "not_recorded",
                "candidate_revision_status": ["current" if item["proposal"]["expected_revision"] == state.current.revision_id else "stale" for item in proposals],
                "relationships": [link.model_dump(mode="json") for link in state.current.references
                                  if link.source_handle == entity.handle]})
        return {"version": "m08-review-view-1", "run_id": self.run.run_id,
                "state_id": state.state_id, "record_revision": state.current.revision_id,
                "status": state.status, "cases": cases, "decisions": len(state.decisions),
                "source_warnings": [item.model_dump(mode="json") for item in self.bundle.warnings],
                "current_reviewed_assertions": [item.model_dump(mode="json") for item in state.current_assertions],
                "original_accounting": self.accounting, "new_model_calls": 0, "new_live_requests": 0,
                "coverage_status": "unsupported", "reviewer_identity": "self_declared_not_authenticated",
                "approval_scope": "explicit_reviewed_assertions_only",
                "response_options": ["approve", "edit", "unresolved"]}
