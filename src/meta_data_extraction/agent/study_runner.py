"""Frozen offline study orchestration over the unchanged M07 fixture runner.

This module does not read evaluation protocols, labels, gold, or alignment helpers.
It measures synthetic repair-policy mechanics, never natural end-to-end quality.
"""
from __future__ import annotations

from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time

from meta_data_extraction.config_validation import load_yaml_mapping

from .contracts import digest
from .evidence import sha256
from .inspect_record import InputPin, RecordRunConfig, _targets
from .inspect_workflow import WorkflowRunConfig, inspect_workflow
from . import inspect_workflow as workflow_inspection
from .paths import RunDirectory, checked_path, input_file, validate_run_id, workspace_root
from .profiles import load_profile
from .providers import ProviderResult
from .record_checks import check_structure
from .record_patches import build_candidate
from .sources import prepare_sources
from .storage import RunConflictError, RunStore, _check_request, _json_bytes, _read_completion
from .study_types import StudySpec
from .support_checks import interpret_support_response, verify_report
from .workflow_graph import _implementation as graph_implementation
from .workflow_types import WorkflowState

MAX_INPUT_BYTES = 1_000_000
MAX_ARTIFACT_BYTES = 32_000_000
LIMITATIONS = [
    "Synthetic repair-policy mechanics; no natural-paper or end-to-end performance claim.",
    "Initial records and seed judgments are shared fixtures; their production cost is not measured.",
    "Standards are disabled; approved guidance and independent empirical labels remain external gates.",
    "Deterministic mock repeats do not represent provider nondeterminism or independent publications.",
    "Raw model responses are retained only as supplied fixtures; workflow outputs retain hashes.",
]


def _read(root: Path, relative: str, roots: tuple[str, ...], maximum=MAX_INPUT_BYTES) -> bytes:
    path = input_file(root, relative, allowed_roots=roots)
    if path.stat().st_size > maximum:
        raise ValueError("Study input exceeds its size bound")
    raw = path.read_bytes()
    if len(raw) > maximum:
        raise ValueError("Study input exceeds its size bound")
    return raw


def _implementation(root: Path) -> dict:
    own = {name: sha256(_read(root, f"src/meta_data_extraction/agent/{name}.py",
                              ("src/meta_data_extraction/agent",)))
           for name in ("study_types", "study_runner", "study_cli")}
    return {"study": own, "inspection": workflow_inspection._implementation(root),
            "workflow": graph_implementation(), "python": platform.python_version(),
            "dependencies": {name: version(name) for name in (
                "pydantic", "PyYAML", "openai", "httpx", "pymupdf4llm", "pymupdf",
                "langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite", "langchain-core", "langsmith")}}


def _prepare(root: Path, spec: StudySpec) -> tuple[dict, dict]:
    pins: dict[str, str] = {}
    cache: dict[str, bytes] = {}

    def pinned(pin: InputPin, roots=("tests/agent/fixtures",)) -> bytes:
        input_file(root, pin.path, allowed_roots=roots)
        if pin.path not in cache:
            cache[pin.path] = _read(root, pin.path, roots)
        raw = cache[pin.path]
        if sha256(raw) != pin.sha256:
            raise ValueError("Study input differs from its frozen fingerprint")
        pins[pin.path] = pin.sha256
        return raw

    configs = {}
    publications = []
    source_owners = {}
    common_budget = None
    common_profile = None
    for publication in spec.publications:
        reference = None
        frozen_publication = None
        for policy in spec.variants:
            config_pin = publication.workflows[policy]
            config = WorkflowRunConfig.model_validate(load_yaml_mapping(
                pinned(config_pin, ("configs/agent",)).decode("utf-8"), source="study workflow config"))
            if config.policy != policy:
                raise ValueError("Study variant differs from its workflow selector")
            budget = config.budget.model_dump(mode="json", exclude={"max_model_calls"})
            if common_budget is None:
                common_budget = budget
            elif common_budget != budget:
                raise ValueError("Study variants must share all non-call budgets")
            fairness = {"seed": config.seed.model_dump(mode="json"),
                        "checker_responses": [p.model_dump(mode="json") for p in config.checker_responses],
                        "raw_retention": config.raw_retention}
            if reference is None:
                reference = fairness
            elif reference != fairness:
                raise ValueError("Study variants must share seed, checker, and tool access")
            seed = RecordRunConfig.model_validate(load_yaml_mapping(
                pinned(config.seed, ("configs/agent",)).decode("utf-8"), source="study seed config"))
            if seed.patch is None or seed.before_response is None or not seed.patch_policy.accept_model_judgments:
                raise ValueError("Study seed requires its explicit synthetic correction and checks")
            pinned(seed.source)
            raw_record = pinned(seed.record)
            before = pinned(seed.before_response).decode("utf-8")
            if seed.after_response is not None:
                pinned(seed.after_response)
            for pin in (*config.checker_responses, *config.policy_responses):
                pinned(pin).decode("utf-8")
            profile = load_profile(root, seed.schema_names)
            if common_profile is None:
                common_profile = profile.profile_id
            elif common_profile != profile.profile_id:
                raise ValueError("Study publications must share the same frozen profile")
            prepared = prepare_sources(root, [seed.source.path])[0]
            if prepared.bundle is None or prepared.bundle.source_sha256 != seed.source.sha256:
                raise ValueError("Study source could not be prepared or changed")
            bundle = prepared.bundle
            record = check_structure(profile, bundle, load_yaml_mapping(raw_record.decode("utf-8"), source="study record"))
            if record.snapshot is None:
                raise ValueError("Study seed must be structurally valid")
            support = interpret_support_response(profile, record.snapshot, bundle, _targets(bundle, seed.targets),
                ProviderResult(status="ok", mode="mock", finish_reason="stop", content=before),
                model_name=workflow_inspection.SEED_MODEL, raw_retention="hash_only")
            verify_report(support, profile, record.snapshot, bundle, root=root)
            candidate = build_candidate(profile, bundle, record.snapshot, seed.patch)
            _targets(bundle, seed.after_targets)
            if (candidate.prepared is None or support.status != "succeeded" or len(support.judgments) != 2
                    or any(j.verdict != "contradicted" for j in support.judgments)
                    or {j.path for j in support.judgments} != set(candidate.prepared.changed_paths)):
                raise ValueError("Study requires the reusable M07 two-assertion synthetic seed")
            frozen_publication = {"publication_id": publication.publication_id, "split": publication.split,
                "data_kind": publication.data_kind, "source": seed.source.model_dump(mode="json"),
                "source_snapshot_sha256": digest(bundle.model_dump(mode="json")),
                "profile": profile.model_dump(mode="json"), "initial_revision": record.snapshot.revision_id,
                "seed_support_sha256": digest(support.model_dump(mode="json")),
                "standards_resources": [], "review_rule": "pending_is_not_approved"}
            configs[(publication.publication_id, policy)] = config
        source_hash = frozen_publication["source"]["sha256"]
        if source_hash in source_owners:
            raise ValueError("The same source cannot count as multiple independent publications")
        source_owners[source_hash] = publication.publication_id
        publications.append(frozen_publication)
    maximum = len(spec.publications) * len(spec.variants) * spec.repeats * sum(spec.caps)
    return {"version": "m09-study-dry-run-1", "spec": spec.model_dump(mode="json"),
            "spec_sha256": digest(spec.model_dump(mode="json")), "readiness": "ready",
            "estimated_cells": len(spec.publications) * len(spec.variants) * len(spec.caps) * spec.repeats,
            "maximum_model_calls": maximum, "max_total_model_calls": spec.max_total_model_calls,
            "live_requests": 0, "frozen": {"input_sha256": pins, "publications": publications,
                "implementation": _implementation(root), "common_budget": common_budget},
            "limitations": LIMITATIONS}, configs


def dry_run(root: Path | str, spec: StudySpec) -> dict:
    """Validate every planned input without creating runs or dispatching providers."""
    root = workspace_root(root)
    spec = StudySpec.model_validate(spec.model_dump(mode="json"))
    return _prepare(root, spec)[0]


def expected_cells(spec: StudySpec, run_id: str) -> list[dict]:
    validate_run_id(run_id)
    spec = StudySpec.model_validate(spec.model_dump(mode="json"))
    spec_hash = digest(spec.model_dump(mode="json"))
    result = []
    for publication in spec.publications:
        for policy in spec.variants:
            for cap in spec.caps:
                for repeat in range(spec.repeats):
                    identity = {"publication_id": publication.publication_id, "variant": policy,
                                "cap": cap, "repeat": repeat}
                    cell_id = "cell-" + digest(identity)[:24]
                    child = "m09-cell-" + digest({"study": run_id, "spec": spec_hash, **identity})[:40]
                    if child == run_id:
                        raise ValueError("Study and child run IDs must differ")
                    result.append({"cell_id": cell_id, **identity, "split": publication.split,
                                   "data_kind": publication.data_kind, "child_run_id": child})
    return result


def _artifacts(root: Path, child: str) -> dict[str, dict]:
    result = {}
    for name in ("request.json", "result.json", "canonical.json", "sidecar.json", "checkpoints.sqlite", "dispatch.sqlite"):
        relative = f"runs/agent/{child}/{name}"
        path = checked_path(root, relative, allowed_roots=(f"runs/agent/{child}",))
        if path.exists():
            raw = _read(root, relative, (f"runs/agent/{child}",), MAX_ARTIFACT_BYTES)
            result[name] = {"path": relative, "sha256": sha256(raw)}
    return result


def _resources(state: WorkflowState) -> dict:
    return {"model_calls": state.model_calls, "uncertain_model_calls": state.uncertain_model_calls,
            "input_tokens_known": state.input_tokens, "output_tokens_known": state.output_tokens,
            "usage_unknown": state.usage_unknown, "elapsed_ms": state.elapsed_ms}


def _accounting(cells: list[dict]) -> dict:
    resources = [cell["original_resources"] for cell in cells]
    return {"original_model_calls_known": sum(r["model_calls"] or 0 for r in resources),
        "uncertain_model_calls": sum(r["uncertain_model_calls"] for r in resources),
        "new_model_calls_known": sum(c["invocation_resources"]["new_model_calls"] or 0 for c in cells),
        "new_model_calls_unknown": any(c["invocation_resources"]["new_model_calls"] is None for c in cells),
        "input_tokens_known": sum(r["input_tokens_known"] or 0 for r in resources),
        "output_tokens_known": sum(r["output_tokens_known"] or 0 for r in resources),
        "usage_unknown": any(r["usage_unknown"] for r in resources), "new_live_requests": 0,
        "cost": None, "cost_status": "not_applicable_mock"}


def _verify_completed(root: Path, result: dict, spec: StudySpec, run_id: str, configs: dict, frozen: dict) -> None:
    if result.get("version") != "m09-study-result-1" or result.get("spec") != spec.model_dump(mode="json"):
        raise RunConflictError("Study result version or scope changed")
    if result.get("spec_sha256") != digest(spec.model_dump(mode="json")):
        raise RunConflictError("Study result fingerprint changed")
    cells = result.get("cells", [])
    expected = expected_cells(spec, run_id)
    if len(cells) != len(expected):
        raise RunConflictError("Study result omitted planned cells")
    for cell, coordinate in zip(cells, expected, strict=True):
        if any(cell.get(key) != value for key, value in coordinate.items()):
            raise RunConflictError("Study result changed cell identity")
        if cell.get("artifacts") != _artifacts(root, coordinate["child_run_id"]):
            raise RunConflictError("Study child artifacts changed")
        resources = cell.get("original_resources", {})
        invocation = cell.get("invocation_resources", {})
        if (set(resources) != {"model_calls", "uncertain_model_calls", "input_tokens_known", "output_tokens_known", "usage_unknown", "elapsed_ms"}
                or type(resources["usage_unknown"]) is not bool
                or type(resources["uncertain_model_calls"]) is not int or resources["uncertain_model_calls"] < 0):
            raise RunConflictError("Study original resource fields changed")
        for name in ("model_calls", "input_tokens_known", "output_tokens_known"):
            if resources[name] is not None and (type(resources[name]) is not int or resources[name] < 0):
                raise RunConflictError("Study known resource subtotal changed")
        if resources["elapsed_ms"] is not None and (type(resources["elapsed_ms"]) not in {int, float}
                or not math.isfinite(resources["elapsed_ms"]) or resources["elapsed_ms"] < 0):
            raise RunConflictError("Study original timing changed")
        if (set(invocation) != {"new_model_calls", "new_live_requests", "elapsed_ms"}
                or invocation["new_live_requests"] != 0 or type(invocation["new_live_requests"]) is not int
                or type(invocation["elapsed_ms"]) not in {int, float}
                or not math.isfinite(invocation["elapsed_ms"]) or invocation["elapsed_ms"] < 0):
            raise RunConflictError("Study invocation accounting changed")
        if cell.get("status") == "completed":
            aliases = {"canonical": "canonical.json", "sidecar": "sidecar.json", "result_artifact": "result.json"}
            if any(cell.get(key) is None or cell.get(key) != cell["artifacts"].get(name) for key, name in aliases.items()):
                raise RunConflictError("Study prediction aliases changed")
            child = RunDirectory(root, coordinate["child_run_id"])
            envelope = json.loads(_read(root, f"runs/agent/{child.run_id}/request.json", (f"runs/agent/{child.run_id}",), MAX_ARTIFACT_BYTES))
            request = envelope["request"]
            request_hash = sha256(_json_bytes(request))
            if envelope != {"version": 1, "request_sha256": request_hash, "request": request}:
                raise RunConflictError("Child request fingerprint changed")
            config = configs[(coordinate["publication_id"], coordinate["variant"])]
            effective = {**config.model_dump(mode="json"), "run_id": child.run_id,
                "budget": {**config.budget.model_dump(mode="json"), "max_model_calls": coordinate["cap"]}}
            if (request.get("version") != "m07-run-1" or request.get("policy") != coordinate["variant"]
                    or request.get("budget") != effective["budget"]
                    or request.get("declared", {}).get("config") != effective
                    or request.get("implementation") != frozen["frozen"]["implementation"]["workflow"]):
                raise RunConflictError("Study child used a different frozen configuration")
            child_result = _read_completion(child, request_hash)["result"]
            state = WorkflowState.model_validate(child_result["state"])
            publication = next(p for p in frozen["frozen"]["publications"] if p["publication_id"] == coordinate["publication_id"])
            if (state.initial_revision != publication["initial_revision"]
                    or state.record.source_sha256 != publication["source"]["sha256"]
                    or state.record.profile_id != publication["profile"]["profile_id"]
                    or state.model_calls + state.uncertain_model_calls > coordinate["cap"]):
                raise RunConflictError("Study child record or cap differs")
            canonical = json.loads(_read(root, cell["canonical"]["path"], (f"runs/agent/{child.run_id}",), MAX_ARTIFACT_BYTES))
            sidecar = json.loads(_read(root, cell["sidecar"]["path"], (f"runs/agent/{child.run_id}",), MAX_ARTIFACT_BYTES))
            if canonical != state.record.canonical or sidecar.get("workflow") != state.model_dump(mode="json"):
                raise RunConflictError("Study child exports differ from their committed state")
            if (resources != _resources(state) or cell.get("workflow_status") != state.status
                    or cell.get("record_revision") != state.record.revision_id
                    or cell.get("unresolved_issues") != sum(i.status != "resolved" for i in state.issues)
                    or cell.get("truncated") != any(p.finish_reason == "length" for e in state.events for p in e.providers)
                    or cell.get("error_code") is not None):
                raise RunConflictError("Study child outcome or resource metadata changed")
            if (cell.get("execution") not in {"executed", "resumed", "cached_replay"}
                    or type(invocation["new_model_calls"]) is not int
                    or not 0 <= invocation["new_model_calls"] <= state.model_calls
                    or (cell["execution"] == "executed" and invocation["new_model_calls"] != state.model_calls)
                    or (cell["execution"] == "cached_replay" and invocation["new_model_calls"] != 0)):
                raise RunConflictError("Study replay consumption changed")
        elif cell.get("status") == "failed":
            if any(cell.get(key) is not None for key in ("canonical", "sidecar", "result_artifact")) or not cell.get("error_code"):
                raise RunConflictError("Failed study cell cannot declare a partial prediction")
            if resources.get("model_calls") is None:
                if (resources != {"model_calls": None, "uncertain_model_calls": coordinate["cap"],
                        "input_tokens_known": None, "output_tokens_known": None, "usage_unknown": True, "elapsed_ms": None}
                        or invocation["new_model_calls"] is not None):
                    raise RunConflictError("Unknown failed-cell usage cannot become known zero")
            elif (type(resources.get("model_calls")) is not int or resources["model_calls"] < 0
                    or type(resources.get("uncertain_model_calls")) is not int or resources["uncertain_model_calls"] < 0
                    or resources["model_calls"] + resources["uncertain_model_calls"] > coordinate["cap"]):
                raise RunConflictError("Failed-cell observed usage exceeds its cap")
            elif "result.json" in cell["artifacts"]:
                raw = _read(root, cell["artifacts"]["result.json"]["path"],
                            (f"runs/agent/{coordinate['child_run_id']}",), MAX_ARTIFACT_BYTES)
                child_state = WorkflowState.model_validate(json.loads(raw)["result"]["state"])
                if resources != _resources(child_state):
                    raise RunConflictError("Failed-cell observed usage differs from its saved result")
        else:
            raise RunConflictError("Study cell has an unknown processing status")
    if result.get("accounting") != _accounting(cells):
        raise RunConflictError("Study aggregate accounting changed")
    if result.get("budget") != {"max_total_model_calls": spec.max_total_model_calls,
                                "reserved_model_calls": frozen["maximum_model_calls"]}:
        raise RunConflictError("Study frozen budget changed")


def run_study(root: Path | str, spec: StudySpec, *, run_id: str, replay_only: bool = False) -> dict:
    """Execute every frozen synthetic cell once; incomplete parent runs fail closed.

    Completed replay verifies all child artifacts without dispatch. Ambiguous child
    failures reserve their entire cap and retain unknown resource totals.
    """
    root = workspace_root(root)
    validate_run_id(run_id)
    if type(replay_only) is not bool:
        raise TypeError("replay_only must be an explicit boolean")
    spec = StudySpec.model_validate(spec.model_dump(mode="json"))
    frozen, configs = _prepare(root, spec)
    planned = expected_cells(spec, run_id)
    request = {"version": "m09-study-run-1", "spec": spec.model_dump(mode="json"),
               "spec_sha256": frozen["spec_sha256"], "dry_run": frozen, "cells": planned}
    if replay_only:
        run = RunDirectory(root, run_id)
        request_hash = sha256(_json_bytes(request))
        _check_request(run, request, request_hash)
        store = RunStore(run, _read_completion(run, request_hash)["result"], request, request_hash)
    else:
        store = RunStore.begin(root, run_id, request)
    if store.completed is not None:
        try:
            _verify_completed(root, store.completed, spec, run_id, configs, frozen)
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise RunConflictError("Completed study no longer validates") from error
        return {"execution": "cached_replay", "new_model_calls": 0, "new_live_requests": 0, "result": store.completed}
    cells = []
    new_known = 0
    invocation_unknown = False
    for coordinate in planned:
        config = configs[(coordinate["publication_id"], coordinate["variant"])]
        config = WorkflowRunConfig.model_validate({**config.model_dump(mode="json"),
            "run_id": coordinate["child_run_id"],
            "budget": {**config.budget.model_dump(mode="json"), "max_model_calls": coordinate["cap"]}})
        started = time.perf_counter()
        cell = {**coordinate, "status": "failed", "execution": "failed", "workflow_status": None,
            "record_revision": None, "canonical": None, "sidecar": None, "result_artifact": None,
            "original_resources": {"model_calls": None, "uncertain_model_calls": coordinate["cap"],
                "input_tokens_known": None, "output_tokens_known": None, "usage_unknown": True, "elapsed_ms": None},
            "invocation_resources": {"new_model_calls": None, "new_live_requests": 0, "elapsed_ms": None},
            "unresolved_issues": None, "truncated": None, "error_code": None}
        try:
            # Recheck every frozen input and implementation before any next child.
            current, _ = _prepare(root, spec)
            if current != frozen:
                raise ValueError("Study frozen inputs changed")
            report = inspect_workflow(root, config, run_id=coordinate["child_run_id"])
            state = report["state"]
            if report["new_live_requests"] != 0:
                raise ValueError("Synthetic study cannot dispatch live requests")
            cell.update(status="completed", execution=report["execution"], workflow_status=state.status,
                record_revision=state.record.revision_id,
                original_resources=_resources(state),
                invocation_resources={"new_model_calls": report["new_model_calls"], "new_live_requests": 0, "elapsed_ms": None},
                unresolved_issues=sum(issue.status != "resolved" for issue in state.issues),
                truncated=any(p.finish_reason == "length" for event in state.events for p in event.providers))
            new_known += report["new_model_calls"]
        except Exception as error:
            cell["error_code"] = "cell_execution_failed_" + type(error).__name__
            invocation_unknown = True
        cell["invocation_resources"]["elapsed_ms"] = (time.perf_counter() - started) * 1000
        try:
            cell["artifacts"] = _artifacts(root, coordinate["child_run_id"])
        except (OSError, ValueError):
            cell["artifacts"] = {}
            cell["status"] = "failed"
            cell["error_code"] = "cell_artifacts_unavailable"
        if cell["status"] == "completed":
            for field, name in (("canonical", "canonical.json"), ("sidecar", "sidecar.json"), ("result_artifact", "result.json")):
                cell[field] = cell["artifacts"].get(name)
            if any(cell[name] is None for name in ("canonical", "sidecar", "result_artifact")):
                cell["status"] = "failed"
                cell["error_code"] = "cell_exports_incomplete"
                cell.update(canonical=None, sidecar=None, result_artifact=None)
        cells.append(cell)
    accounting = _accounting(cells)
    result = {"version": "m09-study-result-1", "spec": spec.model_dump(mode="json"),
        "spec_sha256": frozen["spec_sha256"], "cells": cells,
        "budget": {"max_total_model_calls": spec.max_total_model_calls,
                   "reserved_model_calls": frozen["maximum_model_calls"]},
        "accounting": accounting, "limitations": LIMITATIONS}
    store.complete(result, canonical=None, sidecar={"study_spec": spec.model_dump(mode="json"),
                   "frozen": frozen["frozen"]}, raw_response=None, retain_raw=False)
    return {"execution": "executed", "new_model_calls": None if invocation_unknown else new_known,
            "new_live_requests": 0, "result": result}
