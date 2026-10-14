"""Frozen M09 collection, using the unchanged M03 scorer only offline.

Engineering fixtures establish reproducibility, never independent accuracy or
review-effort evidence. A completed workflow is not necessarily fully supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from meta_data_extraction.agent.contracts import canonical_json
from meta_data_extraction.agent.paths import RunDirectory, input_file, validate_run_id, workspace_root
from meta_data_extraction.agent.storage import RunStore, _check_request, _digest, _json_bytes, _read_completion
from meta_data_extraction.offline.manifest import (
    ArtifactRef, OfflineModel, PredictionRef, Publication, SplitManifest, read_artifact, strict_json,
)
from meta_data_extraction.offline.runner import run_offline
from meta_data_extraction.offline.scoring import EvaluatorSnapshot, summaries, verify_evaluator


class StudyEvaluationProtocol(OfflineModel):
    version: Literal["m09-evaluation-1"] = "m09-evaluation-1"
    stage: Literal["synthetic_engineering", "pilot", "confirmatory"]
    description: str = Field(min_length=1)
    primary_field: str = Field(min_length=1)
    primary_metric: Literal["f1", "precision", "recall", "mean_score", "score"]
    minimum_effect: float = Field(allow_inf_nan=False)
    baseline_variant: str = Field(min_length=1)
    variants: tuple[str, ...] = Field(min_length=1, max_length=3)
    caps: tuple[int, ...] = Field(min_length=1, max_length=4)
    repeats: int = Field(ge=1, le=3, strict=True)
    publications: tuple[Publication, ...] = Field(min_length=1, max_length=20)
    evaluator: EvaluatorSnapshot
    denominator_policy: Literal["last_valid_canonical_all_planned_cells_visible"] = "last_valid_canonical_all_planned_cells_visible"
    pairing_policy: Literal["matched_repeat_then_publication_mean"] = "matched_repeat_then_publication_mean"
    missingness_policy: Literal["unavailable_is_null"] = "unavailable_is_null"
    support_policy: Literal["positive_known_gold_support_only"] = "positive_known_gold_support_only"
    bootstrap_seed: int = Field(default=0, ge=0, strict=True)
    bootstrap_samples: int = Field(default=1000, ge=100, le=2000, strict=True)

    @field_validator("caps", mode="before")
    @classmethod
    def _caps(cls, value):
        if not isinstance(value, (tuple, list)) or any(type(cap) is not int for cap in value):
            raise ValueError("Study caps must be explicit integers")
        return value

    @model_validator(mode="after")
    def _scope(self) -> StudyEvaluationProtocol:
        if len(set(self.variants)) != len(self.variants) or self.baseline_variant not in self.variants:
            raise ValueError("Study variants must be distinct and include the baseline")
        if any(type(cap) is not int or cap < 0 or cap > 32 for cap in self.caps) or len(set(self.caps)) != len(self.caps):
            raise ValueError("Study caps must be distinct bounded integers")
        if any(p.predictions for p in self.publications):
            raise ValueError("Prediction inputs are collected after the protocol is frozen")
        if any(p.data_kind != "synthetic" for p in self.publications) and self.stage == "synthetic_engineering":
            raise ValueError("Engineering stage accepts explicitly synthetic publications only")
        # Reuse the established split/lineage integrity checks without opening gold.
        self.split_manifest()
        if self.primary_field not in {item.field for item in self.evaluator.field_metrics}:
            raise ValueError("Primary field is absent from the frozen evaluator")
        return self

    def split_manifest(self, publications: tuple[Publication, ...] | None = None) -> SplitManifest:
        return SplitManifest(
            experiment_id="m09-protocol", description=self.description,
            primary_outcome=self.primary_field + ":" + self.primary_metric,
            variants=(self.baseline_variant, *(v for v in self.variants if v != self.baseline_variant)),
            publications=self.publications if publications is None else publications,
            split_notes="Publication identities and all derivatives are frozen by the M09 protocol.",
            held_out_policy="Final-test gold requires separate explicit offline scoring authorization.",
        )


def protocol_digest(protocol: StudyEvaluationProtocol) -> str:
    protocol = StudyEvaluationProtocol.model_validate(protocol)
    return hashlib.sha256(canonical_json(protocol.model_dump(mode="json")).encode()).hexdigest()


def load_protocol(root: Path | str, relative: str) -> StudyEvaluationProtocol:
    path = input_file(root, relative, allowed_roots=(
        "configs/agent", "configs/offline", "tests/offline/fixtures", "runs/agent",
    ))
    if path.stat().st_size > 2_000_000:
        raise ValueError("Evaluation protocol exceeds the size limit")
    raw = path.read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError("Evaluation protocol exceeds the size limit")
    value = strict_json(raw)
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 32:
            raise ValueError("Evaluation protocol exceeds the nesting limit")
        if isinstance(item, (list, dict)):
            pending.extend((child, depth + 1) for child in (item.values() if isinstance(item, dict) else item))
    return StudyEvaluationProtocol.model_validate_json(canonical_json(value))


def _completed(root: Path, run_id: str) -> tuple[dict, dict, dict[str, str]]:
    run = RunDirectory(root, validate_run_id(run_id))
    path = input_file(root, f"runs/agent/{run_id}/request.json", allowed_roots=(f"runs/agent/{run_id}",))
    request = strict_json(path.read_bytes())
    if not isinstance(request, dict) or set(request) != {"version", "request_sha256", "request"}:
        raise ValueError("Invalid completed study request")
    content = request["request"]
    request_hash = _digest(_json_bytes(content))
    _check_request(run, content, request_hash)
    envelope = _read_completion(run, request_hash)
    pins = {}
    for name in ("request.json", "result.json", *envelope["artifacts"]):
        target = input_file(root, f"runs/agent/{run_id}/{name}", allowed_roots=(f"runs/agent/{run_id}",))
        pins[target.relative_to(root).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    return content, envelope["result"], pins


def _pin(root: Path, raw: dict, *, child: str, filename: str) -> ArtifactRef:
    ref = ArtifactRef.model_validate(raw)
    if ref.path != f"runs/agent/{child}/{filename}":
        raise ValueError("Study child artifact must use its exact completed run path")
    read_artifact(root, ref, role="sidecar" if filename == "sidecar.json" else "prediction")
    return ref


def _scope(parent: dict, protocol: StudyEvaluationProtocol) -> tuple[Any, list[tuple]]:
    # One-way dependency: offline collection imports runtime declarations; runtime
    # never imports this module or opens an evaluation protocol.
    from meta_data_extraction.agent.study_types import StudySpec

    if parent.get("version") != "m09-study-result-1":
        raise ValueError("Unrecognized completed study version")
    spec = StudySpec.model_validate(parent.get("spec"))
    actual_digest = hashlib.sha256(canonical_json(spec.model_dump(mode="json")).encode()).hexdigest()
    if parent.get("spec_sha256") != actual_digest:
        raise ValueError("Study specification fingerprint differs")
    if spec.evaluation_protocol_sha256 != protocol_digest(protocol):
        raise ValueError("Study evaluation protocol was not frozen before execution")
    if set(spec.variants) != set(protocol.variants) or tuple(spec.caps) != protocol.caps or spec.repeats != protocol.repeats:
        raise ValueError("Evaluation scope differs from the runtime study")
    declared = {p.publication_id: p for p in protocol.publications}
    if len(declared) != len(protocol.publications) or set(declared) != {p.publication_id for p in spec.publications}:
        raise ValueError("Publication scope differs from the runtime study")
    for publication in spec.publications:
        expected = declared[publication.publication_id]
        if (publication.split, publication.data_kind) != (expected.split, expected.data_kind):
            raise ValueError("Publication role differs from the runtime study")
    # Current implementation performs repair-policy diagnostics only. Do not turn
    # a future label into a supported natural/confirmatory execution mode.
    if spec.scope != "synthetic_repair_policy" or protocol.stage != "synthetic_engineering":
        raise ValueError("This study implementation supports synthetic engineering only")
    expected = [(p.publication_id, v, cap, repeat) for p in spec.publications
                for v in spec.variants for cap in spec.caps for repeat in range(spec.repeats)]
    return spec, expected


def _workflow_config(root: Path, spec: Any, coordinate: dict) -> dict:
    from meta_data_extraction.agent.inspect_workflow import load_config
    publication = next(p for p in spec.publications if p.publication_id == coordinate["publication_id"])
    config = load_config(root, publication.workflows[coordinate["variant"]].path).model_dump(mode="json")
    return {**config, "run_id": coordinate["child_run_id"],
            "budget": {**config["budget"], "max_model_calls": coordinate["cap"]}}


def _validate_resources(cell: dict, cap: int) -> None:
    resources = cell.get("original_resources")
    if not isinstance(resources, dict) or set(resources) != {
        "model_calls", "uncertain_model_calls", "input_tokens_known", "output_tokens_known", "usage_unknown", "elapsed_ms",
    }:
        raise ValueError("Invalid original resource accounting")
    if type(resources["usage_unknown"]) is not bool or type(resources["uncertain_model_calls"]) is not int or resources["uncertain_model_calls"] < 0:
        raise ValueError("Invalid original usage uncertainty")
    for name in ("model_calls", "input_tokens_known", "output_tokens_known"):
        value = resources[name]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Invalid original known resource subtotal")
    elapsed = resources["elapsed_ms"]
    if elapsed is not None and (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError("Invalid original elapsed time")
    invocation = cell["invocation_resources"]
    if resources["model_calls"] is None:
        if (cell["status"] != "failed" or resources != {"model_calls": None, "uncertain_model_calls": cap,
                "input_tokens_known": None, "output_tokens_known": None, "usage_unknown": True, "elapsed_ms": None}
                or invocation["new_model_calls"] is not None):
            raise ValueError("Unknown failed usage requires the full uncertain cap and unknown calls")
    else:
        if resources["model_calls"] + resources["uncertain_model_calls"] > cap:
            raise ValueError("Known workflow usage exceeds the study cap")
        new_calls = invocation["new_model_calls"]
        if new_calls is not None and new_calls > resources["model_calls"]:
            raise ValueError("Invocation calls exceed original calls")
        if ((cell["execution"] == "executed" and new_calls != resources["model_calls"])
                or (cell["execution"] == "cached_replay" and new_calls != 0)):
            raise ValueError("Executed or cached invocation accounting differs")


def _cells(root: Path, parent: dict, spec: Any, expected: list[tuple], protocol: StudyEvaluationProtocol,
           planned: list[dict], frozen: dict) -> tuple[dict, dict[str, str]]:
    cells = parent.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Invalid study cell table")
    indexed, pins, child_ids = {}, {}, set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Invalid study cell")
        if type(cell.get("cap")) is not int or type(cell.get("repeat")) is not int:
            raise ValueError("Study cap and repeat require integer coordinates")
        key = (cell.get("publication_id"), cell.get("variant"), cell.get("cap"), cell.get("repeat"))
        if key not in expected or key in indexed:
            raise ValueError("Unexpected or duplicate study cell")
        coordinate = next(item for item in planned if (item["publication_id"], item["variant"], item["cap"], item["repeat"]) == key)
        if any(cell.get(name) != value for name, value in coordinate.items()):
            raise ValueError("Study cell identity differs from its frozen execution plan")
        indexed[key] = cell
        child = validate_run_id(cell["child_run_id"])
        if child in child_ids:
            raise ValueError("A child run cannot represent more than one study cell")
        child_ids.add(child)
        if cell.get("status") not in {"completed", "failed"}:
            raise ValueError("Unrecognized study execution status")
        if cell.get("execution") not in {"executed", "resumed", "cached_replay", "failed"}:
            raise ValueError("Unrecognized study invocation mode")
        if cell["status"] == "completed" and (cell["execution"] == "failed" or cell.get("error_code") is not None):
            raise ValueError("Completed study cell cannot declare an execution failure")
        invocation = cell.get("invocation_resources")
        if not isinstance(invocation, dict) or set(invocation) != {"new_model_calls", "new_live_requests", "elapsed_ms"}:
            raise ValueError("Invalid invocation accounting")
        if type(invocation["new_live_requests"]) is not int or invocation["new_live_requests"] != 0:
            raise ValueError("Synthetic collection requires zero live requests")
        for name in ("new_model_calls", "elapsed_ms"):
            value = invocation[name]
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                                      or (name == "new_model_calls" and type(value) is not int)):
                raise ValueError("Invalid invocation accounting value")
        _validate_resources(cell, key[2])
        declared_artifacts = cell.get("artifacts")
        if not isinstance(declared_artifacts, dict) or not set(declared_artifacts).issubset({
            "request.json", "result.json", "canonical.json", "sidecar.json", "checkpoints.sqlite", "dispatch.sqlite",
        }):
            raise ValueError("Invalid study child artifact inventory")
        from meta_data_extraction.agent.study_runner import _artifacts
        if declared_artifacts != _artifacts(root, child):
            raise ValueError("Study child artifact inventory changed")
        for filename, raw_ref in declared_artifacts.items():
            ref = _pin(root, raw_ref, child=child, filename=filename)
            pins[ref.path] = ref.sha256
        artifact_keys = ("canonical", "sidecar", "result_artifact")
        present = [cell.get(name) is not None for name in artifact_keys]
        if any(present) and not all(present):
            raise ValueError("Partially declared study completion artifacts")
        if not all(present):
            if cell.get("status") == "completed":
                raise ValueError("A completed study cell requires all artifacts")
            if not cell.get("error_code"):
                raise ValueError("A failed study cell requires an explicit failure code")
            if cell["original_resources"]["model_calls"] is not None and "result.json" in declared_artifacts:
                from meta_data_extraction.agent.study_runner import _resources
                from meta_data_extraction.agent.workflow_types import WorkflowState
                _, saved, saved_pins = _completed(root, child)
                saved_state = WorkflowState.model_validate(saved.get("state"))
                if canonical_json(cell["original_resources"]) != canonical_json(_resources(saved_state)):
                    raise ValueError("Failed workflow known usage differs from its saved child state")
                pins.update(saved_pins)
            continue
        if set(declared_artifacts) != {"request.json", "result.json", "canonical.json", "sidecar.json", "checkpoints.sqlite", "dispatch.sqlite"}:
            raise ValueError("Completed workflow requires all checkpoint and export pins")
        child_request, child_result, child_pins = _completed(root, child)
        for name, filename in zip(artifact_keys, ("canonical.json", "sidecar.json", "result.json")):
            ref = _pin(root, cell[name], child=child, filename=filename)
            if child_pins[ref.path] != ref.sha256 or declared_artifacts[filename] != ref.model_dump():
                raise ValueError("Study child completion fingerprint differs")
        from meta_data_extraction.agent.workflow_types import WorkflowState
        state = WorkflowState.model_validate(child_result.get("state"))
        if child_result.get("version") != "m07-result-1" or state.status == "running":
            raise ValueError("Only completed M07 workflow predictions are currently supported")
        if cell.get("workflow_status") != state.status or cell.get("record_revision") != state.record.revision_id:
            raise ValueError("Study workflow status or revision differs from child completion")
        expected_budget = {**frozen["frozen"]["common_budget"], "max_model_calls": key[2]}
        declared = child_request.get("declared", {})
        config = declared.get("config", {})
        if (child_request.get("version") != "m07-run-1" or child_request.get("policy") != key[1]
                or child_result.get("policy") != key[1] or child_request.get("budget") != expected_budget
                or config.get("policy") != key[1] or config.get("budget") != expected_budget
                or config.get("run_id") != child):
            raise ValueError("Workflow policy or budget differs from its study cell")
        if config != _workflow_config(root, spec, coordinate):
            raise ValueError("Workflow configuration differs from the frozen selected variant")
        frozen_publication = next(p for p in frozen["frozen"]["publications"] if p["publication_id"] == key[0])
        initial = WorkflowState.model_validate(child_request.get("initial_state"))
        if (initial.record.revision_id != frozen_publication["initial_revision"]
                or initial.record.profile_id != frozen_publication["profile"]["profile_id"]
                or state.record.profile_id != initial.record.profile_id
                or declared.get("source_snapshot_sha256") != frozen_publication["source_snapshot_sha256"]):
            raise ValueError("Workflow profile or initial source representation differs from its study")
        input_pins = declared.get("input_sha256")
        if not isinstance(input_pins, dict) or any(frozen["frozen"]["input_sha256"].get(name) != value for name, value in input_pins.items()):
            raise ValueError("Workflow fixture inputs differ from the study freeze")
        canonical = strict_json(read_artifact(root, ArtifactRef.model_validate(cell["canonical"]), role="prediction"))
        if canonical_json(canonical) != state.record.canonical_json:
            raise ValueError("Study canonical prediction differs from child state")
        sidecar = strict_json(read_artifact(root, ArtifactRef.model_validate(cell["sidecar"]), role="sidecar"))
        if (not isinstance(sidecar, dict) or set(sidecar) != {"workflow", "source", "profile"}
                or canonical_json(sidecar["workflow"]) != canonical_json(state.model_dump(mode="json"))
                or canonical_json(sidecar["profile"]) != canonical_json(frozen_publication["profile"])
                or hashlib.sha256(canonical_json(sidecar["source"]).encode()).hexdigest() != frozen_publication["source_snapshot_sha256"]):
            raise ValueError("Workflow sidecar differs from its state and frozen source/profile")
        source_refs = next(p.sources for p in protocol.publications if p.publication_id == key[0])
        if not any((ref.path, ref.sha256) == (state.record.source_id, state.record.source_sha256) for ref in source_refs):
            raise ValueError("Study prediction source differs from the offline publication")
        observed = {"model_calls": state.model_calls, "uncertain_model_calls": state.uncertain_model_calls,
                    "input_tokens_known": state.input_tokens, "output_tokens_known": state.output_tokens,
                    "usage_unknown": state.usage_unknown, "elapsed_ms": state.elapsed_ms}
        if canonical_json(cell.get("original_resources")) != canonical_json(observed):
            raise ValueError("Study resource accounting differs from child state")
        if state.model_calls + state.uncertain_model_calls > key[2]:
            raise ValueError("Study child exceeded its declared model-call cap")
        if cell.get("unresolved_issues") != sum(issue.status != "resolved" for issue in state.issues):
            raise ValueError("Study unresolved-issue count differs from child state")
        if any(provider.mode != "mock" for event in state.events for provider in event.providers):
            raise ValueError("Synthetic study cannot collect live provider observations")
        observations = [provider for event in state.events for provider in event.providers]
        if state.model_calls != len(observations):
            raise ValueError("Workflow call count differs from recorded provider observations")
        if (state.input_tokens != sum(p.input_tokens or 0 for p in observations)
                or state.output_tokens != sum(p.output_tokens or 0 for p in observations)
                or state.usage_unknown != (bool(state.uncertain_model_calls) or any(p.input_tokens is None or p.output_tokens is None for p in observations))):
            raise ValueError("Workflow token accounting differs from recorded provider observations")
        if type(cell.get("truncated")) is not bool or cell["truncated"] != any(p.finish_reason == "length" for p in observations):
            raise ValueError("Workflow truncation metadata differs from its observations")
        if invocation["new_model_calls"] is None or invocation["new_model_calls"] > state.model_calls:
            raise ValueError("Invocation calls cannot exceed the original recorded calls")
        if ((cell["execution"] == "executed" and invocation["new_model_calls"] != state.model_calls)
                or (cell["execution"] == "cached_replay" and invocation["new_model_calls"] != 0)):
            raise ValueError("Invocation calls differ from executed or cached replay accounting")
        pins.update(child_pins)
    return indexed, pins


def _prediction(cell: dict | None, variant: str) -> PredictionRef:
    if cell is None:
        return PredictionRef(variant=variant, status="unprocessed", failure_code="study_cell_missing")
    resources = cell.get("original_resources") or {}
    values = {
        "execution_mode": "mock", "request_count": resources.get("model_calls"),
        "input_tokens": None if resources.get("usage_unknown", True) else resources.get("input_tokens_known"),
        "output_tokens": None if resources.get("usage_unknown", True) else resources.get("output_tokens_known"),
        "latency_ms": resources.get("elapsed_ms"), "cost": None,
    }
    if resources.get("uncertain_model_calls"):
        values["request_count"] = None
    if cell.get("canonical") is None:
        return PredictionRef(variant=variant, status="failed", failure_code=cell.get("error_code") or "study_execution_failed", **values)
    return PredictionRef(variant=variant, status="ready", artifact=ArtifactRef.model_validate(cell["canonical"]),
                         sidecar=ArtifactRef.model_validate(cell["sidecar"]), **values)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def resource_summary(cells: list[dict]) -> dict:
    output = {}
    for basis, fields in (("original_resources", ("model_calls", "uncertain_model_calls", "input_tokens_known", "output_tokens_known", "elapsed_ms")),
                          ("invocation_resources", ("new_model_calls", "new_live_requests", "elapsed_ms"))):
        metrics = {}
        for field in fields:
            values, unknown = [], 0
            for cell in cells:
                resources = cell.get(basis) or {}
                value = resources.get(field)
                if type(value) not in (int, float):
                    unknown += 1
                else:
                    values.append(value)
                    if basis == "original_resources" and ((field in {"input_tokens_known", "output_tokens_known"} and resources.get("usage_unknown", True))
                                                           or (field == "model_calls" and resources.get("uncertain_model_calls"))):
                        unknown += 1
            metrics[field] = {"known_subtotal": sum(values) if values else None,
                              "total": sum(values) if not unknown else None, "unknown_cells": unknown}
        output[basis] = metrics
    output["cost"] = {"value": None, "status": "not_applicable_mock"}
    return output


def paired_primary(scored: list[dict], protocol: StudyEvaluationProtocol) -> list[dict]:
    """Average matched repeats within publications, then bootstrap publications.

    A paired delta is eligible only with known positive gold support on both
    sides. No result is manufactured for missing/failed/blocked comparisons.
    """
    grouped = defaultdict(list)
    for stratum in scored:
        for pair in stratum["result"]["comparisons"]:
            if pair.get("field") != protocol.primary_field:
                continue
            left = pair.get("baseline_support") or {}
            right = pair.get("variant_support") or {}
            a = left.get("metrics", {}).get(protocol.primary_metric)
            b = right.get("metrics", {}).get(protocol.primary_metric)
            eligible = (left.get("support_status") == right.get("support_status") == "known"
                        and type(left.get("support_count")) is int and left["support_count"] > 0
                        and type(right.get("support_count")) is int and right["support_count"] > 0)
            if not eligible or type(a) not in (float, int) or type(b) not in (float, int):
                continue
            key = (stratum["cap"], pair["split"], pair["data_kind"], pair["variant"], pair["publication_group"])
            grouped[key].append((stratum["repeat"], b - a))
    outputs = []
    splits = sorted({(p.split, p.data_kind) for p in protocol.publications})
    scope = protocol.split_manifest()
    for cap in protocol.caps:
        for split, kind in splits:
            intended = {scope.publication_group(p.publication_id) for p in protocol.publications
                        if (p.split, p.data_kind) == (split, kind)}
            for variant in protocol.variants:
                if variant == protocol.baseline_variant:
                    continue
                publications = []
                for publication in sorted(intended):
                    observations = grouped.get((cap, split, kind, variant, publication), [])
                    if len({repeat for repeat, _ in observations}) != len(observations):
                        raise ValueError("Duplicate paired publication repeat")
                    publications.append({"publication_group": publication,
                                         "matched_repeats": len(observations),
                                         "planned_repeats": protocol.repeats,
                                         "delta": _mean([delta for _, delta in observations])})
                values = [p["delta"] for p in publications if p["delta"] is not None]
                ci = None
                if len(values) >= 2:
                    rng = random.Random(protocol.bootstrap_seed)
                    samples = sorted(sum(rng.choice(values) for _ in values) / len(values)
                                     for _ in range(protocol.bootstrap_samples))
                    ci = [samples[int((len(samples) - 1) * 0.025)], samples[int((len(samples) - 1) * 0.975)]]
                outputs.append({"cap": cap, "split": split, "data_kind": kind,
                    "baseline": protocol.baseline_variant, "variant": variant,
                    "field": protocol.primary_field, "metric": protocol.primary_metric,
                    "planned_publications": len(intended), "paired_publications": len(values),
                    "mean_publication_delta": _mean(values), "interval_95": ci,
                    "interval_status": "paired_publication_bootstrap" if ci is not None else "insufficient_independent_publications",
                    "bootstrap_seed": protocol.bootstrap_seed, "bootstrap_samples": protocol.bootstrap_samples,
                    "minimum_effect": protocol.minimum_effect,
                    "threshold_assessment": "engineering_only_no_empirical_claim",
                    "publications": publications})
    return outputs


_UNAVAILABLE = {
    "mapping_correctness": "Independent reviewed mapping labels were not supplied.",
    "source_support_errors": "Independent assertion-support labels were not supplied; model verdicts are not gold.",
    "harmful_corrections": "Independent before/after correction labels were not supplied; metric regressions are reported separately.",
    "review_effort_seconds": "No observed human review durations were supplied; simulated responses do not measure expert time.",
}


def render_report(result: dict) -> str:
    """Deterministic local Markdown; no inherited paper output is touched."""
    lines = ["# M09 synthetic engineering study", "", "Empirical status: **blocked**. These are scripted fixture diagnostics, not accuracy or effort evidence.", "",
             "Primary endpoint: `" + result["primary_field"] + "` / `" + result["primary_metric"] + "`.", "",
             "All planned publication/variant/cap/repeat cells remain in the denominator. Last valid canonical predictions may be scored even when the workflow stopped unresolved or exhausted its budget.", "",
             "## Execution coverage", "", "| Status | Cells |", "|---|---:|"]
    for status, count in sorted(result["coverage"]["execution_statuses"].items()):
        lines.append(f"| {status} | {count} |")
    lines += ["", "## Cell outcomes and original resource use", "",
              "| Publication | Variant | Cap / repeat | Workflow state | Unresolved | Original calls | Study invocation calls | Primary score / support |",
              "|---|---|---|---|---:|---:|---:|---|"]
    for cell in result["cells"]:
        original, invoked = cell.get("original_resources") or {}, cell.get("invocation_resources") or {}
        primary = cell["primary"]
        score = "unavailable" if primary["value"] is None else str(primary["value"])
        support = primary.get("support_count")
        lines.append(f'| {cell["publication_id"]} | {cell["variant"]} | {cell["cap"]} / {cell["repeat"]} | '
                     f'{cell.get("workflow_status") or cell["status"]} | {cell.get("unresolved_issues")} | '
                     f'{original.get("model_calls")} | {invoked.get("new_model_calls")} | '
                     f'{score} / {support if support is not None else "unknown"} {primary["support_status"]} |')
    tokens = result["resources"]["original_resources"]["input_tokens_known"]
    lines += ["", f'Original input-token known subtotal: {tokens["known_subtotal"]}; unknown cells: {tokens["unknown_cells"]}. '
              "Null values mean unavailable, not zero. Study invocation usage is historical; collecting/replaying this report makes zero model calls."]
    lines += ["", "## Publication-paired primary differences", "", "| Cap | Variant | Paired / planned publications | Mean delta | 95% interval |", "|---:|---|---:|---:|---|"]
    for row in result["paired_primary"]:
        delta = "unavailable" if row["mean_publication_delta"] is None else f'{row["mean_publication_delta"]:.6g}'
        interval = "insufficient independent publications" if row["interval_95"] is None else str(row["interval_95"])
        lines.append(f'| {row["cap"]} | {row["variant"]} | {row["paired_publications"]} / {row["planned_publications"]} | {delta} | {interval} |')
    lines += ["", "## Inherited per-field metrics and support", "",
              "Pooled metrics are not publication-level means. Conditional attributes and references remain conditional on matched parent entities. No cross-field average is computed.", "",
              "| Cap / repeat | Variant | Category / field | Metrics | Gold support |", "|---|---|---|---|---|"]
    for stratum in result["strata"]:
        for row in stratum["result"]["metric_rows"]:
            if row["category"] not in {"entity_recovery", "conditional_attributes", "references"}:
                continue
            values = {key: value for key, value in row["metrics"].items() if key in {"f1", "precision", "recall", "mean_score", "score"}}
            lines.append(f'| {stratum["cap"]} / {stratum["repeat"]} | {row["variant"]} | {row["category"]} / {row["field"]} | '
                         f'{json.dumps(values, sort_keys=True)} | {row["support_count"]} {row["support_unit"]} ({row["support_status"]}) |')
    lines += ["", "Harmed/improved per-field changes, failed/missing scoring rows, token known subtotals and full resource details are retained in the JSON report.", "",
              "Cached replay calls are separate from the original workflow calls. Unknown usage and costs remain unknown; fixture latency does not establish live end-to-end performance.", "", "## Unavailable independent diagnostics", ""]
    for field, entry in result["independent_diagnostics"].items():
        lines.append(f'- {field}: unavailable. {entry["reason"]}')
    lines += ["", "No model judges its own output as independent gold. No schema-wide average or completeness claim is produced.", ""]
    return "\n".join(lines)


def collect_study(root: Path | str, study_run_id: str, protocol: StudyEvaluationProtocol, *, run_id: str) -> dict:
    """Collect a completed frozen synthetic study and reproduce its fixed scores."""
    root = workspace_root(root)
    validate_run_id(run_id)
    if run_id == study_run_id:
        raise ValueError("Collection must use a fresh distinct run ID")
    protocol = StudyEvaluationProtocol.model_validate_json(protocol.model_dump_json())
    parent_request, parent, parent_pins = _completed(root, study_run_id)
    spec, expected = _scope(parent, protocol)  # Must precede evaluator or gold access.
    if parent_request.get("spec") != spec.model_dump(mode="json") or parent_request.get("spec_sha256") != parent["spec_sha256"]:
        raise ValueError("Completed study scope differs from its immutable request")
    from meta_data_extraction.agent.study_runner import dry_run, expected_cells
    frozen = dry_run(root, spec)
    planned = expected_cells(spec, study_run_id)
    if parent_request.get("dry_run") != frozen or parent_request.get("cells") != planned:
        raise ValueError("Runtime study freeze or planned cells differ from the completed request")
    verify_evaluator(protocol.evaluator, root)
    indexed, child_pins = _cells(root, parent, spec, expected, protocol, planned, frozen)
    # A changed declared artifact is an integrity failure, not a new missing-data
    # observation. An absent declaration remains an explicit scorer limitation.
    for publication in protocol.publications:
        for ref in publication.sources:
            read_artifact(root, ref, role="source")
        for ref in (publication.gold, publication.identity_gold):
            if ref is not None:
                read_artifact(root, ref, role="gold", split=publication.split)
    pins = {**parent_pins, **child_pins}
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    request = {"version": "m09-collection-1", "study_run_id": study_run_id,
               "protocol": protocol.model_dump(mode="json"), "input_pins": pins,
               "implementation": implementation, "model_calls": 0}
    store = RunStore.begin(root, run_id, request)
    scored, new_evaluations = [], 0
    for cap in protocol.caps:
        for repeat in range(protocol.repeats):
            publications = []
            for publication in protocol.publications:
                predictions = tuple(_prediction(indexed.get((publication.publication_id, v, cap, repeat)), v)
                                    for v in protocol.variants)
                publications.append(Publication.model_validate(publication.model_dump(mode="json") |
                                                               {"predictions": [p.model_dump(mode="json") for p in predictions]}))
            manifest = protocol.split_manifest(tuple(publications))
            score_run = "m09-score-" + hashlib.sha256((run_id + ":" + str(cap) + ":" + str(repeat)).encode()).hexdigest()[:32]
            report = run_offline(root, manifest, run_id=score_run, evaluator=protocol.evaluator)
            new_evaluations += report["new_evaluations"]
            _, _, score_pins = _completed(root, score_run)
            sidecar_ref = ArtifactRef(path=f"runs/agent/{score_run}/sidecar.json", sha256=score_pins[f"runs/agent/{score_run}/sidecar.json"])
            sidecar = strict_json(read_artifact(root, sidecar_ref, role="sidecar"))
            publication_metrics = [{"publication_group": item["publication_group"], "variant": item["variant"],
                "metrics": [row.model_dump(mode="json") for row in summaries([item["eligible_report"]], protocol.evaluator)]}
                for item in sidecar["reports"] if item.get("eligible_report") is not None]
            scored.append({"cap": cap, "repeat": repeat, "score_run_id": score_run, "result": report["result"],
                           "publication_metrics": publication_metrics})
    execution = []
    for key in expected:
        cell = indexed.get(key)
        if cell is None:
            execution.append({"publication_id": key[0], "variant": key[1], "cap": key[2], "repeat": key[3],
                              "status": "missing", "workflow_status": None, "unresolved_issues": None,
                              "original_resources": None, "invocation_resources": None, "error_code": "study_cell_missing"})
        else:
            execution.append({name: cell.get(name) for name in ("cell_id", "publication_id", "variant", "cap", "repeat", "status",
                "workflow_status", "unresolved_issues", "truncated", "execution", "original_resources", "invocation_resources", "error_code")})
        group = protocol.split_manifest().publication_group(key[0])
        rows = [row for stratum in scored if (stratum["cap"], stratum["repeat"]) == (key[2], key[3])
                for publication in stratum["publication_metrics"] if (publication["publication_group"], publication["variant"]) == (group, key[1])
                for row in publication["metrics"] if row["field"] == protocol.primary_field]
        primary = rows[0] if rows else None
        execution[-1]["primary"] = {"value": primary["metrics"].get(protocol.primary_metric) if primary else None,
                                    "support_status": primary["support_status"] if primary else "unknown",
                                    "support_count": primary["support_count"] if primary else None}
    result = {"version": "m09-collection-1", "study_run_id": study_run_id, "protocol_sha256": protocol_digest(protocol),
              "empirical_status": "blocked", "engineering_status": "completed", "stage": protocol.stage,
              "primary_field": protocol.primary_field, "primary_metric": protocol.primary_metric,
              "denominator_policy": protocol.denominator_policy, "pooled_metric_policy": "unchanged_inherited_aggregates_per_cap_repeat",
              "model_calls": 0, "coverage": {"planned_cells": len(expected), "submitted_cells": len(indexed),
                  "execution_statuses": dict(Counter(cell["status"] for cell in execution))},
              "cells": execution, "strata": scored, "paired_primary": paired_primary(scored, protocol),
              "resources": resource_summary(execution),
              "independent_diagnostics": {key: {"value": None, "status": "not_supplied", "reason": value} for key, value in _UNAVAILABLE.items()},
              "limitations": ["Scripted synthetic fixtures only; no natural accuracy or expert effort claim.",
                  "Reviewed standards guidance remains unavailable; this is not the matrix A2/A3 standards comparison.",
                  "No independent diagnostic labels; new natural identities require adjudication."]}
    markdown = render_report(result)
    from meta_data_extraction.agent.paths import checked_path
    report_path = checked_path(root, f"runs/agent/{run_id}/report.md", allowed_roots=(f"runs/agent/{run_id}",))
    if store.completed is not None:
        if store.completed != result or not report_path.is_file() or report_path.read_bytes() != markdown.encode():
            raise ValueError("Completed collection report differs from reproduced inputs")
        execution_mode = "cached_replay"
    else:
        with report_path.open("xb") as stream:
            stream.write(markdown.encode())
        store.complete(result, canonical=None, sidecar={"protocol": protocol.model_dump(mode="json"),
                       "input_pins": pins, "report_sha256": hashlib.sha256(markdown.encode()).hexdigest()},
                       raw_response=None, retain_raw=False)
        execution_mode = "offline_collection"
    return {"execution": execution_mode, "new_evaluations": new_evaluations, "result": result,
            "report_path": report_path.relative_to(root).as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--study-run-id", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        result = collect_study(args.root, args.study_run_id, load_protocol(args.root, args.protocol), run_id=args.run_id)
        print(json.dumps({"execution": result["execution"], "new_evaluations": result["new_evaluations"],
                          "report_path": result["report_path"], "empirical_status": result["result"]["empirical_status"],
                          "planned_cells": result["result"]["coverage"]["planned_cells"], "model_calls": 0}))
        return 0
    except (ValueError, OSError, TypeError, KeyError, RuntimeError):
        print(json.dumps({"status": "blocked", "error_code": "invalid_or_incompatible_study_collection", "model_calls": 0}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
