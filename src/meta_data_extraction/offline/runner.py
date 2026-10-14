"""Explicit offline scoring only; never imported by the extraction runtime."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from meta_data_extraction.agent.contracts import FixedExtraction, canonical_json
from meta_data_extraction.agent.paths import checked_path, input_file, workspace_root
from meta_data_extraction.agent.storage import RunStore
from meta_data_extraction.offline.manifest import (
    PredictionRef, SplitManifest, load_manifest, manifest_digest, read_artifact, strict_json,
)
from meta_data_extraction.offline.scoring import (
    EvaluatorSnapshot, ScoringError, evaluate_pair, parity, summaries, verify_evaluator,
)


def _implementation(root: Path) -> dict[str, str]:
    names = ("offline/__init__.py", "offline/manifest.py", "offline/scoring.py", "offline/runner.py",
             "agent/paths.py", "agent/storage.py", "agent/contracts.py")
    return {name: hashlib.sha256(input_file(root, "src/meta_data_extraction/" + name,
                                          allowed_roots=("src/meta_data_extraction",)).read_bytes()).hexdigest()
            for name in names}


def _json(raw: bytes) -> dict:
    value = strict_json(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _stage(store: RunStore, relative: str, raw: bytes) -> Path:
    target = checked_path(store.run.root, f"runs/agent/{store.run.run_id}/inputs/{relative}",
                          allowed_roots=(f"runs/agent/{store.run.run_id}/inputs",))
    target.parent.mkdir(parents=True, exist_ok=True)
    target = checked_path(store.run.root, target.relative_to(store.run.root).as_posix(),
                          allowed_roots=(f"runs/agent/{store.run.run_id}/inputs",))
    with target.open("xb") as stream:
        stream.write(raw)
    return target


def _blocked(field: str, collections: tuple[str, ...]) -> bool:
    if any(field == name or field.startswith(name + ".attrs.") for name in collections):
        return True
    # Reference scores depend on target identities as well as the parent match.
    targets = {"samples": "datasets.attrs.sample_refs", "facilities": "datasets.attrs.facility_refs",
               "bibliographic.affiliations": "bibliographic.authors.attrs.affiliation_refs"}
    return any(targets.get(name) == field for name in collections)


def _resources(prediction, payload: dict | None, sidecar: dict | None) -> dict:
    keys = ("request_count", "input_tokens", "output_tokens", "latency_ms", "cost")
    result = {key: getattr(prediction, key) for key in keys}
    result["execution_mode"] = prediction.execution_mode
    result["basis"] = "manifest_declarations"
    observed = (payload or {}).get("provenance")
    if isinstance(observed, dict):
        for key in ("input_tokens", "output_tokens", "latency_ms"):
            value = observed.get(key)
            if result[key] is None and type(value) in (int, float) and value >= 0:
                result[key] = value
        result["basis"] = "inherited_provenance_and_manifest"
    extraction = (sidecar or {}).get("extraction")
    if isinstance(extraction, dict) and isinstance(extraction.get("provider"), dict):
        provider = extraction["provider"]
        for key in ("input_tokens", "output_tokens", "latency_ms"):
            value = provider.get(key)
            if result[key] is None and type(value) in (int, float) and value >= 0:
                result[key] = value
        if result["execution_mode"] == "unknown" and provider.get("mode") in {"mock", "live"}:
            result["execution_mode"] = provider["mode"]
        result["basis"] = "extraction_sidecar_and_manifest"
    # Missing counts/costs remain unknown: a stored response does not prove no retries.
    return result


def _resource_totals(items: list[dict]) -> dict:
    totals = {}
    for key in ("request_count", "input_tokens", "output_tokens", "latency_ms", "cost"):
        known = [item["resources"][key] for item in items if item["resources"][key] is not None]
        missing = len(items) - len(known)
        totals[key] = {"total": sum(known) if not missing else None,
                       "known_subtotal": sum(known) if known else None, "unknown_records": missing}
    return totals


def _tables(outcomes: list[dict], snapshot: EvaluatorSnapshot, variants: tuple[str, ...]) -> tuple[list, list, list]:
    groups = sorted({(item["split"], item["data_kind"]) for item in outcomes})
    rows, coverage, comparisons = [], [], []
    for split, kind in groups:
        for variant in variants:
            items = [item for item in outcomes if (item["split"], item["data_kind"], item["variant"]) ==
                     (split, kind, variant)]
            reports = [item["eligible_report"] for item in items if item.get("eligible_report") is not None]
            coverage.append({"split": split, "data_kind": kind, "variant": variant,
                             "publications": len(items),
                             "status_counts": {status: sum(i["status"] == status for i in items)
                                               for status in ("scored", "partial", "failed", "blocked", "unprocessed")},
                             "resources_by_execution_mode": {
                                 mode: _resource_totals([i for i in items if i["resources"]["execution_mode"] == mode])
                                 for mode in sorted({i["resources"]["execution_mode"] for i in items})}})
            for metric in summaries(reports, snapshot):
                rows.append({"split": split, "data_kind": kind, "variant": variant, **metric.model_dump(mode="json")})
    # Paired changes are per publication and field, never a schema-wide mean.
    baseline = variants[0]
    by_key = {(i["publication_group"], i["variant"]): i for i in outcomes}
    for publication in sorted({item["publication_group"] for item in outcomes}):
        left = by_key.get((publication, baseline))
        for variant in variants[1:]:
            right = by_key.get((publication, variant))
            if (not left or not right or left.get("eligible_report") is None or right.get("eligible_report") is None
                    or (left["split"], left["data_kind"]) != (right["split"], right["data_kind"])):
                comparisons.append({"publication_group": publication, "baseline": baseline, "variant": variant,
                                    "status": "unpaired"})
                continue
            lm = {row.field: row for row in summaries([left["eligible_report"]], snapshot)}
            rm = {row.field: row for row in summaries([right["eligible_report"]], snapshot)}
            for field in sorted(set(lm) | set(rm)):
                a, b = lm.get(field), rm.get(field)
                metric_name = next((key for key in ("f1", "mean_score", "score")
                                    if a and b and key in a.metrics and key in b.metrics), None)
                av = a.metrics.get(metric_name) if a and metric_name else None
                bv = b.metrics.get(metric_name) if b and metric_name else None
                supported = a and b and a.support_status == b.support_status == "known"
                delta = bv - av if supported and type(av) in (int, float) and type(bv) in (int, float) else None
                comparisons.append({"publication_group": publication, "split": left["split"],
                                    "data_kind": left["data_kind"], "baseline": baseline, "variant": variant,
                                    "field": field, "metric": metric_name, "baseline_value": av, "variant_value": bv,
                                    "baseline_support": a.model_dump(mode="json") if a else None,
                                    "variant_support": b.model_dump(mode="json") if b else None,
                                    "delta": delta, "status": "unsupported_or_blocked" if delta is None else
                                    "improved" if delta > 0 else "harmed" if delta < 0 else "unchanged"})
    return rows, coverage, comparisons


def run_offline(root: Path | str, manifest: SplitManifest, *, run_id: str,
                evaluator: EvaluatorSnapshot, allow_final_gold: bool = False) -> dict:
    """Score only the explicit frozen scope; preserve failures and unknown costs."""
    started = time.perf_counter()
    root = workspace_root(root)
    if type(allow_final_gold) is not bool:
        raise TypeError("Final-gold authorization must be an explicit boolean")
    manifest = SplitManifest.model_validate(manifest.model_dump(mode="json"))
    verify_evaluator(evaluator, root)
    keys = [(manifest.publication_group(p.publication_id), v.variant)
            for p in manifest.publications for v in p.predictions]
    if len(keys) != len(set(keys)):
        raise ValueError("Choose one record per publication group/variant; chunks cannot be independent papers")
    if any(p.split == "future_held_out" and p.gold is not None for p in manifest.publications) and not allow_final_gold:
        raise ValueError("Final held-out gold requires an explicitly authorized offline scoring invocation")
    request = {"version": "m03-run-1", "manifest": manifest.model_dump(mode="json"),
               "manifest_sha256": manifest_digest(manifest), "evaluator": evaluator.model_dump(mode="json"),
               "implementation": _implementation(root), "allow_final_gold": allow_final_gold,
               "model_calls": 0}
    store = RunStore.begin(root, run_id, request)
    # Validate all declared inputs on replay too; never trust stale paths or hashes.
    inputs: dict[tuple[str, str], bytes] = {}
    errors: dict[tuple[str, str], str] = {}

    def load(publication, ref, role):
        if ref is None:
            return None
        key = (role, ref.path)
        if key not in inputs and key not in errors:
            try:
                inputs[key] = read_artifact(root, ref, role=role, split=publication.split,
                                            allow_final_gold=allow_final_gold)
            except (ValueError, OSError):
                errors[key] = "input_unavailable_or_hash_mismatch"
        if key in errors:
            raise ValueError(errors[key])
        return inputs[key]

    prepared: list[tuple[Any, Any, dict]] = []
    for publication in manifest.publications:
        for prediction in publication.predictions:
            data = {}
            try:
                # Sources are provenance only and are never used for scoring/retrieval.
                for source in publication.sources:
                    load(publication, source, "source")
                data["sidecar"] = load(publication, prediction.sidecar, "sidecar")
                if prediction.status == "ready":
                    if publication.gold is None:
                        raise ValueError("gold_not_supplied")
                    if publication.data_kind == "natural" and publication.alignment_status == "frozen_historical":
                        if not manifest.matches_publication(publication.publication_id, Path(publication.gold.path).stem):
                            raise ValueError("gold_publication_identity_mismatch")
                    data["gold"] = load(publication, publication.gold, "gold")
                    data["identity"] = load(publication, publication.identity_gold, "gold")
                    data["prediction"] = load(publication, prediction.artifact, "prediction")
                    data["reference"] = load(publication, prediction.reference_report, "report")
            except ValueError as error:
                data["error"] = str(error)
            prepared.append((publication, prediction, data))
    # An omitted variant is an explicit unprocessed publication, not a smaller denominator.
    groups = {manifest.publication_group(p.publication_id) for p in manifest.publications}
    publications_by_id = {p.publication_id: p for p in manifest.publications}
    for group in sorted(groups):
        for variant in manifest.variants:
            if (group, variant) not in keys:
                prepared.append((publications_by_id[group], PredictionRef(
                    variant=variant, status="unprocessed", failure_code="prediction_not_declared"), {}))
    if store.completed is not None:
        if any(data.get("error") for _, _, data in prepared):
            raise ValueError("Completed replay inputs no longer match the frozen scope")
        for staged in store.completed["staged_inputs"]:
            path = input_file(root, staged["path"], allowed_roots=(f"runs/agent/{run_id}/inputs",))
            if hashlib.sha256(path.read_bytes()).hexdigest() != staged["sha256"]:
                raise ValueError("Staged scoring input no longer matches the completed run")
        verify_evaluator(evaluator, root)
        return {"execution": "cached_replay", "elapsed_ms": (time.perf_counter() - started) * 1000,
                "new_evaluations": 0, "result": store.completed}
    outcomes = []
    staged_inputs = []

    def stage(relative: str, raw: bytes) -> Path:
        path = _stage(store, relative, raw)
        staged_inputs.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()})
        return path

    for index, (publication, prediction, data) in enumerate(prepared):
        item = {"publication_id": publication.publication_id,
                "publication_group": manifest.publication_group(publication.publication_id),
                "split": publication.split, "data_kind": publication.data_kind, "variant": prediction.variant,
                "alignment_status": publication.alignment_status, "status": prediction.status,
                "resources": _resources(prediction, None, None), "report": None, "eligible_report": None,
                "parity": {"status": "not_available", "differences": []}, "diagnostics": []}
        outcomes.append(item)
        if prediction.status != "ready":
            item["diagnostics"].append(prediction.failure_code or "not_processed")
            if data.get("error"):
                item["diagnostics"].append(data["error"])
            elif data.get("sidecar"):
                try:
                    sidecar = _json(data["sidecar"])
                    if "extraction" in sidecar:
                        FixedExtraction.model_validate(sidecar["extraction"])
                    item["resources"] = _resources(prediction, None, sidecar)
                except (ValueError, TypeError, RecursionError):
                    item["diagnostics"].append("invalid_failure_sidecar")
            continue
        if data.get("error"):
            item.update(status="blocked", diagnostics=[data["error"]])
            continue
        try:
            payload = _json(data["prediction"])
            sidecar = _json(data["sidecar"]) if data.get("sidecar") else None
            item["resources"] = _resources(prediction, payload, sidecar)
            if (publication.data_kind == "natural" and publication.alignment_status == "frozen_historical"
                    and "document_id" in payload and (not isinstance(payload["document_id"], str)
                    or not manifest.matches_publication(publication.publication_id, payload["document_id"]))):
                raise ScoringError("prediction_publication_identity_mismatch")
            record = payload.get("output") if "document_id" in payload and "output" in payload else payload
            if not isinstance(record, dict):
                raise ScoringError("nonobject_prediction")
            warnings = payload.get("warnings", [])
            if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
                raise ScoringError("invalid_warning_metadata")
            if warnings:
                item["diagnostics"].append("inherited_extraction_warnings_present")
            if any(w in {
                "LLM output parsed but is not a JSON object; using empty output.",
                "JSON parse failed after fallback extraction; using empty output.",
            } for w in warnings):
                raise ScoringError("reported_extraction_parse_failure")
            if any(w.startswith("LLM response was cut off at the token limit") for w in warnings):
                raise ScoringError("reported_response_truncated")
            if sidecar is not None and "extraction" in sidecar:
                extraction = FixedExtraction.model_validate(sidecar["extraction"])
                if extraction.status == "failed" or extraction.response_truncated:
                    raise ScoringError("failed_extraction_sidecar")
                if extraction.snapshot is None or extraction.snapshot.canonical_json != canonical_json(record):
                    raise ScoringError("sidecar_record_mismatch")
                item["extraction_status"] = extraction.status
                item["evidence_support"] = "not_assessed_by_this_evaluator"
                if extraction.status == "needs_review":
                    item["diagnostics"].append("extraction_needs_review")
            gold_name = Path(publication.gold.path).name  # stem controls inherited paper-specific identities
            gold_path = stage(f"{index}/gold/{gold_name}", data["gold"])
            pred_path = stage(f"{index}/prediction.json", data["prediction"])
            identity_path = stage(f"{index}/identity/{gold_name}", data["identity"]) if data.get("identity") else None
            report = evaluate_pair(root, gold_path, pred_path, evaluator, identity_gold=identity_path)
            item["report"] = report
            blocked = tuple(publication.blocked_collections)
            if publication.alignment_status == "unreviewed":
                blocked = tuple(sorted(set(blocked) | {"samples", "datasets", "bibliographic.funding"}))
                item["diagnostics"].append("new_publication_identity_coverage_unreviewed")
            eligible = deepcopy(report)
            omitted = [field for field in report["fields"] if _blocked(field, blocked)]
            for field in omitted:
                del eligible["fields"][field]
            item["eligible_report"] = eligible
            item["blocked_fields"] = omitted
            item["status"] = "partial" if omitted or item["diagnostics"] else "scored"
            if data.get("reference"):
                item["parity"] = parity(report, _json(data["reference"])).model_dump(mode="json")
        except (ValueError, TypeError, KeyError, RecursionError, OSError) as error:
            # Do not write raw exceptions containing private record values.
            item.update(status="failed", report=None, eligible_report=None)
            item["diagnostics"].append(getattr(error, "code", "invalid_input_or_scoring_failure"))
    verify_evaluator(evaluator, root)
    rows, coverage, comparisons = _tables(outcomes, evaluator, manifest.variants)
    result = {"version": "m03-result-1", "manifest_sha256": manifest_digest(manifest),
              "evaluator_id": evaluator.evaluator_id, "primary_outcome": manifest.primary_outcome,
              "model_calls": 0, "new_evaluations": sum(item["report"] is not None for item in outcomes),
              "staged_inputs": staged_inputs,
              "coverage": coverage, "metric_rows": rows, "comparisons": comparisons,
              "records": [{key: value for key, value in item.items() if key not in {"report", "eligible_report"}}
                          for item in outcomes]}
    store.complete(result, canonical=None, sidecar={"manifest": manifest.model_dump(mode="json"),
                   "evaluator": evaluator.model_dump(mode="json"), "reports": outcomes},
                   raw_response=None, retain_raw=False)
    return {"execution": "offline_scoring", "elapsed_ms": (time.perf_counter() - started) * 1000,
            "new_evaluations": result["new_evaluations"], "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluator", required=True, help="Explicit frozen evaluator JSON under configs/agent or runs/agent")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--allow-final-gold", action="store_true", help="Only for an explicitly authorized final offline scoring step")
    args = parser.parse_args()
    root = workspace_root(args.root)
    lock = input_file(root, args.evaluator, allowed_roots=("configs/agent", "runs/agent"))
    evaluator = EvaluatorSnapshot.model_validate_json(canonical_json(_json(lock.read_bytes())))
    report = run_offline(root, load_manifest(root, args.manifest), run_id=args.run_id,
                         evaluator=evaluator, allow_final_gold=args.allow_final_gold)
    records = report["result"]["records"]
    print(json.dumps({"run_id": args.run_id, "execution": report["execution"],
                      "new_evaluations": report["new_evaluations"], "model_calls": 0,
                      "records": [{key: r[key] for key in ("publication_id", "variant", "status", "parity")}
                                  for r in records]}))
    return int(any(r["status"] in {"failed", "blocked"} or r["parity"]["status"] == "mismatch" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
