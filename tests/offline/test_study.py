"""M09 collection engineering tests use invented records and a stub scorer only."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from meta_data_extraction.agent.contracts import ProviderObservation, canonical_json, snapshot
from meta_data_extraction.agent.storage import RunStore, _digest, _json_bytes
from meta_data_extraction.agent.study_types import StudySpec
from meta_data_extraction.agent.workflow_types import ActionEvent, Finish, WorkflowState
from meta_data_extraction.agent import study_runner
from meta_data_extraction.offline import runner, study
from meta_data_extraction.offline.manifest import ArtifactRef, Publication
from meta_data_extraction.offline.scoring import EvaluatorSnapshot


def write(workspace, relative, value):
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else (json.dumps(value) + "\n").encode()
    path.write_bytes(raw)
    return ArtifactRef(path=relative, sha256=hashlib.sha256(raw).hexdigest())


def pin(workspace, relative):
    return {"path": relative, "sha256": hashlib.sha256((workspace / relative).read_bytes()).hexdigest()}


def fingerprint():
    payload = {"version": "m03-evaluator-1", "files": [], "dependencies": [],
               "field_metrics": [{"field": "datasets.attrs.sample_refs", "metric": "reference", "parent_entity": "datasets"},
                                 {"field": "samples", "metric": "entity", "parent_entity": None}],
               "python_version": "synthetic-test", "metadata_json": "{}"}
    payload["evaluator_id"] = "evaluator:" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return EvaluatorSnapshot.model_validate_json(json.dumps(payload))


@pytest.fixture
def scorer(monkeypatch):
    calls = []
    def evaluate(root, gold, pred, evaluator, identity_gold=None):
        assert gold.is_relative_to(root / "runs/agent") and pred.is_relative_to(root / "runs/agent")
        calls.append(pred)
        quality = json.loads(pred.read_bytes())["fixture_quality"]
        return {"fields": {"datasets.attrs.sample_refs": {"tp": quality, "fp": 1 - quality, "fn": 1 - quality, "n_pairs": 1},
                           "samples": {"tp": 2, "fp": 0, "fn": 0, "n_gold": 2}},
                "evaluator": {}, "identity_sources": {}}
    monkeypatch.setattr(study, "verify_evaluator", lambda *args: None)
    monkeypatch.setattr(runner, "verify_evaluator", lambda *args: None)
    monkeypatch.setattr(runner, "evaluate_pair", evaluate)
    monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic_stub": "a" * 64})
    monkeypatch.setattr(study_runner, "dry_run", lambda root, spec: json.loads(
        (root / "runs/agent/study-parent/request.json").read_bytes())["request"]["dry_run"])
    monkeypatch.setattr(study, "_workflow_config", lambda root, spec, coordinate: {
        "policy": coordinate["variant"], "run_id": coordinate["child_run_id"], "budget": {"max_model_calls": coordinate["cap"]}})
    return calls


def protocol(workspace, *, count=1, caps=(2,), repeats=1, variants=("fixed", "generic_llm")):
    publications = []
    for number in range(count):
        source = write(workspace, f"tests/agent/fixtures/source-{number}.md", f"Invented publication {number}.".encode())
        gold = write(workspace, f"tests/offline/fixtures/gold-{number}.json", {"fixture_gold": number})
        publications.append(Publication(publication_id=f"paper-{number}", split="calibration", data_kind="synthetic",
                                        sources=(source,), gold=gold, alignment_status="synthetic"))
    return study.StudyEvaluationProtocol(stage="synthetic_engineering", description="Invented diagnostics only",
        primary_field="datasets.attrs.sample_refs", primary_metric="f1", minimum_effect=0.0,
        baseline_variant="fixed", variants=variants, caps=caps, repeats=repeats,
        publications=tuple(publications), evaluator=fingerprint(), bootstrap_samples=200)


def child(workspace, publication, coordinate, *, quality=1, status="finished"):
    child_id, variant, cap, repeat = (coordinate[key] for key in ("child_run_id", "variant", "cap", "repeat"))
    source = publication.sources[0]
    bundle = SimpleNamespace(source_id=source.path, source_sha256=source.sha256, representation_sha256="b" * 64)
    record = snapshot({"fixture_quality": quality}, bundle, SimpleNamespace(profile_id="synthetic-profile"))
    initial_record = snapshot({"fixture_quality": 0}, bundle, SimpleNamespace(profile_id="synthetic-profile"))
    initial = WorkflowState(initial_revision=initial_record.revision_id, record=initial_record, issues=())
    observation = ProviderObservation(status="ok", mode="mock", model="synthetic-unit", finish_reason="stop",
                                      input_tokens=None, output_tokens=None, latency_ms=None,
                                      error_code=None, request_id=None, response_sha256=None)
    event = ActionEvent(index=0, policy=variant, action=Finish(outcome="finished", summary="Scripted unit transition"),
                        status="ok", code="unit_fixture", revision_before=initial_record.revision_id,
                        revision_after=record.revision_id, progress=True, providers=(observation,), summary="Invented unit data")
    state = WorkflowState(initial_revision=initial_record.revision_id, record=record, issues=(), status=status,
                          events=(event,), model_calls=1, usage_unknown=True, elapsed_ms=1.0)
    result = {"version": "m07-result-1", "state": state.model_dump(mode="json"), "policy": variant}
    request = {"version": "m07-run-1", "policy": variant, "budget": {"max_model_calls": cap},
               "initial_state": initial.model_dump(mode="json"),
               "declared": {"config": {"policy": variant, "run_id": child_id, "budget": {"max_model_calls": cap}},
                            "source_snapshot_sha256": hashlib.sha256(canonical_json(vars(bundle)).encode()).hexdigest(), "input_sha256": {}}}
    RunStore.begin(workspace, child_id, request).complete(
        result, canonical=record.canonical, sidecar={"workflow": state.model_dump(mode="json"), "source": vars(bundle),
                                                  "profile": {"profile_id": "synthetic-profile"}}, raw_response=None, retain_raw=False)
    for name in ("checkpoints.sqlite", "dispatch.sqlite"):
        (workspace / f"runs/agent/{child_id}/{name}").write_bytes(b"Explicit synthetic inventory bytes; no SQL executed")
    return {**coordinate,
            "child_run_id": child_id, "execution": "executed", "status": "completed", "workflow_status": status,
            "record_revision": record.revision_id, "canonical": pin(workspace, f"runs/agent/{child_id}/canonical.json"),
            "sidecar": pin(workspace, f"runs/agent/{child_id}/sidecar.json"),
            "result_artifact": pin(workspace, f"runs/agent/{child_id}/result.json"),
            "artifacts": {name: pin(workspace, f"runs/agent/{child_id}/{name}") for name in (
                "request.json", "result.json", "canonical.json", "sidecar.json", "checkpoints.sqlite", "dispatch.sqlite")},
            "original_resources": {"model_calls": 1, "uncertain_model_calls": 0, "input_tokens_known": 0,
                                   "output_tokens_known": 0, "usage_unknown": True, "elapsed_ms": 1.0},
            "invocation_resources": {"new_model_calls": 1, "new_live_requests": 0, "elapsed_ms": 1.0},
            "unresolved_issues": 0, "truncated": False}


def parent(workspace, scope, *, omit=(), failures=(), state_status="finished", mutate=None, run_id="study-parent"):
    dummy = {"path": "configs/agent/synthetic-workflow.json", "sha256": "a" * 64}
    spec = StudySpec(description="Synthetic fixture", primary_outcome="Predeclared reference diagnostic",
                     counting_policy="All cells retained", evaluation_protocol_sha256=study.protocol_digest(scope),
                     publications=tuple({"publication_id": p.publication_id, "split": p.split,
                                         "workflows": {v: dummy for v in scope.variants}} for p in scope.publications),
                     variants=scope.variants, caps=scope.caps, repeats=scope.repeats,
                     max_total_model_calls=len(scope.publications) * len(scope.variants) * scope.repeats * sum(scope.caps))
    cells = []
    planned = study_runner.expected_cells(spec, run_id)
    for publication in scope.publications:
        for variant in scope.variants:
            for cap in scope.caps:
                for repeat in range(scope.repeats):
                    key = (publication.publication_id, variant, cap, repeat)
                    if key in omit:
                        continue
                    coordinate = next(c for c in planned if (c["publication_id"], c["variant"], c["cap"], c["repeat"]) == key)
                    value = child(workspace, publication, coordinate,
                                  quality=0 if variant == "fixed" else 1, status=state_status)
                    if key in failures:
                        value.update(status="failed", execution="failed", canonical=None, sidecar=None, result_artifact=None,
                                     error_code="synthetic_timeout", original_resources={"model_calls": None,
                                         "uncertain_model_calls": cap, "input_tokens_known": None, "output_tokens_known": None,
                                         "usage_unknown": True, "elapsed_ms": None},
                                     invocation_resources={"new_model_calls": None, "new_live_requests": 0, "elapsed_ms": None})
                    cells.append(value)
    result = {"version": "m09-study-result-1", "spec": spec.model_dump(mode="json"),
              "spec_sha256": hashlib.sha256(canonical_json(spec.model_dump(mode="json")).encode()).hexdigest(),
              "cells": cells, "budget": {}, "accounting": {}}
    if mutate:
        mutate(result)
    frozen_publications = []
    for publication in scope.publications:
        source = publication.sources[0]
        bundle = SimpleNamespace(source_id=source.path, source_sha256=source.sha256, representation_sha256="b" * 64)
        initial = snapshot({"fixture_quality": 0}, bundle, SimpleNamespace(profile_id="synthetic-profile"))
        frozen_publications.append({"publication_id": publication.publication_id, "initial_revision": initial.revision_id,
                                    "profile": {"profile_id": "synthetic-profile"}, "source_snapshot_sha256": hashlib.sha256(
                                        canonical_json(vars(bundle)).encode()).hexdigest()})
    dry_run = {"spec": spec.model_dump(mode="json"), "frozen": {"publications": frozen_publications, "input_sha256": {}, "common_budget": {}}}
    RunStore.begin(workspace, run_id, {"spec": spec.model_dump(mode="json"), "spec_sha256": hashlib.sha256(
        canonical_json(spec.model_dump(mode="json")).encode()).hexdigest(), "dry_run": dry_run, "cells": planned}).complete(
        result, canonical=None, sidecar={"synthetic": True}, raw_response=None, retain_raw=False)
    return result


def collect(workspace, scope, **kwargs):
    return study.collect_study(workspace, "study-parent", scope, run_id="study-report", **kwargs)


def test_collection_scores_stored_raw_records_and_reproduces_without_rescoring(workspace, scorer):
    scope = protocol(workspace)
    parent(workspace, scope)
    first = collect(workspace, scope)
    assert first["execution"] == "offline_collection" and first["new_evaluations"] == 2
    assert len(scorer) == 2
    result = first["result"]
    assert result["coverage"] == {"planned_cells": 2, "submitted_cells": 2, "execution_statuses": {"completed": 2}}
    assert result["paired_primary"][0]["mean_publication_delta"] == 1.0
    assert result["paired_primary"][0]["interval_95"] is None
    assert result["empirical_status"] == "blocked"
    before = {p.name: p.read_bytes() for p in (workspace / "runs/agent/study-report").iterdir()}
    replay = collect(workspace, scope)
    assert replay["execution"] == "cached_replay" and replay["new_evaluations"] == 0 and len(scorer) == 2
    assert before == {p.name: p.read_bytes() for p in (workspace / "runs/agent/study-report").iterdir()}
    assert not (workspace / "outputs").exists() and not (workspace / "paper").exists()


def test_failed_and_missing_cells_remain_visible_and_unpaired(workspace, scorer):
    scope = protocol(workspace, count=2)
    parent(workspace, scope, omit=(("paper-0", "generic_llm", 2, 0),), failures=(("paper-1", "generic_llm", 2, 0),))
    result = collect(workspace, scope)["result"]
    assert result["coverage"]["execution_statuses"] == {"completed": 2, "missing": 1, "failed": 1}
    assert len(result["strata"][0]["result"]["records"]) == 4
    assert result["paired_primary"][0]["paired_publications"] == 0
    assert result["paired_primary"][0]["mean_publication_delta"] is None
    assert result["resources"]["original_resources"]["model_calls"]["unknown_cells"] == 2


@pytest.mark.parametrize("status", ["budget_exhausted", "unresolved", "failed", "cancelled", "timed_out", "interrupted", "no_progress"])
def test_last_valid_prediction_scored_with_terminal_status_preserved(workspace, scorer, status):
    scope = protocol(workspace)
    parent(workspace, scope, state_status=status)
    result = collect(workspace, scope)["result"]
    assert len(scorer) == 2
    assert all(c["workflow_status"] == status for c in result["cells"])
    assert result["denominator_policy"] == "last_valid_canonical_all_planned_cells_visible"


def test_repeats_are_averaged_within_publication_and_caps_stay_separate(workspace, scorer):
    scope = protocol(workspace, count=2, caps=(1, 3), repeats=2)
    parent(workspace, scope)
    result = collect(workspace, scope)["result"]
    assert len(result["strata"]) == 4 and len(scorer) == 16
    assert len(result["paired_primary"]) == 2
    for row in result["paired_primary"]:
        assert row["paired_publications"] == 2 and row["interval_95"] == [1.0, 1.0]
        assert [p["matched_repeats"] for p in row["publications"]] == [2, 2]
    assert study.render_report(result) == study.render_report(result)


def test_unknown_tokens_and_independent_diagnostics_never_become_zero(workspace, scorer):
    scope = protocol(workspace)
    parent(workspace, scope)
    result = collect(workspace, scope)["result"]
    tokens = result["resources"]["original_resources"]["input_tokens_known"]
    assert tokens == {"known_subtotal": 0, "total": None, "unknown_cells": 2}
    assert all(item["value"] is None and item["status"] == "not_supplied" for item in result["independent_diagnostics"].values())
    assert result["resources"]["cost"]["value"] is None
    text = (workspace / "runs/agent/study-report/report.md").read_text()
    assert "not accuracy or effort evidence" in text and "Pooled metrics are not publication-level means" in text


@pytest.mark.parametrize("field,value", [("stage", "pilot"), ("stage", "confirmatory"), ("minimum_effect", 0.1),
                                        ("primary_metric", "recall"), ("bootstrap_seed", 1)])
def test_post_execution_protocol_changes_rejected_before_any_evaluator_or_gold_read(workspace, scorer, monkeypatch, field, value):
    scope = protocol(workspace)
    parent(workspace, scope)
    changed = study.StudyEvaluationProtocol.model_validate_json(json.dumps(scope.model_dump(mode="json") | {field: value}))
    monkeypatch.setattr(study, "verify_evaluator", lambda *a: pytest.fail("evaluator accessed before matching protocol"))
    monkeypatch.setattr(study, "read_artifact", lambda *a, **k: pytest.fail("artifact accessed before matching protocol"))
    with pytest.raises(ValueError, match="not frozen"):
        collect(workspace, changed)
    assert not scorer


@pytest.mark.parametrize("stage", ["pilot", "confirmatory"])
def test_matching_stage_label_cannot_promote_synthetic_runtime(workspace, scorer, stage):
    scope = protocol(workspace)
    scope = study.StudyEvaluationProtocol.model_validate_json(json.dumps(scope.model_dump(mode="json") | {"stage": stage}))
    parent(workspace, scope)
    with pytest.raises(ValueError, match="synthetic engineering only"):
        collect(workspace, scope)
    assert not scorer


@pytest.mark.parametrize("target", ["canonical", "sidecar", "result_artifact"])
def test_mutated_child_artifacts_fail_closed_before_scoring(workspace, scorer, target):
    scope = protocol(workspace)
    result = parent(workspace, scope)
    path = workspace / result["cells"][0][target]["path"]
    path.write_bytes(b"{}")
    with pytest.raises(ValueError):
        collect(workspace, scope)
    assert not scorer


@pytest.mark.parametrize("role", ["gold", "source"])
def test_mutated_protocol_artifact_fails_closed(workspace, scorer, role):
    scope = protocol(workspace)
    parent(workspace, scope)
    ref = scope.publications[0].gold if role == "gold" else scope.publications[0].sources[0]
    (workspace / ref.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen hash"):
        collect(workspace, scope)
    assert not scorer


@pytest.mark.parametrize("mutation", [
    lambda r: r["cells"].append(r["cells"][0]),
    lambda r: r["cells"][0].update(cap=32),
    lambda r: r["cells"][0].update(workflow_status="failed"),
    lambda r: r["cells"][0].update(record_revision="stale"),
    lambda r: r["cells"][0].update(unresolved_issues=1),
    lambda r: r["cells"][0].update(canonical=None),
    lambda r: r["cells"][0]["original_resources"].update(model_calls=2),
    lambda r: r["cells"][0]["original_resources"].update(input_tokens_known=False),
    lambda r: r["cells"][0]["invocation_resources"].update(new_live_requests=True),
    lambda r: r["cells"][0]["invocation_resources"].update(new_model_calls=99),
    lambda r: r["cells"][0]["invocation_resources"].update(new_model_calls=0),
    lambda r: r["cells"][0].update(execution="cached_replay"),
    lambda r: r["cells"][0].update(truncated=True),
    lambda r: r["cells"][0].update(child_run_id="unrelated-run"),
    lambda r: r["cells"][0].update(execution="imaginary_mode"),
    lambda r: r["cells"][0]["canonical"].update(path="outputs/changed.json"),
    lambda r: r.update(spec_sha256="0" * 64),
])
def test_forged_collection_metadata_rejected(workspace, scorer, mutation):
    scope = protocol(workspace)
    parent(workspace, scope, mutate=mutation)
    with pytest.raises(ValueError):
        collect(workspace, scope)
    assert not scorer


def test_report_tamper_or_disappearance_rejects_replay(workspace, scorer):
    scope = protocol(workspace)
    parent(workspace, scope)
    collect(workspace, scope)
    (workspace / "runs/agent/study-report/report.md").write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        collect(workspace, scope)
    assert len(scorer) == 2


def test_missing_gold_declaration_reported_blocked_without_invented_score(workspace, scorer):
    scope = protocol(workspace)
    value = scope.model_dump(mode="json")
    value["publications"][0]["gold"] = None
    scope = study.StudyEvaluationProtocol.model_validate_json(json.dumps(value))
    parent(workspace, scope)
    result = collect(workspace, scope)["result"]
    assert not scorer
    assert all(r["status"] == "blocked" for r in result["strata"][0]["result"]["records"])
    assert result["paired_primary"][0]["paired_publications"] == 0


@pytest.mark.parametrize("field,value", [("caps", [True]), ("caps", [1, 1]), ("repeats", 0),
                                        ("variants", ["fixed", "fixed"]), ("baseline_variant", "absent"),
                                        ("primary_field", "unlisted"), ("bootstrap_samples", 2001)])
def test_protocol_rejects_ambiguous_or_unbounded_scope(workspace, field, value):
    scope = protocol(workspace)
    with pytest.raises(ValidationError):
        study.StudyEvaluationProtocol.model_validate_json(json.dumps(scope.model_dump(mode="json") | {field: value}))


def test_protocol_file_roundtrip_semantic_hash_and_strict_json(workspace):
    scope = protocol(workspace)
    write(workspace, "configs/agent/protocol.json", scope.model_dump(mode="json"))
    assert study.load_protocol(workspace, "configs/agent/protocol.json") == scope
    assert study.protocol_digest(scope) == hashlib.sha256(canonical_json(scope.model_dump(mode="json")).encode()).hexdigest()
    (workspace / "configs/agent/protocol.json").write_text('{"version":"x","version":"y"}')
    with pytest.raises(ValueError, match="Duplicate"):
        study.load_protocol(workspace, "configs/agent/protocol.json")


def test_cli_safe_summary_and_sanitized_failure(workspace, scorer, monkeypatch, capsys):
    scope = protocol(workspace)
    parent(workspace, scope)
    write(workspace, "configs/agent/protocol.json", scope.model_dump(mode="json"))
    import sys
    args = ["study", "--root", str(workspace), "--study-run-id", "study-parent", "--protocol", "configs/agent/protocol.json", "--run-id", "study-report"]
    monkeypatch.setattr(sys, "argv", args)
    assert study.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["model_calls"] == 0 and output["empirical_status"] == "blocked"
    monkeypatch.setattr(study, "collect_study", lambda *a, **k: (_ for _ in ()).throw(ValueError("private fixture text")))
    assert study.main() == 2
    assert "private fixture text" not in capsys.readouterr().out


@pytest.mark.parametrize("filename", ["checkpoints.sqlite", "dispatch.sqlite", "request.json"])
def test_nonprediction_child_artifact_changes_also_reject_collection(workspace, scorer, filename):
    scope = protocol(workspace)
    result = parent(workspace, scope)
    ref = result["cells"][0]["artifacts"][filename]
    (workspace / ref["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        collect(workspace, scope)
    assert not scorer


def test_failed_cell_cannot_omit_an_existing_child_artifact_from_inventory(workspace, scorer):
    scope = protocol(workspace)
    parent(workspace, scope, failures=(("paper-0", "fixed", 2, 0),),
           mutate=lambda r: r["cells"][0].update(artifacts={}))
    with pytest.raises(ValueError, match="inventory changed"):
        collect(workspace, scope)
    assert not scorer


def test_changed_runtime_freeze_rejected_before_evaluator_or_gold(workspace, scorer, monkeypatch):
    scope = protocol(workspace)
    parent(workspace, scope)
    monkeypatch.setattr(study_runner, "dry_run", lambda *a: {"changed": "config or profile"})
    monkeypatch.setattr(study, "verify_evaluator", lambda *a: pytest.fail("evaluator must not be consulted"))
    with pytest.raises(ValueError, match="freeze"):
        collect(workspace, scope)
    assert not scorer


def modify_child_request(workspace, cell, change):
    folder = workspace / "runs/agent" / cell["child_run_id"]
    request = json.loads((folder / "request.json").read_bytes())
    change(request["request"])
    request["request_sha256"] = _digest(_json_bytes(request["request"]))
    (folder / "request.json").write_bytes(_json_bytes(request))
    result = json.loads((folder / "result.json").read_bytes())
    result["request_sha256"] = request["request_sha256"]
    result["completion_sha256"] = _digest(_json_bytes({k: v for k, v in result.items() if k != "completion_sha256"}))
    (folder / "result.json").write_bytes(_json_bytes(result))
    for name in ("request.json", "result.json"):
        cell["artifacts"][name] = pin(workspace, f'runs/agent/{cell["child_run_id"]}/{name}')
    cell["result_artifact"] = cell["artifacts"]["result.json"]


@pytest.mark.parametrize("change", [
    lambda r: r.update(policy="issue_aware"),
    lambda r: r["budget"].update(max_model_calls=3),
    lambda r: r["declared"]["config"].update(policy="issue_aware"),
    lambda r: r["declared"]["config"].update(run_id="another-run"),
    lambda r: r["declared"].update(source_snapshot_sha256="c" * 64),
    lambda r: r["declared"].update(input_sha256={"invented-source": "a" * 64}),
])
def test_rehashed_child_request_cannot_relabel_policy_cap_or_source(workspace, scorer, change):
    scope = protocol(workspace)
    parent(workspace, scope, mutate=lambda r: modify_child_request(workspace, r["cells"][0], change))
    with pytest.raises(ValueError):
        collect(workspace, scope)
    assert not scorer


@pytest.mark.parametrize("kind", ["oversized", "deep"])
def test_protocol_read_is_bounded(workspace, kind):
    value = b" " * 2_000_001 if kind == "oversized" else ("[" * 34 + "0" + "]" * 34).encode()
    write(workspace, "configs/offline/invalid.json", value)
    with pytest.raises(ValueError, match="limit"):
        study.load_protocol(workspace, "configs/offline/invalid.json")


def test_bootstrap_uses_publications_not_repeat_count_and_excludes_zero_support(workspace):
    scope = protocol(workspace, count=2, repeats=2)
    scored = []
    for repeat in range(2):
        comparisons = []
        for number in range(2):
            delta = [1.0, -0.5][number]
            comparisons.append({"field": scope.primary_field, "publication_group": f"paper-{number}",
                "split": "calibration", "data_kind": "synthetic", "variant": "generic_llm",
                "baseline_support": {"support_status": "known", "support_count": 1, "metrics": {"f1": 0.5}},
                "variant_support": {"support_status": "known", "support_count": 1, "metrics": {"f1": 0.5 + delta}}})
        scored.append({"cap": 2, "repeat": repeat, "result": {"comparisons": comparisons}})
    row = study.paired_primary(scored, scope)[0]
    assert row["paired_publications"] == 2 and row["mean_publication_delta"] == 0.25
    assert row["interval_95"] == [-0.5, 1.0]
    assert study.paired_primary(scored, scope)[0] == row
    for stratum in scored:
        stratum["result"]["comparisons"][0]["variant_support"]["support_count"] = 0
    row = study.paired_primary(scored, scope)[0]
    assert row["paired_publications"] == 1 and row["interval_95"] is None


def test_report_exposes_budget_stops_absolute_scores_support_and_calls(workspace, scorer):
    scope = protocol(workspace)
    parent(workspace, scope, state_status="budget_exhausted")
    collect(workspace, scope)
    report = (workspace / "runs/agent/study-report/report.md").read_text()
    for value in ("budget_exhausted", "Primary score / support", "Original calls", "Study invocation calls",
                  "references / datasets.attrs.sample_refs", "entity_recovery / samples", "gold_references", "unknown cells: 2"):
        assert value in report


def test_cached_child_keeps_original_work_but_records_zero_invocation_calls(workspace, scorer):
    scope = protocol(workspace)
    def cached(result):
        result["cells"][0]["execution"] = "cached_replay"
        result["cells"][0]["invocation_resources"]["new_model_calls"] = 0
    parent(workspace, scope, mutate=cached)
    result = collect(workspace, scope)["result"]
    assert result["resources"]["original_resources"]["model_calls"]["total"] == 2
    assert result["resources"]["invocation_resources"]["new_model_calls"]["total"] == 1


def as_failed(cell):
    cell.update(status="failed", execution="failed", canonical=None, sidecar=None, result_artifact=None,
                error_code="synthetic_failure")


@pytest.mark.parametrize("change", [
    lambda c: (c["original_resources"].update(model_calls=999), c["invocation_resources"].update(new_model_calls=999)),
    lambda c: c["original_resources"].update(model_calls=2),
    lambda c: c["original_resources"].update(input_tokens_known=999),
    lambda c: c["original_resources"].update(usage_unknown=False),
    lambda c: c["invocation_resources"].update(new_model_calls=3),
    lambda c: c["original_resources"].update(uncertain_model_calls=True),
])
def test_failed_known_resources_cannot_bypass_cap_or_saved_child_observations(workspace, scorer, change):
    scope = protocol(workspace)
    def mutate(result):
        cell = result["cells"][0]
        as_failed(cell)
        change(cell)
    parent(workspace, scope, mutate=mutate)
    with pytest.raises(ValueError):
        collect(workspace, scope)
    assert not scorer


@pytest.mark.parametrize("change", [
    lambda c: c["original_resources"].update(uncertain_model_calls=0),
    lambda c: c["original_resources"].update(input_tokens_known=0),
    lambda c: c["original_resources"].update(usage_unknown=False),
    lambda c: c["invocation_resources"].update(new_model_calls=0),
])
def test_unknown_failed_resources_keep_full_reservation_and_unknown_usage(workspace, scorer, change):
    scope = protocol(workspace)
    parent(workspace, scope, failures=(("paper-0", "fixed", 2, 0),), mutate=lambda r: change(r["cells"][0]))
    with pytest.raises(ValueError):
        collect(workspace, scope)
    assert not scorer


def test_failed_cell_can_retain_verified_known_work_without_a_scoring_prediction(workspace, scorer):
    scope = protocol(workspace)
    parent(workspace, scope, mutate=lambda r: as_failed(r["cells"][0]))
    result = collect(workspace, scope)["result"]
    assert result["coverage"]["execution_statuses"] == {"failed": 1, "completed": 1}
    assert result["resources"]["original_resources"]["model_calls"]["total"] == 2
    assert len(scorer) == 1


def modify_child_sidecar(workspace, cell, change):
    folder = workspace / "runs/agent" / cell["child_run_id"]
    sidecar = json.loads((folder / "sidecar.json").read_bytes())
    change(sidecar)
    raw = _json_bytes(sidecar)
    (folder / "sidecar.json").write_bytes(raw)
    completion = json.loads((folder / "result.json").read_bytes())
    completion["artifacts"]["sidecar.json"] = {"sha256": _digest(raw), "bytes": len(raw)}
    completion["completion_sha256"] = _digest(_json_bytes({k: v for k, v in completion.items() if k != "completion_sha256"}))
    (folder / "result.json").write_bytes(_json_bytes(completion))
    for name in ("sidecar.json", "result.json"):
        cell["artifacts"][name] = pin(workspace, f'runs/agent/{cell["child_run_id"]}/{name}')
    cell["sidecar"] = cell["artifacts"]["sidecar.json"]
    cell["result_artifact"] = cell["artifacts"]["result.json"]


@pytest.mark.parametrize("change", [
    lambda s: s.update(extraction={"provider": {"input_tokens": 999, "output_tokens": 888}}),
    lambda s: s["workflow"].update(status="failed"),
    lambda s: s["source"].update(source_sha256="c" * 64),
    lambda s: s["profile"].update(profile_id="another-profile"),
])
def test_rehashed_sidecar_cannot_inject_scoring_metadata_or_change_frozen_state(workspace, scorer, change):
    scope = protocol(workspace)
    parent(workspace, scope, mutate=lambda r: modify_child_sidecar(workspace, r["cells"][0], change))
    with pytest.raises(ValueError, match="sidecar differs"):
        collect(workspace, scope)
    assert not scorer
