"""Offline publication/cell orchestration; labels never enter runtime execution."""
import json
from pathlib import Path
import shutil

import pytest

from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.agent.providers import MockProvider
from meta_data_extraction.agent.storage import RunConflictError, RunIncompleteError
from meta_data_extraction.agent.study_types import StudySpec
import meta_data_extraction.agent.study_runner as runner

CONFIGS = {"fixed": "configs/agent/workflow_fixed.json", "issue_aware": "configs/agent/workflow_issue_aware.json",
           "generic_llm": "configs/agent/workflow_generic.json"}


def pin(root, relative):
    return {"path": relative, "sha256": sha256((root / relative).read_bytes())}


def changed(spec, **values):
    return StudySpec.model_validate(spec.model_dump(mode="json") | values)


def one_cell(spec):
    return changed(spec, variants=["fixed"], publications=[{**spec.publications[0].model_dump(mode="json"),
        "workflows": {"fixed": spec.publications[0].workflows["fixed"].model_dump(mode="json")}}], caps=[5], max_total_model_calls=5)


@pytest.fixture
def study(workspace, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    names = ["schemas/prompt_template.md", "configs/agent/record_inspection.json", *CONFIGS.values()]
    names.extend(str(path.relative_to(repository)) for prefix in ("m06_", "m07_")
                 for path in (repository / "tests/agent/fixtures").glob(prefix + "*"))
    for name in names:
        destination = workspace / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, destination)
    monkeypatch.setattr(runner, "_implementation", lambda root: {"study-runtime": "a" * 64, "workflow": runner.graph_implementation()})
    monkeypatch.setattr(runner.workflow_inspection, "_implementation", lambda root: {"inspection-runtime": "b" * 64})
    spec = StudySpec(description="Invented publication repair study", primary_outcome="Reference diagnostic",
        counting_policy="Retain every planned cell, aggregate by publication; no independent empirical labels.",
        evaluation_protocol_sha256="c" * 64,
        publications=({"publication_id": "invented-a", "workflows": {k: pin(workspace, v) for k, v in CONFIGS.items()}},),
        variants=tuple(CONFIGS), caps=(2, 5), max_total_model_calls=21)
    return workspace, spec


def test_dry_run_validates_all_inputs_without_dispatch_or_writes(study, monkeypatch):
    root, spec = study
    monkeypatch.setattr(MockProvider, "generate", lambda *a: pytest.fail("dry-run dispatch"))
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    report = runner.dry_run(root, spec)
    assert report["estimated_cells"] == 6 and report["maximum_model_calls"] == 21
    assert report["live_requests"] == 0 and report["readiness"] == "ready"
    assert len(report["frozen"]["publications"]) == 1
    assert report["frozen"]["publications"][0]["standards_resources"] == []
    assert before == sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    assert not (root / "runs").exists()


def test_dry_run_never_opens_opaque_evaluation_protocol(study, monkeypatch):
    root, spec = study
    original = Path.read_bytes
    reads = []
    def guarded(path):
        relative = str(path.relative_to(root)).replace("\\", "/") if path.is_relative_to(root) else str(path)
        assert not relative.startswith(("data/gold/", "data/annotations/", "configs/offline/", "src/meta_data_extraction/offline/"))
        assert relative != "configs/alignment_helper.yaml" and relative != ".env"
        reads.append(relative)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    report = runner.dry_run(root, spec)
    assert report["spec"]["evaluation_protocol_sha256"] == "c" * 64
    assert reads


def test_all_cells_accounted_with_caps_unknown_tokens_and_verified_replay(study, monkeypatch):
    root, spec = study
    run = runner.run_study(root, spec, run_id="study-full")
    assert run["execution"] == "executed" and run["new_live_requests"] == 0
    cells = run["result"]["cells"]
    assert len(cells) == 6 and all(c["status"] == "completed" for c in cells)
    assert all(c["original_resources"]["model_calls"] <= c["cap"] for c in cells)
    assert all(c["original_resources"]["usage_unknown"] for c in cells)
    assert all(c["canonical"] and c["sidecar"] and c["result_artifact"] for c in cells)
    generic_low = next(c for c in cells if c["variant"] == "generic_llm" and c["cap"] == 2)
    assert generic_low["workflow_status"] == "budget_exhausted" and generic_low["unresolved_issues"] == 2
    assert generic_low["original_resources"]["model_calls"] == 2
    assert next(c for c in cells if c["variant"] == "generic_llm" and c["cap"] == 5)["original_resources"]["model_calls"] == 5
    accounting = run["result"]["accounting"]
    assert accounting["original_model_calls_known"] == run["new_model_calls"] == 15
    assert accounting["input_tokens_known"] == 0 and accounting["usage_unknown"]
    artifacts = {str(p): p.read_bytes() for p in (root / "runs").rglob("*") if p.is_file()}
    monkeypatch.setattr(runner, "inspect_workflow", lambda *a, **k: pytest.fail("replay dispatch"))
    replay = runner.run_study(root, spec, run_id="study-full", replay_only=True)
    assert replay["execution"] == "cached_replay" and replay["new_model_calls"] == 0
    assert replay["result"] == run["result"]
    assert artifacts == {str(p): p.read_bytes() for p in (root / "runs").rglob("*") if p.is_file()}


@pytest.mark.parametrize("relative", ["tests/agent/fixtures/m06_source.md", "tests/agent/fixtures/m07_checker_after.json",
    "tests/agent/fixtures/m07_policy_finish.json", "configs/agent/record_inspection.json", "configs/agent/workflow_fixed.json"])
def test_all_transitive_pins_checked_before_parent_claim(study, relative, monkeypatch):
    root, spec = study
    (root / relative).write_text("changed", encoding="utf-8")
    monkeypatch.setattr(runner, "inspect_workflow", lambda *a, **k: pytest.fail("dispatch"))
    with pytest.raises(ValueError):
        runner.run_study(root, spec, run_id="bad-pins")
    assert not (root / "runs").exists()


@pytest.mark.parametrize("forbidden", ["data/gold/hidden.json", "data/annotations/hidden.json", "configs/alignment_helper.yaml",
                                      "outputs/hidden.json", "paper/hidden.json", ".env"])
def test_forbidden_workflow_path_is_rejected_before_read(study, forbidden, monkeypatch):
    root, spec = study
    value = spec.model_dump(mode="json")
    value["publications"][0]["workflows"]["fixed"]["path"] = forbidden
    original = Path.read_bytes
    def guarded(path):
        assert str(path) != str(root / forbidden), "forbidden runtime read"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError):
        runner.dry_run(root, StudySpec.model_validate(value))


def repin_config(root, spec, policy, updates):
    value = spec.model_dump(mode="json")
    relative = CONFIGS[policy]
    config = json.loads((root / relative).read_text(encoding="utf-8"))
    config.update(updates)
    (root / relative).write_text(json.dumps(config), encoding="utf-8")
    value["publications"][0]["workflows"][policy] = pin(root, relative)
    return StudySpec.model_validate(value)


def test_different_selector_rejected_before_run(study):
    root, spec = study
    spec = repin_config(root, spec, "fixed", {"policy": "issue_aware"})
    with pytest.raises(ValueError, match="selector"):
        runner.dry_run(root, spec)


def test_noncall_budgets_must_match(study):
    root, spec = study
    budget = json.loads((root / CONFIGS["fixed"]).read_text())["budget"]
    spec = repin_config(root, spec, "fixed", {"budget": budget | {"total_actions": 6}})
    with pytest.raises(ValueError, match="budgets"):
        runner.dry_run(root, spec)


def test_checker_access_must_match_even_when_both_fixtures_exist(study):
    root, spec = study
    config = json.loads((root / CONFIGS["fixed"]).read_text())
    responses = config["checker_responses"]
    spec = repin_config(root, spec, "fixed", {"checker_responses": list(reversed(responses))})
    with pytest.raises(ValueError, match="checker"):
        runner.dry_run(root, spec)


def test_same_source_cannot_count_as_two_publications(study):
    root, spec = study
    pubs = spec.model_dump(mode="json")["publications"]
    spec = changed(spec, publications=[*pubs, pubs[0] | {"publication_id": "invented-b"}], max_total_model_calls=42)
    with pytest.raises(ValueError, match="independent publications"):
        runner.dry_run(root, spec)


def test_coordinates_repeat_without_inventing_independent_documents(study):
    root, spec = study
    spec = changed(spec, repeats=3, max_total_model_calls=63)
    cells = runner.expected_cells(spec, "repeat-study")
    assert len(cells) == 18 and {c["repeat"] for c in cells} == {0, 1, 2}
    assert {c["publication_id"] for c in cells} == {"invented-a"}
    assert len({c["child_run_id"] for c in cells}) == len(cells)
    assert not {c["child_run_id"] for c in cells} & {c["child_run_id"] for c in runner.expected_cells(spec, "other-study")}


def test_failed_cells_remain_visible_with_conservative_uncertainty(study, monkeypatch):
    root, spec = study
    def fail(*a, **k):
        raise OSError("private document text and secret should not be copied")
    monkeypatch.setattr(runner, "inspect_workflow", fail)
    run = runner.run_study(root, spec, run_id="failed-study")
    assert run["new_model_calls"] is None and run["new_live_requests"] == 0
    assert len(run["result"]["cells"]) == 6
    assert all(c["status"] == "failed" and c["canonical"] is None for c in run["result"]["cells"])
    assert run["result"]["accounting"]["uncertain_model_calls"] == 21
    assert "private document" not in json.dumps(run)
    assert runner.run_study(root, spec, run_id="failed-study")["new_model_calls"] == 0


def test_input_change_between_cells_stops_dispatch_and_keeps_remaining_cells(study, monkeypatch):
    root, spec = study
    original = runner.inspect_workflow
    called = []
    def change_after_first(*args, **kwargs):
        called.append(kwargs["run_id"])
        result = original(*args, **kwargs)
        (root / "tests/agent/fixtures/m07_checker_after.json").write_bytes(b"changed between cells")
        return result
    monkeypatch.setattr(runner, "inspect_workflow", change_after_first)
    result = runner.run_study(root, spec, run_id="changed-between-cells")
    assert len(called) == 1 and len(result["result"]["cells"]) == 6
    assert result["result"]["cells"][0]["status"] == "completed"
    assert all(cell["status"] == "failed" for cell in result["result"]["cells"][1:])
    assert result["new_model_calls"] is None


def test_a_provider_exception_after_a_completed_child_is_not_counted_as_zero(study, monkeypatch):
    root, spec = study
    original = runner.inspect_workflow
    def failed_return(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("lost caller acknowledgement")
    monkeypatch.setattr(runner, "inspect_workflow", failed_return)
    result = runner.run_study(root, spec, run_id="lost-return")
    assert all(cell["status"] == "failed" and cell["original_resources"]["model_calls"] is None
               for cell in result["result"]["cells"])
    assert result["new_model_calls"] is None
    assert result["result"]["accounting"]["uncertain_model_calls"] == 21


def test_incomplete_parent_never_automatically_repeats_children(study, monkeypatch):
    root, spec = study
    original = runner.RunStore.complete
    def fail_parent(store, *a, **k):
        if store.run.run_id == "interrupted-study":
            raise OSError("publication interrupted")
        return original(store, *a, **k)
    monkeypatch.setattr(runner.RunStore, "complete", fail_parent)
    with pytest.raises(OSError):
        runner.run_study(root, spec, run_id="interrupted-study")
    monkeypatch.setattr(runner, "inspect_workflow", lambda *a, **k: pytest.fail("ambiguous retry"))
    with pytest.raises(RunIncompleteError):
        runner.run_study(root, spec, run_id="interrupted-study")
    with pytest.raises(RunConflictError):
        runner.run_study(root, spec, run_id="interrupted-study", replay_only=True)


def test_explicit_replay_cannot_create_missing_parent_or_child(study, monkeypatch):
    root, spec = study
    monkeypatch.setattr(runner.RunStore, "begin", lambda *a, **k: pytest.fail("replay claimed run"))
    monkeypatch.setattr(runner, "inspect_workflow", lambda *a, **k: pytest.fail("replay dispatch"))
    with pytest.raises(RunConflictError):
        runner.run_study(root, spec, run_id="missing-replay", replay_only=True)
    assert not (root / "runs").exists()


@pytest.mark.parametrize("name", ["canonical.json", "sidecar.json", "result.json"])
def test_missing_child_export_reports_failed_cell_without_partial_prediction(study, name, monkeypatch):
    root, spec = study
    spec = one_cell(spec)
    original = runner.inspect_workflow
    def missing_export(*args, **kwargs):
        report = original(*args, **kwargs)
        (root / "runs/agent" / kwargs["run_id"] / name).unlink()
        return report
    monkeypatch.setattr(runner, "inspect_workflow", missing_export)
    run = runner.run_study(root, spec, run_id="missing-export")
    cell = run["result"]["cells"][0]
    assert cell["status"] == "failed" and cell["error_code"] == "cell_exports_incomplete"
    assert all(cell[key] is None for key in ("canonical", "sidecar", "result_artifact"))
    assert cell["original_resources"]["model_calls"] == run["new_model_calls"] == 2
    assert runner.run_study(root, spec, run_id="missing-export", replay_only=True)["new_model_calls"] == 0


@pytest.mark.parametrize("field", ["resources", "alias", "status", "revision", "unresolved", "truncated",
                                   "invocation", "aggregate", "budget"])
def test_rehashed_parent_metadata_forgery_is_rejected_against_child_state(study, field):
    root, spec = study
    spec = one_cell(spec)
    runner.run_study(root, spec, run_id="forged-parent")
    path = root / "runs/agent/forged-parent/result.json"
    envelope = json.loads(path.read_bytes())
    result = envelope["result"]
    cell = result["cells"][0]
    if field == "resources":
        cell["original_resources"]["model_calls"] = 1
    elif field == "alias":
        cell["canonical"]["sha256"] = "0" * 64
    elif field == "status":
        cell["workflow_status"] = "unresolved"
    elif field == "revision":
        cell["record_revision"] = "0" * 64
    elif field == "unresolved":
        cell["unresolved_issues"] = 2
    elif field == "truncated":
        cell["truncated"] = True
    elif field == "invocation":
        cell["invocation_resources"]["new_model_calls"] = 0
    elif field == "aggregate":
        result["accounting"]["original_model_calls_known"] = 0
    else:
        result["budget"]["reserved_model_calls"] = 0
    signed = {k: v for k, v in envelope.items() if k != "completion_sha256"}
    envelope["completion_sha256"] = sha256(runner._json_bytes(signed))
    path.write_bytes(runner._json_bytes(envelope))
    with pytest.raises(RunConflictError):
        runner.run_study(root, spec, run_id="forged-parent", replay_only=True)


@pytest.mark.parametrize("name", ["canonical.json", "sidecar.json", "result.json", "checkpoints.sqlite", "dispatch.sqlite"])
def test_completed_parent_rejects_child_artifact_changes(study, name):
    root, spec = study
    spec = one_cell(spec)
    result = runner.run_study(root, spec, run_id="integrity-study")
    child = result["result"]["cells"][0]["child_run_id"]
    (root / "runs/agent" / child / name).write_bytes(b"changed")
    with pytest.raises(RunConflictError):
        runner.run_study(root, spec, run_id="integrity-study")


def test_changed_implementation_or_scope_requires_new_study(study, monkeypatch):
    root, spec = study
    monkeypatch.setattr(runner, "inspect_workflow", lambda *a, **k: (_ for _ in ()).throw(ValueError()))
    runner.run_study(root, spec, run_id="version-study")
    with pytest.raises(RunConflictError):
        runner.run_study(root, changed(spec, primary_outcome="Different endpoint"), run_id="version-study")
    monkeypatch.setattr(runner, "_implementation", lambda root: {"study-runtime": "d" * 64})
    with pytest.raises(RunConflictError):
        runner.run_study(root, spec, run_id="version-study")


def test_runtime_source_does_not_import_offline_evaluation():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "from meta_data_extraction.offline" not in source
    assert "import meta_data_extraction.offline" not in source
