"""M07 CLI fixtures are invented; all provider execution is deterministic mock."""
import json
from pathlib import Path
import shutil
import sys

import pytest

from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.agent.storage import RunConflictError
import meta_data_extraction.agent.inspect_workflow as runner

CONFIGS = {"fixed": "configs/agent/workflow_fixed.json", "issue_aware": "configs/agent/workflow_issue_aware.json",
           "generic_llm": "configs/agent/workflow_generic.json"}
RUN_ID = "m07-cli-unit"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def amended(config, **changes):
    return runner.WorkflowRunConfig.model_validate(config.model_dump() | changes)


def repin(path, relative):
    return {"path": relative, "sha256": sha256(path.read_bytes())}


@pytest.fixture
def scope(workspace, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    names = ["schemas/prompt_template.md", "configs/agent/record_inspection.json", *CONFIGS.values()]
    names.extend(str(path.relative_to(repository)) for prefix in ("m06_", "m07_")
                 for path in (repository / "tests/agent/fixtures").glob(prefix + "*"))
    for name in names:
        destination = workspace / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, destination)
    monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic-cli-runtime": "a" * 64})
    configs = {policy: amended(runner.load_config(workspace, path), run_id=RUN_ID) for policy, path in CONFIGS.items()}
    return workspace, configs


@pytest.mark.parametrize("policy,calls", [("fixed", 2), ("issue_aware", 2), ("generic_llm", 5)])
def test_each_policy_finishes_same_two_corrections_with_honest_mock_accounting(scope, policy, calls):
    root, configs = scope
    report = runner.inspect_workflow(root, configs[policy])
    state = report["state"]
    assert report["execution"] == "executed" and report["new_model_calls"] == calls
    assert report["new_live_requests"] == 0 and state.model_calls == calls
    assert state.status == "finished" and all(issue.status == "resolved" for issue in state.issues)
    assert state.usage_unknown and state.uncertain_model_calls == 0
    assert state.record.canonical["datasets"][0]["dataset_description"] == "Measured at 300 K."
    assert state.record.canonical["datasets"][0]["sample_refs"] == ["sample_1"]
    assert len(state.record.canonical["samples"]) == 2
    assert [event.action.kind for event in state.events] == ["retrieve_source", "propose_update", "finish"]
    directory = root / "runs/agent" / RUN_ID
    assert not (directory / "raw-response.json").exists()
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.suffix == ".json"}
    replay = runner.inspect_workflow(root, configs[policy])
    assert replay["execution"] == "cached_replay" and replay["new_model_calls"] == 0
    assert replay["state"] == state
    assert before == {p.name: p.read_bytes() for p in directory.iterdir() if p.suffix == ".json"}


def test_policy_variants_share_frozen_initial_record_issues_catalog_and_budget(scope, monkeypatch):
    root, configs = scope
    captures = []
    def capture(root, run_id, request, initial, runtime, **kwargs):
        captures.append((request, initial, runtime))
        return {"captured": True}
    monkeypatch.setattr(runner, "run_workflow_graph", capture)
    for config in configs.values():
        assert runner.inspect_workflow(root, config) == {"captured": True}
    states = [item[1] for item in captures]
    assert states[0] == states[1] == states[2]
    assert len(states[0].issues) == 2 and all(issue.evidence == () for issue in states[0].issues)
    assert all(issue.proposal_ids == ("correct_dataset_a",) and not issue.definition_keys for issue in states[0].issues)
    budgets = [item[2].budget for item in captures]
    assert budgets[0] == budgets[1] == budgets[2]
    assert all(capture[0]["seed_new_model_calls"] == 0 for capture in captures)
    assert all(not capture[2].tools.checker.requests for capture in captures)
    assert all(capture[2].tools.checker.model_name == "m07-synthetic-checker" for capture in captures)
    assert captures[2][2].policy_provider.model_name == "m07-synthetic-policy"
    assert all(capture[0]["standards_resources"] == [] for capture in captures)
    assert not (root / "runs/agent" / RUN_ID).exists()


def test_generic_resume_uses_full_script_positions_and_shared_model_allowance(scope):
    root, configs = scope
    config = configs["generic_llm"]
    first = runner.inspect_workflow(root, config, max_steps=1)
    assert first["execution"] == "checkpointed" and first["new_model_calls"] == 1
    assert first["state"].model_calls == 1 and first["state"].status == "running"
    second = runner.inspect_workflow(root, config, max_steps=1)
    assert second["execution"] == "checkpointed" and second["new_model_calls"] == 3
    assert second["state"].model_calls == 4
    assert second["state"].events[-1].action.kind == "propose_update"
    third = runner.inspect_workflow(root, config)
    assert third["execution"] == "resumed" and third["new_model_calls"] == 1
    assert third["state"].model_calls == 5 and third["state"].status == "finished"
    assert runner.inspect_workflow(root, config)["new_model_calls"] == 0


def test_low_generic_model_budget_preserves_original_record_and_unresolved_issues(scope):
    root, configs = scope
    config = configs["generic_llm"]
    config = amended(config, budget={**config.budget.model_dump(), "max_model_calls": 1})
    result = runner.inspect_workflow(root, config)
    assert result["state"].status == "budget_exhausted"
    assert result["state"].model_calls == 1 and result["new_model_calls"] == 1
    assert all(issue.status == "open" for issue in result["state"].issues)
    assert result["state"].record.revision_id == result["state"].initial_revision


@pytest.mark.parametrize("changes", [
    {"mode": "live"}, {"raw_retention": "retain"}, {"run_id": "../outputs"},
    {"policy": "unknown"}, {"checker_responses": []}, {"unknown_option": True},
])
def test_cli_configuration_is_explicit_bounded_and_offline(scope, changes):
    _, configs = scope
    with pytest.raises(ValueError):
        amended(configs["fixed"], **changes)


def test_policy_fixtures_required_only_for_generic_selector(scope):
    _, configs = scope
    with pytest.raises(ValueError, match="generic"):
        amended(configs["fixed"], policy_responses=configs["generic_llm"].policy_responses)
    with pytest.raises(ValueError, match="generic"):
        amended(configs["generic_llm"], policy_responses=[])


@pytest.mark.parametrize("field", ["seed", "checker_responses", "policy_responses"])
def test_every_input_pin_is_verified_before_claiming_a_run(scope, field):
    root, configs = scope
    config = configs["generic_llm"]
    if field == "seed":
        config = amended(config, seed={"path": config.seed.path, "sha256": "0" * 64})
    else:
        values = [pin.model_dump() for pin in getattr(config, field)]
        values[0]["sha256"] = "0" * 64
        config = amended(config, **{field: values})
    with pytest.raises(ValueError, match="declared hash"):
        runner.inspect_workflow(root, config)
    assert not (root / "runs/agent" / RUN_ID).exists()


@pytest.mark.parametrize("field,path", [
    ("seed", "configs/alignment_helper.yaml"), ("seed", ".env"),
    ("checker_responses", "data/gold/example.json"), ("policy_responses", "outputs/prediction.json"),
])
def test_forbidden_inputs_cannot_reach_seed_or_mock_provider(scope, field, path):
    root, configs = scope
    config = configs["generic_llm"]
    pin = {"path": path, "sha256": "a" * 64}
    config = amended(config, **{field: pin if field == "seed" else [pin, *getattr(config, field)[1:]]})
    with pytest.raises(ValueError):
        runner.inspect_workflow(root, config)
    assert not (root / "runs/agent" / RUN_ID).exists()


def test_seed_source_scope_is_checked_before_reading_source(scope):
    root, configs = scope
    config = configs["fixed"]
    path = root / config.seed.path
    seed = json.loads(path.read_bytes())
    seed["source"] = {"path": ".env", "sha256": "a" * 64}
    write_json(path, seed)
    config = amended(config, seed=repin(path, config.seed.path))
    with pytest.raises(ValueError):
        runner.inspect_workflow(root, config)
    assert not (root / "runs/agent" / RUN_ID).exists()


def test_changed_seed_source_hash_rejects_before_graph_dispatch(scope):
    root, configs = scope
    config = configs["fixed"]
    seed = json.loads((root / config.seed.path).read_bytes())
    source = root / seed["source"]["path"]
    source.write_bytes(source.read_bytes() + b"\nChanged source.")
    with pytest.raises(ValueError, match="declared hash"):
        runner.inspect_workflow(root, config)
    assert not (root / "runs/agent" / RUN_ID).exists()


@pytest.mark.parametrize("change", ["seed", "response", "code", "budget"])
def test_changed_execution_scope_requires_new_run_id(scope, monkeypatch, change):
    root, configs = scope
    config = configs["generic_llm"]
    runner.inspect_workflow(root, config)
    if change == "seed":
        path = root / config.seed.path
        seed = json.loads(path.read_bytes())
        seed["run_id"] = "changed-seed-declaration"
        write_json(path, seed)
        config = amended(config, seed=repin(path, config.seed.path))
    elif change == "response":
        values = [pin.model_dump() for pin in config.policy_responses]
        path = root / values[2]["path"]
        value = json.loads(path.read_bytes())
        value["summary"] = "Changed fixture statement."
        write_json(path, value)
        values[2] = repin(path, values[2]["path"])
        config = amended(config, policy_responses=values)
    elif change == "code":
        monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic-cli-runtime": "b" * 64})
    else:
        config = amended(config, budget={**config.budget.model_dump(), "total_actions": 11})
    with pytest.raises(RunConflictError):
        runner.inspect_workflow(root, config)


def test_failed_checker_fixture_cannot_produce_a_successful_correction(scope):
    root, configs = scope
    config = configs["fixed"]
    values = [pin.model_dump() for pin in config.checker_responses]
    path = root / values[0]["path"]
    path.write_bytes(b"not a checker judgment")
    values[0] = repin(path, values[0]["path"])
    config = amended(config, checker_responses=values)
    report = runner.inspect_workflow(root, config)
    state = report["state"]
    assert state.status == "pending_review" and state.pending_reviews
    assert state.record.revision_id == state.initial_revision
    assert any(event.code == "support_check_incomplete" for event in state.events)
    assert all(case.status == "pending" for case in state.pending_reviews)


def test_malformed_policy_action_is_accounted_without_fake_action(scope):
    root, configs = scope
    config = configs["generic_llm"]
    values = [pin.model_dump() for pin in config.policy_responses]
    path = root / values[0]["path"]
    path.write_bytes(b'{"kind":"execute_code","code":"untrusted"}')
    values[0] = repin(path, values[0]["path"])
    config = amended(config, policy_responses=values)
    result = runner.inspect_workflow(root, config)
    assert result["state"].status == "failed" and result["new_model_calls"] == 1
    assert result["state"].events[0].action is None
    assert result["state"].events[0].code == "invalid_policy_action"
    assert result["state"].record.revision_id == result["state"].initial_revision


def test_strict_bounded_config_loader(scope):
    root, configs = scope
    path = root / CONFIGS["fixed"]
    path.write_bytes(b"mode: synthetic_offline\nmode: live\n")
    with pytest.raises(ValueError):
        runner.load_config(root, CONFIGS["fixed"])
    path.write_bytes(b" " * (runner.MAX_INPUT_BYTES + 1))
    with pytest.raises(ValueError, match="size bound"):
        runner.load_config(root, CONFIGS["fixed"])


def test_input_record_and_seed_response_are_read_once_for_hash_and_parse(scope, monkeypatch):
    root, configs = scope
    config = configs["fixed"]
    seed = json.loads((root / config.seed.path).read_bytes())
    watched = {root / seed[key]["path"] for key in ("record", "before_response")}
    original = Path.read_bytes
    reads = []
    def observed(path):
        if path in watched:
            reads.append(path)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", observed)
    runner.inspect_workflow(root, config)
    assert len(reads) == len(watched) and set(reads) == watched


def test_cli_summary_reports_counts_without_document_or_raw_judgment(scope, monkeypatch, capsys):
    root, configs = scope
    write_json(root / CONFIGS["fixed"], configs["fixed"].model_dump(mode="json"))
    monkeypatch.setattr(sys, "argv", ["inspect_workflow", "--root", str(root), "--config", CONFIGS["fixed"]])
    assert runner.main() == 0
    output = capsys.readouterr()
    summary = json.loads(output.out)
    assert summary["status"] == "finished" and summary["new_model_calls"] == 2
    assert summary["new_live_requests"] == 0 and summary["pending_reviews"] == 0
    assert summary["coverage_status"] == "unsupported" and summary["usage_unknown"] is True
    assert "Sample A" not in output.out and "quotation" not in output.out


def test_cli_errors_hide_private_values(scope, monkeypatch, capsys):
    root, _ = scope
    monkeypatch.setattr(sys, "argv", ["inspect_workflow", "--root", str(root), "--config", CONFIGS["fixed"]])
    def rejected(*args):
        raise ValueError("private-document-value-must-not-appear")
    monkeypatch.setattr(runner, "load_config", rejected)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "ValueError" in output.err and "private-document-value" not in output.err
    assert output.out == ""
