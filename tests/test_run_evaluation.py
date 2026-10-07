"""Evaluation report output, population selection, and failure handling."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from meta_data_extraction.evaluation import build_evaluator_metadata, evaluate
from scripts import run_evaluation as runner
from scripts.corpus import paper_ids


CONDITIONS = tuple(f"pan_model_{index:02d}_5_0shot" for index in range(22))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "version: test\nscoring_plan:\n  bibliographic.title: {metric: exact}\n",
        encoding="utf-8",
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for condition in CONDITIONS:
        (config_dir / f"{condition.replace('_5_', '.5_')}.yaml").touch()
    record = {"bibliographic": {"title": "A title"}}
    for paper in ("paper_1", "paper_2"):
        _write_json(tmp_path / "gold" / f"{paper}.json", record)
        for condition in CONDITIONS:
            _write_json(tmp_path / "experiment" / condition / f"{paper}.json", {
                "document_id": paper, "output": record,
            })
    monkeypatch.setattr(runner, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(runner, "GOLD_DIR", tmp_path / "gold")
    monkeypatch.setattr(runner, "PROCESSED_DIR", tmp_path / "experiment")
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path / "reports")
    monkeypatch.setattr(runner, "PLAN_PATH", plan)
    monkeypatch.setattr(runner, "paper_ids", lambda split: ("paper_1", "paper_2"))


def test_saves_only_complete_per_paper_reports(inputs, tmp_path, capsys):
    assert runner.main([]) == 0
    assert "22 conditions, 44 scored, 0 failed" in capsys.readouterr().out
    reports = sorted((tmp_path / "reports").glob("*/*"))
    assert len(reports) == 44
    assert all(path.name.endswith("_evaluation.json") for path in reports)
    assert not list((tmp_path / "experiment").glob("*/evaluations"))
    for path in reports:
        paper = path.name.removesuffix("_evaluation.json")
        expected = evaluate(
            tmp_path / "gold" / f"{paper}.json",
            tmp_path / "experiment" / path.parent.name / f"{paper}.json",
            plan_path=tmp_path / "plan.yaml",
        )
        assert json.loads(path.read_text(encoding="utf-8")) == expected


def test_removed_options_are_rejected(inputs):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--papers", "paper_1"])
    assert exc.value.code == 2


def test_fixed_run_uses_exactly_22_conditions_and_30_evaluation_papers(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "paper_ids", paper_ids)
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    monkeypatch.setattr(runner, "CONFIG_DIR", config_dir)
    conditions = {path.stem.replace(".", "_") for path in config_dir.glob("pan_*.yaml")}
    for paper in paper_ids():
        _write_json(tmp_path / "gold" / f"{paper}.json", {})
        for condition in conditions:
            _write_json(tmp_path / "experiment" / condition / f"{paper}.json", {})
    assert runner.main([]) == 0
    folders = list((tmp_path / "reports").iterdir())
    assert {folder.name for folder in folders} == conditions
    assert len(folders) == 22
    for folder in folders:
        actual = {path.name.removesuffix("_evaluation.json") for path in folder.glob("*.json")}
        assert actual == set(paper_ids("evaluation"))
        assert len(actual) == 30


@pytest.mark.parametrize("missing", ["gold", "prediction"])
def test_missing_selected_input_fails_and_removes_stale_report(inputs, tmp_path, caplog, missing):
    condition = CONDITIONS[0]
    report = tmp_path / "reports" / condition / "paper_1_evaluation.json"
    _write_json(report, {"stale": True})
    directory = tmp_path / "gold" if missing == "gold" else tmp_path / "experiment" / condition
    (directory / "paper_1.json").unlink()
    assert runner.main([]) == 1
    assert "paper_1" in caplog.text
    assert not report.exists()
    assert (report.parent / "paper_2_evaluation.json").is_file()


def test_failed_paper_does_not_stop_other_papers_or_conditions(inputs, tmp_path, capsys):
    broken = tmp_path / "experiment" / CONDITIONS[0] / "paper_1.json"
    broken.write_text("not JSON", encoding="utf-8")
    assert runner.main([]) == 1
    assert "22 conditions, 43 scored, 1 failed" in capsys.readouterr().out
    assert len(list((tmp_path / "reports").glob("*/*_evaluation.json"))) == 43


@pytest.mark.parametrize("field", ["plan_version", "alignment_helper_version"])
def test_rejects_inconsistent_report_metadata(inputs, tmp_path, monkeypatch, caplog, field):
    def stale_report(*args, **kwargs):
        report = evaluate(*args, **kwargs)
        report["evaluator"][field] = "stale"
        return report
    monkeypatch.setattr(runner, "evaluate", stale_report)
    assert runner.main([]) == 1
    assert "unexpected evaluator provenance" in caplog.text
    assert not list((tmp_path / "reports").glob("*/*_evaluation.json"))


def test_invalid_plan_fails_before_scoring(inputs, tmp_path, monkeypatch):
    (tmp_path / "plan.yaml").write_text("version: [", encoding="utf-8")
    scorer = Mock()
    monkeypatch.setattr(runner, "evaluate", scorer)
    assert runner.main([]) == 1
    scorer.assert_not_called()
    assert not (tmp_path / "reports").exists()


def test_missing_configured_condition_fails(inputs, tmp_path, caplog):
    config = next((tmp_path / "configs").glob("*.yaml"))
    config.rename(config.with_name("pan_missing_condition.yaml"))
    assert runner.main([]) == 1
    assert "Prediction condition not found" in caplog.text


def test_incomplete_config_set_fails_before_scoring(inputs, tmp_path, monkeypatch, caplog):
    next((tmp_path / "configs").glob("*.yaml")).unlink()
    scorer = Mock()
    monkeypatch.setattr(runner, "evaluate", scorer)
    assert runner.main([]) == 1
    assert "Expected 22 experiment configs" in caplog.text
    scorer.assert_not_called()


def test_empty_configured_condition_is_not_success(inputs, tmp_path, capsys):
    for path in (tmp_path / "experiment" / CONDITIONS[0]).glob("*.json"):
        path.unlink()
    assert runner.main([]) == 1
    assert "42 scored, 2 failed" in capsys.readouterr().out


def test_output_directory_error_is_reported(inputs, tmp_path, caplog):
    (tmp_path / "reports").write_text("not a directory", encoding="utf-8")
    assert runner.main([]) == 1
    assert "Cannot write reports" in caplog.text


def test_single_condition_rescores_when_version_changes_at_same_path(inputs, tmp_path):
    plan = tmp_path / "plan.yaml"
    prediction_dir = tmp_path / "experiment" / CONDITIONS[0]
    args = dict(
        gold_dir=tmp_path / "gold", output_dir=tmp_path / "reports",
        papers=["paper_1"], plan_path=plan,
    )
    assert runner.run_condition(
        prediction_dir, **args, expected_evaluator=build_evaluator_metadata(plan),
    ) == (1, 0)
    plan.write_text(
        plan.read_text(encoding="utf-8").replace("version: test", "version: 9.8.7"),
        encoding="utf-8",
    )
    assert runner.run_condition(
        prediction_dir, **args, expected_evaluator=build_evaluator_metadata(plan),
    ) == (1, 0)
    report = tmp_path / "reports" / CONDITIONS[0] / "paper_1_evaluation.json"
    assert json.loads(report.read_text(encoding="utf-8"))["evaluator"]["plan_version"] == "9.8.7"


def test_changed_scoring_rule_is_used_without_a_version_bump(inputs, tmp_path):
    plan = tmp_path / "plan.yaml"
    prediction_dir = tmp_path / "experiment" / CONDITIONS[0]
    _write_json(prediction_dir / "paper_1.json", {"bibliographic": {"title": "A TITLE"}})
    args = dict(
        gold_dir=tmp_path / "gold", output_dir=tmp_path / "reports",
        papers=["paper_1"], plan_path=plan,
    )
    report = tmp_path / "reports" / CONDITIONS[0] / "paper_1_evaluation.json"
    assert runner.run_condition(
        prediction_dir, **args, expected_evaluator=build_evaluator_metadata(plan),
    ) == (1, 0)
    before = json.loads(report.read_text(encoding="utf-8"))
    plan.write_text(
        plan.read_text(encoding="utf-8").replace(
            "{metric: exact}", "{metric: normalized_exact, normalizer: casefold}",
        ), encoding="utf-8",
    )
    assert runner.run_condition(
        prediction_dir, **args, expected_evaluator=build_evaluator_metadata(plan),
    ) == (1, 0)
    after = json.loads(report.read_text(encoding="utf-8"))
    assert before["evaluator"] == after["evaluator"]
    assert before["fields"]["bibliographic.title"]["score"] == 0
    assert after["fields"]["bibliographic.title"]["score"] == 1
