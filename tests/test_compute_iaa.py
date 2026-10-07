"""Fixed IAA population, pooled scores, and failed-rerun handling."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from meta_data_extraction.evaluation import evaluate
from scripts import compute_iaa as runner
from scripts.compute_paper_results import load_iaa_results


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    pairs = {"batch_a": ["first", "second"], "batch_b": ["first", "third"],
             "batch_c": ["second", "third"]}
    corpus = {"annotation_pairs": pairs, "papers": []}
    for index, (batch, annotators) in enumerate(pairs.items()):
        for number in range(10):
            paper = f"paper_{index * 10 + number:02d}"
            corpus["papers"].append({"arxiv_id": paper, "split": "evaluation", "annotation_batch": batch})
            for annotator in annotators:
                write_json(tmp_path / "annotations" / annotator / f"{paper}.json", {
                    "bibliographic": {"title": "same"}, "tags": ["item"],
                })
            write_json(tmp_path / "gold" / f"{paper}.json", {"bibliographic": {"title": "reconciled"}})
    corpus["papers"].append({"arxiv_id": "calibration", "split": "calibration"})
    write_json(tmp_path / "corpus.json", corpus)
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "version: test\nscoring_plan:\n  bibliographic.title: {metric: exact}\n  tags: {metric: set_f1}\n",
        encoding="utf-8",
    )
    for name, path in {
        "CORPUS_PATH": tmp_path / "corpus.json", "PLAN_PATH": plan,
        "ANNOTATIONS_DIR": tmp_path / "annotations", "IDENTITY_GOLD_DIR": tmp_path / "gold",
        "OUTPUT_DIR": tmp_path / "reports",
    }.items():
        monkeypatch.setattr(runner, name, path)


def test_complete_run_saves_30_reports_and_one_pooled_summary(inputs, tmp_path, capsys):
    assert runner.main([]) == 0
    assert "Total: 30/30 scored, 0 missing, 0 failed" in capsys.readouterr().out
    reports = list((tmp_path / "reports").glob("*/*_evaluation.json"))
    assert len(reports) == 30
    assert {path.name for path in reports} == {f"paper_{index:02d}_evaluation.json" for index in range(30)}
    assert len(list((tmp_path / "reports").rglob("*.json"))) == 31
    summary = json.loads((tmp_path / "reports/summary.json").read_text(encoding="utf-8"))
    assert summary["n_papers_total"] == summary["n_papers_scored"] == 30
    assert summary["n_papers_missing"] == summary["n_papers_failed"] == 0
    assert summary["aggregated"]["bibliographic.title"]["mean_score"] == 1
    for batch, paper, first, second in (
        ("batch_a", "paper_00", "first", "second"),
        ("batch_b", "paper_10", "first", "third"),
        ("batch_c", "paper_20", "second", "third"),
    ):
        expected = evaluate(
            tmp_path / "annotations" / first / f"{paper}.json",
            tmp_path / "annotations" / second / f"{paper}.json",
            plan_path=tmp_path / "plan.yaml", identity_gold_path=tmp_path / "gold" / f"{paper}.json",
        )
        saved = json.loads((tmp_path / "reports" / batch / f"{paper}_evaluation.json").read_text(encoding="utf-8"))
        assert saved == expected
        assert saved["evaluator"] == summary["evaluator"]


def test_pooled_f1_uses_counts_instead_of_averaging_batch_scores(inputs, tmp_path):
    write_json(tmp_path / "annotations/second/paper_00.json", {"tags": ["wrong"]})
    for index in range(10, 20):
        write_json(tmp_path / "annotations/third" / f"paper_{index:02d}.json", {
            "tags": ["item"] + [f"extra_{number}" for number in range(9)],
        })
    assert runner.main([]) == 0
    summary = json.loads((tmp_path / "reports/summary.json").read_text(encoding="utf-8"))
    score = summary["aggregated"]["tags"]
    assert (score["tp"], score["fp"], score["fn"]) == (29, 91, 1)
    assert score["f1"] == pytest.approx(58 / 150)


@pytest.mark.parametrize("directory", ["annotations/first", "annotations/second", "gold"])
def test_missing_input_removes_stale_report_and_marks_summary_incomplete(inputs, tmp_path, directory):
    report_path = tmp_path / "reports/batch_a/paper_00_evaluation.json"
    write_json(report_path, {"stale": True})
    write_json(tmp_path / "reports/summary.json", {"stale": True})
    (tmp_path / directory / "paper_00.json").unlink()
    assert runner.main([]) == 1
    assert not report_path.exists()
    summary_path = tmp_path / "reports/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert (summary["n_papers_scored"], summary["n_papers_missing"], summary["n_papers_failed"]) == (29, 1, 0)
    assert (tmp_path / "reports/batch_c/paper_29_evaluation.json").is_file()
    with pytest.raises(ValueError, match="complete formal N=30"):
        load_iaa_results(summary_path, tmp_path / "plan.yaml")


def test_invalid_annotation_is_a_failure_and_other_papers_continue(inputs, tmp_path):
    (tmp_path / "annotations/second/paper_00.json").write_text("not JSON", encoding="utf-8")
    assert runner.main([]) == 1
    summary = json.loads((tmp_path / "reports/summary.json").read_text(encoding="utf-8"))
    assert (summary["n_papers_scored"], summary["n_papers_missing"], summary["n_papers_failed"]) == (29, 0, 1)
    assert (tmp_path / "reports/batch_c/paper_29_evaluation.json").is_file()


def test_rerun_recomputes_changed_annotations(inputs, tmp_path):
    assert runner.main([]) == 0
    write_json(tmp_path / "annotations/second/paper_00.json", {"bibliographic": {"title": "different"}})
    assert runner.main([]) == 0
    report = json.loads((tmp_path / "reports/batch_a/paper_00_evaluation.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "reports/summary.json").read_text(encoding="utf-8"))
    assert report["fields"]["bibliographic.title"]["score"] == 0
    assert summary["aggregated"]["bibliographic.title"]["mean_score"] == pytest.approx(29 / 30)


def test_mismatched_evaluator_metadata_is_not_saved(inputs, tmp_path, monkeypatch):
    def stale_report(*args, **kwargs):
        report = evaluate(*args, **kwargs)
        report["evaluator"]["alignment_helper_version"] = "stale"
        return report
    monkeypatch.setattr(runner, "evaluate", stale_report)
    assert runner.main([]) == 1
    assert not list((tmp_path / "reports").glob("*/*_evaluation.json"))
    summary = json.loads((tmp_path / "reports/summary.json").read_text(encoding="utf-8"))
    assert summary["n_papers_failed"] == 30
    assert summary["aggregated"] == {}


@pytest.mark.parametrize("problem", ["invalid_plan", "missing_paper", "duplicate_paper"])
def test_invalid_inputs_fail_before_scoring(inputs, tmp_path, monkeypatch, problem):
    if problem == "invalid_plan":
        (tmp_path / "plan.yaml").write_text("version: [", encoding="utf-8")
    else:
        path = tmp_path / "corpus.json"
        corpus = json.loads(path.read_text(encoding="utf-8"))
        if problem == "missing_paper":
            corpus["papers"].pop(0)
        else:
            corpus["papers"].append(dict(corpus["papers"][0]))
        write_json(path, corpus)
    scorer = Mock()
    monkeypatch.setattr(runner, "evaluate", scorer)
    assert runner.main([]) == 1
    scorer.assert_not_called()
    assert not (tmp_path / "reports").exists()


def test_batch_selection_is_not_supported(inputs):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--batches", "batch_a", "batch_a", "batch_a"])
    assert exc.value.code == 2


def test_batch_directory_error_does_not_leave_a_stale_pooled_summary(inputs, tmp_path, caplog):
    write_json(tmp_path / "reports/summary.json", {"stale": True})
    (tmp_path / "reports/batch_b").write_text("not a directory", encoding="utf-8")
    assert runner.main([]) == 1
    assert "Cannot write IAA reports" in caplog.text
    assert not (tmp_path / "reports/summary.json").exists()
