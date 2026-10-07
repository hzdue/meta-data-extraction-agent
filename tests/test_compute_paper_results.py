"""Paper readers require complete inputs and preserve pooled-count scoring."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import compute_iaa, run_evaluation
from scripts import compute_paper_results as results
from tests.report_expectations import formal_reports, micro_f1


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def model_reports(tmp_path, monkeypatch):
    corpus = {
        "annotation_pairs": {"batch": ["first", "second"]},
        "papers": [
            {"arxiv_id": f"paper_{index:02d}", "split": "evaluation", "annotation_batch": "batch"}
            for index in range(30)
        ] + [{"arxiv_id": "calibration", "split": "calibration"}],
    }
    write_json(tmp_path / "data/corpus.json", corpus)
    monkeypatch.setattr(results, "ROOT", tmp_path)
    monkeypatch.setattr(results, "EVALUATION_DIR", tmp_path / "evaluation")
    monkeypatch.setattr(results, "IAA_SUMMARY_PATH", tmp_path / "absent_iaa.json")
    directory = tmp_path / "evaluation/model"
    evaluator = results.build_evaluator_metadata(results.DEFAULT_PLAN_PATH)
    for index in range(30):
        counts = {"tp": 0, "fp": 9, "fn": 1} if index == 0 else {"tp": 1, "fp": 0, "fn": 0}
        write_json(directory / f"paper_{index:02d}_evaluation.json", {
            "evaluator": evaluator, "fields": {"samples": counts},
        })
    return directory


def test_model_reader_pools_counts_without_other_conditions_or_iaa(model_reports):
    aggregate = results.load_model_results("model")["samples"]
    assert (aggregate["tp"], aggregate["fp"], aggregate["fn"]) == (29, 9, 1)
    assert aggregate["n_papers"] == 30
    assert aggregate["f1"] == pytest.approx(58 / 68)
    assert aggregate["f1"] != pytest.approx(29 / 30)


@pytest.mark.parametrize("change", ["missing", "extra", "replace_with_calibration"])
def test_model_reader_requires_exact_evaluation_membership(model_reports, change):
    path = model_reports / "paper_00_evaluation.json"
    if change in {"extra", "replace_with_calibration"}:
        (model_reports / "calibration_evaluation.json").write_bytes(path.read_bytes())
    if change in {"missing", "replace_with_calibration"}:
        path.unlink()
    with pytest.raises(ValueError, match="exactly match the 30 evaluation papers"):
        results.load_model_results("model")


def test_model_reader_rejects_wrong_population_size(model_reports):
    path = model_reports.parents[1] / "data/corpus.json"
    corpus = json.loads(path.read_text(encoding="utf-8"))
    corpus["papers"] = corpus["papers"][1:]
    write_json(path, corpus)
    with pytest.raises(ValueError, match="Expected 30 evaluation papers, found 29"):
        results.load_model_results("model")


@pytest.fixture
def iaa_summary(tmp_path, monkeypatch):
    aggregate = {
        field: {"f1": 0.8, "tp": 8, "fp": 3, "fn": 1}
        for _, field in results.PRIMARY_FIELDS
    }
    path = tmp_path / "summary.json"
    write_json(path, {
        "n_papers_total": 30, "n_papers_scored": 30,
        "n_papers_missing": 0, "n_papers_failed": 0,
        "evaluator": results.build_evaluator_metadata(results.DEFAULT_PLAN_PATH),
        "aggregated": aggregate,
    })
    monkeypatch.setattr(results, "IAA_SUMMARY_PATH", path)
    monkeypatch.setattr(results, "EVALUATION_DIR", tmp_path / "absent_models")
    return path


def test_iaa_reader_needs_no_model_outputs_and_preserves_numeric_support(iaa_summary):
    aggregate = results.load_iaa_results()
    assert len(aggregate) == 6
    assert all((value["f1"], value["tp"], value["fp"], value["fn"]) == (0.8, 8, 3, 1)
               for value in aggregate.values())


@pytest.mark.parametrize("key,value", [
    ("n_papers_total", 31), ("n_papers_scored", 29),
    ("n_papers_missing", 1), ("n_papers_failed", 1),
])
def test_iaa_reader_rejects_incomplete_summary(iaa_summary, key, value):
    summary = json.loads(iaa_summary.read_text(encoding="utf-8"))
    summary[key] = value
    write_json(iaa_summary, summary)
    with pytest.raises(ValueError, match="complete formal N=30"):
        results.load_iaa_results()


def test_active_config_and_output_paths_agree():
    assert results.DEFAULT_PLAN_PATH.name == "evaluation_metrics.yaml"
    assert compute_iaa.PLAN_PATH.resolve() == results.DEFAULT_PLAN_PATH
    assert run_evaluation.OUTPUT_DIR.resolve() == results.EVALUATION_DIR
    assert compute_iaa.OUTPUT_DIR.resolve() / "summary.json" == results.IAA_SUMMARY_PATH
    assert "datasets" in results.load_plan(results.DEFAULT_PLAN_PATH)


def test_report_version_must_follow_active_config(model_reports, tmp_path, monkeypatch):
    plan = tmp_path / "evaluation_metrics.yaml"
    plan.write_text("version: 9.8.7\nscoring_plan:\n  samples: {metric: exact}\n", encoding="utf-8")
    monkeypatch.setattr(results, "DEFAULT_PLAN_PATH", plan)
    for path in model_reports.glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["evaluator"] = results.build_evaluator_metadata(plan)
        write_json(path, report)
    assert results.load_model_results("model")["samples"]["n_papers"] == 30
    report["evaluator"]["plan_version"] = "old-version"
    write_json(path, report)
    with pytest.raises(ValueError, match="expected.*9.8.7"):
        results.load_model_results("model")


@pytest.mark.parametrize("reader", ["model", "iaa"])
def test_readers_reject_stale_helper_version_after_successful_read(request, reader):
    if reader == "model":
        path = request.getfixturevalue("model_reports") / "paper_00_evaluation.json"
        read = lambda: results.load_model_results("model")
    else:
        path = request.getfixturevalue("iaa_summary")
        read = results.load_iaa_results
    read()
    report = json.loads(path.read_text(encoding="utf-8"))
    report["evaluator"]["alignment_helper_version"] = "stale-version"
    write_json(path, report)
    with pytest.raises(ValueError, match="alignment helper version"):
        read()


@pytest.fixture(scope="module")
def generated_reports(tmp_path_factory):
    """Recompute released predictions and annotations into temporary reports."""
    directory = tmp_path_factory.mktemp("paper_reports")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_evaluation, "OUTPUT_DIR", directory / "evaluation")
        patch.setattr(compute_iaa, "OUTPUT_DIR", directory / "iaa")
        assert run_evaluation.main([]) == 0
        assert compute_iaa.main([]) == 0
    return directory


@pytest.fixture(scope="module")
def paper_results(generated_reports):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(results, "EVALUATION_DIR", generated_reports / "evaluation")
        patch.setattr(results, "IAA_SUMMARY_PATH", generated_reports / "iaa/summary.json")
        return results.compute_paper_results()


def test_benchmark_statistics_preserve_published_counts():
    assert results.compute_benchmark_statistics() == {
        "Corpus and annotation": {
            "Papers": 40, "Calibration papers": 10,
            "Evaluation papers": 30, "Working annotations": 90,
        },
        "Repeated entities": {
            "Affiliations": 179, "Authors": 308, "Funding items": 104,
            "Facilities": 57, "Samples": 127, "Datasets": 118,
        },
        "Paper-local reference edges": {
            "Author--affiliation": 354, "Dataset--sample": 321, "Dataset--facility": 73,
        },
    }


@pytest.fixture
def benchmark_inputs(tmp_path):
    gold = tmp_path / "data/gold"
    gold.mkdir(parents=True)
    shutil.copyfile(results.ROOT / "data/corpus.json", tmp_path / "data/corpus.json")
    for paper in results.load_corpus(tmp_path / "data/corpus.json")["papers"]:
        name = f"{paper['arxiv_id']}.json"
        shutil.copyfile(results.ROOT / "data/gold" / name, gold / name)
    return tmp_path


def test_benchmark_counts_need_no_annotation_files(benchmark_inputs):
    assert not (benchmark_inputs / "data/annotations").exists()
    assert results.compute_benchmark_statistics(benchmark_inputs) == results.compute_benchmark_statistics()


def test_benchmark_counts_require_every_selected_gold_record(benchmark_inputs):
    next((benchmark_inputs / "data/gold").iterdir()).unlink()
    with pytest.raises(FileNotFoundError):
        results.compute_benchmark_statistics(benchmark_inputs)


def test_benchmark_counts_require_the_published_split(benchmark_inputs):
    path = benchmark_inputs / "data/corpus.json"
    corpus = json.loads(path.read_text(encoding="utf-8"))
    paper = next(paper for paper in corpus["papers"] if paper["split"] == "evaluation")
    paper["split"] = "calibration"
    del paper["annotation_batch"]
    write_json(path, corpus)
    with pytest.raises(ValueError, match="10 calibration and 30 evaluation"):
        results.compute_benchmark_statistics(benchmark_inputs)


@pytest.mark.parametrize("field", ["datasets", "affiliation_refs"])
def test_benchmark_counts_reject_non_list_collections(benchmark_inputs, field):
    path = next((benchmark_inputs / "data/gold").iterdir())
    record = json.loads(path.read_text(encoding="utf-8"))
    if field == "datasets":
        record[field] = {}
    else:
        record["bibliographic"]["authors"][0][field] = "affiliation_1"
    write_json(path, record)
    with pytest.raises(ValueError, match="must be a list"):
        results.compute_benchmark_statistics(benchmark_inputs)


def test_zero_shot_and_instruction_values_match_independent_report_counts(paper_results, generated_reports):
    zero = {row["path"]: row for row in paper_results["zero_shot"]["headline"]}
    instructions = {row["path"]: row for row in paper_results["instruction_effects"]}
    assert paper_results["zero_shot"]["models"] == list(results.ZERO_SHOT_RUNS)
    for index, conditions in enumerate(results.ZERO_SHOT_RUNS.values()):
        full = formal_reports(conditions["full"], generated_reports / "evaluation")
        no_instruction = formal_reports(conditions["no_inst"], generated_reports / "evaluation")
        for _, path in results.PRIMARY_FIELDS:
            expected = micro_f1(full, path)
            assert zero[path]["scores"][index] == pytest.approx(expected)
            assert instructions[path]["deltas"][index] == pytest.approx(
                expected - micro_f1(no_instruction, path)
            )
    assert round(zero["datasets"]["scores"][1], 3) == 0.609
    assert all(delta == 0 for delta in instructions["bibliographic.authors"]["deltas"])
    for path in ("samples", "datasets"):
        assert all(delta > 0 for delta in instructions[path]["deltas"])
    facility_deltas = instructions["facilities"]["deltas"]
    assert sum(delta > 0 for delta in facility_deltas) == 3
    assert sum(delta < 0 for delta in facility_deltas) == 4


def test_few_shot_deltas_match_independent_report_counts(paper_results, generated_reports):
    few_shot = paper_results["few_shot"]
    assert few_shot["models"] == list(results.FEW_SHOT_RUNS)
    assert few_shot["shots"] == [0, 1, 3]
    for index, conditions in enumerate(results.FEW_SHOT_RUNS.values()):
        reports = {
            shots: formal_reports(run, generated_reports / "evaluation")
            for shots, run in conditions.items()
        }
        for row in few_shot["outcomes"]:
            scores = [micro_f1(reports[shots], row["path"]) for shots in (0, 1, 3)]
            assert row["deltas"][index] == pytest.approx([0, scores[1] - scores[0], scores[2] - scores[0]])
            if row["path"] in ("bibliographic.authors", "bibliographic.affiliations"):
                assert scores == [1, 1, 1]


def test_iaa_headline_and_appendix_preserve_scores_and_support(paper_results, generated_reports):
    aggregate = json.loads(
        (generated_reports / "iaa/summary.json").read_text(encoding="utf-8")
    )["aggregated"]
    for row in paper_results["iaa"]["headline"]:
        raw = aggregate[row["path"]]
        assert row["agreement"] == raw["f1"]
        assert row["matched"] == raw["tp"]
        assert row["unmatched"] == raw["fp"] + raw["fn"]
    assert aggregate["bibliographic.affiliations"]["f1"] == 1
    assert aggregate["bibliographic.authors"]["f1"] == 1
    assert round(aggregate["facilities"]["f1"], 3) == 0.753
    assert aggregate["facilities"]["tp"] > 0
    assert aggregate["facilities"]["fp"] + aggregate["facilities"]["fn"] > 0
    paths = [
        "bibliographic.affiliations", "bibliographic.authors", "bibliographic.funding",
        "facilities", "samples", "datasets",
    ]
    for section in ("iaa", "zero_shot"):
        assert [row["path"] for row in paper_results[section]["headline"]] == paths
    assert [row["path"] for row in paper_results["instruction_effects"]] == paths
    assert all(len(row["scores"]) == 7 for row in paper_results["zero_shot"]["headline"])
    assert all(len(row["deltas"]) == 7 for row in paper_results["instruction_effects"])
    assert len(paper_results["metrics"]) == 44
    for section in ("iaa", "zero_shot"):
        fields = paper_results[section]["fields"]
        assert set(fields) == set(paper_results["metrics"])
        assert fields["bibliographic.subtitle"]["support"] == [0, 30]
    headings = "document_overview.section_headings"
    assert paper_results["iaa"]["fields"][headings]["order_score"] == aggregate[headings]["mean_order"]
    assert len(paper_results["zero_shot"]["fields"][headings]["order_scores"]) == 7


@pytest.mark.parametrize("damage", ["missing_field", "incomplete_field"])
def test_computation_rejects_incomplete_aggregates(monkeypatch, damage):
    metrics = {"samples": "entity"}
    aggregate = {} if damage == "missing_field" else {"samples": {"n_papers": 29}}
    monkeypatch.setattr(results, "load_metrics", lambda: metrics)
    monkeypatch.setattr(results, "load_evaluation_results", lambda: {"model": aggregate})
    monkeypatch.setattr(results, "load_iaa_results", lambda: {"samples": {"n_papers": 30}})
    with pytest.raises(ValueError, match="scoring plan|all 30 papers"):
        results.compute_paper_results()


def test_cli_recomputes_and_overwrites_the_saved_json(tmp_path, monkeypatch):
    path = tmp_path / "paper/results.json"
    monkeypatch.setattr(results, "OUTPUT_PATH", path)
    calls = []

    def compute():
        calls.append(1)
        return {"n_papers": 30, "test_invocation": len(calls)}

    monkeypatch.setattr(results, "compute_paper_results", compute)
    assert results.main([]) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["test_invocation"] == 1
    assert results.main([]) == 0
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved.pop("test_invocation") == 2
    assert saved == {"n_papers": 30}
    assert list(path.parent.iterdir()) == [path]


def test_failed_computation_removes_old_results(tmp_path, monkeypatch, capsys):
    path = tmp_path / "results.json"
    write_json(path, {"old": True})
    monkeypatch.setattr(results, "OUTPUT_PATH", path)
    monkeypatch.setattr(results, "EVALUATION_DIR", tmp_path / "missing-evaluations")
    assert results.main([]) == 1
    assert not path.exists()
    assert "Missing evaluation directory" in capsys.readouterr().err


@pytest.fixture
def loaded_results(monkeypatch):
    monkeypatch.setattr(results, "PRIMARY_FIELDS", (("Samples", "samples"),))
    monkeypatch.setattr(results, "ZERO_SHOT_RUNS", {
        "Model A": {"full": "a0", "no_inst": "a_bare"},
        "Model B": {"full": "b0", "no_inst": "b_bare"},
    })
    monkeypatch.setattr(results, "FEW_SHOT_RUNS", {
        "Model A": {0: "a0", 1: "a1", 3: "a3"},
        "Model B": {0: "b0", 1: "b1", 3: "b3"},
    })
    metrics = {
        "samples": "entity",
        "bibliographic.title": "rouge_l",
        "bibliographic.subtitle": "normalized_exact",
        "samples.attrs.material": "normalized_exact",
        "samples.attrs.empty": "normalized_exact",
        "document_overview.section_headings": "ordered_set_f1",
    }
    iaa = {
        "samples": {"f1": 0.8, "tp": 8, "fp": 3, "fn": 1},
        "bibliographic.title": {"mean_score": 0.6, "n_non_null": 12, "n_papers": 30},
        "bibliographic.subtitle": {"mean_score": 1, "n_non_null": 0, "n_papers": 30},
        "samples.attrs.material": {"mean_score": 0.5, "n_pairs_non_null": 2, "n_pairs": 3},
        "samples.attrs.empty": {"mean_score": 0, "n_pairs_non_null": 0, "n_pairs": 0},
        "document_overview.section_headings": {"f1": 0.8, "mean_order": 0.1},
    }
    first, second = deepcopy(iaa), deepcopy(iaa)
    second["samples"]["f1"] = 0.3
    for aggregate in (first, second):
        aggregate["bibliographic.title"]["n_non_null"] = 30
    first["samples.attrs.material"].update(n_pairs_non_null=1, n_pairs=7)
    second["samples.attrs.material"].update(n_pairs_non_null=4, n_pairs=5)
    first["document_overview.section_headings"].update(f1=0.9, mean_order=0.6)
    second["document_overview.section_headings"].update(f1=0.5, mean_order=-0.2)
    evaluations = {
        "a0": first, "b0": second,
        "a_bare": {"samples": {"f1": 0.5}}, "b_bare": {"samples": {"f1": 0.6}},
        "a1": {"samples": {"f1": 0.7}}, "a3": {"samples": {"f1": 0.9}},
        "b1": {"samples": {"f1": 0.5}}, "b3": {"samples": {"f1": 0.2}},
    }

    def reject_file_read(*args, **kwargs):
        pytest.fail("Result calculations must use the supplied data without reading files")

    monkeypatch.setattr(Path, "read_text", reject_file_read)
    return iaa, evaluations, metrics


def test_iaa_calculation_preserves_support_and_distinct_heading_order(loaded_results):
    iaa, _, metrics = loaded_results
    actual = results.compute_iaa_results(iaa, metrics)
    assert actual["headline"] == [{
        "outcome": "Samples", "path": "samples", "agreement": 0.8,
        "matched": 8, "unmatched": 4,
    }]
    fields = actual["fields"]
    assert fields["bibliographic.title"]["support"] == [12, 30]
    assert fields["bibliographic.subtitle"]["support"] == [0, 30]
    assert fields["samples.attrs.material"]["support"] == [2, 3]
    assert fields["samples.attrs.empty"]["support"] == [0, 0]
    assert fields["document_overview.section_headings"] == {"score": 0.8, "order_score": 0.1}
    assert fields["samples"] == {"score": 0.8}


def test_zero_shot_calculation_preserves_model_order_and_support_maxima(loaded_results):
    _, evaluations, metrics = loaded_results
    actual = results.compute_zero_shot_results(evaluations, metrics)
    assert actual["models"] == ["Model A", "Model B"]
    assert actual["headline"] == [{
        "outcome": "Samples", "path": "samples", "scores": [0.8, 0.3],
    }]
    fields = actual["fields"]
    assert fields["bibliographic.title"]["support"] == [30, 30]
    assert fields["bibliographic.subtitle"]["support"] == [0, 30]
    assert fields["samples.attrs.material"]["support"] == [4, 7]
    assert fields["samples.attrs.empty"]["support"] == [0, 0]
    assert fields["document_overview.section_headings"] == {
        "scores": [0.9, 0.5], "order_scores": [0.6, -0.2],
    }
    assert fields["samples"] == {"scores": [0.8, 0.3]}


def test_comparison_calculations_use_the_correct_direction_and_each_model_baseline(loaded_results):
    _, evaluations, _ = loaded_results
    instructions = results.compute_instruction_effects(evaluations)
    assert instructions[0]["deltas"] == pytest.approx([0.3, -0.3])
    few_shot = results.compute_few_shot_results(evaluations)
    assert few_shot["models"] == ["Model A", "Model B"]
    assert few_shot["shots"] == [0, 1, 3]
    assert few_shot["outcomes"][0]["deltas"][0] == pytest.approx([0, -0.1, 0.1])
    assert few_shot["outcomes"][0]["deltas"][1] == pytest.approx([0, 0.2, -0.1])
