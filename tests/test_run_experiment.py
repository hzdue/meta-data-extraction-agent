"""Paper selection, resumable extraction, and command exit status."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from meta_data_extraction.app_config import AppConfig
from meta_data_extraction.components.tracker import LocalTracker
from meta_data_extraction.data_models import ExtractionResult, Provenance
from scripts import run_experiment
from scripts.corpus import paper_ids


EXEMPLARS = ["2309.13729", "2104.10884", "2304.08002"]
EVALUATION_IDS = list(paper_ids("evaluation"))


def _config(shot_mode="zero_shot"):
    extractor = {"shot_mode": shot_mode}
    if shot_mode == "few_shot":
        extractor["few_shot"] = {"example_ids": EXEMPLARS}
    return AppConfig(domain_name="photon_and_neutron", extractor=extractor)


@pytest.mark.parametrize("shot_mode", ["zero_shot", "few_shot"])
@pytest.mark.parametrize("limit", [None, 3])
def test_defaults_to_evaluation_split_before_applying_limit(shot_mode, limit):
    selected = run_experiment.select_papers(_config(shot_mode), limit=limit)
    assert len(EVALUATION_IDS) == 30
    assert selected == EVALUATION_IDS[:limit]
    assert not set(selected) & set(paper_ids("calibration"))


@pytest.mark.parametrize("shot_mode", ["zero_shot", "few_shot"])
def test_explicit_papers_preserve_order_and_allow_nonexemplar_calibration(shot_mode):
    selected = ["2503.18475", "1806.08116", EVALUATION_IDS[0]]
    assert run_experiment.select_papers(_config(shot_mode), selected) == selected
    assert run_experiment.select_papers(_config(shot_mode), selected, 2) == selected[:2]


@pytest.mark.parametrize("explicit", [None, EXEMPLARS[:1]])
def test_few_shot_rejects_exemplars_for_default_and_explicit_targets(monkeypatch, explicit):
    monkeypatch.setattr(run_experiment, "paper_ids", lambda *args: EXEMPLARS[:1])
    with pytest.raises(ValueError, match="Exemplar leakage"):
        run_experiment.select_papers(_config("few_shot"), explicit)


def test_zero_shot_can_explicitly_target_few_shot_exemplar_papers():
    assert run_experiment.select_papers(_config(), EXEMPLARS) == EXEMPLARS


@pytest.fixture
def run_context(tmp_path, monkeypatch):
    config = _config()
    config.tracker.output_dir = str(tmp_path / "results")
    config_path = tmp_path / "test.run.yaml"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    raw = tmp_path / "raw"
    monkeypatch.setattr(run_experiment, "RAW_DIR", raw)
    pipeline = Mock()
    pipeline.run.side_effect = AssertionError("Unexpected extraction")
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr(run_experiment.Pipeline, "from_config", factory)
    tracker = LocalTracker(config.tracker.output_dir)
    return SimpleNamespace(
        config_path=config_path, args=["--config", str(config_path)],
        raw=raw, tracker=tracker, pipeline=pipeline, factory=factory,
        status=tracker.config_dir(config_path.stem) / "status.md",
    )


def _write_existing_result(context, paper_id):
    path = context.tracker.result_path(paper_id, context.config_path.stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"existing": true}', encoding="utf-8")
    return path


def _write_pdf(context, paper_id):
    path = context.raw / paper_id / "paper.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_invalid_limit_fails_before_pipeline_creation(run_context, caplog, limit):
    assert run_experiment.main(run_context.args + ["--limit", limit]) == 1
    assert "--limit must be a positive integer" in caplog.text
    run_context.factory.assert_not_called()


def test_exemplar_target_fails_before_pipeline_creation(run_context, caplog):
    config = _config("few_shot")
    config.tracker.output_dir = str(run_context.tracker.output_dir)
    run_context.config_path.write_text(config.model_dump_json(), encoding="utf-8")
    assert run_experiment.main(run_context.args + ["--papers", EXEMPLARS[0]]) == 1
    assert "Exemplar leakage" in caplog.text
    run_context.factory.assert_not_called()


def test_completed_default_run_needs_neither_pdfs_nor_pipeline(run_context):
    paths = [_write_existing_result(run_context, paper_id) for paper_id in EVALUATION_IDS]
    assert run_experiment.main(run_context.args) == 0
    run_context.factory.assert_not_called()
    assert all(path.read_text(encoding="utf-8") == '{"existing": true}' for path in paths)
    status = run_context.status.read_text(encoding="utf-8")
    assert "extracted: 0, skipped: 30, failed: 0, missing PDF: 0" in status


@pytest.mark.parametrize("force", [False, True])
def test_missing_pdf_fails_without_creating_pipeline(run_context, force):
    paper_id = EVALUATION_IDS[0]
    args = run_context.args + ["--papers", paper_id]
    if force:
        _write_existing_result(run_context, paper_id)
        args.append("--force")
    assert run_experiment.main(args) == 1
    run_context.factory.assert_not_called()
    assert f"| `{paper_id}` | missing_pdf |" in run_context.status.read_text(encoding="utf-8")


def test_limit_caps_selection_before_skipping_completed_results(run_context):
    _write_existing_result(run_context, EVALUATION_IDS[0])
    assert run_experiment.main(run_context.args + ["--limit", "1"]) == 0
    run_context.factory.assert_not_called()
    assert "Attempted: 1" in run_context.status.read_text(encoding="utf-8")


def test_force_reextracts_and_writes_result_and_usage(run_context):
    paper_id = EVALUATION_IDS[0]
    result_path = _write_existing_result(run_context, paper_id)
    pdf_path = _write_pdf(run_context, paper_id)
    result = ExtractionResult(
        document_id=paper_id, domain_name="photon_and_neutron",
        output={"title": "New extraction"}, raw_output='{"title": "New extraction"}',
        provenance=Provenance(
            config_name=run_context.config_path.stem,
            input_tokens=100, output_tokens=20, latency_ms=125,
        ),
        warnings=["Recovered JSON"],
    )

    def extract(path):
        run_context.tracker.log_run(result)
        return result

    run_context.pipeline.run.side_effect = extract
    assert run_experiment.main(run_context.args + ["--papers", paper_id, "--force"]) == 0
    run_context.factory.assert_called_once()
    run_context.pipeline.run.assert_called_once_with(pdf_path)
    assert ExtractionResult.model_validate_json(result_path.read_text(encoding="utf-8")) == result
    assert f"| `{paper_id}` | extracted | 100/20 | 125 | Recovered JSON |" in run_context.status.read_text(encoding="utf-8")


def test_failed_paper_does_not_stop_later_papers(run_context):
    first, second = EVALUATION_IDS[:2]
    paths = [_write_pdf(run_context, paper_id) for paper_id in (first, second)]
    run_context.pipeline.run.side_effect = [
        RuntimeError("Extraction failed"),
        ExtractionResult(
            document_id=second, domain_name="photon_and_neutron", provenance=Provenance(),
        ),
    ]
    assert run_experiment.main(run_context.args + ["--papers", first, second]) == 1
    run_context.factory.assert_called_once()
    assert run_context.pipeline.run.call_args_list == [call(path) for path in paths]
    status = run_context.status.read_text(encoding="utf-8")
    assert f"| `{first}` | failed |" in status
    assert f"| `{second}` | extracted |" in status


def test_pipeline_setup_error_is_reported_once(run_context, caplog):
    first, second = EVALUATION_IDS[:2]
    for paper_id in (first, second):
        _write_pdf(run_context, paper_id)
    run_context.factory.side_effect = EnvironmentError("API key not found")
    assert run_experiment.main(run_context.args + ["--papers", first, second]) == 1
    assert "API key not found" in caplog.text
    run_context.factory.assert_called_once()
    run_context.pipeline.run.assert_not_called()
