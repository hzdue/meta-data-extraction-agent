"""Use a real temporary SQLite MLflow store and synthetic Qwen fixtures only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
torch = pytest.importorskip("torch")
from mlflow import MlflowClient

from meta_data_extraction.finetune.tracking import (
    TrackingConfig, RunTracker, attach_evaluation, read_tracking_config,
)
from meta_data_extraction.finetune.training import attach_lora, build_trainer
from meta_data_extraction.finetune import validation as validation_module
from tests.test_finetune_training import PREPARATION, StopAfterTwo, config, records, tiny_model, tokenizer
from tests.test_finetune_validation import validation_rows


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("FINETUNE_MLFLOW_ARTIFACT_LOCATION", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def settings(tmp_path):
    return TrackingConfig(experiment_name="synthetic-finetune-test",
                          tracking_uri="sqlite:///" + (tmp_path / "tracking.db").as_posix(),
                          artifact_location=(tmp_path / "artifacts").as_uri())


def contract(cfg=None):
    return {"config": cfg.model_dump() if cfg else {"method": "base"},
            "data": {"tokenizer": PREPARATION, "dataset_hashes": {"synthetic": "fixture"}},
            "runtime": {"fixture": True}, "sources": {"uv.lock": "synthetic-lock-hash"}}


def artifact_files(client, run_id, path=None):
    result = []
    for item in client.list_artifacts(run_id, path):
        result.extend(artifact_files(client, run_id, item.path) if item.is_dir else [item.path])
    return result


def trainer_for(output, cfg, tracker=None, callbacks=None):
    return build_trainer(attach_lora(tiny_model(), cfg, PREPARATION), tokenizer(), records(cfg.method), cfg,
                         output, cpu=True, validation=validation_rows(), tracking=tracker, callbacks=callbacks)


def test_tracking_requires_explicit_persistent_configuration(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        TrackingConfig().require_ready()
    with pytest.raises(ValueError, match="absolute"):
        TrackingConfig(tracking_uri="sqlite:///relative.db", artifact_location="/artifacts").require_ready()
    with pytest.raises(ValueError, match="artifact_location"):
        TrackingConfig(tracking_uri="sqlite:///" + (tmp_path / "db").as_posix()).require_ready()
    with pytest.raises(ValueError, match="credentials"):
        TrackingConfig(tracking_uri="https://user:password@example.invalid").require_ready()
    path = tmp_path / "tracking.yaml"
    path.write_text("tracking_uri: null\nartifact_location: null\n")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.example.invalid")
    cfg = read_tracking_config(path, require_ready=True)
    assert cfg.tracking_uri == "https://tracking.example.invalid"
    assert not (tmp_path / "mlruns").exists()  # Parsing never connects or creates a store.


def test_three_methods_share_experiment_and_reject_artifact_location_drift(settings, tmp_path):
    trackers = []
    for method in ("base", "json_only", "structure_aware"):
        tracker = RunTracker.start(settings, tmp_path / method,
                                   contract(None if method == "base" else config(method=method)), method=method)
        tracker.finish("FINISHED")
        trackers.append(tracker)
    assert len({t.record["experiment_id"] for t in trackers}) == 1
    assert [t.record["run_name"] for t in trackers] == ["base", "json_only_seed42", "structure_aware_seed42"]
    for tracker in trackers:
        run = tracker.client.get_run(tracker.run_id)
        assert run.data.params["method"] == tracker.record["method"]
        assert "provenance/run_manifest.json" in artifact_files(tracker.client, tracker.run_id)
    assert "optimizer" not in trackers[0].client.get_run(trackers[0].run_id).data.params
    drift = settings.model_copy(update={"artifact_location": (tmp_path / "different_artifacts").as_uri()})
    with pytest.raises(ValueError, match="different artifact"):
        RunTracker.start(drift, tmp_path / "new", contract(), method="base")


def test_metrics_are_idempotent_and_conflicting_replays_fail(settings, tmp_path):
    tracker = RunTracker.start(settings, tmp_path / "run", contract(), method="base")
    tracker.log_metrics({"example/loss": 1.25}, 2)
    tracker.log_metrics({"example/loss": 1.25}, 2)
    tracker.seen.clear()  # Exercise server-side recovery after a lost local acknowledgement.
    tracker.log_metrics({"example/loss": 1.25}, 2)
    assert len(tracker.client.get_metric_history(tracker.run_id, "example/loss")) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        tracker.log_metrics({"example/loss": 2.0}, 2)
    with pytest.raises(ValueError, match="finite"):
        tracker.log_metrics({"bad": float("nan")}, 2)
    tracker.finish("FINISHED")


@pytest.mark.parametrize("method", ["json_only", "structure_aware"])
def test_logged_training_resumes_same_run_without_duplicate_curves(settings, tmp_path, method):
    cfg = config(method=method, max_steps=3)
    output = tmp_path / "run"
    manifest = contract(cfg)
    tracker = RunTracker.start(settings, output, manifest, method=method)
    interrupted = trainer_for(output, cfg, tracker, [StopAfterTwo()])
    interrupted.train()
    tracker.finish("KILLED", trainer=interrupted)
    assert tracker.client.get_run(tracker.run_id).info.status == "KILLED"
    old_id = tracker.run_id
    resumed_tracker = RunTracker.start(settings, output, manifest, method=method, resume=True)
    assert resumed_tracker.run_id == old_id and resumed_tracker.record["attempt"] == 2
    resumed = trainer_for(output, cfg, resumed_tracker)
    resumed.train(resume_from_checkpoint=str(output / "checkpoint-2"))
    resumed_tracker.finish("FINISHED", trainer=resumed)
    run = tracker.client.get_run(old_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics["exposure/examples/total"] == 6
    assert sorted(m.step for m in tracker.client.get_metric_history(old_id, "validation/loss")) == [0, 2, 3]
    assert sorted(m.step for m in tracker.client.get_metric_history(old_id, "train/loss")) == [1, 2, 3]
    assert run.data.metrics["resources/attempt_1/final/elapsed_seconds"] > 0
    assert run.data.metrics["resources/attempt_2/final/elapsed_seconds"] > 0
    assert not any("cuda_elapsed_seconds" in k for k in run.data.metrics)
    assert run.data.metrics["resources/attempt_2/final/training/wall_seconds"] > 0
    assert run.data.metrics["resources/attempt_2/final/validation/wall_seconds"] > 0
    assert run.data.metrics["resources/attempt_2/final/optimizer/wall_seconds"] > 0
    plain = trainer_for(tmp_path / "plain", cfg)
    plain.train()
    assert plain.validation.history == resumed.validation.history
    for (name, p), (other, q) in zip(plain.model.named_parameters(), resumed.model.named_parameters()):
        assert name == other
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert len(tracker.client.search_runs([tracker.record["experiment_id"]])) == 1
    assert "selected/selection.json" in artifact_files(tracker.client, old_id)


@pytest.mark.parametrize("winner", [0, 2])
def test_selected_adapter_and_selected_exposure_are_logged_separately(settings, tmp_path, monkeypatch, winner):
    losses = iter([1.0, 2.0, 3.0] if winner == 0 else [3.0, 1.0, 2.0])
    def measured(trainer, rows):
        value = next(losses)
        return [{"paper_id": r["paper_id"], "loss": value, "supervised_tokens": 3} for r in rows]
    monkeypatch.setattr(validation_module, "validation_losses", measured)
    cfg, output = config(max_steps=3), tmp_path / "run"
    tracker = RunTracker.start(settings, output, contract(cfg), method=cfg.method)
    trainer = trainer_for(output, cfg, tracker)
    trainer.train()
    tracker.finish("FINISHED", trainer=trainer)
    run = tracker.client.get_run(tracker.run_id)
    assert run.data.metrics["selected/step"] == winner
    assert run.data.metrics["exposure/examples/total"] == 6
    assert run.data.metrics["selected/examples/total"] == (0 if winner == 0 else 4)
    files = artifact_files(tracker.client, tracker.run_id)
    assert not any(p.endswith(("optimizer.pt", "paper.pdf", "model.safetensors")) and "adapter_model" not in p for p in files)
    if winner:
        assert "selected/adapter/adapter_model.safetensors" in files
        assert run.data.tags["selected.checkpoint"] == "checkpoint-2"
    else:
        assert "selected/adapter/adapter_model.safetensors" not in files
        assert run.data.tags["selected.kind"] == "base"


def test_resume_rejects_changed_backend_or_contract(settings, tmp_path):
    cfg = config()
    tracker = RunTracker.start(settings, tmp_path / "run", contract(cfg), method=cfg.method)
    tracker.finish("FAILED", error=RuntimeError("synthetic failure"))
    changed = deepcopy(contract(cfg))
    changed["config"]["seed"] = 43
    with pytest.raises(ValueError, match="Cannot resume"):
        RunTracker.start(settings, tmp_path / "run", changed, method=cfg.method, resume=True)
    assert tracker.client.get_run(tracker.run_id).data.tags["finetune.error_type"] == "RuntimeError"


def test_later_evaluation_attaches_to_existing_run_and_does_not_change_selection(settings, tmp_path):
    output = tmp_path / "base"
    tracker = RunTracker.start(settings, output, contract(), method="base")
    tracker.finish("FINISHED")
    results = tmp_path / "results.json"
    results.write_text('{"synthetic": true}')
    run_id = attach_evaluation(output, {"entity_f1": 0.75}, {"predictions": results}, {"version": "fixture"})
    assert run_id == tracker.run_id
    attach_evaluation(output, {"entity_f1": 0.75}, {"predictions": results}, {"version": "fixture"})
    assert len(tracker.client.get_metric_history(run_id, "test/entity_f1")) == 1
    assert tracker.client.get_run(run_id).info.status == "FINISHED"
    assert "evaluation/predictions/results.json" in artifact_files(tracker.client, run_id)
    assert len(tracker.client.search_runs([tracker.record["experiment_id"]])) == 1
    results.write_text('{"synthetic": "changed"}')
    with pytest.raises(ValueError, match="Different evaluation"):
        attach_evaluation(output, {"entity_f1": 0.75}, {"predictions": results}, {"version": "fixture"})


def test_evaluation_requires_completed_training_and_excludes_pdfs(settings, tmp_path):
    output = tmp_path / "run"
    cfg = config()
    tracker = RunTracker.start(settings, output, contract(cfg), method=cfg.method)
    with pytest.raises(ValueError, match="Complete the training"):
        attach_evaluation(output, {"f1": 0.5}, {}, {})
    tracker.finish("FINISHED")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="not PDFs"):
        attach_evaluation(output, {"f1": 0.5}, {"predictions": pdf}, {})
    assert "test/f1" not in tracker.client.get_run(tracker.run_id).data.metrics
