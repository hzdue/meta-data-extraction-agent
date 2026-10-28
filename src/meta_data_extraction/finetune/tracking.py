"""Explicit MLflow runs, committed metrics, selected adapters and later evaluation."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Literal
from urllib.parse import urlsplit

from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from pydantic import BaseModel, ConfigDict
from sqlalchemy.engine import make_url
import torch
from transformers import TrainerCallback
import yaml

from .parsing import text_hash
from .validation import write_json


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"] = "1.0.0"
    experiment_name: str = "pan-qwen35-4b-finetuning"
    tracking_uri: str | None = None
    artifact_location: str | None = None
    upload_selected_adapter: bool = True

    def require_ready(self):
        if not self.tracking_uri:
            raise ValueError("Set MLFLOW_TRACKING_URI or tracking_uri in the shared tracking config before training")
        uri = self.tracking_uri
        if uri.startswith("sqlite:"):
            database = make_url(uri).database
            if not database or not Path(database).is_absolute():
                raise ValueError("Use an absolute persistent SQLite database path")
            if not self.artifact_location:
                raise ValueError("SQLite tracking requires an explicit persistent artifact_location")
        elif urlsplit(uri).scheme in {"http", "https"}:
            parsed = urlsplit(uri)
            if not parsed.hostname or parsed.username or parsed.password or parsed.query:
                raise ValueError("Use a tracking server URL without embedded credentials or query parameters")
        else:
            raise ValueError("Use an absolute SQLite URI or an HTTP(S) MLflow tracking server")
        if not self.experiment_name.strip():
            raise ValueError("MLflow experiment_name cannot be empty")
        if self.artifact_location:
            location = self.artifact_location
            if not urlsplit(location).scheme and not Path(location).is_absolute():
                raise ValueError("artifact_location must be an absolute path or storage URI")


def read_tracking_config(path: Path, *, require_ready=False) -> TrackingConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    for field, variable in (("tracking_uri", "MLFLOW_TRACKING_URI"),
                            ("artifact_location", "FINETUNE_MLFLOW_ARTIFACT_LOCATION")):
        if os.environ.get(variable):
            values[field] = os.environ[variable]
    config = TrackingConfig.model_validate(values)
    if require_ready:
        config.require_ready()
    return config


def checksum(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exposure_metrics(exposure, prefix):
    result = {}
    for category, counts in exposure.items():
        result[f"{prefix}/{category}/total"] = sum(counts.values())
        result.update({f"{prefix}/{category}/{task}": value for task, value in counts.items()})
    return result


class ResourceMeter:
    """CUDA-event elapsed time is not kernel-utilization time; CPU runs omit it."""
    def __init__(self, cuda=False):
        self.cuda = cuda
        self.started = time.perf_counter()
        self.phases = {}
        self.pending = {}
        if cuda:
            torch.cuda.reset_peak_memory_stats()

    def begin(self, phase):
        event = torch.cuda.Event(enable_timing=True) if self.cuda else None
        if event:
            event.record()
        self.pending[phase] = (time.perf_counter(), event)

    def end(self, phase):
        start, event = self.pending.pop(phase)
        values = self.phases.setdefault(phase, {"wall_seconds": 0.0})
        if event:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            values["cuda_elapsed_seconds"] = values.get("cuda_elapsed_seconds", 0.0) + event.elapsed_time(end) / 1000
        values["wall_seconds"] += time.perf_counter() - start

    @contextmanager
    def phase(self, name):
        self.begin(name)
        try:
            yield
        finally:
            self.end(name)

    def snapshot(self):
        values = {"elapsed_seconds": time.perf_counter() - self.started}
        values.update({f"{phase}/{key}": value for phase, counts in self.phases.items() for key, value in counts.items()})
        if self.cuda:
            values.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          cuda_elapsed_seconds_total=sum(p.get("cuda_elapsed_seconds", 0.0) for p in self.phases.values()))
        return values


class RunTracker:
    """One scientific run can contain several interrupted/resumed execution attempts."""
    def __init__(self, settings, output, client, record):
        self.settings, self.output, self.client, self.record = settings, output, client, record
        self.run_id = record["run_id"]
        self.seen = {}
        self.resources = None

    @classmethod
    def start(cls, settings: TrackingConfig, output: Path, contract: dict, *, method: str, resume=False):
        settings.require_ready()
        if method not in {"base", "json_only", "structure_aware"}:
            raise ValueError("Unknown fine-tuning method")
        client = MlflowClient(tracking_uri=settings.tracking_uri)
        digest = text_hash(json.dumps(contract, sort_keys=True, ensure_ascii=False))
        path = output / "mlflow_run.json"
        if resume:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (record["settings"] != settings.model_dump() or record["contract_sha256"] != digest
                    or record["method"] != method):
                raise ValueError("Cannot resume a different MLflow backend, contract or method")
            run = client.get_run(record["run_id"])
            if (run.info.lifecycle_stage != "active" or run.info.experiment_id != record["experiment_id"]
                    or run.data.tags.get("finetune.contract_sha256") != digest
                    or run.data.tags.get("finetune.method") != method):
                raise ValueError("Saved MLflow run does not match this experiment/contract")
            if run.info.status == "FINISHED":
                raise ValueError("This MLflow run is already complete")
            record["attempt"] = int(run.data.tags.get("finetune.attempt", "0")) + 1
            client.update_run(record["run_id"], status="RUNNING")
        else:
            if path.exists():
                raise ValueError("MLflow run already exists; resume it instead")
            experiment = client.get_experiment_by_name(settings.experiment_name)
            location = settings.artifact_location
            if location and Path(location).is_absolute():
                location = Path(location).as_uri()
            if experiment is None:
                experiment_id = client.create_experiment(settings.experiment_name, artifact_location=location)
                experiment = client.get_experiment(experiment_id)
            elif experiment.lifecycle_stage != "active" or (location and experiment.artifact_location.rstrip("/") != location.rstrip("/")):
                raise ValueError("Existing experiment has a different artifact location or is deleted")
            name = "base" if method == "base" else f"{method}_seed{contract['config']['seed']}"
            run = client.create_run(experiment.experiment_id, run_name=name,
                                    tags={"finetune.method": method, "finetune.contract_sha256": digest,
                                          "finetune.phase": "evaluation_only" if method == "base" else "training"})
            record = {"version": "1.0.0", "run_id": run.info.run_id, "run_name": name,
                      "experiment_id": experiment.experiment_id, "artifact_uri": run.info.artifact_uri,
                      "settings": settings.model_dump(), "contract_sha256": digest,
                      "method": method, "attempt": 1}
        output.mkdir(parents=True, exist_ok=True)
        write_json(path, record)  # Persist identity before any artifact/metric upload.
        tracker = cls(settings, output, client, record)
        try:
            client.set_tag(tracker.run_id, "finetune.attempt", str(record["attempt"]))
            tracker.log_contract(contract)
        except Exception:
            client.set_terminated(tracker.run_id, "FAILED")
            raise
        return tracker

    def log_contract(self, contract):
        params = {"method": self.record["method"], "contract_sha256": self.record["contract_sha256"]}
        params.update({k: str(v) for k, v in contract.get("config", {}).items() if not isinstance(v, (dict, list))})
        params["method"] = self.record["method"]
        tokenizer = contract.get("data", {}).get("tokenizer", {})
        params.update({f"model.{k}": str(tokenizer[k]) for k in ("model_id", "revision") if k in tokenizer})
        params.update({f"runtime.{k}": str(v) for k, v in contract.get("runtime", {}).items()})
        params["training_task_ratio"] = {"base": "none", "json_only": "D only", "structure_aware": "D:E:R=1:1:1 per pass"}[self.record["method"]]
        params.update(decoding="greedy; thinking disabled", validation_weighting="equal per paper")
        if self.record["method"] != "base":
            params.update(optimizer="adamw_torch", warmup_steps="0")
        if "uv.lock" in contract.get("sources", {}):
            params["uv_lock_sha256"] = contract["sources"]["uv.lock"]
        self.client.log_batch(self.run_id, params=[Param(k, str(v)) for k, v in params.items()], synchronous=True)
        self.client.log_dict(self.run_id, contract, "provenance/run_manifest.json")
        self.client.log_dict(self.run_id, self.record, "provenance/mlflow_run.json")
        path = self.output / "resolved_config.yaml"
        path.write_text(yaml.safe_dump(contract.get("config", {}), sort_keys=False), encoding="utf-8")
        self.client.log_artifact(self.run_id, str(path), "provenance")

    def log_metrics(self, values: dict, step: int, *, strict=True):
        pending = []
        timestamp = int(time.time() * 1000)
        for key, value in values.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"MLflow metric must be finite: {key}")
            if key not in self.seen:
                self.seen[key] = {m.step: m.value for m in self.client.get_metric_history(self.run_id, key)}
            if step in self.seen[key]:
                if strict and self.seen[key][step] != float(value):
                    raise ValueError(f"Conflicting metric for committed step: {key}@{step}")
                continue
            pending.append(Metric(key, float(value), timestamp, step))
        if pending:
            self.client.log_batch(self.run_id, metrics=pending, synchronous=True)
            for item in pending:
                self.seen[item.key][step] = item.value

    def sync_progress(self, trainer):
        # Only called after a checkpoint commit (or step zero), so resumed runs
        # never upload losses for discarded, uncheckpointed optimizer updates.
        for row in trainer.state.log_history:
            values = {f"train/{key}": row[key] for key in
                      ("loss", "learning_rate", "grad_norm", "entropy", "mean_token_accuracy") if key in row}
            if values:
                self.log_metrics(values, row["step"])
        if trainer.validation:
            for entry in trainer.validation.history:
                values = {"validation/loss": entry["loss"]}
                values.update({f"validation/{p['paper_id']}/loss": p["loss"] for p in entry["papers"]})
                for diagnostic in entry["diagnostics"]:
                    prefix = f"validation/{diagnostic['paper_id']}"
                    values[f"{prefix}/generation_failure"] = int(diagnostic["status"] != "scored")
                    values[f"{prefix}/hit_output_limit"] = int(diagnostic["hit_output_limit"])
                    for entity, f1 in (diagnostic["entity_f1"] or {}).items():
                        values[f"{prefix}/{entity}_f1"] = f1
                    for field, scores in (diagnostic["conditional_reference_f1"] or {}).items():
                        values[f"{prefix}/{field}_n_pairs"] = scores["n_pairs"]
                        if scores["f1"] is not None:
                            values[f"{prefix}/{field}_conditional_f1"] = scores["f1"]
                values.update(exposure_metrics(entry["exposure"], "exposure"))
                self.log_metrics(values, entry["step"])
        self.log_metrics({"optimizer_updates": trainer.state.global_step,
                          **exposure_metrics(trainer.exposure, "exposure")}, trainer.state.global_step)
        for name in ("validation_history.json", "selection.json"):
            if (self.output / name).exists():
                self.client.log_artifact(self.run_id, str(self.output / name), "progress")
        if self.resources:
            prefix = f"resources/attempt_{self.record['attempt']}"
            self.log_metrics({f"{prefix}/{k}": v for k, v in self.resources.snapshot().items()}, trainer.state.global_step, strict=False)

    def finish(self, status: str, *, trainer=None, error=None):
        try:
            if self.resources:
                step = trainer.state.global_step if trainer else 0
                values = self.resources.snapshot()
                self.client.log_dict(self.run_id, values, f"attempts/{self.record['attempt']}/resources.json")
                self.log_metrics({f"resources/attempt_{self.record['attempt']}/final/{k}": v for k, v in values.items()}, step)
            if error:
                self.client.set_tag(self.run_id, "finetune.error_type", type(error).__name__)
            if trainer and (self.output / "selection.json").exists():
                self.log_selection()
            for name in ("training_summary.json", "trainer_state.json"):
                if (self.output / name).exists():
                    self.client.log_artifact(self.run_id, str(self.output / name), "progress")
            if (self.output / "validation").exists():
                self.client.log_artifacts(self.run_id, str(self.output / "validation"), "validation")
        except Exception:
            self.client.set_terminated(self.run_id, "FAILED")
            raise
        self.client.set_terminated(self.run_id, status)

    def log_selection(self):
        selection = json.loads((self.output / "selection.json").read_text(encoding="utf-8"))
        last_step = json.loads((self.output / "validation_history.json").read_text(encoding="utf-8"))["history"][-1]["step"]
        self.log_metrics({"selected/step": selection["step"], "selected/loss": selection["loss"],
                          **exposure_metrics(selection["exposure"], "selected")}, last_step)
        self.client.set_tag(self.run_id, "selected.kind", selection["kind"])
        self.client.set_tag(self.run_id, "selected.checkpoint", selection["checkpoint"] or "base")
        self.client.log_artifact(self.run_id, str(self.output / "selection.json"), "selected")
        if selection["kind"] == "adapter":
            path = self.output / selection["checkpoint"]
            if checksum(path / "adapter_model.safetensors") != selection["adapter_sha256"]:
                raise ValueError("Selected adapter checksum changed")
            self.client.set_tag(self.run_id, "selected.adapter_sha256", selection["adapter_sha256"])
            self.client.set_tag(self.run_id, "selected.persistent_path", str(path.resolve()))
            if self.settings.upload_selected_adapter:
                for name in ("adapter_model.safetensors", "adapter_config.json"):
                    self.client.log_artifact(self.run_id, str(path / name), "selected/adapter")


class TrackingCallback(TrainerCallback):
    def __init__(self, trainer, tracker):
        self.trainer, self.tracker = trainer, tracker

    def on_train_begin(self, args, state, control, **kwargs):
        self.tracker.client.set_tag(self.tracker.run_id, "finetune.resume_step", str(state.global_step))
        self.tracker.sync_progress(self.trainer)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.tracker.resources.begin("optimizer")

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.tracker.resources.end("optimizer")

    def on_save(self, args, state, control, **kwargs):
        self.tracker.sync_progress(self.trainer)


def attach_evaluation(output: Path, metrics: dict, artifacts: dict[str, Path], evaluator: dict):
    """Step 7 can attach saved results to a base/M1/M2 run without retraining."""
    record = json.loads((output / "mlflow_run.json").read_text(encoding="utf-8"))
    settings = TrackingConfig.model_validate(record["settings"])
    tracker = RunTracker(settings, output, MlflowClient(tracking_uri=settings.tracking_uri), record)
    run = tracker.client.get_run(tracker.run_id)
    if (run.info.experiment_id != record["experiment_id"] or run.info.lifecycle_stage != "active"
            or run.data.tags.get("finetune.contract_sha256") != record["contract_sha256"]
            or run.data.tags.get("finetune.method") != record["method"]):
        raise ValueError("Evaluation run ID does not match the saved experiment/contract")
    if record["method"] != "base" and run.info.status != "FINISHED":
        raise ValueError("Complete the training run before attaching held-out evaluation")
    files = {}
    for name, path in artifacts.items():
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError("Evaluation artifact group must be a single path component")
        candidates = list(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_dir():
                continue
            if candidate.suffix.lower() not in {".json", ".jsonl", ".txt", ".csv", ".tsv", ".md"} or candidate.is_symlink():
                raise ValueError("Evaluation artifacts must be prediction/report files, not PDFs or model weights")
            relative = candidate.relative_to(path).as_posix() if path.is_dir() else candidate.name
            files[f"{name}/{relative}"] = checksum(candidate)
    manifest = {"metrics": metrics, "evaluator": evaluator, "files": files}
    digest = text_hash(json.dumps(manifest, sort_keys=True, allow_nan=False))
    previous = run.data.tags.get("finetune.evaluation_sha256")
    if previous is not None and previous != digest:
        raise ValueError("Different evaluation results are already attached to this run")
    tracker.log_metrics({f"test/{k}": v for k, v in metrics.items()}, 0)
    tracker.client.log_dict(tracker.run_id, manifest, "evaluation/manifest.json")
    tracker.client.log_dict(tracker.run_id, evaluator, "evaluation/evaluator.json")
    for name, path in artifacts.items():
        if path.is_dir():
            tracker.client.log_artifacts(tracker.run_id, str(path), f"evaluation/{name}")
        else:
            tracker.client.log_artifact(tracker.run_id, str(path), f"evaluation/{name}")
    tracker.client.set_tag(tracker.run_id, "finetune.evaluation_logged", "true")
    tracker.client.set_tag(tracker.run_id, "finetune.evaluation_sha256", digest)
    return tracker.run_id
