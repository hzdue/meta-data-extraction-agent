"""Isolated hardware probes using frozen examples and the production trainer.

Each GPU case runs in its own process/output directory. A failed case is evidence
of a feasibility problem, never permission to truncate or change the experiment.
"""
from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import subprocess
import traceback

import torch
from transformers import TrainerCallback, set_seed

from .tracking import ResourceMeter, RunTracker, read_tracking_config
from .training import attach_lora, build_trainer, load_base, prepare_output, run_contract
from .training_data import TrainingConfig, file_hash
from .validation import generate_answer, paper_mean, score_generation, validation_context, validation_losses, validate_rows, write_json


CASES = ("short", "longest_D", "longest_E", "longest_R", "validation_loss", "validation_generation")


def case_rows(case, records, validation):
    """Select full records deterministically; validation is never a training case."""
    if case not in CASES:
        raise ValueError(f"Unknown smoke case: {case}")
    if case.startswith("validation_"):
        return validation
    task = "D" if case == "short" else case.removeprefix("longest_")
    candidates = [r for r in records if r["task"] == task]
    if not candidates:
        raise ValueError(f"No training examples for task {task}; use the structure-aware config for E/R")
    candidates.sort(key=lambda r: (len(r["input_ids"]), r["example_id"]))
    return [candidates[0] if case == "short" else candidates[-1]]


def describe_rows(rows):
    return [{"example_id": r["example_id"], "task": r["task"],
             "sequence_tokens": len(r["input_ids"]),
             "supervised_tokens": sum(x != -100 for x in r["labels"][1:]),
             **({"prompt_tokens": len(r["prompt_ids"])} if "prompt_ids" in r else {})}
            for r in rows]


def probe_plan(records, validation, config):
    validate_rows(validation, {r["example_id"].split(":")[0] for r in records},
                  config.max_seq_length, config.validation_max_new_tokens)
    tasks = {r["task"] for r in records}
    return {case: describe_rows(case_rows(case, records, validation)) for case in CASES
            if not case.startswith("longest_") or case[-1] in tasks}


def weight_hashes(model):
    """Hash all weights in bounded CPU chunks, including frozen vision/embeddings."""
    hashes = {"frozen": hashlib.sha256(), "adapter": hashlib.sha256()}
    for name, parameter in model.named_parameters():
        digest = hashes["adapter" if parameter.requires_grad else "frozen"]
        digest.update(f"{name}:{parameter.dtype}:{tuple(parameter.shape)}".encode())
        for chunk in parameter.detach().reshape(-1).split(4 * 1024 * 1024):
            digest.update(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return {key: digest.hexdigest() for key, digest in hashes.items()}


class CheckGradients(TrainerCallback):
    def __init__(self):
        self.updates = []

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        nonzero = 0
        for name, parameter in model.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            if not parameter.requires_grad or ".lora_" not in name:
                raise ValueError(f"Unexpected gradient on frozen/base parameter: {name}")
            if not bool(torch.isfinite(grad).all()):
                raise ValueError(f"Nonfinite gradient: {name}")
            nonzero += int(bool(torch.count_nonzero(grad)))
        if not nonzero:
            raise ValueError("No nonzero adapter gradients before the optimizer update")
        self.updates.append({"step": state.global_step + 1, "nonzero_gradient_tensors": nonzero})


def hardware_info():
    result = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "visible_devices": torch.cuda.device_count(), "cuda_available": torch.cuda.is_available(),
              "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))}
    result["optional_kernels"] = {}
    for package in ("flash-linear-attention", "fla-core", "causal-conv1d"):
        try:
            result["optional_kernels"][package] = version(package)
        except PackageNotFoundError:
            result["optional_kernels"][package] = None
    if torch.cuda.is_available():
        result.update(gpu=torch.cuda.get_device_name(),
                      native_bf16=torch.cuda.is_bf16_supported(including_emulation=False),
                      total_bytes=torch.cuda.get_device_properties(0).total_memory,
                      free_bytes=torch.cuda.mem_get_info()[0])
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        result["nvidia_smi"] = query.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        result["nvidia_smi"] = None
    return result


def exercise_case(case, trainer, rows, validation, config, output, report):
    """Also exercised on tiny CPU models; production callers require CUDA BF16."""
    if not case.startswith("validation_"):
        report["phase"] = "weight_fingerprints"
        write_json(output / "smoke_report.json", report)
        before = weight_hashes(trainer.model)
        report["phase"] = "forward_backward_optimizer"
        write_json(output / "smoke_report.json", report)
        result = trainer.train()
        after = weight_hashes(trainer.model)
        if before["frozen"] != after["frozen"]:
            raise ValueError("Frozen base/vision weights changed")
        if before["adapter"] == after["adapter"]:
            raise ValueError("Optimizer did not change adapter weights")
        if not math.isfinite(result.training_loss):
            raise ValueError("Nonfinite training loss")
        if trainer.state.global_step != 1:
            raise ValueError("Expected exactly one smoke optimizer update")
        trainer.save_model(str(output / "adapter"))
        return {"training_loss": result.training_loss, "exposure": trainer.exposure,
                "weight_hashes_before": before, "weight_hashes_after": after,
                "checkpoint": "checkpoint-1", "adapter": "adapter"}

    report["phase"] = case
    write_json(output / "smoke_report.json", report)
    # Exercise the same step-zero baseline path as the production callback.
    with validation_context(trainer.model, base=True):
        if case == "validation_loss":
            def monitored_rows():
                for row in validation:
                    report["active_paper"] = row["paper_id"]
                    write_json(output / "smoke_report.json", report)
                    yield row
            papers = validation_losses(trainer, monitored_rows())
            return {"papers": papers, "loss": paper_mean(papers)}
        diagnostics = []
        for row in validation:
            report["active_paper"] = row["paper_id"]
            write_json(output / "smoke_report.json", report)
            generated = generate_answer(trainer.model, trainer.processing_class, row,
                                        config.validation_max_new_tokens)
            diagnostics.append(score_generation(row, generated, output / "validation"))
            # Keep completed papers even if the next paper fails or the job stops.
            write_json(output / "generation_diagnostics.json", diagnostics)
        return {"diagnostics": diagnostics, "generation_budget": config.validation_max_new_tokens}


def run_probe(root, config, tokenizer, records, validation, summary, architecture,
              preparation, case, output: Path, *, offline=False):
    plan = probe_plan(records, validation, config)
    if case not in plan:
        raise ValueError(f"Case {case} is unavailable for {config.method}")
    hardware = hardware_info()
    if not hardware["cuda_available"] or not hardware.get("native_bf16") or hardware["visible_devices"] != 1:
        raise ValueError("GPU probes require one visible CUDA GPU with native BF16")
    settings = read_tracking_config(root / config.tracking_config, require_ready=True)
    # Only the duration/checkpoint interval/output change. Batch size, accumulation,
    # precision, attention, LoRA, optimizer and context remain production settings.
    values = config.model_dump()
    values.update(max_steps=1, save_steps=1, validation_steps=1, output_dir=str(output))
    probe_config = TrainingConfig.model_validate(values)
    contract = run_contract(root, probe_config, summary, architecture)
    contract["smoke"] = {"case": case, "examples": plan[case], "production_config": config.model_dump(),
                         "not_a_comparative_run": True}
    for name in ("scripts/smoke_finetune.py", "src/meta_data_extraction/finetune/smoke.py"):
        contract["sources"][name] = file_hash(root / name)
    prepare_output(output, contract, None)
    report = {"status": "running", "phase": "model_load", "case": case, "hardware": hardware,
              "examples": plan[case], "production_ready": False}
    write_json(output / "smoke_report.json", report)
    tracker = RunTracker.start(settings, output, contract, method=config.method)
    tracker.client.set_tag(tracker.run_id, "finetune.phase", "smoke")
    tracker.client.set_tag(tracker.run_id, "finetune.smoke_case", case)
    tracker.record["run_name"] = f"smoke_{config.method}_{case}"
    tracker.client.set_tag(tracker.run_id, "mlflow.runName", tracker.record["run_name"])
    write_json(output / "mlflow_run.json", tracker.record)
    tracker.client.log_dict(tracker.run_id, tracker.record, "provenance/mlflow_run.json")
    report["mlflow_run_id"] = tracker.run_id
    trainer = None
    tracker.resources = ResourceMeter(cuda=True)
    try:
        set_seed(config.seed)
        model = attach_lora(load_base(preparation, probe_config, offline=offline), probe_config, preparation)
        chosen = case_rows(case, records, validation)
        # Repeating one complete example fills an entire accumulation window.
        # Validation rows are never passed to the training dataset.
        training_rows = (chosen if not case.startswith("validation_") else records[:1]) * config.gradient_accumulation_steps
        gradient_check = CheckGradients()
        trainer = build_trainer(model, tokenizer, training_rows, probe_config, output,
                                tracking=tracker, callbacks=[gradient_check])
        trainer.args.logging_nan_inf_filter = False
        # Training/optimizer phases are already timed by the shared trainer.
        # Avoid overlapping CUDA events being summed twice in resource totals.
        with tracker.resources.phase("validation") if case.startswith("validation_") else nullcontext():
            details = exercise_case(case, trainer, chosen, validation, probe_config, output, report)
        report.update(status="passed", phase="complete", results=details,
                      gradient_checks=gradient_check.updates)
    except BaseException as exc:
        trace_path = output / "smoke_traceback.txt"
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")
        report.update(status="out_of_memory" if isinstance(exc, torch.OutOfMemoryError) else "failed",
                      error_type=type(exc).__name__, error=str(exc), resources=tracker.resources.snapshot())
        write_json(output / "smoke_report.json", report)
        try:
            tracker.client.log_artifact(tracker.run_id, str(output / "smoke_report.json"), "smoke")
            tracker.client.log_artifact(tracker.run_id, str(trace_path), "smoke")
            tracker.finish("KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED", trainer=trainer, error=exc)
        except Exception as logging_error:
            exc.add_note(f"Smoke failure logging also failed: {type(logging_error).__name__}")
        raise
    report["resources"] = tracker.resources.snapshot()
    write_json(output / "smoke_report.json", report)
    tracker.client.log_artifact(tracker.run_id, str(output / "smoke_report.json"), "smoke")
    tracker.finish("FINISHED", trainer=trainer)
    return report
