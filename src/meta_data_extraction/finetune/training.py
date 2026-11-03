"""Shared LoRA trainer. Real pilot runs are single-process, single-GPU BF16."""
from __future__ import annotations

from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import subprocess
import sys

from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import AutoConfig, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from .training_data import AnswerCollator, TrainingConfig, file_hash
from .memory import chunk_text_mlps
from .validation import ValidationCallback, write_json


PROJECTIONS = {
    "self_attn": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "linear_attn": {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"},
    "mlp": {"gate_proj", "up_proj", "down_proj"},
}


def language_targets(model) -> list[str]:
    """Explicit full paths avoid accidentally adapting the vision MLP/attention."""
    targets = []
    for i, layer_type in enumerate(model.config.text_config.layer_types):
        attention = "self_attn" if layer_type == "full_attention" else "linear_attn"
        if layer_type not in {"full_attention", "linear_attention"}:
            raise ValueError(f"Unexpected Qwen layer type: {layer_type}")
        for component in (attention, "mlp"):
            for projection in sorted(PROJECTIONS[component]):
                name = f"model.language_model.layers.{i}.{component}.{projection}"
                if not isinstance(model.get_submodule(name), torch.nn.Linear):
                    raise ValueError(f"Expected a language linear projection: {name}")
                targets.append(name)
    if not targets:
        raise ValueError("No language LoRA targets found")
    return targets


def inspect_model(preparation: dict, *, offline: bool) -> dict:
    """Instantiate the pinned architecture on meta tensors; load no weights."""
    config = AutoConfig.from_pretrained(preparation["model_id"], revision=preparation["revision"],
                                      trust_remote_code=False, local_files_only=offline)
    if config.model_type != "qwen3_5":
        raise ValueError("Expected Qwen3.5 multimodal architecture")
    if preparation["max_seq_length"] > config.text_config.max_position_embeddings:
        raise ValueError("Training cap exceeds the model context")
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(config)
    targets = language_targets(model)
    return {"model_class": type(model).__name__, "target_modules": targets,
            "target_module_count": len(targets), "vision_target_count": 0,
            "base_parameter_count": sum(p.numel() for p in model.parameters())}


def attach_lora(model, config: TrainingConfig, preparation: dict):
    if bool(getattr(model, "is_loaded_in_4bit", False)) != config.load_in_4bit:
        raise ValueError("Base quantization differs from the training configuration")
    targets = language_targets(model)
    adapter = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.lora_rank, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, bias="none",
        target_modules="(?:" + "|".join(re.escape(name) for name in targets) + ")",
        modules_to_save=None,
    )
    model = get_peft_model(model, adapter)
    model.peft_config["default"].base_model_name_or_path = preparation["model_id"]
    model.peft_config["default"].revision = preparation["revision"]
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(".language_model.layers." not in name or ".lora_" not in name for name in trainable):
        raise ValueError("Only language-layer LoRA weights may be trainable")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    chunk_text_mlps(model, config.mlp_chunk_size)
    return model


def load_base(preparation: dict, config: TrainingConfig, *, offline=False):
    options = {}
    if config.load_in_4bit:
        options = {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16), "device_map": {"": 0}}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        preparation["model_id"], revision=preparation["revision"], trust_remote_code=False,
        local_files_only=offline, dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
        **options,
    )
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
        # Retain FP32 stability parameters, but avoid an FP32 copy of the large
        # frozen vocabulary matrix. Apply identically for training and reload.
        for module in (model.get_input_embeddings(), model.get_output_embeddings()):
            module.to(dtype=torch.bfloat16)
    return model


def reload_adapter(base_model, path: Path, preparation: dict, *, trainable=False, config: TrainingConfig | None = None):
    precision_file = path / "inference_precision.json"
    if config is not None and not precision_file.is_file():
        raise ValueError("Adapter is missing its inference precision contract")
    if precision_file.exists():
        precision = json.loads(precision_file.read_text(encoding="utf-8"))
        quantized = bool(getattr(base_model, "is_loaded_in_4bit", False))
        if quantized != (precision["base"] == "nf4_double_quant"):
            raise ValueError("Adapter/base inference precision mismatch")
        if config is not None and precision != config.precision_policy():
            raise ValueError("Adapter/config inference precision mismatch")
    saved = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if (saved["base_model_name_or_path"], saved["revision"]) != (preparation["model_id"], preparation["revision"]):
        raise ValueError("Adapter base model/revision mismatch")
    model = PeftModel.from_pretrained(base_model, str(path), is_trainable=trainable)
    if config is not None:
        chunk_text_mlps(model, config.mlp_chunk_size)
    return model


class StopAfterCheckpoint(TrainerCallback):
    """A controlled interruption after a fully committed validation checkpoint."""
    def __init__(self, step: int):
        self.step = step

    def on_save(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            control.should_training_stop = True
        return control


class PilotTrainer(SFTTrainer):
    """Keep task metadata out of model inputs and persist actual token exposure."""
    def __init__(self, *args, **kwargs):
        self.exposure = {"examples": {}, "supervised_tokens": {}, "total_tokens": {}}
        self.validation = None
        self.tracking = None
        self.offload_activations = False
        self.inference_precision = None
        super().__init__(*args, **kwargs)
        # SFT computes summed token loss divided by all target tokens in the
        # accumulation window. Ask Trainer to count those tokens even though the
        # multimodal Qwen wrapper itself declares accepts_loss_kwargs=False.
        self.model_accepts_loss_kwargs = True
        self.loss_is_scaled_for_ga = True

    def training_step(self, model, inputs, num_items_in_batch=None):
        context = self.tracking.resources.phase("training") if self.tracking else nullcontext()
        with context:
            return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        tasks = inputs.pop("tasks")
        # Saved tensors are copied back to their original device by autograd.
        # Keep TRL's fused answer loss and accumulation-window token weighting.
        offload = (torch.autograd.graph.save_on_cpu(pin_memory=self.args.device.type == "cuda")
                   if self.offload_activations and model.training and torch.is_grad_enabled() else nullcontext())
        with offload:
            result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        if model.training:
            for i, task in enumerate(tasks):
                values = {"examples": 1,
                          "supervised_tokens": int((inputs["labels"][i, 1:] != -100).sum()),
                          "total_tokens": int(inputs["attention_mask"][i].sum())}
                for key, value in values.items():
                    self.exposure[key][task] = self.exposure[key].get(task, 0) + value
        return result

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.is_world_process_zero():
            path = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            write_json(path / "exposure.json", {"global_step": self.state.global_step, **self.exposure})
            if self.validation is not None:
                write_json(path / "validation_state.json", self.validation.state_dict())

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        if self.is_world_process_zero() and self.inference_precision is not None:
            write_json(Path(output_dir or self.args.output_dir) / "inference_precision.json", self.inference_precision)

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if resume_from_checkpoint:
            saved = json.loads((Path(resume_from_checkpoint) / "exposure.json").read_text(encoding="utf-8"))
            state = json.loads((Path(resume_from_checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))
            if saved["global_step"] != state["global_step"]:
                raise ValueError("Checkpoint exposure/state steps differ")
            self.exposure = {k: saved[k] for k in self.exposure}
            if self.validation is not None:
                self.validation.restore(Path(resume_from_checkpoint), state["global_step"])
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)


def build_trainer(model, tokenizer, records, config: TrainingConfig, output: Path, *,
                  validation=None, tracking=None, cpu=False, callbacks=None,
                  selection_policy="loss"):
    if config.max_steps is None:
        raise ValueError("Set a finite max_steps after the server feasibility check")
    args = SFTConfig(
        output_dir=str(output), max_steps=config.max_steps, seed=config.seed, data_seed=config.seed,
        per_device_train_batch_size=config.microbatch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=0, weight_decay=config.weight_decay, max_grad_norm=config.max_grad_norm,
        optim="adamw_torch", bf16=not cpu, fp16=False, use_cpu=cpu,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False, padding_free=False, max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        completion_only_loss=False, assistant_only_loss=False,  # explicit labels are authoritative
        remove_unused_columns=False, loss_type="nll", label_names=["labels"],
        eval_strategy="no", save_strategy="steps", save_steps=config.save_steps,
        save_only_model=False, save_total_limit=None,
        logging_strategy="steps", logging_steps=1, report_to=[],
        dataloader_num_workers=0, dataloader_pin_memory=not cpu,
        accelerator_config={"use_seedable_sampler": True},
        disable_tqdm=True,
    )
    trainer = PilotTrainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_list(records),
        data_collator=AnswerCollator(tokenizer.pad_token_id, config.max_seq_length),
        callbacks=callbacks,
    )
    trainer.offload_activations = config.offload_activations
    trainer.inference_precision = config.precision_policy()
    if trainer.args.world_size != 1 or trainer.args.n_gpu > 1:
        raise ValueError("This pilot trainer currently requires one process and one visible GPU")
    if validation is not None:
        trainer.validation = ValidationCallback(trainer, validation, config,
                                                selection_policy=selection_policy)
        trainer.add_callback(trainer.validation)
    if tracking is not None:
        from .tracking import ResourceMeter, TrackingCallback

        trainer.tracking = tracking
        tracking.resources = ResourceMeter(cuda=trainer.args.device.type == "cuda")
        # After validation: publish only checkpoint-committed selection/history.
        trainer.add_callback(TrackingCallback(trainer, tracking))
    return trainer


def run_contract(root: Path, config: TrainingConfig, summary: dict, architecture: dict) -> dict:
    files = ["pyproject.toml", "uv.lock", "scripts/run_finetune.py",
             "src/meta_data_extraction/finetune/training.py",
             "src/meta_data_extraction/finetune/training_data.py",
             "src/meta_data_extraction/finetune/memory.py",
             "src/meta_data_extraction/finetune/validation.py",
             "src/meta_data_extraction/finetune/schema.py",
             "src/meta_data_extraction/finetune/tracking.py", config.tracking_config,
             "src/meta_data_extraction/evaluation.py", "src/meta_data_extraction/alignment.py",
             "src/meta_data_extraction/normalization.py", "src/meta_data_extraction/config_validation.py",
             "configs/evaluation_metrics.yaml", "configs/alignment_helper.yaml"]
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    prepared = json.loads((root / config.prepared_dir / "preparation_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((root / prepared["config"]["split_manifest"]).read_text(encoding="utf-8"))
    parsers = {json.dumps(p["parser"], sort_keys=True) for p in prepared["papers"]}
    optional_versions = {}
    for package in ("bitsandbytes", "flash-linear-attention", "fla-core", "einops", "causal-conv1d"):
        try:
            optional_versions[package] = version(package)
        except PackageNotFoundError:
            optional_versions[package] = None
    return {"config": config.model_dump(), "inference_precision": config.precision_policy(),
            "data": summary, "architecture": architecture,
            "preparation": {"inputs": prepared["inputs"], "config": prepared["config"],
                            "runtime": prepared["runtime"], "parsers": [json.loads(p) for p in sorted(parsers)]},
            "split": {key: split[key] for key in ("train", "validation", "test")},
            "code": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
            "sources": {p: file_hash(root / p) for p in files},
            "runtime": {"python": sys.version.split()[0], **optional_versions,
                        **{p: version(p) for p in ("torch", "transformers", "tokenizers", "peft", "trl", "accelerate", "mlflow")}}}


def prepare_output(output: Path, contract: dict, resume: Path | None):
    """Reject accidental overwrites and incompatible resumes before model loading."""
    manifest = output / "run_manifest.json"
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty; choose a new path or resume a checkpoint")
        output.mkdir(parents=True, exist_ok=True)
        write_json(manifest, contract)
        return
    if resume.resolve().parent != output.resolve():
        raise ValueError("Resume checkpoint must belong to the configured output directory")
    if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")) != contract:
        raise ValueError("Resume configuration/data/code/runtime differs from the saved run")
    required = ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "rng_state.pth", "exposure.json", "validation_state.json",
                "inference_precision.json")
    for name in required:
        if not (resume / name).is_file():
            raise ValueError(f"Incomplete checkpoint: {name}")
    step = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))["global_step"]
    if step >= contract["config"]["max_steps"]:
        raise ValueError("Checkpoint already reached this run's step cap")
    # validation_state.json is written atomically after the model/optimizer save.
    # Ignore a later partial save left by an interruption, allowing it to be rerun.
    latest = max(int(p.name.split("-")[-1]) for p in output.glob("checkpoint-*")
                 if p.is_dir() and p.name.split("-")[-1].isdigit()
                 and all((p / name).is_file() for name in required))
    if step != latest:
        raise ValueError("Resume the latest complete checkpoint to avoid replacing later checkpoints")


def run_training(root: Path, config: TrainingConfig, tokenizer, records, summary, architecture,
                 preparation: dict, *, validation, offline=False, resume: Path | None = None,
                 stop_after_step: int | None = None, smoke: bool = False,
                 selection_policy="loss", experiment: dict | None = None):
    from .tracking import RunTracker, read_tracking_config

    if config.max_steps is None:
        raise ValueError("Set max_steps in both configs after the hardware check, or pass --max-steps for a smoke run")
    if stop_after_step is not None and (stop_after_step <= 0 or stop_after_step >= config.max_steps
                                       or stop_after_step % config.save_steps):
        raise ValueError("stop_after_step must be a checkpoint boundary before max_steps")
    if validation is None:
        raise ValueError("Server training requires the held-out validation examples")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("Real training requires a CUDA GPU with native BF16; use --dry-run for CPU checks")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES for this pilot")
    tracking_settings = read_tracking_config(root / config.tracking_config, require_ready=True)
    set_seed(config.seed)
    output = (root / config.output_dir).resolve()
    contract = run_contract(root, config, summary, architecture)
    experiment_phase = experiment.get("phase", "ablation") if experiment else None
    if experiment_phase not in {None, "ablation", "followup_phase_b"}:
        raise ValueError(f"Unsupported training experiment phase: {experiment_phase}")
    contract["phase"] = experiment_phase or ("smoke" if smoke else "comparison")
    if experiment:
        contract["ablation" if experiment_phase == "ablation" else "experiment"] = experiment
    prepare_output(output, contract, resume)
    tracker = RunTracker.start(tracking_settings, output, contract, method=config.method, resume=resume is not None)
    tracker.client.set_tag(tracker.run_id, "finetune.phase", contract["phase"])
    if experiment:
        if experiment_phase == "ablation":
            tracker.client.set_tag(tracker.run_id, "finetune.ablation_condition", experiment["condition"])
            tracker.client.set_tag(tracker.run_id, "finetune.ablation_family", experiment["family"])
            run_name = f"ablation_{experiment['condition']}_seed{config.seed}"
        else:
            tracker.client.set_tag(tracker.run_id, "finetune.experiment_condition", experiment["condition"])
            tracker.client.set_tag(tracker.run_id, "finetune.experiment_family", experiment["family"])
            run_name = f"{experiment_phase}_{experiment['condition']}_seed{config.seed}"
        tracker.client.set_tag(tracker.run_id, "mlflow.runName", run_name)
    if smoke:
        tracker.client.set_tag(tracker.run_id, "mlflow.runName", f"smoke_{config.method}_integrated")
    trainer = None
    try:
        set_seed(config.seed)
        model = attach_lora(load_base(preparation, config, offline=offline), config, preparation)
        trainer = build_trainer(model, tokenizer, records, config, output, validation=validation,
                                tracking=tracker, selection_policy=selection_policy)
        if stop_after_step is not None:
            trainer.add_callback(StopAfterCheckpoint(stop_after_step))
        tracker.client.log_dict(tracker.run_id, {"gpu": torch.cuda.get_device_name(),
                                "cuda_runtime": torch.version.cuda,
                                "device_total_bytes": torch.cuda.get_device_properties(0).total_memory},
                                f"attempts/{tracker.record['attempt']}/hardware.json")
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        trainer._save_checkpoint(trainer.model, trial=None)
        trainer.save_model(str(output / "adapter"))
        trainer.save_state()
        complete = trainer.state.global_step == config.max_steps
        selection = trainer.validation.publish(complete=complete)
        write_json(output / "training_summary.json", {
            "global_step": trainer.state.global_step, "metrics": result.metrics,
            "exposure": trainer.exposure, "selection": selection, "mlflow_run_id": tracker.run_id,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            **({"ablation" if experiment_phase == "ablation" else "experiment": experiment}
               if experiment else {}),
        })
    except BaseException as exc:
        try:
            tracker.finish("KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED", trainer=trainer, error=exc)
        except Exception as logging_error:
            exc.add_note(f"MLflow finalization also failed: {type(logging_error).__name__}")
        raise
    tracker.finish("FINISHED" if complete else "KILLED", trainer=trainer)
    return {"output_dir": str(output), "global_step": trainer.state.global_step, "mlflow_run_id": tracker.run_id}
