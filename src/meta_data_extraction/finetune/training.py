"""Shared LoRA trainer. Real pilot runs are single-process, single-GPU BF16."""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import re

from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration, set_seed
from trl import SFTConfig, SFTTrainer

from .training_data import AnswerCollator, TrainingConfig, file_hash


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
    return model


def load_base(preparation: dict, config: TrainingConfig, *, offline=False):
    return Qwen3_5ForConditionalGeneration.from_pretrained(
        preparation["model_id"], revision=preparation["revision"], trust_remote_code=False,
        local_files_only=offline, dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
    )


def reload_adapter(base_model, path: Path, preparation: dict, *, trainable=False):
    saved = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if (saved["base_model_name_or_path"], saved["revision"]) != (preparation["model_id"], preparation["revision"]):
        raise ValueError("Adapter base model/revision mismatch")
    return PeftModel.from_pretrained(base_model, str(path), is_trainable=trainable)


class PilotTrainer(SFTTrainer):
    """Keep task metadata out of model inputs and persist actual token exposure."""
    def __init__(self, *args, **kwargs):
        self.exposure = {"examples": {}, "supervised_tokens": {}, "total_tokens": {}}
        super().__init__(*args, **kwargs)
        # SFT computes summed token loss divided by all target tokens in the
        # accumulation window. Ask Trainer to count those tokens even though the
        # multimodal Qwen wrapper itself declares accepts_loss_kwargs=False.
        self.model_accepts_loss_kwargs = True
        self.loss_is_scaled_for_ga = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        tasks = inputs.pop("tasks")
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

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if resume_from_checkpoint:
            saved = json.loads((Path(resume_from_checkpoint) / "exposure.json").read_text(encoding="utf-8"))
            state = json.loads((Path(resume_from_checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))
            if saved["global_step"] != state["global_step"]:
                raise ValueError("Checkpoint exposure/state steps differ")
            self.exposure = {k: saved[k] for k in self.exposure}
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)


def build_trainer(model, tokenizer, records, config: TrainingConfig, output: Path, *, cpu=False, callbacks=None):
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
    if trainer.args.world_size != 1 or trainer.args.n_gpu > 1:
        raise ValueError("This pilot trainer currently requires one process and one visible GPU")
    return trainer


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_contract(root: Path, config: TrainingConfig, summary: dict, architecture: dict) -> dict:
    files = ["pyproject.toml", "uv.lock", "scripts/run_finetune.py",
             "src/meta_data_extraction/finetune/training.py",
             "src/meta_data_extraction/finetune/training_data.py"]
    return {"config": config.model_dump(), "data": summary, "architecture": architecture,
            "sources": {p: file_hash(root / p) for p in files},
            "runtime": {p: version(p) for p in ("torch", "transformers", "tokenizers", "peft", "trl", "accelerate")}}


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
    for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                 "optimizer.pt", "scheduler.pt", "rng_state.pth", "exposure.json"):
        if not (resume / name).is_file():
            raise ValueError(f"Incomplete checkpoint: {name}")
    step = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))["global_step"]
    if step >= contract["config"]["max_steps"]:
        raise ValueError("Checkpoint already reached this run's step cap")
    latest = max(int(p.name.split("-")[-1]) for p in output.glob("checkpoint-*")
                 if p.is_dir() and p.name.split("-")[-1].isdigit())
    if step != latest:
        raise ValueError("Resume the latest checkpoint to avoid replacing later checkpoints")


def run_training(root: Path, config: TrainingConfig, tokenizer, records, summary, architecture,
                 preparation: dict, *, offline=False, resume: Path | None = None):
    if config.max_steps is None:
        raise ValueError("Set max_steps in both configs after the hardware check, or pass --max-steps for a smoke run")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("Real training requires a CUDA GPU with native BF16; use --dry-run for CPU checks")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES for this pilot")
    set_seed(config.seed)
    output = (root / config.output_dir).resolve()
    contract = run_contract(root, config, summary, architecture)
    prepare_output(output, contract, resume)
    model = attach_lora(load_base(preparation, config, offline=offline), config, preparation)
    trainer = build_trainer(model, tokenizer, records, config, output)
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    # The final step may not coincide with the regular save interval.
    trainer._save_checkpoint(trainer.model, trial=None)
    trainer.save_model(str(output / "adapter"))
    trainer.save_state()
    write_json(output / "training_summary.json", {
        "global_step": trainer.state.global_step, "metrics": result.metrics,
        "exposure": trainer.exposure, "selection": "not implemented; Step 4",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    })
    return {"output_dir": str(output), "global_step": trainer.state.global_step}
