"""Paper-balanced validation, diagnostic generation, and recoverable selection."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from transformers import GenerationConfig, Trainer, TrainerCallback

from meta_data_extraction.evaluation import evaluate
from .parsing import text_hash


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def paper_mean(papers: list[dict]) -> float:
    if len(papers) != 2 or len({p["paper_id"] for p in papers}) != 2:
        raise ValueError("Validation requires exactly two distinct papers")
    if any(not math.isfinite(p["loss"]) or p["loss"] < 0 or p["supervised_tokens"] <= 0 for p in papers):
        raise ValueError("Every paper needs a finite nonnegative loss and answer tokens")
    return math.fsum(p["loss"] for p in papers) / 2


def best_entry(history: list[dict]) -> dict:
    """Selection sees only loss and step, never the generated F1 diagnostics."""
    if not history or history[0]["step"] != 0:
        raise ValueError("Selection must include the unadapted step-zero baseline")
    steps = [r["step"] for r in history]
    if steps != sorted(set(steps)) or any(type(s) is not int or s < 0 for s in steps):
        raise ValueError("Validation steps must be unique and increasing")
    for row in history:
        if row["loss"] != paper_mean(row["papers"]):
            raise ValueError("Stored selection loss is not the equal-paper mean")
    return min(history, key=lambda row: (row["loss"], row["step"]))


@contextmanager
def validation_context(model, *, base=False):
    """Evaluation cannot consume training randomness or leave dropout disabled."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    training = model.training
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            with model.disable_adapter() if base else nullcontext():
                yield
    finally:
        model.train(training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def validate_rows(rows: list[dict], training_ids: set[str], cap: int, max_new_tokens: int):
    if len(rows) != 2 or len({r["paper_id"] for r in rows}) != 2:
        raise ValueError("Exactly two distinct validation papers are required")
    for row in rows:
        if row["split"] != "validation" or row["task"] != "D" or row["paper_id"] in training_ids:
            raise ValueError("Only held-out validation D examples may be used")
        if row["example_id"] != f"{row['paper_id']}:D":
            raise ValueError("Validation example/paper identity mismatch")
        labels, ids, prefix = row["labels"], row["input_ids"], row["prompt_ids"]
        positions = [i for i, token in enumerate(labels) if token != -100]
        if (len(labels) != len(ids) or not prefix or not positions or positions[0] != len(prefix)
                or ids[:len(prefix)] != prefix or any(labels[i] != ids[i] for i in positions)):
            raise ValueError("Generation prefix and supervised answer boundaries disagree")
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("Validation answer labels must be contiguous")
        if len(ids) > cap or len(prefix) + max_new_tokens > cap:
            raise ValueError("Validation sequence/output reservation exceeds the context cap")


def validation_losses(trainer, rows: list[dict]) -> list[dict]:
    """One forward-only paper at a time; no pooled-token or training-loss reuse."""
    result = []
    for row in rows:
        batch = trainer._prepare_inputs(trainer.data_collator([row]))
        batch.pop("tasks")
        count = int((batch["labels"][:, 1:] != -100).sum())
        with trainer.compute_loss_context_manager():
            outputs = trainer.model(**batch, use_cache=False, fused_lm_head=True)
        if int(outputs.label_mask.sum()) != count:
            raise ValueError("Fused validation loss used an unexpected label mask")
        loss = float(-outputs.log_probs.float().sum() / count)
        result.append({"paper_id": row["paper_id"], "loss": loss, "supervised_tokens": count})
        del batch, outputs
    paper_mean(result)  # Reject NaN/Inf before it can influence selection.
    return result


def generate_answer(model, tokenizer, row: dict, max_new_tokens: int) -> dict:
    """The model receives only the audited D prompt, never the gold completion."""
    inputs = torch.tensor([row["prompt_ids"]], device=next(model.parameters()).device)
    generation = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
        repetition_penalty=1.0, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=False,
    )
    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            generation_config=generation, logits_to_keep=1)
    new_ids = output[0, inputs.shape[1]:].tolist()
    ended = bool(new_ids and new_ids[-1] == tokenizer.eos_token_id)
    # Remove only the actual end marker, retaining other unexpected special tokens
    # so that malformed responses are not silently made valid by decoding.
    text = tokenizer.decode(new_ids[:-1] if ended else new_ids, skip_special_tokens=False)
    return {"text": text, "generated_tokens": len(new_ids), "ended_with_eos": ended,
            "hit_output_limit": len(new_ids) >= max_new_tokens and not ended}


def _reject_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def score_generation(row: dict, generated: dict, directory: Path) -> dict:
    """Reuse the existing evaluator; invalid outputs remain explicit failures."""
    pid = row["paper_id"]
    raw_path = directory / "raw" / f"{pid}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(generated["text"], encoding="utf-8")
    result = {"paper_id": pid, **{k: v for k, v in generated.items() if k != "text"},
              "raw_file": f"raw/{pid}.txt", "json_valid": False,
              "entity_f1": None, "conditional_reference_f1": None}
    try:
        prediction = json.loads(generated["text"], parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        if not isinstance(prediction, dict):
            raise ValueError("Expected a JSON object")
        result["json_valid"] = True
    except ValueError as exc:
        return {**result, "status": "invalid_json", "error": str(exc)}
    gold_path, pred_path = directory / "gold" / f"{pid}.json", directory / "predictions" / f"{pid}.json"
    write_json(gold_path, row["gold"])
    write_json(pred_path, prediction)
    try:
        report = evaluate(gold_path, pred_path)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return {**result, "status": "scoring_failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(directory / "reports" / f"{pid}.json", report)
    fields = report["fields"]
    return {**result, "status": "scored",
            "entity_f1": {k: fields[k]["f1"] for k in ("facilities", "samples", "datasets")},
            "conditional_reference_f1": {
                k: {"f1": fields[f"datasets.attrs.{k}"].get("f1"),
                    "n_pairs": fields[f"datasets.attrs.{k}"]["n_pairs"]}
                for k in ("sample_refs", "facility_refs")},
            "report_file": f"reports/{pid}.json"}


class ValidationCallback(TrainerCallback):
    def __init__(self, trainer, rows, config):
        self.trainer, self.rows, self.config = trainer, rows, config
        self.output = Path(trainer.args.output_dir)
        train_ids = {r["example_id"].split(":")[0] for r in trainer.train_dataset}
        validate_rows(rows, train_ids, config.max_seq_length, config.validation_max_new_tokens)
        self.data_hash = text_hash(json.dumps(rows, ensure_ascii=False, sort_keys=True))
        self.history = []

    def state_dict(self):
        return {"version": "1.0.0", "validation_sha256": self.data_hash,
                "validation_steps": self.config.validation_steps,
                "max_new_tokens": self.config.validation_max_new_tokens, "history": self.history}

    def restore(self, checkpoint: Path, step: int):
        saved = json.loads((checkpoint / "validation_state.json").read_text(encoding="utf-8"))
        if {k: v for k, v in saved.items() if k != "history"} != {k: v for k, v in self.state_dict().items() if k != "history"}:
            raise ValueError("Resumed validation data or settings changed")
        best_entry(saved["history"])
        if saved["history"][-1]["step"] != step:
            raise ValueError("Checkpoint validation history is incomplete")
        self.history = saved["history"]

    def measure(self, step: int):
        trainer = self.trainer
        timer = trainer.tracking.resources.phase("validation") if trainer.tracking else nullcontext()
        with timer, validation_context(trainer.model, base=step == 0):
            papers = validation_losses(trainer, self.rows)
            directory = self.output / "validation" / f"step-{step}"
            diagnostics = [score_generation(row, generate_answer(trainer.model, trainer.processing_class, row,
                           self.config.validation_max_new_tokens), directory) for row in self.rows]
        entry = {"step": step, "loss": paper_mean(papers), "papers": papers,
                 "exposure": deepcopy(trainer.exposure), "diagnostics": diagnostics,
                 "diagnostics_directory": f"validation/step-{step}"}
        self.history.append(entry)
        best_entry(self.history)
        # SFTTrainer.log drains pending training metrics, and on_log clears the
        # scheduled training-log flag. Keep validation separate from both.
        should_log = trainer.control.should_log
        Trainer.log(trainer, {"validation_loss": entry["loss"],
                              **{f"validation_loss/{p['paper_id']}": p["loss"] for p in papers}})
        trainer.control.should_log = should_log

    def publish(self, *, complete=False):
        winner = best_entry(self.history)
        cfg = self.trainer.model.peft_config["default"]
        selected = {"status": "complete" if complete else "in_progress",
                    "metric": "mean_of_per_paper_target_token_nll", "tie_break": "earliest_step",
                    "step": winner["step"], "loss": winner["loss"], "papers": winner["papers"],
                    "exposure": winner["exposure"], "base_model": cfg.base_model_name_or_path,
                    "revision": cfg.revision,
                    "kind": "base" if winner["step"] == 0 else "adapter",
                    "checkpoint": None, "adapter_sha256": None}
        if winner["step"]:
            relative = f"checkpoint-{winner['step']}"
            with (self.output / relative / "adapter_model.safetensors").open("rb") as stream:
                selected["adapter_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            selected["checkpoint"] = relative
        write_json(self.output / "validation_history.json", self.state_dict())
        write_json(self.output / "selection.json", selected)
        return selected

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.history:
            if state.global_step != 0:
                raise ValueError("Missing validation history for resumed training")
            self.measure(0)
        self.publish()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.config.validation_steps == 0 or state.global_step == state.max_steps:
            self.measure(state.global_step)
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self.publish()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.publish(complete=state.global_step == state.max_steps)
        return control
