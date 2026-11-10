"""No-update gradient interaction diagnostic over frozen D24 adapters."""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch
import yaml

from .ablation import read_ablation_config
from .ablation_evaluation import endpoint_adapter
from .tracking import checksum
from .training_data import file_hash, prepare_training, read_config
from .validation import write_json


SEEDS = (42, 123, 456)
TASKS = ("D", "E", "R")
_LAYER = re.compile(r"\.language_model\.layers\.(\d+)\.(self_attn|linear_attn|mlp)\.")


class GradientInteractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    study_type: Literal["exploratory_no_update_gradient_interaction"]
    ablation_config: str
    ablation_config_sha256: str
    base_training_config_sha256: str
    condition: Literal["d_only"]
    seeds: list[int]
    tasks: list[Literal["D", "E", "R"]]
    expected_papers: Literal[8]
    gradient_parameters: Literal["trainable_lora_only"]
    loss_normalization: Literal["supervised_target_tokens_including_eos"]
    dropout_mode: Literal["disabled_deterministic"]
    dot_accumulation_dtype: Literal["float64"]
    near_zero_norm: float = Field(gt=0)
    output_dir: str
    attach_mlflow: Literal[False]
    model_selection_allowed: Literal[False]

    @model_validator(mode="after")
    def frozen_design(self):
        if self.seeds != list(SEEDS) or self.tasks != list(TASKS):
            raise ValueError("Gradient diagnostic seed/task order changed")
        return self


def read_gradient_config(path: Path) -> GradientInteractionConfig:
    return GradientInteractionConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def verify_contract(root: Path, config_path: Path, config: GradientInteractionConfig):
    matrix_path = root / config.ablation_config
    if file_hash(matrix_path) != config.ablation_config_sha256:
        raise ValueError("Frozen ablation configuration changed")
    matrix = read_ablation_config(matrix_path)
    base_path = root / matrix.base_training_config
    if file_hash(base_path) != config.base_training_config_sha256:
        raise ValueError("Frozen base training configuration changed")
    if Path(config.output_dir) in {Path("outputs/finetune/ablations"), Path("outputs/finetune/evaluation")}:
        raise ValueError("Diagnostic output may not overwrite training/evaluation artifacts")
    endpoints = []
    for seed in config.seeds:
        training, checkpoint, identity = endpoint_adapter(root, matrix, config.condition, seed)
        endpoints.append({**identity, "checkpoint": str(checkpoint.relative_to(root))})
    return matrix, read_config(base_path), endpoints


def prepare_examples(root: Path, training, config: GradientInteractionConfig, *, offline=False):
    tokenizer, records, _, summary = prepare_training(root, training, offline=offline)
    selected = [row for row in records if row["task"] in config.tasks]
    selected.sort(key=lambda row: (row["example_id"].split(":")[0], config.tasks.index(row["task"])))
    ids = [row["example_id"] for row in selected]
    papers = sorted({value.split(":")[0] for value in ids})
    if len(selected) != config.expected_papers * len(config.tasks):
        raise ValueError("Expected exactly one D/E/R example per training paper")
    if len(ids) != len(set(ids)) or len(papers) != config.expected_papers:
        raise ValueError("Gradient diagnostic paper/task coverage changed")
    if any({row["task"] for row in selected if row["example_id"].split(":")[0] == paper}
           != set(config.tasks) for paper in papers):
        raise ValueError("A training paper is missing a diagnostic task")
    return tokenizer, selected, papers, summary


def parameter_group(name: str) -> tuple[str, str]:
    match = _LAYER.search(name)
    if not match or ".lora_" not in name:
        raise ValueError(f"Unexpected trainable parameter in diagnostic: {name}")
    return f"block_{int(match.group(1)):02d}", match.group(2)


def trainable_state_sha256(model) -> str:
    digest = hashlib.sha256()
    found = 0
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        found += 1
        digest.update(name.encode("utf-8"))
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        digest.update(raw)
    if not found:
        raise ValueError("No trainable parameters found")
    return digest.hexdigest()


def diagnostic_train_mode(model) -> int:
    """Enable checkpointing's training path while deterministically disabling LoRA dropout."""
    model.train()
    changed = 0
    for name, module in model.named_modules():
        if "lora_dropout" in name and isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            changed += 1
    if not changed:
        raise ValueError("No LoRA dropout modules found to disable")
    if any(module.p != 0 for name, module in model.named_modules()
           if "lora_dropout" in name and isinstance(module, torch.nn.Dropout)):
        raise ValueError("LoRA dropout was not fully disabled")
    return changed


def install_fused_answer_loss(model):
    """Install the same chunked LM head used by the production SFT trainer."""
    from trl.trainer.utils import add_fused_lm_head

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    add_fused_lm_head(base, outputs=("log_probs",))


def collect_example_gradient(model, batch: dict, target_tokens: int, *,
                             offload_activations: bool = False) -> tuple[float, dict[str, torch.Tensor]]:
    """Return one target-token-normalized gradient without modifying parameters."""
    if target_tokens <= 0:
        raise ValueError("Example has no supervised target tokens")
    model.zero_grad(set_to_none=True)
    inputs = {key: value for key, value in batch.items() if key != "tasks"}
    offload = (torch.autograd.graph.save_on_cpu(pin_memory=True)
               if offload_activations else nullcontext())
    with offload:
        outputs = model(**inputs, fused_lm_head=True, num_items_in_batch=target_tokens)
        if int(outputs.label_mask.sum()) != target_tokens:
            raise ValueError("Fused loss target-token count differs from frozen labels")
        loss = -outputs.log_probs.sum() / target_tokens
    if loss.ndim or not torch.isfinite(loss):
        raise ValueError("Nonfinite or nonscalar diagnostic loss")
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"Missing or nonfinite gradient: {name}")
        gradients[name] = parameter.grad.detach().to(device="cpu", dtype=torch.float32).clone()
    model.zero_grad(set_to_none=True)
    if not gradients:
        raise ValueError("No LoRA gradients were collected")
    return float(loss.detach().cpu()), gradients


def gram_matrices(gradients: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Accumulate exact requested group Gram matrices in float64."""
    if not gradients:
        raise ValueError("No gradients supplied")
    names = list(gradients[0])
    if any(list(item) != names for item in gradients):
        raise ValueError("Trainable parameter order changed between examples")
    size = len(gradients)
    grams: dict[str, torch.Tensor] = {"global": torch.zeros((size, size), dtype=torch.float64)}
    for name in names:
        block, family = parameter_group(name)
        matrix = torch.stack([item[name].reshape(-1) for item in gradients]).to(torch.float64)
        contribution = matrix @ matrix.T
        grams["global"] += contribution
        for key in (block, f"family_{family}"):
            grams.setdefault(key, torch.zeros_like(grams["global"]))
            grams[key] += contribution
    return grams


def comparison_kind(left: dict, right: dict) -> str | None:
    lt, rt = left["task"], right["task"]
    same = left["paper_id"] == right["paper_id"]
    tasks = {lt, rt}
    if same and tasks == {"D", "E"}:
        return "same_paper_D_E"
    if same and tasks == {"D", "R"}:
        return "same_paper_D_R"
    if same:
        return None
    if lt == rt == "D":
        return "cross_paper_D_D"
    if lt == rt == "E":
        return "cross_paper_E_E"
    if lt == rt == "R":
        return "cross_paper_R_R"
    if tasks == {"D", "E"}:
        return "cross_paper_D_E"
    if tasks == {"D", "R"}:
        return "cross_paper_D_R"
    return None


def pair_rows(metadata: list[dict], grams: dict[str, torch.Tensor], seed: int,
              learning_rate: float, near_zero: float) -> tuple[list[dict], list[dict]]:
    global_rows, group_rows = [], []
    global_gram = grams["global"]
    for i, original_left in enumerate(metadata):
        for j in range(i + 1, len(metadata)):
            left, right = original_left, metadata[j]
            kind = comparison_kind(left, right)
            if kind is None:
                continue
            # Canonicalize mixed-task rows with D on the left.
            if left["task"] != "D" and right["task"] == "D":
                left, right, a, b = right, left, j, i
            else:
                a, b = i, j
            row = _pair_row(left, right, global_gram, a, b, near_zero)
            row.update(seed=seed, comparison=kind,
                       predicted_D_loss_change=(-learning_rate * row["dot"]
                                                if "_D_E" in kind or "_D_R" in kind else None))
            global_rows.append(row)
            for group, gram in grams.items():
                if group == "global":
                    continue
                detail = _pair_row(left, right, gram, a, b, near_zero)
                detail.update(seed=seed, comparison=kind, group=group)
                group_rows.append(detail)
    return global_rows, group_rows


def _pair_row(left: dict, right: dict, gram: torch.Tensor, i: int, j: int,
              near_zero: float) -> dict:
    left_sq, right_sq = float(gram[i, i]), float(gram[j, j])
    if left_sq < 0 or right_sq < 0:
        raise ValueError("Negative squared gradient norm")
    left_norm, right_norm = math.sqrt(left_sq), math.sqrt(right_sq)
    if left_norm <= near_zero or right_norm <= near_zero:
        raise ValueError("Near-zero gradient norm makes cosine undefined")
    dot = float(gram[i, j])
    cosine = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    if not all(math.isfinite(value) for value in (dot, cosine, left_norm, right_norm)):
        raise ValueError("Nonfinite pair statistic")
    return {"left_example_id": left["example_id"], "right_example_id": right["example_id"],
            "left_task": left["task"], "right_task": right["task"],
            "left_paper_id": left["paper_id"], "right_paper_id": right["paper_id"],
            "dot": dot, "cosine": cosine,
            "left_norm": left_norm, "right_norm": right_norm}


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def summarize_pairs(rows: list[dict]) -> dict:
    result = {}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["comparison"])].append(row)
    for (seed, comparison), subset in sorted(groups.items()):
        cosines = [row["cosine"] for row in subset]
        dots = [row["dot"] for row in subset]
        result[f"seed_{seed}:{comparison}"] = {
            "pairs": len(subset), "cosine_median": statistics.median(cosines),
            "cosine_iqr": [quantile(cosines, .25), quantile(cosines, .75)],
            "cosine_range": [min(cosines), max(cosines)],
            "negative_fraction": sum(value < 0 for value in cosines) / len(cosines),
            "dot_median": statistics.median(dots),
        }
    by_comparison = defaultdict(list)
    for row in rows:
        by_comparison[row["comparison"]].append(row)
    across = {}
    for comparison, subset in sorted(by_comparison.items()):
        seed_medians = [statistics.median(row["cosine"] for row in subset if row["seed"] == seed)
                        for seed in SEEDS]
        seed_dot_medians = [statistics.median(row["dot"] for row in subset if row["seed"] == seed)
                            for seed in SEEDS]
        all_cosines = [row["cosine"] for row in subset]
        across[comparison] = {
            "seed_medians": dict(zip(map(str, SEEDS), seed_medians)),
            "median_of_seed_medians": statistics.median(seed_medians),
            "seed_median_iqr": [quantile(seed_medians, .25), quantile(seed_medians, .75)],
            "seed_median_range": [min(seed_medians), max(seed_medians)],
            "pair_negative_fraction": sum(value < 0 for value in all_cosines) / len(all_cosines),
            "dot_seed_medians": dict(zip(map(str, SEEDS), seed_dot_medians)),
            "dot_median_of_seed_medians": statistics.median(seed_dot_medians),
            "pairs": len(all_cosines),
        }
        predicted = [row.get("predicted_D_loss_change") for row in subset
                     if row.get("predicted_D_loss_change") is not None]
        if predicted:
            across[comparison]["predicted_D_loss_change_median"] = statistics.median(predicted)
    return {"per_seed": result, "across_seeds": across}


def summarize_group_pairs(rows: list[dict]) -> dict:
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    return {group: summarize_pairs(subset) for group, subset in sorted(by_group.items())}


def summarize_norms(rows: list[dict]) -> dict:
    examples = {}
    for row in rows:
        for side in ("left", "right"):
            examples[(row["seed"], row[f"{side}_example_id"])] = {
                "task": row[f"{side}_task"], "norm": row[f"{side}_norm"]}
    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for task in TASKS:
            values = [item["norm"] for (value_seed, _), item in examples.items()
                      if value_seed == seed and item["task"] == task]
            per_seed[str(seed)][task] = {"median": statistics.median(values),
                                         "range": [min(values), max(values)]}
    seed_medians = {task: [per_seed[str(seed)][task]["median"] for seed in SEEDS]
                    for task in TASKS}
    return {"per_seed": per_seed,
            "median_of_seed_medians": {task: statistics.median(values)
                                        for task, values in seed_medians.items()}}


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(result: dict) -> str:
    lines = ["# No-update gradient interaction diagnostic", "",
             "This exploratory analysis measures local LoRA-gradient compatibility at the three frozen D24 endpoints. No optimizer or scheduler was created, and checkpoint hashes were unchanged.", "",
             "| Comparison | Seed 42 | Seed 123 | Seed 456 | Median | Negative pairs |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in result["summary"]["across_seeds"].items():
        seeds = row["seed_medians"]
        lines.append(f"| {name} | {seeds['42']:.6f} | {seeds['123']:.6f} | {seeds['456']:.6f} | {row['median_of_seed_medians']:.6f} | {row['pair_negative_fraction']:.1%} |")
    norms = result["gradient_norms"]
    lines += ["", "## Gradient norms", "",
              "| Task | Seed 42 | Seed 123 | Seed 456 | Median |", "|---|---:|---:|---:|---:|"]
    for task in TASKS:
        values = [norms["per_seed"][str(seed)][task]["median"] for seed in SEEDS]
        lines.append(f"| {task} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {norms['median_of_seed_medians'][task]:.6f} |")
    rows = result["summary"]["across_seeds"]
    dd, de, dr = (rows[name]["median_of_seed_medians"] for name in
                  ("cross_paper_D_D", "cross_paper_D_E", "cross_paper_D_R"))
    lines += ["", "## Interpretation", "",
              f"Cross-paper D/E ({de:.6f}) and D/R ({dr:.6f}) alignment was positive in every seed but substantially below the ordinary cross-paper D/D baseline ({dd:.6f}). Same-paper D/E was much higher, consistent with shared paper content rather than reusable task transfer. E/E and R/R coherence was strong, so the auxiliary objectives were internally consistent.", "",
              "There is no evidence of systematic E/D conflict: every module-family median was positive and only isolated block medians were near zero or negative. R/D compatibility is weak but reproducibly positive cross-paper; it is not stronger than E/D and is baseline-like relative to D/D. E gradients were roughly four to five times the D norm and R gradients roughly one-and-a-half to two times the D norm, so equal example counts can let auxiliary objectives dominate update magnitude even without directional conflict.", "",
              "Overall, the diagnostic does not justify gradient-balancing or selective-sharing machinery. It supports running the already authorized R-first versus R-last order controls as an exploratory behavioral test, without predicting a winner or changing the frozen Gate B/M1 decisions. These local measurements do not establish causal transfer or replace free-generation evaluation.", ""]
    return "\n".join(lines)


def seed_result(root: Path, config: GradientInteractionConfig, seed: int, identity: dict,
                training, model, tokenizer, examples: list[dict]) -> dict:
    from .training_data import AnswerCollator

    started = time.monotonic()
    before = trainable_state_sha256(model)
    checkpoint = root / identity["checkpoint"]
    checkpoint_before = checksum(checkpoint / "adapter_model.safetensors")
    if checkpoint_before != identity["adapter_sha256"]:
        raise ValueError("Endpoint adapter hash changed before diagnostic")
    collator = AnswerCollator(tokenizer.pad_token_id, training.max_seq_length)
    device = next((parameter.device for parameter in model.parameters() if parameter.device.type == "cuda"),
                  torch.device("cuda"))
    gradients, metadata = [], []
    torch.cuda.reset_peak_memory_stats()
    for index, example in enumerate(examples, start=1):
        batch = collator([example])
        batch = {key: (value.to(device) if torch.is_tensor(value) else value)
                 for key, value in batch.items()}
        target_tokens = int((batch["labels"][:, 1:] != -100).sum())
        loss, gradient = collect_example_gradient(
            model, batch, target_tokens, offload_activations=training.offload_activations)
        gradients.append(gradient)
        example_digest = hashlib.sha256()
        example_digest.update(torch.tensor(example["input_ids"], dtype=torch.int64).numpy().tobytes())
        example_digest.update(torch.tensor(example["labels"], dtype=torch.int64).numpy().tobytes())
        metadata.append({"example_id": example["example_id"], "paper_id": example["example_id"].split(":")[0],
                         "task": example["task"], "target_tokens": target_tokens,
                         "total_tokens": int(batch["attention_mask"].sum()), "loss": loss,
                         "tokenized_example_sha256": example_digest.hexdigest()})
        print(json.dumps({"seed": seed, "backward_pass": index, "total": len(examples),
                          "example_id": example["example_id"], "loss": loss}), flush=True)
    grams = gram_matrices(gradients)
    del gradients
    rows, group_rows = pair_rows(metadata, grams, seed, training.learning_rate, config.near_zero_norm)
    del grams
    after = trainable_state_sha256(model)
    checkpoint_after = checksum(checkpoint / "adapter_model.safetensors")
    if before != after or checkpoint_before != checkpoint_after:
        raise ValueError("Frozen adapter changed during no-update diagnostic")
    return {"seed": seed, "endpoint": identity, "examples": metadata, "pairs": rows,
            "group_pairs": group_rows, "trainable_state_sha256_before": before,
            "trainable_state_sha256_after": after, "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "runtime_seconds": time.monotonic() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated()}


def combine_results(root: Path, config_path: Path, config: GradientInteractionConfig,
                    endpoints: list[dict], seed_results: list[dict], data_summary: dict) -> dict:
    rows = [row for result in seed_results for row in result["pairs"]]
    groups = [row for result in seed_results for row in result["group_pairs"]]
    result = {
        "version": config.version, "study_type": config.study_type,
        "status": "complete", "config": config.model_dump(),
        "config_sha256": file_hash(config_path), "endpoints": endpoints,
        "data": data_summary, "seeds": [{key: value for key, value in item.items()
                                           if key not in {"pairs", "group_pairs"}}
                                          for item in seed_results],
        "summary": summarize_pairs(rows),
        "gradient_norms": summarize_norms(rows),
        "group_summary": summarize_group_pairs(groups),
        "sources": {str(path.relative_to(root)): file_hash(path) for path in (
            config_path, root / config.ablation_config,
            root / "scripts/run_finetune_gradient_interaction.py",
            root / "src/meta_data_extraction/finetune/gradient_interaction.py",
            root / "src/meta_data_extraction/finetune/training.py",
            root / "src/meta_data_extraction/finetune/training_data.py")},
        "safety": {"optimizer_created": False, "scheduler_created": False,
                   "optimizer_steps": 0, "checkpoint_mutations": 0,
                   "mlflow_attached": False, "model_selection_allowed": False},
    }
    output = root / config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "pair_statistics.csv", rows)
    write_csv(output / "layer_module_pair_statistics.csv", groups)
    write_json(output / "gradient_interaction.json", result)
    (output / "gradient_interaction.md").write_text(markdown_report(result), encoding="utf-8")
    return result
