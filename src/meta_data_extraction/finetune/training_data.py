"""Read-only verification and explicit answer labels for the frozen pilot data."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
import yaml

from .parsing import text_hash
from .tokenization import encode_record, load_tokenizer


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    method: Literal["json_only", "structure_aware"]
    prepare_config: str
    prepared_dir: str
    output_dir: str
    seed: int = 42
    max_steps: PositiveInt | None = None
    save_steps: PositiveInt = 10
    validation_steps: PositiveInt = 10
    validation_max_new_tokens: PositiveInt = 16384
    learning_rate: float = Field(default=1e-4, gt=0)
    lr_scheduler_type: Literal["constant"] = "constant"
    weight_decay: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    microbatch_size: Literal[1] = 1
    gradient_accumulation_steps: PositiveInt = 4
    gradient_checkpointing: Literal[True] = True
    packing: Literal[False] = False
    max_seq_length: PositiveInt = 65536
    dtype: Literal["bfloat16"] = "bfloat16"
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"
    lora_rank: PositiveInt = 16
    lora_alpha: PositiveInt = 32
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)

    @model_validator(mode="after")
    def matching_checkpoint_interval(self):
        if self.save_steps != self.validation_steps:
            raise ValueError("save_steps and validation_steps must match so every candidate is recoverable")
        return self


def read_config(path: Path) -> TrainingConfig:
    return TrainingConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def file_hash(path: Path) -> str:
    """Match preparation's UTF-8/LF hashes across Windows and Linux checkouts."""
    return text_hash(path.read_text(encoding="utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_prepared(root: Path, config: TrainingConfig) -> tuple[dict, dict, dict]:
    """Verify transferred artifacts without PDFs or the old preparation environment.

    The preparation manifest is historical. Only pyproject/lock drift is allowed;
    preparation code, schemas, cached text, split and payloads must still match.
    """
    directory = root / config.prepared_dir
    manifest = read_json(directory / "preparation_manifest.json")
    preparation = yaml.safe_load((root / config.prepare_config).read_text(encoding="utf-8"))
    if preparation != manifest["config"]:
        raise ValueError("Preparation config differs from the frozen artifact manifest")
    if config.max_seq_length != preparation["max_seq_length"]:
        raise ValueError("Training context cap must match the prepared length audit")
    if config.validation_max_new_tokens != preparation["max_new_tokens"]:
        raise ValueError("Validation generation budget must match the prepared output reservation")
    split_path = root / preparation["split_manifest"]
    if file_hash(split_path) != manifest["inputs"]["split_manifest_sha256"]:
        raise ValueError("Frozen split manifest changed")
    splits = read_json(split_path)
    groups = [set(splits[key]) for key in ("train", "validation", "test")]
    if [len(s) for s in groups] != [8, 2, 30] or len(set.union(*groups)) != 40:
        raise ValueError("Expected disjoint 8/2/30 paper sets")
    drift = {}
    for name, original in manifest["sources"].items():
        current = file_hash(root / name)
        if current != original["sha256"]:
            if name not in {"pyproject.toml", "uv.lock"}:
                raise ValueError(f"Preparation source changed: {name}")
            drift[name] = {"preparation_sha256": original["sha256"], "training_sha256": current}
    for paper in manifest["papers"]:
        pid = paper["paper_id"]
        if file_hash(directory / "text" / f"{pid}.md") != paper["text_sha256"]:
            raise ValueError(f"Cached paper text changed: {pid}")
        metadata = read_json(directory / "text" / f"{pid}.json")
        if any(metadata.get(k) != v for k, v in paper.items() if k not in {"paper_id", "split"}):
            raise ValueError(f"Cached parser metadata changed: {pid}")
    datasets = {}
    specifications = {
        "train_json_only.jsonl": ("train", ("D",)),
        "train_structure_aware.jsonl": ("train", ("D", "E", "R")),
        "validation.jsonl": ("validation", ("D",)),
    }
    for name, (split, tasks) in specifications.items():
        path = directory / name
        expected = manifest["datasets"][name]
        if file_hash(path) != expected["sha256"]:
            raise ValueError(f"Prepared dataset hash changed: {name}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        pairs = Counter((r["paper_id"], r["task"]) for r in rows)
        if len(rows) != expected["records"] or pairs != Counter((p, t) for p in splits[split] for t in tasks):
            raise ValueError(f"Incorrect paper/task coverage: {name}")
        if any(r["split"] != split or r["example_id"] != f"{r['paper_id']}:{r['task']}" for r in rows):
            raise ValueError(f"Incorrect split/example labels: {name}")
        datasets[name] = rows
    if datasets["train_json_only.jsonl"] != [r for r in datasets["train_structure_aware.jsonl"] if r["task"] == "D"]:
        raise ValueError("Full-JSON examples differ between methods")
    return datasets, manifest, drift


def tokenize_records(tokenizer, records: list[dict], cap: int) -> list[dict]:
    result = []
    for record in records:
        ids, start, end = encode_record(tokenizer, record)
        if not 0 < start < end <= len(ids) or len(ids) > cap:
            raise ValueError(f"Invalid/oversized sequence: {record['example_id']} ({len(ids)} > {cap})")
        actual = (len(ids), start, end - start)
        saved = record["token_counts"]
        if actual != (saved["sequence_tokens"], saved["prompt_tokens"], saved["supervised_tokens"]):
            raise ValueError(f"Token counts changed: {record['example_id']}")
        labels = [-100] * len(ids)
        labels[start:end] = ids[start:end]
        result.append({"input_ids": ids, "labels": labels, "task": record["task"],
                       "example_id": record["example_id"]})
    return result


def prepare_training(root: Path, config: TrainingConfig, *, offline: bool = False):
    datasets, manifest, drift = verify_prepared(root, config)
    tokenizer, identity = load_tokenizer(manifest["config"], local_files_only=offline)
    if identity != manifest["tokenizer"]:
        raise ValueError("Tokenizer/runtime identity differs from preparation")
    train = tokenize_records(tokenizer, datasets[f"train_{config.method}.jsonl"], config.max_seq_length)
    validation = tokenize_records(tokenizer, datasets["validation.jsonl"], config.max_seq_length)
    for encoded, original in zip(validation, datasets["validation.jsonl"]):
        start = original["token_counts"]["prompt_tokens"]
        encoded.update(paper_id=original["paper_id"], split="validation",
                       prompt_ids=encoded["input_ids"][:start],
                       gold=json.loads(original["completion"][0]["content"]))
    summary = {
        "method": config.method, "training_examples": len(train),
        "training_papers": len({r["example_id"].split(":")[0] for r in train}),
        "task_counts": dict(Counter(r["task"] for r in train)),
        "validation_examples": len(validation), "max_seq_length": config.max_seq_length,
        "validation_papers": [r["paper_id"] for r in validation],
        "longest_training_sequence": max(len(r["input_ids"]) for r in train),
        "longest_validation_sequence": max(len(r["input_ids"]) for r in validation),
        "supervised_tokens_per_pass": sum(sum(t != -100 for t in r["labels"]) for r in train),
        "total_tokens_per_pass": sum(len(r["input_ids"]) for r in train),
        "preparation_environment_changes": drift,
        "preparation_manifest_sha256": file_hash(root / config.prepared_dir / "preparation_manifest.json"),
        "tokenizer": identity, "dataset_hashes": manifest["datasets"],
    }
    return tokenizer, train, validation, summary


class AnswerCollator:
    """Right-pad without changing existing labels or applying another template."""
    def __init__(self, pad_token_id: int, cap: int):
        self.pad_token_id, self.cap = pad_token_id, cap

    def __call__(self, examples):
        import torch

        length = max(len(e["input_ids"]) for e in examples)
        if length > self.cap:
            raise ValueError("Batch exceeds context cap; truncation is forbidden")
        ids, labels, masks = [], [], []
        for example in examples:
            n = len(example["input_ids"])
            if n != len(example["labels"]) or not any(x != -100 for x in example["labels"][1:]):
                raise ValueError("Missing/misaligned answer labels")
            ids.append(example["input_ids"] + [self.pad_token_id] * (length - n))
            labels.append(example["labels"] + [-100] * (length - n))
            masks.append([1] * n + [0] * (length - n))
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(masks), "tasks": [e["task"] for e in examples]}
