#!/usr/bin/env python
"""Reproducible no-retraining audit of the completed fine-tuning comparison."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import re

import torch
import yaml

from meta_data_extraction.finetune.tokenization import render_prompt
from meta_data_extraction.finetune.training import build_trainer, load_base, reload_adapter
from meta_data_extraction.finetune.training_data import prepare_training, read_config
from meta_data_extraction.finetune.validation import validation_context, write_json


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("m0", "m1", "m2")
FIELDS = ("facilities", "samples", "datasets",
          "datasets.attrs.sample_refs", "datasets.attrs.facility_refs")
REFERENCE_PATTERN = re.compile(
    r'(?ms)^      "(?:sample_refs|facility_refs)": (?:\[\]|\[\n.*?^      \])[,]?'
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prf(counts: dict) -> dict:
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 0.0 if not precision or not recall else 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_bootstrap(base_reports: list[dict], model_reports: list[dict], *, samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    result = {}
    for field in FIELDS:
        deltas = []
        for _ in range(samples):
            indices = [rng.randrange(len(base_reports)) for _ in base_reports]
            pooled = []
            for reports in (base_reports, model_reports):
                counts = {key: sum(reports[index]["fields"][field].get(key, 0) for index in indices)
                          for key in ("tp", "fp", "fn")}
                pooled.append(prf(counts)["f1"])
            deltas.append(pooled[1] - pooled[0])
        base_counts = {key: sum(report["fields"][field].get(key, 0) for report in base_reports)
                       for key in ("tp", "fp", "fn")}
        model_counts = {key: sum(report["fields"][field].get(key, 0) for report in model_reports)
                        for key in ("tp", "fp", "fn")}
        result[field] = {
            "delta_f1": prf(model_counts)["f1"] - prf(base_counts)["f1"],
            "paired_bootstrap_95_ci": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
            "bootstrap_samples": samples,
        }
    return result


def generation_stats(rows: list[dict]) -> dict:
    tokens = sum(row["generated_tokens"] for row in rows)
    seconds = sum(row["latency_ms"] for row in rows) / 1000
    return {"papers": len(rows), "total_output_tokens": tokens,
            "mean_output_tokens": tokens / len(rows), "total_seconds": seconds,
            "mean_seconds": seconds / len(rows), "tokens_per_second": tokens / seconds}


def reference_mask(tokenizer, record: dict, expected_ids: list[int]) -> torch.Tensor:
    prefix = render_prompt(tokenizer, record["prompt"])
    content = record["completion"][0]["content"]
    rendered = tokenizer.apply_chat_template(
        record["prompt"] + record["completion"], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != expected_ids or not rendered.startswith(prefix):
        raise ValueError(f"Tokenizer offsets differ from prepared record: {record['example_id']}")
    spans = [(len(prefix) + match.start(), len(prefix) + match.end())
             for match in REFERENCE_PATTERN.finditer(content)]
    expected_fields = 2 * len(json.loads(content)["datasets"])
    if len(spans) != expected_fields:
        raise ValueError(f"Reference span coverage changed: {record['example_id']}")
    mask = []
    for start, end in encoded["offset_mapping"][1:]:
        mask.append(any(start < span_end and end > span_start for span_start, span_end in spans))
    return torch.tensor(mask, dtype=torch.bool)


def validation_reference_loss(root: Path, *, offline: bool) -> dict:
    config = read_config(root / "configs/finetune_json_only.yaml")
    tokenizer, training, validation, _ = prepare_training(root, config, offline=offline)
    prepared = yaml.safe_load((root / config.prepare_config).read_text(encoding="utf-8"))
    selection = read_json(root / config.output_dir / "selection.json")
    model = reload_adapter(
        load_base(prepared, config, offline=offline),
        root / config.output_dir / selection["checkpoint"], prepared, config=config,
    )
    trainer = build_trainer(model, tokenizer, training, config,
                            root / "outputs/finetune/evaluation/audit_trainer")
    raw_records = read_jsonl(root / config.prepared_dir / "validation.jsonl")
    papers, total_nll, reference_nll = [], 0.0, 0.0
    total_tokens = reference_tokens = 0
    with validation_context(model):
        for row, record in zip(validation, raw_records, strict=True):
            batch = trainer._prepare_inputs(trainer.data_collator([row]))
            batch.pop("tasks")
            outputs = trainer.model(**batch, use_cache=False, fused_lm_head=True)
            losses = -outputs.log_probs[0].float().cpu()
            label_mask = outputs.label_mask[0].cpu()
            refs = reference_mask(tokenizer, record, row["input_ids"])
            if losses.shape != refs.shape or not torch.equal(label_mask, torch.tensor(row["labels"][1:]) != -100):
                raise ValueError(f"Fused loss/token alignment changed: {row['paper_id']}")
            refs &= label_mask
            paper_total_nll = float(losses[label_mask].sum())
            paper_reference_nll = float(losses[refs].sum())
            paper_total_tokens = int(label_mask.sum())
            paper_reference_tokens = int(refs.sum())
            papers.append({"paper_id": row["paper_id"], "target_tokens": paper_total_tokens,
                           "reference_tokens": paper_reference_tokens,
                           "reference_token_fraction": paper_reference_tokens / paper_total_tokens,
                           "mean_target_nll": paper_total_nll / paper_total_tokens,
                           "mean_reference_nll": paper_reference_nll / paper_reference_tokens,
                           "reference_nll_fraction": paper_reference_nll / paper_total_nll})
            total_nll += paper_total_nll
            reference_nll += paper_reference_nll
            total_tokens += paper_total_tokens
            reference_tokens += paper_reference_tokens
            del batch, outputs
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return {"definition": "tokens overlapping sample_refs/facility_refs keys and list values",
            "papers": papers, "target_tokens": total_tokens, "reference_tokens": reference_tokens,
            "reference_token_fraction": reference_tokens / total_tokens,
            "mean_target_nll": total_nll / total_tokens,
            "mean_reference_nll": reference_nll / reference_tokens,
            "mean_nonreference_nll": (total_nll - reference_nll) / (total_tokens - reference_tokens),
            "reference_nll_fraction": reference_nll / total_nll}


def funding_failures(root: Path) -> list[dict]:
    result = []
    for row in read_jsonl(root / "outputs/finetune/evaluation/m1/generation.jsonl"):
        errors = [error for error in row["schema_errors"] if "grant_id: null" in error]
        if not errors:
            continue
        prediction = read_json(root / "outputs/finetune/evaluation/m1" / row["prediction_file"])["output"]
        null_entries = [entry for entry in prediction["bibliographic"]["funding"]
                        if entry.get("grant_id") is None]
        result.append({"paper_id": row["paper_id"], "schema_errors": errors,
                       "predicted_null_grant_entries": null_entries,
                       "gold_funding": read_json(root / "data/gold" / f"{row['paper_id']}.json")["bibliographic"]["funding"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--with-validation-loss", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/finetune/evaluation/m1_audit.json"))
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    reports = {condition: [read_json(path) for path in sorted(
        (ROOT / f"outputs/finetune/evaluation/{condition}/reports").glob("*.json"))]
        for condition in CONDITIONS}
    if any(len(value) != 30 for value in reports.values()):
        raise ValueError("The audit requires thirty reports for every condition")
    field_metrics = {}
    for condition in CONDITIONS:
        aggregate = read_json(ROOT / f"outputs/finetune/evaluation/{condition}/metrics.json")["all_papers"]["fields"]
        field_metrics[condition] = {field: prf(aggregate[field]) for field in FIELDS}
    audit = {
        "version": "1.0.0", "seed": args.seed,
        "field_metrics": field_metrics,
        "m1_minus_m0_bootstrap": paired_bootstrap(
            reports["m0"], reports["m1"], samples=args.bootstrap_samples, seed=args.seed),
        "generation": {condition: generation_stats(read_jsonl(
            ROOT / f"outputs/finetune/evaluation/{condition}/generation.jsonl"))
            for condition in CONDITIONS},
        "m1_schema_failures": funding_failures(ROOT),
        "loss_normalization": {
            "supervised": "target JSON tokens and EOS only",
            "masked": "paper/instruction prompt, empty-thinking prefix, template newline and padding",
            "reduction": "sum of shifted supervised-token NLL divided by supervised-token count",
            "paper_aggregation": "validation loss is the equal mean of two per-paper target-token means",
        },
    }
    if args.with_validation_loss:
        if not torch.cuda.is_available():
            raise ValueError("--with-validation-loss requires CUDA")
        audit["m1_validation_reference_loss"] = validation_reference_loss(ROOT, offline=args.offline)
    write_json(ROOT / args.output, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
