"""Audit complete pseudo-label SFT sequences with the pinned chat tokenizer."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from meta_data_extraction.utils import build_prompt


ROOT = Path(__file__).resolve().parents[1]
TRAINING_LIMIT = 65_536
COMPARISON_LIMIT = 32_768
SCHEMAS = ["generic.json", "photon_and_neutron.json"]
SUPERVISED_ROLES = ("pseudo_train", "pseudo_validation")
TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def token_ids(encoded) -> list[int]:
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        return encoded["input_ids"]
    return encoded


def quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)

    def value(percent: float) -> int:
        return ordered[round((len(ordered) - 1) * percent)]

    return {
        "minimum": ordered[0],
        "p50": value(0.50),
        "p90": value(0.90),
        "p95": value(0.95),
        "p99": value(0.99),
        "maximum": ordered[-1],
    }


def audit(root: Path, model: str, revision: str, output: Path) -> dict:
    split_path = root / "data/finetune/pseudo_label_supervision/splits.json"
    manifest_path = root / "data/share_600_papers/manifest.jsonl"
    text_dir = root / "data/finetune/pseudo_label_supervision/prepared/text"
    split = read_json(split_path)
    manifest = {row["paper_id"]: row for row in read_jsonl(manifest_path)}
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, local_files_only=True, trust_remote_code=True)
    model_limit = tokenizer.model_max_length

    roles = {
        paper_id: role
        for role in SUPERVISED_ROLES
        for paper_id in split[role]
    }
    if Counter(roles.values()) != Counter({"pseudo_train": 508,
                                           "pseudo_validation": 57}):
        raise ValueError("Expected the frozen 508/57 supervised split")

    records = []
    for index, paper_id in enumerate(sorted(roles), 1):
        row = manifest[paper_id]
        prediction_relative = row["global_seed_prediction"]["path"]
        if prediction_relative is None:
            raise ValueError(f"Supervised paper lacks a prediction: {paper_id}")
        document = (text_dir / f"{paper_id}.md").read_bytes().decode("utf-8")
        prediction_path = root / "data/share_600_papers" / prediction_relative
        target_object = read_json(prediction_path)
        target = json.dumps(
            target_object, ensure_ascii=False, indent=2, allow_nan=False)
        prompt = build_prompt(SCHEMAS, document)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
        sequence = token_ids(tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        ))
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        target_tokens = len(tokenizer.encode(target, add_special_tokens=False))
        total = len(sequence)
        records.append({
            "paper_id": paper_id,
            "split_role": roles[paper_id],
            "pages": row["page_count"],
            "document_chars": len(document),
            "target_chars": len(target),
            "prompt_tokens_without_chat_wrapper": prompt_tokens,
            "target_tokens_without_chat_wrapper": target_tokens,
            "sequence_tokens": total,
            "chat_wrapper_tokens": total - prompt_tokens - target_tokens,
            "margin_to_training_limit": TRAINING_LIMIT - total,
            "exceeds_32768": total > COMPARISON_LIMIT,
            "exceeds_65536": total > TRAINING_LIMIT,
            "exceeds_model_limit": total > model_limit,
            "prediction_sha256": file_hash(prediction_path),
        })
        if index % 25 == 0 or index == len(roles):
            print(f"Tokenized {index}/{len(roles)} examples", flush=True)

    records.sort(key=lambda item: (-item["sequence_tokens"], item["paper_id"]))
    totals = [row["sequence_tokens"] for row in records]
    report = {
        "version": "1.0.0",
        "status": "complete",
        "scope": {
            "included_roles": list(SUPERVISED_ROLES),
            "included_papers": len(records),
            "excluded_future_test_papers": len(split["future_test"]),
            "excluded_missing_prediction_papers": len(
                split["excluded_missing_prediction"]),
            "future_test_predictions_read": False,
        },
        "sequence_policy": {
            "schemas": SCHEMAS,
            "prompt_template": "schemas/prompt_template.md",
            "chat_roles": ["user", "assistant"],
            "thinking_enabled": False,
            "add_generation_prompt": False,
            "assistant_end_token_included": True,
            "target_json": "UTF-8, ensure_ascii=false, indent=2; values unchanged",
            "truncation": False,
        },
        "tokenizer": {
            "model": model,
            "revision": revision,
            "class": tokenizer.__class__.__name__,
            "vocabulary_size": len(tokenizer),
            "model_max_length": model_limit,
            "chat_template_sha256": text_hash(tokenizer.chat_template),
        },
        "inputs": {
            "split_manifest_sha256": file_hash(split_path),
            "source_manifest_sha256": file_hash(manifest_path),
            "prompt_template_sha256": file_hash(root / "schemas/prompt_template.md"),
            "schema_sha256": {
                name: file_hash(root / "schemas" / name) for name in SCHEMAS
            },
        },
        "limits": {
            "comparison": COMPARISON_LIMIT,
            "project_training": TRAINING_LIMIT,
            "tokenizer_model": model_limit,
        },
        "summary": {
            "sequence_tokens": quantiles(totals),
            "over_32768": sum(row["exceeds_32768"] for row in records),
            "over_65536": sum(row["exceeds_65536"] for row in records),
            "over_model_limit": sum(row["exceeds_model_limit"] for row in records),
            "by_role": {
                role: {
                    "papers": sum(row["split_role"] == role for row in records),
                    "over_65536": sum(
                        row["split_role"] == role and row["exceeds_65536"]
                        for row in records),
                }
                for role in SUPERVISED_ROLES
            },
        },
        "records_longest_first": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default=TOKENIZER_REVISION)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/finetune/pseudo_label_supervision/prepared/length_report.json")
    args = parser.parse_args(argv)
    report = audit(ROOT, args.model, args.revision, args.output.resolve())
    summary = report["summary"]
    print(
        f"Audited {report['scope']['included_papers']} examples: "
        f"{summary['over_65536']} exceed 65,536 tokens; "
        f"{summary['over_model_limit']} exceed the tokenizer model limit",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
