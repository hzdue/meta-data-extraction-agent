"""Verify frozen pseudo-label records and adapt them to the shared trainer.

This module is deliberately read-only. It does not load model weights, select a
checkpoint, or train. Validation targets remain teacher pseudo-labels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from meta_data_extraction.utils import build_prompt, load_merged_schema

from .pseudo_label_records import (
    SCHEMAS,
    TOKENIZER_MODEL,
    TOKENIZER_REVISION,
    canonical_json,
    validate_prediction,
)
from .tokenization import encode_record, load_tokenizer


DEFAULT_PREPARED_DIR = "data/finetune/pseudo_label_supervision/prepared"
MAX_SEQUENCE_TOKENS = 65_536
TRAINING_SEQUENCE_CUTOFF = 32_660
LENGTH_FILTER_MANIFEST = (
    "data/finetune/pseudo_label_supervision/prepared/"
    "length_filtered_training_manifest.json"
)
DATASETS = {
    "pseudo_train": "train_pseudo_d.jsonl",
    "pseudo_validation": "validation_pseudo_d.jsonl",
}
SPLIT_COUNTS = {
    "pseudo_train": 508,
    "pseudo_validation": 57,
    "future_test": 30,
    "excluded_missing_prediction": 5,
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    """Read physical JSONL lines; str.splitlines() also splits embedded U+0085."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _complete_sequence_tokens(row: dict) -> int:
    counts = row["token_counts"]
    return counts.get("sequence_tokens", counts.get("complete_sequence"))


def build_length_filter_manifest(root: Path, records: list[dict], *,
                                 prepared_dir: str = DEFAULT_PREPARED_DIR) -> dict:
    """Derive the frozen resource filter from already audited training records."""
    source_relative = f"{prepared_dir}/{DATASETS['pseudo_train']}"
    rows = sorted(
        ({"paper_id": row["paper_id"],
          "sequence_tokens": _complete_sequence_tokens(row)} for row in records),
        key=lambda row: row["paper_id"],
    )
    if (len(rows) != len(records)
            or len({row["paper_id"] for row in rows}) != len(rows)
            or any(type(row["sequence_tokens"]) is not int
                   or row["sequence_tokens"] <= 0 for row in rows)):
        raise ValueError("Length-filter source records are invalid")
    included = [row for row in rows
                if row["sequence_tokens"] <= TRAINING_SEQUENCE_CUTOFF]
    excluded = [row for row in rows
                if row["sequence_tokens"] > TRAINING_SEQUENCE_CUTOFF]
    if (len(rows), len(included), len(excluded)) != (504, 442, 62):
        raise ValueError("Length-filter population differs from the reviewed 504/442/62 split")
    return {
        "version": "1.0.0",
        "status": "frozen",
        "source": {
            "path": source_relative,
            "sha256": file_hash(repository_path(root, source_relative)),
        },
        "policy": {
            "field": "complete_sequence_tokens",
            "operator": "<=",
            "maximum": TRAINING_SEQUENCE_CUTOFF,
            "truncation": False,
            "reason": "RTX 5070 Ti backward-pass memory envelope validated by the prior pilot",
        },
        "counts": {
            "source": len(rows),
            "included": len(included),
            "excluded_over_limit": len(excluded),
        },
        "summary": {
            "included_maximum": max(row["sequence_tokens"] for row in included),
            "excluded_minimum": min(row["sequence_tokens"] for row in excluded),
        },
        "included": included,
        "excluded": excluded,
    }


def apply_length_filter(root: Path, records: list[dict], summary: dict, *,
                        prepared_dir: str = DEFAULT_PREPARED_DIR):
    """Verify and apply the reviewed filter without truncating any record."""
    manifest_path = repository_path(root, LENGTH_FILTER_MANIFEST)
    manifest = read_json(manifest_path)
    expected = build_length_filter_manifest(root, records, prepared_dir=prepared_dir)
    if manifest != expected:
        raise ValueError("Frozen pseudo training length-filter manifest changed")
    included = {row["paper_id"] for row in manifest["included"]}
    filtered = [row for row in records if row["paper_id"] in included]
    if (len(filtered) != manifest["counts"]["included"]
            or max(len(row["input_ids"]) for row in filtered)
            != manifest["summary"]["included_maximum"]):
        raise ValueError("Applied pseudo training filter differs from its manifest")
    filter_relative = str(manifest_path.relative_to(root.resolve()))
    filtered_summary = {
        **summary,
        "training_examples": len(filtered),
        "training_papers": len(filtered),
        "task_counts": {"D": len(filtered)},
        "longest_training_sequence": max(len(row["input_ids"]) for row in filtered),
        "supervised_tokens_per_pass": sum(
            row["token_counts"]["supervised_tokens"] for row in filtered
        ),
        "total_tokens_per_pass": sum(len(row["input_ids"]) for row in filtered),
        "training_filter": {
            "manifest": filter_relative,
            "manifest_sha256": file_hash(manifest_path),
            **manifest["policy"],
            **manifest["counts"],
            **manifest["summary"],
        },
        "dataset_hashes": {
            **summary["dataset_hashes"],
            filter_relative: {
                "sha256": file_hash(manifest_path),
                "bytes": manifest_path.stat().st_size,
            },
        },
    }
    return filtered, filtered_summary, manifest


def verify_hash(path: Path, expected: str, *, allow_line_endings=False) -> dict | None:
    """Verify bytes, optionally accepting only an LF/CRLF representation change."""
    actual = file_hash(path)
    if actual == expected:
        return None
    if allow_line_endings:
        lf = path.read_bytes().replace(b"\r\n", b"\n")
        variants = {"LF": lf, "CRLF": lf.replace(b"\n", b"\r\n")}
        for line_endings, content in variants.items():
            if hashlib.sha256(content).hexdigest() == expected:
                return {
                    "prepared_sha256": expected,
                    "current_sha256": actual,
                    "prepared_line_endings": line_endings,
                    "policy": "LF_CRLF_only",
                }
    raise ValueError(f"Artifact hash changed: {path}")


def repository_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"Expected a repository-relative artifact path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact leaves the repository: {relative}")
    return path


def _verify_splits(split: dict, sources: list[dict], manifest: dict,
                   exclusions: dict, benchmark: dict) -> dict[str, set[str]]:
    groups = {name: set(split[name]) for name in SPLIT_COUNTS}
    if (split["status"] != "frozen"
            or any(len(split[name]) != count or len(groups[name]) != count
                   for name, count in SPLIT_COUNTS.items())
            or len(set.union(*groups.values())) != 600):
        raise ValueError("Expected frozen, unique, disjoint 508/57/30/5 paper sets")

    source_ids = [row["paper_id"] for row in sources]
    if len(source_ids) != 600 or len(set(source_ids)) != 600 or set(source_ids) != set.union(*groups.values()):
        raise ValueError("Source manifest and frozen split coverage differ")
    missing = {
        row["paper_id"] for row in sources
        if row["global_seed_prediction"]["path"] is None
    }
    if missing != groups["excluded_missing_prediction"]:
        raise ValueError("Missing-prediction exclusions differ from the source manifest")

    benchmark_ids = set(benchmark["train"] + benchmark["validation"] + benchmark["test"])
    if benchmark_ids & set(source_ids):
        raise ValueError("Pseudo corpus overlaps the existing benchmark")
    leakage = split["leakage_checks"]
    if (leakage["existing_benchmark_overlap"] != []
            or leakage["future_test_predictions_permitted_in_training"] is not False):
        raise ValueError("Frozen leakage checks do not preserve future-test isolation")

    formal = manifest["formal_validation"]
    invalid = set(formal["excluded_paper_ids"])
    supervised = groups["pseudo_train"] | groups["pseudo_validation"]
    if (len(invalid) != len(formal["excluded_paper_ids"])
            or invalid != set(exclusions)
            or any(not reasons for reasons in exclusions.values())
            or not invalid <= supervised
            or formal["failed_and_excluded"] != len(invalid)
            or formal["passed"] != len(supervised - invalid)):
        raise ValueError("Schema-invalid exclusions disagree with the frozen preparation")

    eligible = {role: groups[role] - invalid for role in DATASETS}
    expected_counts = {
        "pseudo_train": len(eligible["pseudo_train"]),
        "pseudo_validation": len(eligible["pseudo_validation"]),
        "future_test": len(groups["future_test"]),
        "excluded_missing_prediction": len(groups["excluded_missing_prediction"]),
        "excluded_schema_invalid_prediction": len(invalid),
    }
    if manifest["counts"] != expected_counts or expected_counts["pseudo_train"] % 4:
        raise ValueError("Prepared counts differ or training count is not divisible by four")
    return eligible


def _verify_coverage(rows: list[dict], role: str, expected: set[str]) -> None:
    if (Counter(row["paper_id"] for row in rows) != Counter(expected)
            or any(row["split_role"] != role or row["task"] != "D_complete_json"
                   for row in rows)):
        raise ValueError(f"Incorrect paper/task/split coverage: {role}")
    for row in rows:
        messages = row["messages"]
        if (len(messages) != 2
                or [message["role"] for message in messages] != ["user", "assistant"]
                or any(set(message) != {"role", "content"}
                       or not isinstance(message["content"], str) for message in messages)):
            raise ValueError(f"Expected unmodified user/assistant messages: {row['paper_id']}")


def verify_pseudo_prepared(root: Path, prepared_dir: str = DEFAULT_PREPARED_DIR):
    """Verify frozen records without opening future-test predictions or source PDFs."""
    directory = repository_path(root, prepared_dir)
    manifest = read_json(directory / "preparation_manifest.json")
    if manifest["version"] != "1.0.0" or manifest["status"] != "prepared_pending_gpu_probe":
        raise ValueError("Unsupported pseudo preparation manifest")

    expected_policy = {
        "task": "D_complete_json",
        "target_values_modified": False,
        "target_whitespace_canonicalized": True,
        "future_test_predictions_read": False,
        "truncation": False,
    }
    if any(manifest["policy"].get(key) != value for key, value in expected_policy.items()):
        raise ValueError("Pseudo preparation policy changed")

    inputs = manifest["inputs"]
    split_path = repository_path(root, inputs["split_manifest"])
    source_path = repository_path(root, inputs["source_manifest"])
    length_path = directory / "length_report.json"
    verify_hash(split_path, inputs["split_manifest_sha256"])
    verify_hash(source_path, inputs["source_manifest_sha256"])
    verify_hash(length_path, inputs["length_report_sha256"])

    expected_names = {
        *DATASETS.values(),
        "future_test_manifest.jsonl",
        "review_sample.jsonl",
        "quality_report.md",
        "semantic_review.md",
        "label_validation_errors.json",
    }
    expected_outputs = {f"{prepared_dir}/{name}" for name in expected_names}
    if set(manifest["outputs"]) != expected_outputs:
        raise ValueError("Prepared output inventory changed")
    for relative, expected in manifest["outputs"].items():
        path = repository_path(root, relative)
        verify_hash(path, expected["sha256"])
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Artifact byte count changed: {relative}")

    preparation_source = root / "src/meta_data_extraction/finetune/pseudo_label_records.py"
    source_hash = hashlib.sha256(
        preparation_source.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    if source_hash != manifest["preparation_source_sha256"]:
        raise ValueError("Pseudo record preparation code changed")

    line_endings = {}
    source_files = [(inputs["prompt_template"], inputs["prompt_template_sha256"])]
    source_files.extend((f"schemas/{name}", digest) for name, digest in inputs["schemas"].items())
    for relative, expected in source_files:
        difference = verify_hash(
            repository_path(root, relative), expected, allow_line_endings=True
        )
        if difference:
            line_endings[relative] = difference
    if set(inputs["schemas"]) != set(SCHEMAS):
        raise ValueError("Pseudo schema inventory changed")
    if ((inputs["tokenizer_model"], inputs["tokenizer_revision"])
            != (TOKENIZER_MODEL, TOKENIZER_REVISION)):
        raise ValueError("Pseudo tokenizer must use the pinned Qwen revision")

    split = read_json(split_path)
    sources = read_jsonl(source_path)
    if split["source_manifest_sha256"] != inputs["source_manifest_sha256"]:
        raise ValueError("Split and preparation source manifests disagree")
    benchmark_path = repository_path(
        root, split["leakage_checks"]["existing_benchmark_manifest"]
    )
    eligible = _verify_splits(
        split,
        sources,
        manifest,
        read_json(directory / "label_validation_errors.json"),
        read_json(benchmark_path),
    )

    future = read_jsonl(directory / "future_test_manifest.jsonl")
    if (Counter(row["paper_id"] for row in future) != Counter(split["future_test"])
            or any(row["teacher_prediction_read"] is not False
                   or row["teacher_prediction_permitted_in_training"] is not False
                   or "messages" in row for row in future)):
        raise ValueError("Future-test manifest violates isolation")

    lengths = read_json(length_path)
    if (lengths["status"] != "complete"
            or lengths["limits"]["project_training"] != MAX_SEQUENCE_TOKENS
            or lengths["scope"]["future_test_predictions_read"] is not False
            or lengths["sequence_policy"]["thinking_enabled"] is not False
            or lengths["sequence_policy"]["truncation"] is not False):
        raise ValueError("Length-audit policy changed")
    for key in ("split_manifest_sha256", "source_manifest_sha256", "prompt_template_sha256"):
        if lengths["inputs"][key] != inputs[key]:
            raise ValueError(f"Length-audit provenance differs: {key}")
    if lengths["inputs"]["schema_sha256"] != inputs["schemas"]:
        raise ValueError("Length-audit schema provenance differs")

    audited = lengths["records_longest_first"]
    supervised_ids = split["pseudo_train"] + split["pseudo_validation"]
    if Counter(row["paper_id"] for row in audited) != Counter(supervised_ids):
        raise ValueError("Length-audit paper coverage differs")
    token_by_id = {row["paper_id"]: row for row in audited}
    source_by_id = {row["paper_id"]: row for row in sources}
    datasets = {
        role: read_jsonl(directory / filename)
        for role, filename in DATASETS.items()
    }

    # Complete all split checks before following any record-specific path.
    for role, rows in datasets.items():
        _verify_coverage(rows, role, eligible[role])

    schema = load_merged_schema(SCHEMAS)
    for role, rows in datasets.items():
        for row in rows:
            paper_id = row["paper_id"]
            provenance = row["provenance"]
            source = source_by_id[paper_id]
            length = token_by_id[paper_id]
            prediction_relative = source["global_seed_prediction"]["path"]
            expected_provenance = {
                **inputs,
                "source_text": f"{prepared_dir}/text/{paper_id}.md",
                "source_pdf": (
                    source_path.parent / source["pdf"]["path"]
                ).relative_to(root).as_posix(),
                "source_pdf_sha256": source["pdf"]["sha256"],
                "prediction": (
                    source_path.parent / prediction_relative
                ).relative_to(root).as_posix(),
                "prediction_sha256": source["global_seed_prediction"]["sha256"],
                "prediction_status": source["global_seed_prediction"]["status"],
            }
            if any(provenance.get(key) != value for key, value in expected_provenance.items()):
                raise ValueError(f"Record provenance differs: {paper_id}")

            expected_counts = {
                "prompt_without_chat_wrapper": length["prompt_tokens_without_chat_wrapper"],
                "target_without_chat_wrapper": length["target_tokens_without_chat_wrapper"],
                "complete_sequence": length["sequence_tokens"],
            }
            if (length["split_role"] != role
                    or length["prediction_sha256"] != provenance["prediction_sha256"]
                    or row["token_counts"] != expected_counts
                    or length["sequence_tokens"] > MAX_SEQUENCE_TOKENS):
                raise ValueError(f"Record length audit differs: {paper_id}")

            text_path = repository_path(root, provenance["source_text"])
            verify_hash(text_path, provenance["source_text_sha256"])
            metadata = read_json(text_path.with_suffix(".json"))
            if (metadata["paper_id"] != paper_id
                    or metadata["split_role"] != role
                    or metadata["text_sha256"] != provenance["source_text_sha256"]
                    or metadata["pdf_sha256"] != provenance["source_pdf_sha256"]):
                raise ValueError(f"Cached parser provenance differs: {paper_id}")

            prediction_path = repository_path(root, provenance["prediction"])
            verify_hash(prediction_path, provenance["prediction_sha256"])
            prediction = read_json(prediction_path)
            if (row["messages"][1]["content"] != canonical_json(prediction)
                    or validate_prediction(prediction, schema)):
                raise ValueError(f"Teacher target changed or fails schema validation: {paper_id}")
            document = text_path.read_bytes().decode("utf-8")
            if row["messages"][0]["content"] != build_prompt(SCHEMAS, document):
                raise ValueError(f"Prepared prompt differs from current renderer: {paper_id}")

    maximum = max(
        row["token_counts"]["complete_sequence"]
        for rows in datasets.values() for row in rows
    )
    if manifest["policy"]["maximum_sequence_tokens"] != maximum:
        raise ValueError("Prepared maximum sequence length differs")
    return datasets, manifest, {
        "line_ending_differences": line_endings,
        "benchmark_split_sha256": file_hash(benchmark_path),
        "length_audit_tokenizer": lengths["tokenizer"],
    }


def tokenize_pseudo_records(tokenizer, records: list[dict]) -> list[dict]:
    """Create explicit answer-only labels, including assistant EOS."""
    result = []
    for row in records:
        messages = row["messages"]
        chat_record = {"prompt": messages[:1], "completion": messages[1:]}
        ids, start, end = encode_record(tokenizer, chat_record)
        actual_counts = {
            "prompt_without_chat_wrapper": len(tokenizer.encode(
                messages[0]["content"], add_special_tokens=False
            )),
            "target_without_chat_wrapper": len(tokenizer.encode(
                messages[1]["content"], add_special_tokens=False
            )),
            "complete_sequence": len(ids),
        }
        if actual_counts != row["token_counts"] or len(ids) > MAX_SEQUENCE_TOKENS:
            raise ValueError(
                f"Token counts changed or sequence exceeds context cap: {row['paper_id']}"
            )
        if not 0 < start < end <= len(ids) or ids[end - 1] != tokenizer.eos_token_id:
            raise ValueError(f"Invalid answer/EOS boundaries: {row['paper_id']}")

        labels = [-100] * len(ids)
        labels[start:end] = ids[start:end]
        encoded = {
            "input_ids": ids,
            "labels": labels,
            "task": "D",
            "example_id": f"{row['paper_id']}:D",
            "paper_id": row["paper_id"],
            "split_role": row["split_role"],
            "target_source": "pseudo_label",
            "quality_flags": row["quality_flags"],
            "provenance": row["provenance"],
            "token_counts": {
                "sequence_tokens": len(ids),
                "prompt_tokens": start,
                "supervised_tokens": end - start,
                "post_eos_tokens": len(ids) - end,
            },
        }
        if row["split_role"] == "pseudo_validation":
            # `gold` is the existing scorer's API key, not a human-label claim.
            encoded.update(
                split="validation",
                prompt_ids=ids[:start],
                gold=json.loads(messages[1]["content"]),
            )
        result.append(encoded)
    return result


def prepare_pseudo_training(root: Path, *, prepared_dir: str = DEFAULT_PREPARED_DIR,
                            offline=False):
    """Return tokenizer, train rows, validation rows and a CPU preflight summary."""
    datasets, manifest, verification = verify_pseudo_prepared(root, prepared_dir)
    tokenizer, identity = load_tokenizer(
        {
            "model_id": TOKENIZER_MODEL,
            "revision": TOKENIZER_REVISION,
            "enable_thinking": False,
        },
        local_files_only=offline,
    )
    audited = verification["length_audit_tokenizer"]
    observed = {
        "model": identity["model_id"],
        "revision": identity["revision"],
        "class": identity["class"],
        "vocabulary_size": len(tokenizer),
        "model_max_length": identity["model_max_length"],
        "chat_template_sha256": identity["chat_template_sha256"],
    }
    if observed != audited:
        raise ValueError("Tokenizer identity differs from the pseudo length audit")

    train = tokenize_pseudo_records(tokenizer, datasets["pseudo_train"])
    validation = tokenize_pseudo_records(tokenizer, datasets["pseudo_validation"])
    from .validation import validate_rows
    validate_rows(
        validation,
        {row["paper_id"] for row in train},
        MAX_SEQUENCE_TOKENS,
        16_384,
    )
    summary = {
        "method": "json_only",
        "corpus": "pseudo_label_supervision",
        "training_examples": len(train),
        "training_papers": len(train),
        "task_counts": {"D": len(train)},
        "validation_examples": len(validation),
        "validation_papers": [row["paper_id"] for row in validation],
        "validation_target_source": "pseudo_label",
        "max_seq_length": MAX_SEQUENCE_TOKENS,
        "longest_training_sequence": max(len(row["input_ids"]) for row in train),
        "longest_validation_sequence": max(len(row["input_ids"]) for row in validation),
        "supervised_tokens_per_pass": sum(
            row["token_counts"]["supervised_tokens"] for row in train
        ),
        "total_tokens_per_pass": sum(len(row["input_ids"]) for row in train),
        "counts": manifest["counts"],
        "excluded_schema_invalid_papers": manifest["formal_validation"]["excluded_paper_ids"],
        "future_test_predictions_read": False,
        "source_pdfs_read": False,
        "preparation_manifest_sha256": file_hash(
            root / prepared_dir / "preparation_manifest.json"
        ),
        "inputs": manifest["inputs"],
        "dataset_hashes": manifest["outputs"],
        "tokenizer": identity,
        "verification": verification,
        "sources": {
            name: file_hash(root / name)
            for name in (
                "src/meta_data_extraction/finetune/pseudo_training_data.py",
                "src/meta_data_extraction/finetune/tokenization.py",
                "src/meta_data_extraction/utils.py",
            )
        },
    }
    return tokenizer, train, validation, summary
