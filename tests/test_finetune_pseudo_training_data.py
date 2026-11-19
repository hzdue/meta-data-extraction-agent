"""Pseudo-label loader invariants before any GPU or model-weight work."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from meta_data_extraction.finetune.pseudo_training_data import (
    LENGTH_FILTER_MANIFEST,
    TRAINING_SEQUENCE_CUTOFF,
    apply_length_filter,
    build_length_filter_manifest,
    prepare_pseudo_training,
    read_jsonl,
    verify_hash,
    verify_pseudo_prepared,
)
from meta_data_extraction.finetune.training_data import AnswerCollator


ROOT = Path(__file__).resolve().parents[1]


def test_jsonl_reader_does_not_split_embedded_unicode_next_line(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [{"paper_id": "one", "text": "left\u0085right"},
               {"paper_id": "two", "text": "ordinary"}]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    assert read_jsonl(path) == records
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_hash_policy_accepts_only_line_ending_representation_changes(tmp_path):
    path = tmp_path / "source.txt"
    path.write_bytes(b"first\nsecond\n")
    expected = hashlib.sha256(b"first\r\nsecond\r\n").hexdigest()

    difference = verify_hash(path, expected, allow_line_endings=True)
    assert difference["prepared_line_endings"] == "CRLF"
    assert difference["policy"] == "LF_CRLF_only"
    with pytest.raises(ValueError, match="Artifact hash changed"):
        verify_hash(path, hashlib.sha256(b"changed\n").hexdigest(), allow_line_endings=True)


def test_real_frozen_package_verifies_without_future_test_labels():
    prepared = ROOT / "data/finetune/pseudo_label_supervision/prepared"
    if not (prepared / "preparation_manifest.json").exists():
        pytest.skip("Run pseudo-label preparation first")

    datasets, manifest, verification = verify_pseudo_prepared(ROOT)

    assert {role: len(rows) for role, rows in datasets.items()} == {
        "pseudo_train": 504,
        "pseudo_validation": 57,
    }
    assert manifest["policy"]["future_test_predictions_read"] is False
    assert manifest["formal_validation"]["failed_and_excluded"] == 4
    assert set(verification["line_ending_differences"]) == {
        "schemas/prompt_template.md",
        "schemas/generic.json",
        "schemas/photon_and_neutron.json",
    }


def test_real_records_tokenize_to_answer_only_trainer_rows_if_cached():
    prepared = ROOT / "data/finetune/pseudo_label_supervision/prepared"
    if not (prepared / "preparation_manifest.json").exists():
        pytest.skip("Run pseudo-label preparation first")
    try:
        tokenizer, train, validation, summary = prepare_pseudo_training(
            ROOT, offline=True
        )
    except OSError:
        pytest.skip("Pinned tokenizer is not cached")

    assert (len(train), len(validation)) == (504, 57)
    assert summary["longest_training_sequence"] == 63_524
    assert summary["longest_validation_sequence"] == 45_958
    assert summary["future_test_predictions_read"] is False
    assert summary["source_pdfs_read"] is False
    assert summary["validation_target_source"] == "pseudo_label"
    assert all(row["target_source"] == "pseudo_label" for row in train + validation)
    for row in train + validation:
        start = row["token_counts"]["prompt_tokens"]
        end = start + row["token_counts"]["supervised_tokens"]
        assert row["labels"][:start] == [-100] * start
        assert row["labels"][start:end] == row["input_ids"][start:end]
        assert row["labels"][end:] == [-100] * (len(row["labels"]) - end)
        assert row["labels"][end - 1] == tokenizer.eos_token_id
    assert all(row["split"] == "validation" and row["prompt_ids"] for row in validation)
    assert "2101.12483" in {row["paper_id"] for row in validation}
    batch = AnswerCollator(tokenizer.pad_token_id, 65_536)(train[:2])
    assert batch["input_ids"].shape[0] == 2
    assert batch["tasks"] == ["D", "D"]


def test_frozen_length_filter_manifest_matches_verified_source():
    prepared = ROOT / "data/finetune/pseudo_label_supervision/prepared"
    manifest_path = ROOT / LENGTH_FILTER_MANIFEST
    if not (prepared / "preparation_manifest.json").exists() or not manifest_path.exists():
        pytest.skip("Run pseudo-label and length-filter preparation first")
    datasets, _, _ = verify_pseudo_prepared(ROOT)
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = build_length_filter_manifest(ROOT, datasets["pseudo_train"])

    assert actual == expected
    assert actual["policy"]["maximum"] == TRAINING_SEQUENCE_CUTOFF
    assert actual["counts"] == {
        "source": 504, "included": 442, "excluded_over_limit": 62,
    }
    assert actual["summary"] == {
        "included_maximum": 32_530, "excluded_minimum": 32_694,
    }


def test_length_filter_rejects_manifest_or_population_drift_if_cached(tmp_path):
    try:
        _, train, _, summary = prepare_pseudo_training(ROOT, offline=True)
    except OSError:
        pytest.skip("Pinned tokenizer is not cached")
    filtered, filtered_summary, manifest = apply_length_filter(ROOT, train, summary)
    assert len(filtered) == 442
    assert filtered_summary["training_examples"] == 442
    assert max(len(row["input_ids"]) for row in filtered) == 32_530
    changed = list(train)
    changed[0] = {**changed[0], "token_counts": {
        **changed[0]["token_counts"], "sequence_tokens": 40_000,
    }}
    with pytest.raises(ValueError, match="population differs|manifest changed"):
        apply_length_filter(ROOT, changed, summary)
