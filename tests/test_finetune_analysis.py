"""Reproducibility checks for the no-retraining fine-tuning audit."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("trl")

from meta_data_extraction.finetune.tokenization import encode_record, load_tokenizer
from scripts import analyze_finetune_results as audit


ROOT = Path(__file__).resolve().parents[1]


def test_precision_recall_and_paired_bootstrap_recompute_pooled_counts():
    assert audit.prf({"tp": 3, "fp": 1, "fn": 2}) == {
        "tp": 3, "fp": 1, "fn": 2, "precision": 0.75, "recall": 0.6,
        "f1": pytest.approx(2 / 3),
    }
    base = [{"fields": {field: {"tp": 1, "fp": 1, "fn": 1} for field in audit.FIELDS}}
            for _ in range(4)]
    model = [{"fields": {field: {"tp": 2, "fp": 0, "fn": 0} for field in audit.FIELDS}}
             for _ in range(4)]
    result = audit.paired_bootstrap(base, model, samples=100, seed=42)
    for field in audit.FIELDS:
        assert result[field]["delta_f1"] == pytest.approx(0.5)
        assert result[field]["paired_bootstrap_95_ci"] == pytest.approx([0.5, 0.5])


def test_reference_token_spans_match_frozen_validation_targets_if_tokenizer_cached():
    prepare = yaml.safe_load((ROOT / "configs/finetune_prepare.yaml").read_text(encoding="utf-8"))
    try:
        tokenizer, _ = load_tokenizer(prepare, local_files_only=True)
    except OSError:
        pytest.skip("Pinned tokenizer is not cached")
    records = [json.loads(line) for line in
               (ROOT / "data/finetune/prepared/validation.jsonl").read_text(encoding="utf-8").splitlines()]
    counts = []
    for record in records:
        input_ids, _, _ = encode_record(tokenizer, record)
        counts.append(int(audit.reference_mask(tokenizer, record, input_ids).sum()))
    assert counts == [236, 249]


def test_generation_stats_distinguish_length_from_throughput():
    stats = audit.generation_stats([
        {"generated_tokens": 100, "latency_ms": 10_000},
        {"generated_tokens": 200, "latency_ms": 20_000},
    ])
    assert stats == {"papers": 2, "total_output_tokens": 300, "mean_output_tokens": 150,
                     "total_seconds": 30, "mean_seconds": 15, "tokens_per_second": 10}
