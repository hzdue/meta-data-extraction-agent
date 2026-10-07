"""Regression tests for alignment-aware cross-reference scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meta_data_extraction.evaluation import (
    _spec_from_yaml,
    evaluate,
    load_plan,
    _score_references,
)


def test_aligned_local_reference_ids_receive_full_credit():
    result = _score_references(
        ["pred_sample_1"],
        ["sample_1"],
        {"pred_sample_1": "sample_1"},
    )
    assert result == {"tp": 1, "fp": 0, "fn": 0, "f1": 1.0}


def test_unknown_predicted_reference_counts_as_false_positive():
    result = _score_references(
        ["unknown_sample"],
        [],
        {},
    )
    assert result == {"tp": 0, "fp": 1, "fn": 0, "f1": 0.0}


def test_reference_f1_is_symmetric_when_one_side_has_an_extra_entity():
    forward = _score_references(
        ["pred_sample_1", "pred_sample_2"],
        ["sample_1"],
        {"pred_sample_1": "sample_1", "pred_sample_2": None},
    )
    reverse = _score_references(
        ["sample_1"],
        ["pred_sample_1", "pred_sample_2"],
        {"sample_1": "pred_sample_1"},
    )
    assert forward == {"tp": 1, "fp": 1, "fn": 0, "f1": pytest.approx(2 / 3)}
    assert forward["fp"] == reverse["fn"] == 1
    assert forward["fn"] == reverse["fp"] == 0
    assert forward["f1"] == reverse["f1"] == pytest.approx(2 / 3)


def test_repeated_dangling_id_preserves_set_semantics():
    result = _score_references(
        ["unknown_sample", "unknown_sample"],
        [],
        {},
    )
    assert result["fp"] == 1


def test_active_reference_specs_point_to_their_entity_collections():
    plan = load_plan("configs/evaluation_metrics.yaml")
    authors = plan["bibliographic.authors"][1]
    datasets = plan["datasets"][1]
    assert authors.attr_scorers["affiliation_refs"] == (
        "reference", "bibliographic.affiliations"
    )
    assert datasets.attr_scorers["sample_refs"] == ("reference", "samples")
    assert datasets.attr_scorers["facility_refs"] == ("reference", "facilities")


def test_reference_policy_option_is_rejected():
    with pytest.raises(ValueError, match="unsupported keys.*unmatched_pred_refs"):
        _spec_from_yaml({
            "metric": "reference", "target": "samples", "unmatched_pred_refs": "drop"
        })


def _primary_metrics(result: dict) -> tuple[tuple[str, object], ...]:
    return tuple(
        (key, result[key])
        for key in ("score", "mean_score", "f1", "order")
        if key in result
    )


def test_primary_iaa_metrics_are_symmetric_on_all_30_papers():
    corpus = json.loads(Path("data/corpus.json").read_text(encoding="utf-8"))
    for batch_name, pair in corpus["annotation_pairs"].items():
        first, second = pair
        for paper in corpus["papers"]:
            if paper.get("annotation_batch") != batch_name:
                continue
            arxiv_id = paper["arxiv_id"]
            first_path = Path("data/annotations") / first / f"{arxiv_id}.json"
            second_path = Path("data/annotations") / second / f"{arxiv_id}.json"
            identity_gold = Path("data/gold") / f"{arxiv_id}.json"
            forward = evaluate(
                first_path, second_path, identity_gold_path=identity_gold
            )
            reverse = evaluate(
                second_path, first_path, identity_gold_path=identity_gold
            )
            for field, forward_result in forward["fields"].items():
                assert _primary_metrics(forward_result) == _primary_metrics(
                    reverse["fields"][field]
                ), f"direction-dependent IAA metric: {batch_name}/{arxiv_id}/{field}"
