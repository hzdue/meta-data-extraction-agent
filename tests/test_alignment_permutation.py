"""Regression tests for permutation-invariant entity assignment."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from meta_data_extraction.alignment import (
    _prepare_dataset_identities,
    _prepare_funding_identities,
    _prepare_sample_identities,
    _match_datasets,
    align_facilities,
    _match_funding,
    _match_samples,
)
from meta_data_extraction.evaluation import evaluate


def _sample_pairs(predictions, entries):
    return {
        (predictions[pred]["id"], entries[gold]["gold_id"], stage)
        for pred, (gold, stage) in _match_samples(predictions, entries).items()
    }


def _dataset_pairs(predictions, entries):
    return {
        (predictions[pred]["id"], entries[gold]["gold_id"], stage)
        for pred, (gold, stage) in _match_datasets(predictions, entries).items()
    }


def test_sample_stage_solves_competing_candidates_globally():
    entries = _prepare_sample_identities([
        {
            "id": "sample_alpha",
            "sample_name": "alpha base",
            "chemical_composition": None,
        },
        {
            "id": "sample_beta",
            "sample_name": "beta base",
            "chemical_composition": None,
        },
    ], "synthetic-permutation")
    predictions = [
        {
            "id": "sample_broad",
            "sample_name": "base",
            "chemical_composition": None,
        },
        {
            "id": "sample_specific",
            "sample_name": "alpha base extra",
            "chemical_composition": None,
        },
    ]
    expected = {
        ("sample_broad", "sample_beta", "unique_name_containment"),
        ("sample_specific", "sample_alpha", "unique_name_containment"),
    }
    assert _sample_pairs(predictions, entries) == expected
    assert _sample_pairs(list(reversed(predictions)), entries) == expected


def test_dataset_stage_solves_competing_candidates_globally():
    entries = _prepare_dataset_identities([
        {
            "id": "dataset_alpha",
            "measurement_technique": "method",
            "dataset_name": "alpha dataset",
        },
        {
            "id": "dataset_beta",
            "measurement_technique": "method",
            "dataset_name": "beta dataset",
        },
    ], "synthetic-permutation")
    predictions = [
        {
            "id": "dataset_broad",
            "measurement_technique": "method",
            "dataset_name": "alpha beta",
        },
        {
            "id": "dataset_specific",
            "measurement_technique": "method",
            "dataset_name": "alpha signal",
        },
    ]
    expected = {
        ("dataset_broad", "dataset_beta", "shared_technique_name_discriminator"),
        ("dataset_specific", "dataset_alpha", "shared_technique_name_discriminator"),
    }
    assert _dataset_pairs(predictions, entries) == expected
    assert _dataset_pairs(list(reversed(predictions)), entries) == expected


def test_facility_duplicate_winner_uses_stable_local_id():
    predictions = [
        {"id": "facility_b", "name": "ESRF"},
        {"id": "facility_a", "name": "ESRF"},
    ]
    gold = [{"id": "facility_1", "name": "European Synchrotron Radiation Facility"}]

    def selected(items):
        matches = align_facilities(items, gold)
        return {
            (items[pair.pred_index]["id"], gold[pair.gold_index]["id"], pair.stage)
            for pair in matches.pairs
        }

    expected = {("facility_a", "facility_1", "alias_exact")}
    assert selected(predictions) == expected
    assert selected(list(reversed(predictions))) == expected


@pytest.mark.parametrize("gold_grant, alternate", [
    ("DMR-2105081", "Grant No. DMR-2105081"),
    ("DMR2105081", "DMR-2105081"),
])
def test_funding_duplicate_prefers_exact_grant_surface_independent_of_order(gold_grant, alternate):
    predictions = [
        {
            "funder": "National Science Foundation",
            "grant_id": alternate,
        },
        {"funder": "National Science Foundation", "grant_id": gold_grant},
    ]
    entries = _prepare_funding_identities([
        {"grant_id": gold_grant, "funder": "National Science Foundation"}
    ])

    def selected(items):
        matches = _match_funding(items, entries)
        return {(items[pred]["grant_id"], target, stage)
                for pred, (target, stage) in matches.items()}

    expected = {(gold_grant, 0, "grant_then_funder")}
    assert selected(predictions) == expected
    assert selected(list(reversed(predictions))) == expected


@pytest.mark.parametrize(
    "relative_path",
    [
        "pan_gpt5_5_full_1shot/1901.08819.json",
        "pan_qwen3_5_4b_full_1shot/2212.04169.json",
        "pan_gpt5_5_no_inst_0shot/2410.00268.json",
        "pan_gpt5_5_full_3shot/2605.14792.json",
    ],
)
def test_affected_reports_are_invariant_to_entity_list_reversal(tmp_path, relative_path):
    extraction_path = Path("data/experiment") / relative_path
    gold_path = Path("data/gold") / extraction_path.name
    original_data = json.loads(extraction_path.read_text(encoding="utf-8"))
    permuted_data = copy.deepcopy(original_data)
    output = permuted_data["output"]
    for key in ("samples", "datasets", "facilities"):
        output[key] = list(reversed(output.get(key) or []))
    bibliographic = output.get("bibliographic") or {}
    bibliographic["funding"] = list(reversed(bibliographic.get("funding") or []))
    permuted_path = tmp_path / extraction_path.name
    permuted_path.write_text(json.dumps(permuted_data), encoding="utf-8")

    original = evaluate(gold_path, extraction_path)
    permuted = evaluate(gold_path, permuted_path)
    assert original["fields"] == permuted["fields"]


def test_funding_match_diagnostics_show_grant_and_funder_keys():
    path = Path("data/gold/2212.04169.json")
    report = evaluate(path, path)
    stages = report["fields"]["bibliographic.funding"]["match_stages"]
    assert stages
    for item in stages:
        assert set(item["pred"]) == {"grant_id", "funder"}
        assert set(item["gold"]) == {"grant_id", "funder"}
        assert item["pred"]["grant_id"] is not None
        assert item["gold"]["grant_id"] is not None
