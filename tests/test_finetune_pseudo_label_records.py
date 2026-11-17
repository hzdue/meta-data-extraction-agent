"""Pseudo-label validation and D-record preparation invariants."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from meta_data_extraction.finetune import pseudo_label_records as records
from meta_data_extraction.utils import load_merged_schema


ROOT = Path(__file__).resolve().parents[1]


def source_prediction():
    path = (ROOT / "data/share_600_papers/papers/1001.2151/"
            "global_seed_prediction.json")
    if not path.exists():
        pytest.skip("Pseudo-label source corpus is not available")
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_prediction_passes_complete_schema_validation():
    prediction = source_prediction()
    assert records.validate_prediction(
        prediction, load_merged_schema(records.SCHEMAS)) == []


def test_dangling_reference_fails_validation():
    prediction = deepcopy(source_prediction())
    prediction["datasets"][0]["sample_refs"] = ["sample_999"]
    errors = records.validate_prediction(
        prediction, load_merged_schema(records.SCHEMAS))
    assert any("dangling reference 'sample_999'" in error for error in errors)


def test_unknown_nested_field_fails_validation():
    prediction = deepcopy(source_prediction())
    prediction["datasets"][0]["invented"] = "not in schema"
    errors = records.validate_prediction(
        prediction, load_merged_schema(records.SCHEMAS))
    assert any("unknown=['invented']" in error for error in errors)


def test_review_sample_is_deterministic_and_balanced():
    rows = []
    for subject_index, subject in enumerate(sorted({
            "diffraction", "imaging_tomography", "methods_light",
            "multi_technique", "reflectometry", "small_angle_scattering",
            "spectroscopy"})):
        for index, length_bin in enumerate((
                "short_1_10", "medium_11_20", "long_21_40",
                "very_long_41_plus")):
            rows.append({
                "paper_id": f"{subject_index:04d}.{index:04d}",
                "subject_group": subject,
                "page_bin": length_bin,
                "split_role": "pseudo_validation" if index == 0 else "pseudo_train",
            })
    first = records.select_review_sample(rows)
    second = records.select_review_sample(list(reversed(rows)))
    assert [row["paper_id"] for row in first] == [row["paper_id"] for row in second]
    assert len(first) == 28
    assert sum(row["split_role"] == "pseudo_validation" for row in first) == 7


def test_prepared_records_are_complete_and_future_test_is_isolated():
    output = ROOT / "data/finetune/pseudo_label_supervision/prepared"
    manifest_path = output / "preparation_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Run pseudo-label record preparation first")

    def lines(name):
        with (output / name).open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    train = lines("train_pseudo_d.jsonl")
    validation = lines("validation_pseudo_d.jsonl")
    future = lines("future_test_manifest.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalid = set(manifest["formal_validation"]["excluded_paper_ids"])

    assert (len(train), len(validation), len(future)) == (504, 57, 30)
    assert manifest["status"] == "prepared_pending_gpu_probe"
    assert manifest["policy"]["future_test_predictions_read"] is False
    supervised_ids = {row["paper_id"] for row in train + validation}
    assert len(supervised_ids) == 561
    assert not (supervised_ids & invalid)
    assert not (supervised_ids & {row["paper_id"] for row in future})
    for row in train + validation:
        assert [message["role"] for message in row["messages"]] == [
            "user", "assistant"]
        assert isinstance(json.loads(row["messages"][1]["content"]), dict)
        assert row["token_counts"]["complete_sequence"] <= 65_536
    for row in future:
        assert row["teacher_prediction_read"] is False
        assert row["teacher_prediction_permitted_in_training"] is False
        assert "messages" not in row
