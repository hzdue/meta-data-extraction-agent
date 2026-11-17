"""Pseudo-label split and external-PDF cache invariants."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest
import yaml

from meta_data_extraction.finetune import pseudo_label_preparation as preparation


ROOT = Path(__file__).resolve().parents[1]


def synthetic_records(count=120):
    subjects = sorted(preparation.EXPECTED_SUBJECT_GROUPS)
    result = []
    for index in range(count):
        year = 2010 + index % 17
        result.append({
            "paper_id": f"{year % 100:02d}{index % 12 + 1:02d}.{index:05d}",
            "subject_group": subjects[index % len(subjects)],
            "page_count": 2 + index % 60,
            "published": f"{year:04d}-{index % 12 + 1:02d}-01",
        })
    return result


def test_balanced_selection_is_deterministic_unique_and_order_independent():
    records = synthetic_records()
    first = preparation.balanced_select(records, 30, 42, "future_test")
    second = preparation.balanced_select(list(reversed(records)), 30, 42, "future_test")
    assert [row["paper_id"] for row in first] == [row["paper_id"] for row in second]
    assert len(first) == len({row["paper_id"] for row in first}) == 30
    expected = preparation.proportional_quotas(
        Counter(row["subject_group"] for row in records), 30)
    assert Counter(row["subject_group"] for row in first) == expected


def test_reviewed_config_is_strict():
    config = yaml.safe_load(
        (ROOT / "configs/finetune_pseudo_prepare.yaml").read_text(encoding="utf-8"))
    preparation.check_config(config)
    changed = {**config, "split_counts": {**config["split_counts"], "future_test": 29}}
    with pytest.raises(ValueError):
        preparation.check_config(changed)


def test_frozen_split_covers_source_without_benchmark_overlap():
    config = yaml.safe_load(
        (ROOT / "configs/finetune_pseudo_prepare.yaml").read_text(encoding="utf-8"))
    path = ROOT / config["split_manifest"]
    if not path.exists():
        pytest.skip("Run pseudo-label split preparation first")
    split = json.loads(path.read_text(encoding="utf-8"))
    assert {key: len(split[key]) for key in preparation.SPLIT_NAMES} == config["split_counts"]
    sets = [set(split[key]) for key in preparation.SPLIT_NAMES]
    assert len(set.union(*sets)) == sum(map(len, sets)) == 595
    assert len(split["excluded_missing_prediction"]) == 5
    assert split["leakage_checks"]["existing_benchmark_overlap"] == []


def test_future_test_predictions_are_forbidden_by_contract():
    config = yaml.safe_load(
        (ROOT / "configs/finetune_pseudo_prepare.yaml").read_text(encoding="utf-8"))
    path = ROOT / config["split_manifest"]
    if not path.exists():
        pytest.skip("Run pseudo-label split preparation first")
    split = json.loads(path.read_text(encoding="utf-8"))
    assert split["leakage_checks"]["future_test_predictions_permitted_in_training"] is False
    assert not (set(split["future_test"]) & set(split["pseudo_train"]))
    assert not (set(split["future_test"]) & set(split["pseudo_validation"]))


def test_targeted_cache_clear_removes_only_requested_pair(tmp_path):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    requested = "1507.03694"
    retained = "1508.00500"
    for paper_id in (requested, retained):
        (text_dir / f"{paper_id}.md").write_text("text", encoding="utf-8")
        (text_dir / f"{paper_id}.json").write_text("{}", encoding="utf-8")

    removed = preparation.clear_external_cache(tmp_path, requested)

    assert {path.name for path in removed} == {
        f"{requested}.md", f"{requested}.json"}
    assert not (text_dir / f"{requested}.md").exists()
    assert not (text_dir / f"{requested}.json").exists()
    assert (text_dir / f"{retained}.md").exists()
    assert (text_dir / f"{retained}.json").exists()


def test_cache_paths_reject_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        preparation.clear_external_cache(tmp_path, "../outside")


def test_exact_text_read_preserves_crlf_for_cache_hashing(tmp_path):
    path = tmp_path / "paper.md"
    raw = b"first\r\nsecond\r\n"
    path.write_bytes(raw)

    text = preparation.read_text_exact(path)

    assert text == "first\r\nsecond\r\n"
    assert preparation.text_hash(text) == preparation.file_hash(path)


def test_parse_one_tiny_pdf_writes_reusable_provenance(tmp_path):
    source = tmp_path / "source"
    paper_id = "1234.56789"
    paper_dir = source / "papers" / paper_id
    paper_dir.mkdir(parents=True)
    pdf_path = paper_dir / "paper.pdf"
    with preparation.pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "A small scientific paper for parser testing.")
        pdf.save(pdf_path)
    output = tmp_path / "prepared"

    metadata = preparation.parse_external_one(
        ROOT, output, source.relative_to(ROOT), paper_id,
        f"papers/{paper_id}/paper.pdf", "pseudo_train")

    assert metadata["paper_id"] == paper_id
    assert metadata["preparation_source_sha256"]
    assert "summary_source_sha256" not in metadata
    assert preparation.load_external_cached(
        ROOT, output, source.relative_to(ROOT), paper_id,
        f"papers/{paper_id}/paper.pdf", "pseudo_train")[1] == metadata

