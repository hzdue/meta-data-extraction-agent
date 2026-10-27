"""Step 2 data invariants; ordinary tests need no network or model weights."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from meta_data_extraction.finetune.data import (
    COLLECTIONS, canonical_json, make_record, project_entities, project_references,
    reconstruct, task_prompt,
)
from meta_data_extraction.finetune.parsing import save_unchanged_or_new, text_hash
from meta_data_extraction.finetune.tokenization import encode_record, record_lengths
from meta_data_extraction.utils import build_prompt
from scripts import prepare_finetune_data as preparation

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ["generic.json", "photon_and_neutron.json"]


@pytest.fixture
def gold():
    return json.loads((ROOT / "data/gold/2010.02965.json").read_text(encoding="utf-8"))


def test_all_calibration_projections_reconstruct_without_mutation():
    manifest = json.loads((ROOT / "data/finetune/splits.json").read_text(encoding="utf-8"))
    for pid in manifest["train"] + manifest["validation"]:
        gold = json.loads((ROOT / "data/gold" / f"{pid}.json").read_text(encoding="utf-8"))
        before = canonical_json(gold)
        entities, references = project_entities(gold), project_references(gold)
        assert reconstruct(entities, references) == {key: gold[key] for key in COLLECTIONS}
        assert all("sample_refs" not in d and "facility_refs" not in d for d in entities["datasets"])
        assert len(references["references"]) == 2 * len(gold["datasets"])
        assert canonical_json(gold) == before


def test_empty_collections_nulls_unicode_and_empty_references(gold):
    gold["datasets"][0]["sample_refs"] = []
    gold["samples"][0]["name"] = "μ sample"
    entities, references = project_entities(gold), project_references(gold)
    assert references["references"][0]["target_ids"] == []
    assert reconstruct(entities, references) == {key: gold[key] for key in COLLECTIONS}
    assert "μ sample" in canonical_json(entities)
    empty = {key: [] for key in COLLECTIONS}
    assert reconstruct(project_entities(empty), project_references(empty)) == empty


@pytest.mark.parametrize("problem", ["missing", "duplicate", "reordered", "dangling", "repeated_target"])
def test_bad_reference_projection_rejected(gold, problem):
    refs = project_references(gold)
    if problem == "missing":
        refs["references"].pop()
    elif problem == "duplicate":
        refs["references"].append(deepcopy(refs["references"][0]))
    elif problem == "reordered":
        refs["references"].reverse()
    elif problem == "dangling":
        refs["references"][0]["target_ids"] = ["sample_missing"]
    else:
        refs["references"][0]["target_ids"] = ["sample_1", "sample_1"]
    with pytest.raises(ValueError):
        reconstruct(project_entities(gold), refs)


def test_full_json_prompt_is_existing_prompt_verbatim():
    text = "literal {{SCHEMA_JSON}} in a paper"
    assert task_prompt("D", text, SCHEMAS) == build_prompt(SCHEMAS, text)


def test_reference_inventory_contains_no_reference_answers(gold):
    entities = project_entities(gold)
    prompt = task_prompt("R", "paper", SCHEMAS, entities)
    inventory = prompt.split("## Entity inventory (reference fields removed)\n\n")[1].split("\n\n## Document")[0]
    assert json.loads(inventory) == entities
    with pytest.raises(ValueError, match="Reference answers"):
        task_prompt("R", "paper", SCHEMAS, {key: gold[key] for key in COLLECTIONS})


@pytest.mark.parametrize(("split", "task"), [("test", "D"), ("test", "R"), ("validation", "E"), ("validation", "R")])
def test_test_and_validation_auxiliary_targets_forbidden(gold, split, task):
    with pytest.raises(ValueError):
        make_record("paper", split, task, "text", gold, SCHEMAS, {})


def test_records_are_deterministic_and_completions_are_plain_gold(gold):
    first = make_record("paper", "train", "D", "text", gold, SCHEMAS, {"x": 1})
    assert first == make_record("paper", "train", "D", "text", gold, SCHEMAS, {"x": 1})
    assert json.loads(first["completion"][0]["content"]) == gold
    assert first["completion"][0]["reasoning_content"] == ""


def test_existing_outputs_are_never_silently_replaced(tmp_path):
    path = tmp_path / "output.txt"
    save_unchanged_or_new(path, "first\n")
    save_unchanged_or_new(path, "first\n")
    with pytest.raises(ValueError, match="differs"):
        save_unchanged_or_new(path, "changed\n")
    assert path.read_text() == "first\n"


@pytest.mark.parametrize("name", ["pyproject.toml", "uv.lock", "prepare.py"])
def test_source_provenance_survives_windows_linux_line_endings(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"example\r\n")
    original = preparation.source_fingerprint(path)
    path.write_bytes(b"example\n")
    assert preparation.source_fingerprint(path) == original
    path.write_bytes(b"changed\n")
    assert preparation.source_fingerprint(path) != original


class ToyTokenizer:
    """Character tokenizer for boundary/overflow tests, not real length estimates."""
    eos_token_id = 3
    model_max_length = 1000000

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and not enable_thinking
        prefix = messages[0]["content"] + "[assistant][empty thinking]"
        return prefix if add_generation_prompt else prefix + messages[1]["content"] + "\x03\n"

    def encode(self, value, add_special_tokens, truncation):
        assert not add_special_tokens and not truncation
        return [ord(c) for c in value]


def test_template_prefix_eos_and_postfix_boundaries(gold):
    tokenizer = ToyTokenizer()
    record = make_record("paper", "train", "D", "paper text", gold, SCHEMAS, {})
    ids, start, end = encode_record(tokenizer, record)
    assert "".join(map(chr, ids[start:end - 1])) == canonical_json(gold)
    assert ids[end - 1] == tokenizer.eos_token_id
    assert ids[end:] == [ord("\n")]
    assert record_lengths(tokenizer, record, len(ids))["fits"]
    assert not record_lengths(tokenizer, record, len(ids) - 1)["fits"]


def test_actual_pinned_tokenizer_boundaries_if_cached(gold):
    pytest.importorskip("transformers")
    from meta_data_extraction.finetune.tokenization import load_tokenizer
    config = yaml.safe_load((ROOT / "configs/finetune_prepare.yaml").read_text(encoding="utf-8"))
    try:
        tokenizer, _ = load_tokenizer(config, local_files_only=True)
    except OSError:
        pytest.skip("Pinned tokenizer is not cached; no network allowed in tests")
    for task in ("D", "E", "R"):
        record = make_record("paper", "train", task, "Scientific paper μ", gold, SCHEMAS, {})
        ids, start, end = encode_record(tokenizer, record)
        assert tokenizer.decode(ids[start:end - 1]) == record["completion"][0]["content"]
        assert "</think>" in tokenizer.decode(ids[:start])
        assert ids[end - 1] == tokenizer.eos_token_id
        template_ids = tokenizer.apply_chat_template(record["prompt"] + record["completion"],
                         tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=False)
        assert ids == template_ids


@pytest.mark.parametrize(("key", "value"), [("max_seq_length", 0), ("parse_workers", -1),
                                           ("max_new_tokens", 65536), ("enable_thinking", True)])
def test_invalid_preparation_config_rejected(key, value):
    config = yaml.safe_load((ROOT / "configs/finetune_prepare.yaml").read_text(encoding="utf-8"))
    config[key] = value
    with pytest.raises(ValueError):
        preparation.check_config(config)


def test_complete_build_never_reads_test_targets(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "configs/finetune_prepare.yaml").read_text(encoding="utf-8"))
    config["max_seq_length"] = 900000
    config["max_new_tokens"] = 50000
    manifest = json.loads((ROOT / config["split_manifest"]).read_text(encoding="utf-8"))
    monkeypatch.setattr(preparation, "load_tokenizer", lambda *a, **k: (ToyTokenizer(), {"chat_template_sha256": "test"}))
    def cached(root, output, pid):
        return "paper text", {"paper_id": pid, "text_sha256": text_hash("paper text"),
                              "warnings": [], "low_word_presence_pages": []}
    monkeypatch.setattr(preparation, "load_cached", cached)
    original_read = preparation.read_json
    opened_gold = []
    def guarded_read(path):
        assert path.stem not in manifest["test"], "Test target was consulted"
        opened_gold.append(path.stem)
        return original_read(path)
    monkeypatch.setattr(preparation, "read_json", guarded_read)
    report = preparation.prepare(ROOT, tmp_path, manifest, config)
    assert report["counts"] == {"train_json_only.jsonl": 8, "train_structure_aware.jsonl": 24, "validation.jsonl": 2}
    assert len(report["inference"]) == 40 and len(report["supervised_examples"]) == 26
    assert set(opened_gold) == set(manifest["train"] + manifest["validation"])
    d = [json.loads(line) for line in (tmp_path / "train_json_only.jsonl").read_text(encoding="utf-8").splitlines()]
    der = [json.loads(line) for line in (tmp_path / "train_structure_aware.jsonl").read_text(encoding="utf-8").splitlines()]
    assert d == [row for row in der if row["task"] == "D"]
    preparation.prepare(ROOT, tmp_path, manifest, config, verify_only=True)
    overflow_output = tmp_path / "overflow"
    config["max_seq_length"] = 20
    with pytest.raises(ValueError, match="length checks failed"):
        preparation.prepare(ROOT, overflow_output, manifest, config)
    assert (overflow_output / "length_report.json").exists()
    assert not list(overflow_output.glob("*.jsonl"))
