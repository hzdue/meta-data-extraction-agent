"""Check released exemplars, prompt rendering, and few-shot configurations."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from meta_data_extraction.app_config import ExtractorConfig
from meta_data_extraction.utils import (
    build_few_shot_prompt,
    load_few_shot_examples,
)

REPO = Path(__file__).resolve().parents[1]
GOLD_DIR = REPO / "data" / "gold"
FEW_SHOT_DIR = REPO / "data" / "few_shot"
SCHEMAS = ["generic.json", "photon_and_neutron.json"]

EXAMPLE_IDS = ["2309.13729", "2104.10884", "2304.08002"]

@pytest.mark.parametrize("arxiv_id", EXAMPLE_IDS)
def test_exemplar_matches_released_gold(arxiv_id):
    gold = json.loads((GOLD_DIR / f"{arxiv_id}.json").read_text(encoding="utf-8"))
    payload = json.loads((FEW_SHOT_DIR / f"{arxiv_id}.json").read_text(encoding="utf-8"))
    assert payload["expected_output"] == gold


@pytest.mark.parametrize("arxiv_id", EXAMPLE_IDS)
def test_exemplar_cross_references_resolve(arxiv_id):
    """Every exemplar reference must resolve within its expected output."""
    out = json.loads((FEW_SHOT_DIR / f"{arxiv_id}.json").read_text(encoding="utf-8"))[
        "expected_output"
    ]
    bib = out["bibliographic"]
    aff_ids = {a.get("id") for a in (bib.get("affiliations") or [])}
    sample_ids = {s.get("id") for s in (out.get("samples") or [])}
    facility_ids = {f.get("id") for f in (out.get("facilities") or [])}

    for author in bib.get("authors") or []:
        for ref in author.get("affiliation_refs") or []:
            assert ref in aff_ids, f"{arxiv_id}: dangling affiliation_ref {ref!r}"
    for dataset in out.get("datasets") or []:
        for ref in dataset.get("sample_refs") or []:
            assert ref in sample_ids, f"{arxiv_id}: dangling sample_ref {ref!r}"
        for ref in dataset.get("facility_refs") or []:
            assert ref in facility_ids, f"{arxiv_id}: dangling facility_ref {ref!r}"


def test_rendered_prompt_leaks_no_private_annotator_blocks():
    prompt = build_few_shot_prompt(SCHEMAS, "TARGET", load_few_shot_examples(EXAMPLE_IDS))
    assert "_annotation" not in prompt
    assert "_evidence" not in prompt


def test_one_shot_renders_only_the_first_exemplar():
    examples = load_few_shot_examples(EXAMPLE_IDS)
    prompt = build_few_shot_prompt(SCHEMAS, "TARGET", examples[:1])
    assert "### Example 1" in prompt
    assert "### Example 2" not in prompt
    assert examples[1]["document_text"][:200] not in prompt
    assert examples[2]["document_text"][:200] not in prompt


def test_three_shot_renders_all_exemplars_in_config_order():
    examples = load_few_shot_examples(EXAMPLE_IDS)
    assert [example["arxiv_id"] for example in examples] == EXAMPLE_IDS
    prompt = build_few_shot_prompt(SCHEMAS, "TARGET", examples)
    positions = [prompt.index(e["document_text"][:200]) for e in examples]
    assert positions == sorted(positions), "exemplars rendered out of order"
    assert prompt.count("### Example ") == 3


def test_target_document_is_rendered_last_and_placeholders_all_consumed():
    prompt = build_few_shot_prompt(SCHEMAS, "TARGET_SENTINEL", load_few_shot_examples(EXAMPLE_IDS))
    assert "{{" not in prompt
    assert prompt.rstrip().endswith("TARGET_SENTINEL")


def test_exemplar_containing_a_template_token_is_rejected():
    """Rendering substitutes placeholders sequentially; an exemplar carrying a
    literal template token would corrupt the prompt."""
    poisoned = [{"arxiv_id": "fake", "document_text": "x {{DOCUMENT_TEXT}} y", "expected_output": {}}]
    with pytest.raises(ValueError, match="literal template token"):
        build_few_shot_prompt(SCHEMAS, "TARGET", poisoned)


def test_build_few_shot_prompt_requires_at_least_one_exemplar():
    with pytest.raises(ValueError, match="at least one exemplar"):
        build_few_shot_prompt(SCHEMAS, "TARGET", [])


def test_extractor_config_defaults_to_zero_shot():
    config = ExtractorConfig()
    assert config.shot_mode == "zero_shot"
    assert config.few_shot is None


@pytest.mark.parametrize(
    "payload",
    [
        {"shot_mode": "few_shot"},
        {"shot_mode": "few_shot", "few_shot": {"example_ids": []}},
        {"shot_mode": "zero_shot", "few_shot": {"example_ids": ["2309.13729"]}},
        {"shot_mode": "two_shot"},
    ],
)
def test_extractor_config_rejects_inconsistent_shot_settings(payload):
    with pytest.raises(ValidationError):
        ExtractorConfig(**payload)


FEW_SHOT_CONFIGS = sorted((REPO / "configs").glob("pan_*_full_[13]shot.yaml"))


def test_few_shot_configs_are_exactly_the_expected_set():
    """The four tested models each have one-shot and three-shot conditions."""
    assert [p.name for p in FEW_SHOT_CONFIGS] == [
        "pan_gpt5.5_full_1shot.yaml",
        "pan_gpt5.5_full_3shot.yaml",
        "pan_qwen3.5_27b_full_1shot.yaml",
        "pan_qwen3.5_27b_full_3shot.yaml",
        "pan_qwen3.5_4b_full_1shot.yaml",
        "pan_qwen3.5_4b_full_3shot.yaml",
        "pan_qwen3.5_9b_full_1shot.yaml",
        "pan_qwen3.5_9b_full_3shot.yaml",
    ]


@pytest.mark.parametrize("config_path", FEW_SHOT_CONFIGS, ids=lambda p: p.stem)
def test_few_shot_config_example_ids_are_quoted_strings(config_path):
    """YAML parses a bare ``2309.13729`` as a float. Our three ids survive
    ``str()`` round-tripping, but an id like ``2201.00100`` would silently
    become ``2201.001`` and point at a nonexistent exemplar. The ids must be
    quoted in the YAML, and loaded as ``str``."""
    import yaml

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ids = raw["extractor"]["few_shot"]["example_ids"]
    assert all(isinstance(i, str) for i in ids), (
        f"{config_path.name}: example_ids parsed as {[type(i).__name__ for i in ids]}; "
        "quote them in the YAML"
    )


@pytest.mark.parametrize("config_path", FEW_SHOT_CONFIGS, ids=lambda p: p.stem)
def test_few_shot_config_matches_example_order_and_shot_count(config_path):
    """1-shot must reference exactly example_ids[0]; 3-shot the full ordered
    list, following the published exemplar order."""
    from meta_data_extraction.utils import load_config

    config = load_config(config_path)
    assert config.extractor.shot_mode == "few_shot"
    ids = config.extractor.few_shot.example_ids

    n_shot = 1 if "_1shot" in config_path.stem else 3
    assert ids == EXAMPLE_IDS[:n_shot]
    assert config.schemas == SCHEMAS, "Few-shot conditions use the full-instruction schemas"
