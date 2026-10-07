"""Tests for LLMExtractor._parse_output."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from meta_data_extraction.components.extractor import LLMExtractor


def make_extractor() -> LLMExtractor:
    llm = MagicMock()
    llm.model_name = "test-model"
    return LLMExtractor(llm=llm, domain_name="generic", schemas=["generic.json"])


def test_clean_json_returns_dict():
    extractor = make_extractor()
    raw = json.dumps({"title": "My Paper", "authors": []})
    output, warnings = extractor._parse_output(raw)
    assert output == {"title": "My Paper", "authors": []}
    assert warnings == []


def test_json_in_markdown_fences_is_stripped():
    extractor = make_extractor()
    raw = '```json\n{"title": "My Paper"}\n```'
    output, warnings = extractor._parse_output(raw)
    assert output == {"title": "My Paper"}
    assert warnings == []


def test_json_in_plain_fences_is_stripped():
    extractor = make_extractor()
    raw = '```\n{"title": "My Paper"}\n```'
    output, warnings = extractor._parse_output(raw)
    assert output == {"title": "My Paper"}
    assert warnings == []


def test_fallback_recovers_json_with_trailing_text():
    extractor = make_extractor()
    raw = '{"title": "My Paper"}\n\nHere is a summary of the extracted metadata.'
    output, warnings = extractor._parse_output(raw)
    assert output == {"title": "My Paper"}
    assert any("fallback" in w.lower() or "partial" in w.lower() for w in warnings)


def test_total_garbage_returns_empty_with_warning():
    extractor = make_extractor()
    output, warnings = extractor._parse_output("Sorry, I cannot extract metadata from this.")
    assert output == {}
    assert len(warnings) == 1
    assert "parse failed" in warnings[0].lower()


def test_llm_returns_list_instead_of_object():
    extractor = make_extractor()
    raw = json.dumps(["item1", "item2"])
    output, warnings = extractor._parse_output(raw)
    assert output == {}
    assert any("not a json object" in w.lower() for w in warnings)
