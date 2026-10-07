"""Tests for schema utilities: load_merged_schema, build_skeleton, Document.id."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from meta_data_extraction.utils import build_skeleton, load_merged_schema


def test_merged_schema_contains_keys_from_both_schemas():
    merged = load_merged_schema(["generic.json", "photon_and_neutron.json"])
    # generic fields
    assert "bibliographic" in merged
    assert "document_overview" in merged
    # PaN-specific fields
    assert "facilities" in merged
    assert "samples" in merged
    assert "datasets" in merged
    assert "facility_names" not in merged


def test_merged_schema_strips_meta():
    merged = load_merged_schema(["generic.json"])
    assert "_meta" not in merged


def test_build_skeleton_string_field_becomes_ellipsis():
    schema = {"title": {"type": "string"}}
    skeleton = build_skeleton(schema)
    assert skeleton["title"] == "..."


def test_build_skeleton_nullable_field_becomes_none():
    schema = {"doi": {"type": "string|null"}}
    skeleton = build_skeleton(schema)
    assert skeleton["doi"] is None


def test_build_skeleton_list_field_becomes_empty_list():
    schema = {"authors": {"type": "list[author]"}}
    skeleton = build_skeleton(schema)
    assert skeleton["authors"] == []


def test_build_skeleton_nested_namespace_is_recursed():
    schema = {
        "bibliographic": {
            "title": {"type": "string|null"},
            "doi": {"type": "string|null"},
        }
    }
    skeleton = build_skeleton(schema)
    assert "bibliographic" in skeleton
    assert skeleton["bibliographic"]["title"] is None
    assert skeleton["bibliographic"]["doi"] is None


def test_document_id_uses_parent_directory_name():
    """Regression: Document.id must be the arxiv ID, not 'paper'."""
    from meta_data_extraction.components.parser import PdfParser
    import pymupdf4llm

    with patch.object(pymupdf4llm, "to_markdown", return_value="# Title\n\nBody."):
        parser = PdfParser()
        doc = parser.parse(Path("data/raw/2405.14324/paper.pdf"))

    assert doc.id == "2405.14324"
    assert doc.id != "paper"
