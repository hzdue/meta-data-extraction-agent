"""Tests for PdfParser._sections_from_markdown."""
from __future__ import annotations

from meta_data_extraction.components.parser import _sections_from_markdown


def test_document_with_headings_produces_correct_sections():
    text = "Preamble text.\n\n# Introduction\n\nIntro body.\n\n## Methods\n\nMethods body."
    sections = _sections_from_markdown(text)
    titles = [s.title for s in sections]
    assert None in titles          # preamble
    assert "Introduction" in titles
    assert "Methods" in titles


def test_heading_levels_are_captured():
    text = "# Top\n\nBody.\n\n## Sub\n\nSub body."
    sections = _sections_from_markdown(text)
    levels = {s.title: s.level for s in sections}
    assert levels["Top"] == 1
    assert levels["Sub"] == 2


def test_no_headings_returns_single_preamble_section():
    text = "This document has no headings at all."
    sections = _sections_from_markdown(text)
    assert len(sections) == 1
    assert sections[0].title is None
    assert sections[0].level is None
    assert "no headings" in sections[0].text


def test_empty_body_sections_are_skipped():
    text = "# Section A\n\n# Section B\n\nOnly B has a body."
    sections = _sections_from_markdown(text)
    titles = [s.title for s in sections]
    assert "Section A" not in titles
    assert "Section B" in titles


def test_empty_document_returns_no_sections():
    sections = _sections_from_markdown("   ")
    assert sections == []


def test_section_ids_are_sequential():
    text = "# A\n\nBody A.\n\n# B\n\nBody B.\n\n# C\n\nBody C."
    sections = _sections_from_markdown(text)
    assert [s.id for s in sections] == ["section_1", "section_2", "section_3"]
