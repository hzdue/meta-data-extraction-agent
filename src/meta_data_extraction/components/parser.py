from __future__ import annotations

import re
from pathlib import Path

import pymupdf4llm

from meta_data_extraction.base.parser import BaseParser
from meta_data_extraction.data_models import Document, Section


class PdfParser(BaseParser):
    """Convert PDFs to Markdown and derive sections from Markdown headings."""

    def parse(self, path: Path) -> Document:
        path = Path(path)
        text = pymupdf4llm.to_markdown(str(path))
        return Document(
            id=path.parent.name,
            text=text,
            source_format="pdf",
            source_path=str(path),
            parser_name=self.__class__.__name__,
            sections=_sections_from_markdown(text),
        )


def _sections_from_markdown(text: str) -> list[Section]:
    """Split Markdown into sections, retaining an untitled preamble.

    Skip sections without body text.
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
    matches = list(heading_re.finditer(text))

    if not matches:
        body = text.strip()
        return [Section(id="section_1", title=None, level=None, text=body)] if body else []

    sections: list[Section] = []
    section_id = 1

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(
            Section(id=f"section_{section_id}", title=None, level=None, text=preamble)
        )
        section_id += 1

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        if not body:
            continue

        sections.append(
            Section(
                id=f"section_{section_id}",
                title=title,
                level=level,
                text=body,
            )
        )
        section_id += 1

    return sections
