from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Section(BaseModel):
    id: str
    title: str | None = None
    level: int | None = None
    text: str


class Figure(BaseModel):
    """Figure reference with an optional extracted-image path."""

    id: str
    caption: str | None = None
    path: str | None = None


class Table(BaseModel):
    """Table reference with optional Markdown content_text and an image path."""

    id: str
    caption: str | None = None
    content_text: str | None = None
    path: str | None = None


class Document(BaseModel):
    """Internal representation of a parsed paper."""

    id: str
    text: str
    source_format: str | None = None
    source_path: str | None = None
    parser_name: str | None = None  # class name of the parser used
    warnings: list[str] = Field(default_factory=list)  # non-fatal parsing issues
    sections: list[Section] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)


class Provenance(BaseModel):
    """Extraction metadata; model and decoding settings live in the named config."""

    config_name: str | None = None
    prompt_version: str | None = None
    schemas: dict[str, str | None] = Field(default_factory=dict)

    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


class ExtractionResult(BaseModel):
    """One document's parsed metadata, raw response, provenance, and warnings.

    Warnings accumulate across extraction and validation.
    """

    document_id: str
    domain_name: str
    output: dict[str, Any] = Field(default_factory=dict)
    raw_output: str | None = None
    provenance: Provenance | None = None
    warnings: list[str] = Field(default_factory=list)
