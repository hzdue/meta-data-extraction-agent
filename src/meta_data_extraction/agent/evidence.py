"""Immutable text evidence with exact Unicode offsets; no inferred page mapping."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from meta_data_extraction.data_models import Document

REGION_VERSION = "m01-regions-1"
Hash = str
RegionKind = Literal["heading", "paragraph", "caption", "table", "code"]


class EvidenceError(ValueError):
    """Evidence does not resolve in the supplied immutable representation."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class ParserProvenance(FrozenModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    options: tuple[tuple[str, str], ...] = ()


class SourceWarning(FrozenModel):
    code: Literal[
        "page_mapping_unavailable", "images_unavailable", "truncated", "parser_warning",
    ]
    message: str


class SourceRegion(FrozenModel):
    region_id: str
    kind: RegionKind
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    section_path: tuple[str, ...] = ()
    page: None = None


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _representation_hash(source_hash: str, text: str, parser: ParserProvenance,
                         original_char_count: int, source_format: str,
                         warnings: tuple[SourceWarning, ...]) -> str:
    return _digest({"source_sha256": source_hash, "text": text,
                    "parser": parser.model_dump(mode="json"),
                    "original_char_count": original_char_count, "regions": REGION_VERSION,
                    "format": source_format, "warnings": [w.model_dump() for w in warnings]})


_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
_CAPTION = re.compile(r"^\s*(?:\*{1,2}|_{1,2})?(?:fig(?:ure)?\.?|table)\s+\d+[a-z]?\b", re.I)
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _regions(text: str, representation_hash: str, *, markdown: bool) -> tuple[SourceRegion, ...]:
    """Scan once by position; repeated text never needs ambiguous string lookup."""
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    regions = []
    headings: list[tuple[int, str]] = []
    table_lines: set[int] = set()
    row = 0
    while markdown and row + 1 < len(lines):
        cells = lines[row + 1].strip().strip("|").split("|")
        if "|" in lines[row] and len(cells) >= 2 and all(
            re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells
        ):
            end = row + 2
            while end < len(lines) and lines[end].strip() and "|" in lines[end]:
                end += 1
            table_lines.update(range(row, end))
            row = end
        else:
            row += 1
    i = 0

    def kind_at(index: int) -> RegionKind:
        if not markdown:
            return "paragraph"
        line = lines[index]
        if _FENCE.match(line):
            return "code"
        if _HEADING.match(line):
            return "heading"
        if index in table_lines:
            return "table"
        if _CAPTION.match(line):
            return "caption"
        return "paragraph"

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        begin = i
        kind = kind_at(i)
        if kind == "heading":
            match = _HEADING.match(lines[i])
            assert match is not None
            level, title = len(match[1]), match[2]
            headings = [(depth, name) for depth, name in headings if depth < level]
            headings.append((level, title))
            i += 1
        elif kind == "code":
            fence = _FENCE.match(lines[i])[1]
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                i += 1
                if len(line) >= len(fence) and set(line) == {fence[0]}:
                    break
        else:
            i += 1
            while i < len(lines) and lines[i].strip():
                following = kind_at(i)
                if (kind == "table" and following != "table") or (
                    kind != "table" and following != "paragraph"
                ):
                    break
                i += 1
        start, end = offsets[begin], offsets[i]
        section_path = tuple(name for _, name in headings)
        region_id = _digest([representation_hash, start, end, kind, section_path])
        regions.append(SourceRegion(region_id=region_id, kind=kind, start=start,
                                    end=end, section_path=section_path))
    return tuple(regions)


class DocumentBundle(FrozenModel):
    source_id: str = Field(min_length=1)
    source_sha256: Hash = Field(pattern=r"^[a-f0-9]{64}$")
    representation_sha256: Hash = Field(pattern=r"^[a-f0-9]{64}$")
    region_version: Literal["m01-regions-1"] = REGION_VERSION
    source_format: Literal["text", "markdown", "pdf"]
    parser: ParserProvenance
    text: str
    original_char_count: int = Field(ge=0, strict=True)
    regions: tuple[SourceRegion, ...]
    warnings: tuple[SourceWarning, ...]

    @model_validator(mode="after")
    def _integrity(self) -> DocumentBundle:
        if not self.text.strip() or self.original_char_count < len(self.text):
            raise ValueError("Bundle must contain nonempty text with a valid original length")
        expected = _representation_hash(
            self.source_sha256, self.text, self.parser, self.original_char_count,
            self.source_format, self.warnings,
        )
        if self.representation_sha256 != expected:
            raise ValueError("Representation hash does not match the text/provenance")
        if self.regions != _regions(self.text, expected, markdown=self.source_format != "text"):
            raise ValueError("Regions do not match the versioned text segmentation")
        if self.original_char_count > len(self.text) and not any(
            warning.code == "truncated" for warning in self.warnings
        ):
            raise ValueError("Truncated text requires an explicit warning")
        codes = {warning.code for warning in self.warnings}
        if "page_mapping_unavailable" not in codes or (
            self.source_format == "pdf" and "images_unavailable" not in codes
        ):
            raise ValueError("Missing required page/image availability warning")
        return self


def bundle_document(document: Document, *, source_id: str, source_bytes: bytes,
                    source_format: Literal["text", "markdown", "pdf"],
                    parser: ParserProvenance, max_chars: int | None = None) -> DocumentBundle:
    """Snapshot an inherited Document without changing its text or canonical output.

    The trusted caller must supply the bytes actually parsed; the file adapter in
    sources.py checks that association. This function does not discover/read files.
    """
    if max_chars is not None and (type(max_chars) is not int or max_chars < 1):
        raise ValueError("max_chars must be a positive integer or None")
    # Inherited Document is mutable; validating an existing model instance alone
    # would trust mutations/model_construct without revalidating its field data.
    document = Document.model_validate(document.model_dump() if isinstance(document, Document) else document)
    original = document.text
    text = original if max_chars is None else original[:max_chars]
    if not text.strip():
        raise EvidenceError("Parser produced no usable text")
    warnings = [SourceWarning(code="page_mapping_unavailable", message=(
        "Offsets address stored text, not PDF pages; no page mapping is available."
    ))]
    warnings.extend(SourceWarning(code="parser_warning", message=w) for w in document.warnings)
    if source_format == "pdf":
        warnings.append(SourceWarning(code="images_unavailable", message=(
            "The inherited text parser supplies no page-image evidence; image-only content may be missing."
        )))
    if len(text) < len(original):
        warnings.append(SourceWarning(code="truncated", message=(
            f"Retained {len(text)} of {len(original)} Unicode code points; the suffix is unavailable."
        )))
    source_hash = sha256(source_bytes)
    representation = _representation_hash(source_hash, text, parser, len(original),
                                          source_format, tuple(warnings))
    return DocumentBundle(
        source_id=source_id, source_sha256=source_hash, representation_sha256=representation,
        source_format=source_format, parser=parser, text=text, original_char_count=len(original),
        regions=_regions(text, representation, markdown=source_format != "text"), warnings=tuple(warnings),
    )


class EvidenceRef(FrozenModel):
    source_id: str
    source_sha256: Hash
    representation_sha256: Hash
    region_id: str
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    quotation: str = Field(min_length=1)
    page: None = None


def resolve_evidence(bundle: DocumentBundle, ref: EvidenceRef) -> str:
    if (ref.source_id, ref.source_sha256, ref.representation_sha256) != (
        bundle.source_id, bundle.source_sha256, bundle.representation_sha256,
    ):
        raise EvidenceError("Evidence belongs to a different source or representation")
    region = next((r for r in bundle.regions if r.region_id == ref.region_id), None)
    if region is None or not (region.start <= ref.start < ref.end <= region.end):
        raise EvidenceError("Unknown region or offsets outside that region")
    quotation = bundle.text[ref.start:ref.end]
    if quotation != ref.quotation:
        raise EvidenceError("Evidence quotation does not match the stored text")
    return quotation


def evidence_ref(bundle: DocumentBundle, region_id: str, *, start: int | None = None,
                 end: int | None = None) -> EvidenceRef:
    region = next((r for r in bundle.regions if r.region_id == region_id), None)
    if region is None:
        raise EvidenceError("Unknown region")
    start = region.start if start is None else start
    end = region.end if end is None else end
    ref = EvidenceRef(source_id=bundle.source_id, source_sha256=bundle.source_sha256,
                      representation_sha256=bundle.representation_sha256, region_id=region_id,
                      start=start, end=end, quotation=bundle.text[start:end])
    resolve_evidence(bundle, ref)
    return ref
