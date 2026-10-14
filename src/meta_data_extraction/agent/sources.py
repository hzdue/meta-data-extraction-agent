"""Prepare only explicit allowlisted source files, with inspectable failure states."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from meta_data_extraction.agent.config import SOURCE_ROOTS
from meta_data_extraction.agent.evidence import (
    DocumentBundle, EvidenceError, FrozenModel, ParserProvenance, bundle_document, sha256,
)
from meta_data_extraction.agent.paths import PathPolicyError, checked_path, workspace_root
from meta_data_extraction.data_models import Document


class PreparationFailure(FrozenModel):
    code: Literal["read_failed", "unsupported_format", "decode_failed", "parse_failed",
                  "empty_text", "source_changed"]
    message: str


class PreparationResult(FrozenModel):
    source_id: str
    source_sha256: str | None
    status: Literal["ready", "partial", "failed"]
    bundle: DocumentBundle | None = None
    error: PreparationFailure | None = None

    @model_validator(mode="after")
    def _consistent(self) -> PreparationResult:
        if self.status == "failed":
            if self.bundle is not None or self.error is None:
                raise ValueError("Failed preparation requires an error and no bundle")
        elif self.bundle is None or self.error is not None or (
            self.source_id, self.source_sha256
        ) != (self.bundle.source_id, self.bundle.source_sha256):
            raise ValueError("Successful preparation requires a matching bundle and no error")
        elif self.status != ("partial" if any(
            w.code in {"truncated", "parser_warning", "images_unavailable"}
            for w in self.bundle.warnings
        ) else "ready"):
            raise ValueError("Preparation status must reflect bundle incompleteness")
        return self


def prepare_sources(root: Path | str, sources: tuple[str, ...] | list[str], *,
                    max_chars: int | None = None) -> tuple[PreparationResult, ...]:
    """No discovery, downloads, source instructions, model calls, or gold loading.

    Policy violations raise before reading any file. Per-file I/O/parser failures
    remain in the returned result list and cannot become empty successful bundles.
    """
    root = workspace_root(root)
    if max_chars is not None and (type(max_chars) is not int or max_chars < 1):
        raise ValueError("max_chars must be a positive integer or None")
    if not sources:
        raise ValueError("Supply at least one source file")
    paths = tuple(checked_path(root, source, allowed_roots=SOURCE_ROOTS) for source in sources)
    if len(set(paths)) != len(paths):
        raise ValueError("Supply each source path only once")
    if any(path.is_dir() for path in paths):
        raise PathPolicyError("Supply individual files, never directories")
    return tuple(_prepare(path, root, max_chars) for path in paths)


def _prepare(path: Path, root: Path, max_chars: int | None) -> PreparationResult:
    source_id = path.relative_to(root).as_posix()
    source_hash = None

    def failed(code, message):
        return PreparationResult(source_id=source_id, source_sha256=source_hash, status="failed",
                                 error=PreparationFailure(code=code, message=message))

    try:
        # Recheck the allowlist/links immediately before opening each source.
        path = checked_path(root, source_id, allowed_roots=SOURCE_ROOTS)
        raw = path.read_bytes()
    except OSError as exc:
        return failed("read_failed", f"Could not read supplied source ({type(exc).__name__})")
    source_hash = sha256(raw)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        source_format = "text" if suffix == ".txt" else "markdown"
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return failed("decode_failed", "Text input is not valid UTF-8; no replacement decoding used")
        document = Document(id=source_id, text=text, source_format=source_format,
                            source_path=str(path), parser_name="utf8")
        parser = ParserProvenance(name="utf8", version="1", options=(("encoding", "utf-8"),))
    elif suffix == ".pdf":
        source_format = "pdf"
        try:
            # Text-only use does not import the optional heavy PDF stack.
            from meta_data_extraction.components.parser import PdfParser

            document = PdfParser().parse(path)
            parser = ParserProvenance(name="PdfParser/pymupdf4llm", version=version("pymupdf4llm"),
                                      options=(("adapter", "inherited-defaults-m01-1"),
                                               ("pymupdf_version", version("pymupdf"))))
            checked_path(root, source_id, allowed_roots=SOURCE_ROOTS)
            if path.read_bytes() != raw:
                return failed("source_changed", "Source bytes changed during PDF parsing")
        except Exception as exc:
            return failed("parse_failed", f"PDF parser failed ({type(exc).__name__}); output unavailable")
    else:
        return failed("unsupported_format", "Supported inputs are UTF-8 .txt/.md/.markdown or .pdf")
    try:
        document = Document.model_validate(document)
        bundle = bundle_document(document, source_id=source_id, source_bytes=raw,
                                 source_format=source_format, parser=parser, max_chars=max_chars)
    except EvidenceError:
        return failed("empty_text", "Parser produced no usable text within the selected character limit")
    except (ValueError, TypeError) as exc:
        return failed("parse_failed", f"Invalid parser output ({type(exc).__name__})")
    partial = any(w.code in {"truncated", "parser_warning", "images_unavailable"} for w in bundle.warnings)
    return PreparationResult(source_id=source_id, source_sha256=source_hash,
                             status="partial" if partial else "ready", bundle=bundle)
