"""Resumable full-PDF parsing, using the existing PdfParser without overrides."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import pymupdf
import pymupdf4llm

from meta_data_extraction.components.parser import PdfParser


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_unchanged_or_new(path: Path, content: str) -> None:
    """Idempotent outputs, never silently replace different existing artifacts."""
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing artifact differs: {path}. Use a new output directory.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def pretty_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def parser_identity(root: Path) -> dict:
    return {
        "name": "PdfParser", "options": "repository defaults; complete PDF; no truncation",
        "pymupdf": version("pymupdf"), "pymupdf4llm": version("pymupdf4llm"),
        "layout_enabled": bool(pymupdf4llm._use_layout),
        "source_sha256": text_hash((root / "src/meta_data_extraction/components/parser.py").read_text(encoding="utf-8")),
    }


def load_cached(root: Path, output: Path, paper_id: str) -> tuple[str, dict] | None:
    text_path, meta_path = output / "text" / f"{paper_id}.md", output / "text" / f"{paper_id}.json"
    if not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8")
    if (metadata["parser"] != parser_identity(root)
            or metadata["pdf_sha256"] != file_hash(root / "data/raw" / paper_id / "paper.pdf")
            or metadata["text_sha256"] != text_hash(text)):
        raise ValueError(f"Stale or modified Markdown cache: {paper_id}")
    return text, metadata


def parse_one(root: Path, output: Path, paper_id: str) -> dict:
    if re.fullmatch(r"\d{4}\.\d{4,5}", paper_id) is None:
        raise ValueError("Expected a bare corpus arXiv ID")
    cached = load_cached(root, output, paper_id)
    if cached:
        return cached[1]
    pdf_path = root / "data/raw" / paper_id / "paper.pdf"
    pymupdf.TOOLS.mupdf_warnings(reset=True)
    document = PdfParser().parse(pdf_path)
    text = document.text
    if not text.strip():
        raise ValueError(f"No Markdown extracted for {paper_id}")
    warnings = pymupdf.TOOLS.mupdf_warnings(reset=True).strip().splitlines()
    # Diagnostic only: global word presence is not a reading-order/layout proof.
    words = lambda value: set(re.findall(r"[^\W_]{3,}", value.casefold()))
    markdown_words = words(text)
    coverage = []
    with pymupdf.open(pdf_path) as pdf:
        page_count = len(pdf)
        plain_pages = [page.get_text() for page in pdf]
    for index, plain in enumerate(plain_pages, 1):
        page_words = words(plain)
        coverage.append({"page": index, "plain_text_chars": len(plain),
                         "word_presence": round(len(page_words & markdown_words) / len(page_words), 4) if page_words else None})
    metadata = {
        "paper_id": paper_id, "pdf_sha256": file_hash(pdf_path),
        "parser": parser_identity(root), "text_sha256": text_hash(text),
        "text_chars": len(text), "pages": page_count,
        "plain_text_chars": len("\n\n".join(plain_pages)),
        "warnings": warnings, "page_word_presence": coverage,
        "low_word_presence_pages": [row["page"] for row in coverage
                                    if row["word_presence"] is not None and row["word_presence"] < 0.75],
    }
    save_unchanged_or_new(output / "text" / f"{paper_id}.md", text)
    save_unchanged_or_new(output / "text" / f"{paper_id}.json", pretty_json(metadata))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper", required=True)
    args = parser.parse_args()
    result = parse_one(args.root, args.output, args.paper)
    print(f"{args.paper}: {result['pages']} pages, {result['text_chars']} Markdown chars", flush=True)
