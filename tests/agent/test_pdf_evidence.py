"""Actual offline conversion of a generated PDF, not a benchmark publication."""
import hashlib

import pymupdf

from meta_data_extraction.agent.retrieval import SourceCorpus
from meta_data_extraction.agent.sources import prepare_sources


def test_generated_pdf_converts_to_resolvable_text_evidence(workspace):
    relative = "tests/agent/fixtures/generated.pdf"
    path = workspace / relative
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Synthetic M01 fixture\n\nFigure 1: Sample A was measured at 80 K.")
        pdf.save(path)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    result, = prepare_sources(workspace, [relative])
    assert result.status == "partial"  # text exists; page/image evidence is unavailable
    assert result.source_sha256 == source_hash
    assert result.bundle.parser.name == "PdfParser/pymupdf4llm"
    assert dict(result.bundle.parser.options)["pymupdf_version"]
    assert "80 K" in result.bundle.text
    corpus = SourceCorpus(bundles=(result.bundle,))
    selected = corpus.search("80 K", max_chars=1000, limit=1)
    assert "80 K" in corpus.resolve(selected.passages[0].evidence)
    assert selected.passages[0].evidence.page is None
    assert selected.incomplete_sources == (relative,)
