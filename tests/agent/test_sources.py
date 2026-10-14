"""Offline source preparation tests; every PDF parser call is a synthetic stub."""
import builtins
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from meta_data_extraction.agent.paths import PathPolicyError
from meta_data_extraction.agent.sources import PreparationResult, prepare_sources
from meta_data_extraction.data_models import Document


def write_source(workspace, name, raw):
    relative = f"tests/agent/fixtures/{name}"
    (workspace / relative).write_bytes(raw)
    return relative


@pytest.fixture
def pdf_stub(workspace, monkeypatch):
    # Importing the inherited class is safe: parse itself is always replaced.
    from meta_data_extraction.components.parser import PdfParser
    import meta_data_extraction.agent.sources as sources

    relative = write_source(workspace, "synthetic.pdf", b"%PDF-synthetic-fixture-only\n")
    monkeypatch.setattr(sources, "version", lambda name: "fixture-pdf-version")

    def install(result):
        calls = []

        def parse(self, path):
            calls.append(path)
            if isinstance(result, Exception):
                raise result
            return result(path) if callable(result) else result

        monkeypatch.setattr(PdfParser, "parse", parse)
        return relative, calls

    return install


def test_preparation_reads_only_explicit_files(workspace, monkeypatch):
    chosen = write_source(workspace, "chosen.md", b"Selected source.\n")
    write_source(workspace, "unselected.md", b"Unselected source must stay unread.\n")
    original_read = Path.read_bytes
    reads = []

    def checked_read(path):
        reads.append(path)
        assert path == workspace / chosen
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", checked_read)
    results = prepare_sources(workspace, [chosen])
    assert reads == [workspace / chosen]
    assert len(results) == 1
    assert results[0].source_id == chosen
    assert results[0].bundle.text == "Selected source.\n"


@pytest.mark.parametrize("source", [
    "data/gold/never-read.json", "data/annotations/never-read.json",
    "data/experiment/never-read.json", "configs/alignment_helper.yaml",
    "outputs/evaluation/never-read.json", "data/raw/*.md", ".",
    "tests/agent/fixtures", "tests/agent/fixtures/../outside.md",
])
def test_policy_rejects_whole_batch_before_source_reads(workspace, monkeypatch, source):
    def denied_read(path):
        pytest.fail("A source was read before the complete selection passed policy")

    monkeypatch.setattr(Path, "read_bytes", denied_read)
    with pytest.raises(PathPolicyError):
        prepare_sources(workspace, ["tests/agent/fixtures/synthetic_source.md", source])


def test_empty_and_duplicate_selection_fail(workspace):
    with pytest.raises(ValueError, match="at least one"):
        prepare_sources(workspace, [])
    source = "tests/agent/fixtures/synthetic_source.md"
    with pytest.raises(ValueError, match="only once"):
        prepare_sources(workspace, [source, source.replace("/", "\\")])


def test_missing_source_is_retained_as_failure_next_to_success(workspace):
    missing = "tests/agent/fixtures/missing.md"
    present = write_source(workspace, "present.md", b"Available source.")
    failure, success = prepare_sources(workspace, [missing, present])
    assert failure.source_id == missing
    assert failure.status == "failed"
    assert failure.source_sha256 is None
    assert failure.bundle is None
    assert failure.error.code == "read_failed"
    assert success.status == "ready"
    assert success.bundle.text == "Available source."


@pytest.mark.parametrize("name,raw,code", [
    ("unknown.csv", b"sample,value\na,1", "unsupported_format"),
    ("invalid.md", b"temperature \xff K", "decode_failed"),
    ("empty.txt", b"", "empty_text"),
    ("whitespace.md", b" \r\n\t", "empty_text"),
])
def test_bad_input_has_explicit_error_and_no_bundle(workspace, name, raw, code):
    source = write_source(workspace, name, raw)
    result, = prepare_sources(workspace, [source])
    assert result.status == "failed"
    assert result.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.bundle is None
    assert result.error.code == code


@pytest.mark.parametrize("name,expected_format", [
    ("exact.TXT", "text"), ("exact.md", "markdown"), ("exact.markdown", "markdown"),
])
def test_raw_hash_and_unicode_crlf_text_are_preserved(workspace, name, expected_format):
    text = "# μ diffraction\r\n\r\nλ = 1.54 Å; temperature 300 K 🧪.\r\n"
    raw = text.encode("utf-8")
    source = write_source(workspace, name, raw)
    result, = prepare_sources(workspace, [source])
    assert result.status == "ready"
    assert result.bundle.text == text
    assert result.bundle.original_char_count == len(text)
    assert result.bundle.source_format == expected_format
    assert result.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.bundle.source_sha256 == result.source_sha256
    assert result.source_sha256 != hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest()
    assert result.bundle.parser.name == "utf8"
    assert all(region.page is None for region in result.bundle.regions)
    assert "page_mapping_unavailable" in {w.code for w in result.bundle.warnings}


def test_text_preparation_does_not_import_pdf_stack(workspace, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name not in {"meta_data_extraction.components.parser", "pymupdf4llm"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result, = prepare_sources(workspace, ["tests/agent/fixtures/synthetic_source.md"])
    assert result.status == "ready"


def test_unavailable_pdf_dependency_is_recorded_as_failure(workspace, monkeypatch):
    source = write_source(workspace, "unavailable.pdf", b"%PDF-fixture-with-no-parser\n")
    original_import = builtins.__import__

    def unavailable_import(name, *args, **kwargs):
        if name == "meta_data_extraction.components.parser":
            raise ImportError("synthetic missing parser dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable_import)
    result, = prepare_sources(workspace, [source])
    assert result.status == "failed"
    assert result.bundle is None
    assert result.error.code == "parse_failed"
    assert "ImportError" in result.error.message


def test_character_limit_is_explicit_and_does_not_change_source_hash(workspace):
    text = "μ 🧪 300 K\nThe remaining source is unavailable."
    source = write_source(workspace, "limited.md", text.encode("utf-8"))
    full, = prepare_sources(workspace, [source])
    limited, = prepare_sources(workspace, [source], max_chars=7)
    assert limited.status == "partial"
    assert limited.bundle.text == text[:7]
    assert limited.bundle.original_char_count == len(text)
    assert limited.source_sha256 == full.source_sha256
    assert limited.bundle.representation_sha256 != full.bundle.representation_sha256
    assert "truncated" in {w.code for w in limited.bundle.warnings}
    assert all(region.end <= 7 for region in limited.bundle.regions)
    unchanged, = prepare_sources(workspace, [source], max_chars=len(text))
    assert unchanged.status == "ready"
    assert unchanged.bundle == full.bundle


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "7"])
def test_character_limit_requires_positive_integer(workspace, limit):
    with pytest.raises(ValueError, match="positive integer"):
        prepare_sources(workspace, ["tests/agent/fixtures/synthetic_source.md"], max_chars=limit)


def test_limit_that_retains_only_whitespace_is_not_empty_success(workspace):
    source = write_source(workspace, "leading-spaces.md", b"   Visible after whitespace.")
    result, = prepare_sources(workspace, [source], max_chars=3)
    assert result.status == "failed"
    assert result.error.code == "empty_text"
    assert result.bundle is None


def test_pdf_adapter_preserves_inherited_text_and_warns_about_unavailable_content(workspace, pdf_stub):
    text = "# Results\n\nFigure 2. Sample A was measured at 300 K.\n"
    inherited = Document(id="inherited-id", text=text, warnings=["Missing font information"])
    source, calls = pdf_stub(inherited)
    result, = prepare_sources(workspace, [source])
    assert calls == [workspace / source]
    assert result.status == "partial"
    assert result.source_id == source
    assert result.bundle.text == text
    assert result.bundle.source_format == "pdf"
    assert result.bundle.parser.name == "PdfParser/pymupdf4llm"
    assert result.bundle.parser.version == "fixture-pdf-version"
    warnings = {warning.code: warning.message for warning in result.bundle.warnings}
    assert warnings["parser_warning"] == "Missing font information"
    assert {"images_unavailable", "page_mapping_unavailable"} <= warnings.keys()
    assert all(region.page is None for region in result.bundle.regions)
    assert inherited.model_dump() == Document(
        id="inherited-id", text=text, warnings=["Missing font information"],
    ).model_dump()


def test_pdf_parser_exception_becomes_redacted_failure(workspace, pdf_stub):
    source, calls = pdf_stub(RuntimeError("synthetic sensitive diagnostic"))
    result, = prepare_sources(workspace, [source])
    assert calls == [workspace / source]
    assert result.status == "failed"
    assert result.bundle is None
    assert result.error.code == "parse_failed"
    assert "RuntimeError" in result.error.message
    assert "synthetic sensitive diagnostic" not in result.model_dump_json()


@pytest.mark.parametrize("output,expected_code", [
    (Document(id="empty", text=""), "empty_text"),
    (Document(id="spaces", text=" \n"), "empty_text"),
    (None, "parse_failed"),
    ({"id": "missing-text"}, "parse_failed"),
    ({"id": "invalid-text", "text": None}, "parse_failed"),
    (Document.model_construct(id="invalid-instance", text=None), "parse_failed"),
], ids=["empty", "whitespace", "none", "missing-field", "invalid-field", "invalid-instance"])
def test_invalid_pdf_output_is_never_successful_empty_bundle(workspace, pdf_stub, output, expected_code):
    source, calls = pdf_stub(output)
    result, = prepare_sources(workspace, [source])
    assert calls == [workspace / source]
    assert result.status == "failed"
    assert result.bundle is None
    assert result.error.code == expected_code


def test_source_change_during_pdf_parse_is_explicit_failure(workspace, pdf_stub):
    original_hash = hashlib.sha256(b"%PDF-synthetic-fixture-only\n").hexdigest()

    def change_source(path):
        path.write_bytes(b"%PDF-changed-fixture-only\n")
        return Document(id="changed", text="Parser text from an unstable source.")

    source, calls = pdf_stub(change_source)
    result, = prepare_sources(workspace, [source])
    assert calls == [workspace / source]
    assert result.status == "failed"
    assert result.source_sha256 == original_hash
    assert result.bundle is None
    assert result.error.code == "source_changed"


def test_replayed_result_cannot_hide_partial_status_or_warnings(workspace):
    source = write_source(workspace, "replay.md", b"Full source text that is explicitly truncated.")
    result, = prepare_sources(workspace, [source], max_chars=5)
    assert PreparationResult.model_validate_json(result.model_dump_json()) == result
    hidden_status = result.model_dump()
    hidden_status["status"] = "ready"
    with pytest.raises(ValidationError):
        PreparationResult.model_validate(hidden_status)
    hidden_warning = result.model_dump()
    hidden_warning["bundle"]["warnings"] = ()
    with pytest.raises(ValidationError):
        PreparationResult.model_validate(hidden_warning)
