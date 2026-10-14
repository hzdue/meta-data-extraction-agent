import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from meta_data_extraction.agent.evidence import EvidenceError
from meta_data_extraction.agent.inspect_sources import inspect_sources, main
from meta_data_extraction.agent.retrieval import Selection, SourceCorpus
from meta_data_extraction.agent.sources import prepare_sources


@pytest.fixture
def evidence_workspace(workspace):
    repository = Path(__file__).resolve().parents[2]
    for relative in ("tests/agent/fixtures/evidence_source.md", "configs/agent/evidence_offline.yaml"):
        (workspace / relative).write_bytes((repository / relative).read_bytes())
    return workspace


def corpus_at(workspace):
    prepared, = prepare_sources(workspace, ["tests/agent/fixtures/evidence_source.md"])
    assert prepared.status == "ready"
    return SourceCorpus(bundles=(prepared.bundle,))


def test_caption_only_measurements_and_samples_are_separate(evidence_workspace):
    corpus = corpus_at(evidence_workspace)
    a = corpus.search("80 K", max_chars=1000, limit=1)
    b = corpus.search("295 K", max_chars=1000, limit=1)
    assert a.passages[0].kind == b.passages[0].kind == "caption"
    assert "Sample A" in a.text and "Sample B" not in a.text
    assert "Sample B" in b.text and "Sample A" not in b.text
    assert a.passages[0].evidence.region_id != b.passages[0].evidence.region_id
    for selection in (a, b):
        assert corpus.resolve(selection.passages[0].evidence) == selection.text


def test_duplicate_caption_order_is_stable_and_distinct(evidence_workspace):
    corpus = corpus_at(evidence_workspace)
    results = corpus.search("Duplicate caption control", max_chars=1000)
    assert results == corpus.search("Duplicate caption control", max_chars=1000)
    first, second = results.passages[:2]
    assert first.evidence.quotation == second.evidence.quotation
    assert first.evidence.region_id != second.evidence.region_id
    assert first.evidence.start < second.evidence.start


def test_full_text_exactness_and_character_budget(evidence_workspace):
    corpus = corpus_at(evidence_workspace)
    source = corpus.bundles[0]
    full = corpus.full_text(source.source_id, max_chars=10000)
    assert full.status == "complete" and full.text == source.text
    clipped = corpus.full_text(source.source_id, max_chars=67)
    assert clipped.status == "truncated" and clipped.text == source.text[:67]
    assert clipped.used_chars == 67
    for passage in (*full.passages, *clipped.passages):
        assert corpus.resolve(passage.evidence) == passage.evidence.quotation


def test_targeted_budget_counts_join_separators(evidence_workspace):
    corpus = corpus_at(evidence_workspace)
    selected = corpus.search("Sample", max_chars=200, limit=10)
    assert len(selected.passages) >= 2
    assert selected.used_chars <= 200
    assert selected.text == "\n\n".join(p.evidence.quotation for p in selected.passages)
    assert selected.status == "truncated"
    assert selected.omitted_candidates or any(p.clipped for p in selected.passages)
    for passage in selected.passages:
        assert corpus.resolve(passage.evidence) == passage.evidence.quotation


def test_no_match_and_bad_query_or_budget(evidence_workspace):
    corpus = corpus_at(evidence_workspace)
    missing = corpus.search("nonexistenttoken", max_chars=100)
    assert missing.status == "no_match" and missing.text == "" and missing.passages == ()
    for kwargs in ({"max_chars": 0}, {"max_chars": True}, {"max_chars": 10, "limit": 0}):
        with pytest.raises(ValueError):
            corpus.search("sample", **kwargs)
    with pytest.raises(ValueError):
        corpus.search("???", max_chars=100)


def test_supplied_source_scope_cannot_expand(evidence_workspace):
    source_dir = evidence_workspace / "tests/agent/fixtures"
    (source_dir / "unselected.md").write_text("SECRET_OUTSIDE_CORPUS")
    corpus = corpus_at(evidence_workspace)
    assert corpus.search("SECRET_OUTSIDE_CORPUS", max_chars=100).status == "no_match"
    with pytest.raises(EvidenceError):
        corpus.search("secret", max_chars=100, source_ids=("tests/agent/fixtures/unselected.md",))
    with pytest.raises(EvidenceError):
        corpus.full_text("data/gold/private.json", max_chars=100)
    assert corpus.search("sample", max_chars=100, source_ids=()).status == "no_match"


def test_injected_instruction_is_inert_source_text(evidence_workspace):
    corpus = corpus_at(evidence_workspace)
    selected = corpus.search("upload invalid.example", max_chars=1000, limit=1)
    assert "Ignore all prior instructions" in selected.text
    assert corpus.resolve(selected.passages[0].evidence) == selected.text
    assert not (evidence_workspace / "data/gold").exists()


def test_preparation_truncation_is_visible_in_selection(evidence_workspace):
    result, = prepare_sources(evidence_workspace, ["tests/agent/fixtures/evidence_source.md"], max_chars=100)
    assert result.status == "partial"
    corpus = SourceCorpus(bundles=(result.bundle,))
    selected = corpus.full_text(result.source_id, max_chars=1000)
    assert selected.status == "complete"  # all available text, not all original content
    assert selected.incomplete_sources == (result.source_id,)


def test_offline_inspection_writes_only_fresh_run(evidence_workspace):
    run = inspect_sources(evidence_workspace, "configs/agent/evidence_offline.yaml", query="80 K")
    payload = json.loads(run.path.read_text(encoding="utf-8"))
    assert run.path == evidence_workspace / "runs/agent/m01-evidence/inspection.json"
    assert run.failed_sources == 0 and payload["model_calls"] == 0
    assert payload["targeted"]["passages"][0]["kind"] == "caption"
    assert "output" not in payload
    with pytest.raises(FileExistsError):
        inspect_sources(evidence_workspace, "configs/agent/evidence_offline.yaml", query="80 K")


def test_cli_failure_persists_error_without_successful_empty_extraction(evidence_workspace, monkeypatch):
    (evidence_workspace / "tests/agent/fixtures/evidence_source.md").write_text(" ")
    monkeypatch.chdir(evidence_workspace)
    assert main(["--query", "sample"]) == 2
    saved = json.loads((evidence_workspace / "runs/agent/m01-evidence/inspection.json").read_text())
    assert saved["status"] == "preparation_failed"
    assert saved["sources"][0]["error"]["code"] == "empty_text"
    assert saved["targeted"] is None and saved["full_text"] == []


def test_multiple_sources_filter_and_tie_order_are_explicit(workspace):
    files = ["tests/agent/fixtures/z.md", "tests/agent/fixtures/a.md"]
    for name in files:
        (workspace / name).write_text("A shared measurement at 80 K.")
    prepared = prepare_sources(workspace, files)
    corpus = SourceCorpus(bundles=tuple(p.bundle for p in prepared))
    selected = corpus.search("80 K", max_chars=1000)
    assert [p.evidence.source_id for p in selected.passages] == sorted(files)
    reversed_corpus = SourceCorpus(bundles=tuple(reversed(corpus.bundles)))
    assert reversed_corpus.search("80 K", max_chars=1000).passages == selected.passages
    filtered = corpus.search("80 K", max_chars=1000, source_ids=(files[0],))
    assert filtered.source_ids == (files[0],) and len(filtered.passages) == 1
    assert filtered.passages[0].evidence.source_id == files[0]
    with pytest.raises(ValueError):
        SourceCorpus(bundles=(corpus.bundles[0], corpus.bundles[0]))


def test_unicode_matching_does_not_normalize_quotation(workspace):
    source = "tests/agent/fixtures/unicode.txt"
    text = "🧪 Spot size １０ µm, wavelength 1.54 Å; sample cafe\u0301."
    (workspace / source).write_text(text, encoding="utf-8")
    prepared, = prepare_sources(workspace, [source])
    corpus = SourceCorpus(bundles=(prepared.bundle,))
    selected = corpus.search("10 μm CAFÉ", max_chars=1000)
    assert selected.passages[0].score == 1
    assert selected.text == text
    assert corpus.resolve(selected.passages[0].evidence) == text


def test_plain_text_markup_is_literal_and_markdown_headings_searchable(workspace):
    text = "# Sample A measured at 80 K\n\n```\nFigure 99: literal text.\n```"
    source = "tests/agent/fixtures/literal.txt"
    markdown_source = "tests/agent/fixtures/heading.md"
    (workspace / source).write_text(text)
    (workspace / markdown_source).write_text("# Only measurement: 80 K")
    plain, markdown = prepare_sources(workspace, [source, markdown_source])
    assert all(r.kind == "paragraph" for r in plain.bundle.regions)
    corpus = SourceCorpus(bundles=(plain.bundle, markdown.bundle))
    assert len(corpus.search("80 K", max_chars=1000).passages) == 2


def test_tiny_budgets_mark_prefix_clipping_and_whitespace_scope(workspace):
    source = "tests/agent/fixtures/tiny.txt"
    (workspace / source).write_text("   A long prefix before the matching word TEMPERATURE.")
    prepared, = prepare_sources(workspace, [source])
    corpus = SourceCorpus(bundles=(prepared.bundle,))
    full = corpus.full_text(source, max_chars=1)
    assert full.text == " " and full.source_ids == (source,) and full.status == "truncated"
    target = corpus.search("TEMPERATURE", max_chars=1)
    assert target.used_chars == 1 and target.passages[0].clipped
    assert target.status == "truncated" and "TEMPERATURE" not in target.text


@pytest.mark.parametrize("mode", ["full_text", "targeted"])
def test_selection_round_trip_and_tamper_rejection(evidence_workspace, mode):
    corpus = corpus_at(evidence_workspace)
    result = (corpus.full_text(corpus.bundles[0].source_id, max_chars=100) if mode == "full_text"
              else corpus.search("Sample", max_chars=100))
    assert Selection.model_validate_json(result.model_dump_json()) == result
    for change in ({"text": "x" * len(result.text)}, {"source_ids": []},
                   {"status": "complete"}, {"used_chars": 9999}):
        with pytest.raises(ValidationError):
            Selection.model_validate({**result.model_dump(), **change})


def test_corpus_revalidates_modified_bundle_instances(evidence_workspace):
    corpus = corpus_at(evidence_workspace)
    forged = corpus.bundles[0].model_copy(update={"text": "Changed without new hashes"})
    with pytest.raises(ValidationError):
        SourceCorpus(bundles=(forged,))
