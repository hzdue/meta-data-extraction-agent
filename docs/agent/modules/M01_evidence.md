# M01 — Source preparation and evidence retrieval

Date: 2026-10-07. Scope: M01 only; no model calls or controller implementation.

## Question, example and baseline

Can a quotation resolve to the exact supplied source region, and can simple
targeted retrieval find it? For example, a temperature present only in Figure 2's
caption must resolve to that caption, without merging two same-composition samples.

Reuse M00 allowlisted source paths and isolated runs, inherited `Document`, and
the unchanged `PdfParser` for explicitly selected PDFs. Existing Section objects
strip whitespace and have no offsets; do not use text search to reconstruct their
positions. Initial safe baseline: 84 M00 tests plus 13 inherited section/schema
tests passed (97 total). The documented explicit `PYTHONPATH` workaround is still
needed; the standard command runner still fails to initialize.

## Minimal contract and design

- `DocumentBundle`: immutable source ID/path, raw-byte hash, parser provenance,
  representation hash, exact text, deterministic heading/paragraph/caption/table/
  fenced-code regions, and explicit warnings. Offsets are Python Unicode code
  points, half-open `[start, end)`, in the stored representation, never PDF bytes.
- `EvidenceRef`: source and representation hashes, source/region IDs, absolute
  offsets and exact quotation. Resolution rejects unknown/stale/mismatched refs.
  M01 has no page mapping; page fields cannot invent a location.
- Source preparation: explicit files only; UTF-8 text/Markdown or inherited PDF
  adapter. Unreadable/unsupported/invalid/empty outputs are explicit failures.
  Optional character truncation is recorded, never silently called complete.
- Retrieval: a supplied-bundle-only corpus; full-text baseline and deterministic
  lexical region ranking, bounded by characters and result count. Return explicit
  no-match/truncation status and resolvable evidence for selected text. Source
  instructions/links are inert text, never tools or downloads.

Trade-off: offset-preserving Markdown heuristics and character budgets are small
and reproducible, but do not establish semantic relevance, model-token costs,
layout recovery or evidence recall. No embedding service or LangGraph is needed.

## Required verification

Offline cases: same-composition samples with different conditions, caption-only
measurement, duplicate captions, empty/broken parser output, Unicode/CRLF exact
offsets, source-edit invalidation, injected instructions, supplied-source isolation,
serialization/tampering checks, budgets/no-match behavior, and synthetic PDF
adapter checks. Re-run the agent suite and affected inherited parser/schema tests;
verify protected assets and baseline hashes unchanged.

Implementation: ready. Offline tests: pass. Integration: pass.
Empirical E01: not_run; a reviewed question/evidence set and experiment protocol
are not supplied. ChatAI approval remains recorded; M01 needs no inference.

Next dependency after this module: M02 typed extraction/provider boundary.

## Implemented interfaces and interpretation

- `sources.prepare_sources(root, explicit_paths, max_chars=None)` returns one
  `PreparationResult` per source, preserving failures. Policy violations reject
  the whole selection before reads. Text uses exact UTF-8 decoding; PDF conversion
  lazily uses inherited `PdfParser`, recording PyMuPDF4LLM and PyMuPDF versions,
  and rejects a source changed during parsing. The mutable inherited Document is
  revalidated before taking an immutable snapshot.
- `evidence.bundle_document(...)` is the trusted parsed-Document adapter. It
  requires the actual source bytes; the file adapter establishes that association.
  Raw hashes identify the file; representation hashes additionally bind exact
  text, parser/options, source format, original length, warnings and region
  algorithm `m01-regions-1`. Saved snapshots reject altered text/regions/warnings.
- `evidence_ref(bundle, region_id, start=..., end=...)` creates a whole-region or
  subspan reference; `resolve_evidence(bundle, ref)` verifies identity, hashes,
  bounds and exact quotation. All offsets use the immutable parsed text. A source
  edit, even outside the quotation, invalidates the old reference. No entity
  assertions or canonical record revisions are created in M01; those are M02+.
- `SourceCorpus(bundles=...)` revalidates immutable snapshots; `full_text` selects
  an exact per-source prefix, while `search` scores query-term overlap over regions
  and breaks ties by source ID then offset. NFKC/casefold affects matching only.
  Source filters only narrow the corpus. Selected text, evidence and counts are
  validated on serialization/reload. Scores are lexical overlap, not confidence.

Plain `.txt` is split into paragraphs. Markdown and PDF-derived Markdown also
recognize ATX headings, numbered Figure/Fig./Table captions, pipe tables with or
without outer pipes, and fenced code. Headings are both searchable regions and
section context. HTML/layout/unnumbered captions are not specialized regions.
Duplicate text is distinguished by position; samples are never deduplicated or
attached to measurements by this module.

`complete` means the selection includes the available representation/candidates,
not that the publication was fully parsed or its facts exhaustively recovered.
`incomplete_sources` reports source truncation, parser warnings and PDF image
limitations. Top-k or character omissions are marked `truncated`. The character
budget includes targeted join separators. Prefix clipping can omit query words
near a long region's end; the overlap score describes the original region and
`clipped` makes this visible. A no-match result is not evidence of factual absence.

## Worked example and checks

The invented source has two Fe₂O₃ samples, measured separately at 80 K and 295 K.
Query `80 K` selects the Figure 2 caption about Sample A. The recorded quotation
is the exact 64-code-point span `[293, 357)` in the current fixture representation,
under its source and representation hashes, with `page: null`. The duplicate
Figure 4 captions retain separate region IDs. These are synthetic diagnostics,
not expert support judgments or measured extraction accuracy.

Executed with current-checkout `PYTHONPATH` and `PYTHONDONTWRITEBYTECODE=1`:

```text
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/test_sections.py tests/test_utils.py -q
177 passed in 2.55s

uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_sources --config configs/agent/evidence_offline.yaml --run-id m01-20261007-check --query "80 K" --max-chars 160 --limit 1
Created runs/agent/m01-20261007-check/inspection.json; prepared; model_calls=0.
```

The 177 final checks comprise 84 existing M00 tests, 80 new M01 cases and 13
inherited regressions. The generated-PDF integration uses an invented one-page
file in pytest temporary storage and actually runs the inherited converter.
Other PDF success/error cases are mocked. The initial baseline passed 97 tests.
During development one focused test caught an invalid mutable Document escaping
validation (36 passed, 1 failed); the boundary was fixed. A subsequent 156-test
agent run passed before the final plain-text/replay/multi-source/PDF checks.
Independent read-only reviews informed the table, text-mode and replay checks.

Final integrity checks: `git diff --check` passed; `git diff --exit-code HEAD --
data outputs paper schemas configs pyproject.toml uv.lock
src/meta_data_extraction/pipeline.py src/meta_data_extraction/components
src/meta_data_extraction/evaluation.py src/meta_data_extraction/alignment.py
src/meta_data_extraction/normalization.py` found no protected tracked changes.
`git check-ignore -v runs/agent/m01-20261007-check/inspection.json` matched the
existing `/runs/agent/` rule. All 15 M00 audit fingerprints and nine per-task
snapshots (including AGENTS.md, dependencies, parser and M00 implementation) were
unchanged. Fourteen M01-authored/updated files were checked for trailing whitespace
and secret patterns, with no findings. Git still warns about inaccessible global
ignore settings and future LF/CRLF conversion; those are environment advisories.
Incidental Windows tool cache files were preserved in ignored
`tmp/m01-host-cache/` after verifying source and destination stay in this workspace.

No live inference, `.env` access, downloads, benchmark regeneration, final-test
gold inspection or released-output replay occurred. No new dependencies or
configured lint/type checkers were added. Full environment sync remains unverified
for the M00 reasons; the existing environment and explicit source binding work.

## Changed files and decision

New implementation: `src/meta_data_extraction/agent/evidence.py`, `sources.py`,
`retrieval.py`, `inspect_sources.py`. New configuration/fixture:
`configs/agent/evidence_offline.yaml`, `tests/agent/fixtures/evidence_source.md`.
New tests: `tests/agent/test_evidence.py`, `test_sources.py`, `test_retrieval.py`,
`test_pdf_evidence.py`. Documentation: this record, `docs/agent/PROGRESS.md`,
`COMMANDS.md`, and `DATA_POLICY.md`. No inherited source, canonical schema, package
identity, dependency lock, M00 implementation or live-permission guidance changed.

Keep the minimal deterministic implementation. E01 remains `not_run`: reviewed
labels/protocol and publication-level evidence recall are absent. Real-paper
PDF/OCR/layout performance, exact model-token budgets, standards lookup, live
provider/rate-limiter behavior and semantic support judgments remain untested or
outside M01. The inherited PDF parser supplies no reliable page/image mapping.
Application path checks are not protection against adversarial filesystem races
or deliberately mislabeled gold copied into an allowed source directory.

Proceed next with M02 only when requested: fixed extraction, provider boundary,
explicit model/parse/usage failure states, and evidence-linked export. The standing
ChatAI approval must then be implemented with shared persistent request limiting
before any live test; it is unused by M01.
