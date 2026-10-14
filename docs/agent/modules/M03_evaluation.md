# M03 — Offline experiment harness and frozen evaluation

Date: 2026-10-07. Scope: M03 only; no runtime tuning, new inference or M04.

## Question, interfaces and baseline

Can fixed extraction outputs be scored and compared without changing the target,
reading gold at runtime, or overwriting released results? Inputs are an explicit
publication/split manifest, hashed predictions and separately supplied gold, plus
a frozen evaluator snapshot. Outputs are per-publication reports, separate entity,
conditional-attribute and reference rows with support, failure/resource summaries,
and a parity result against explicitly supplied released reports.

Starting architecture: M00–M02 isolate inputs, source/evidence snapshots, fixed
extraction and immutable run storage. The inherited `evaluation.evaluate()` reads
files and returns a report, while its batch script replaces released outputs.
Reuse the pure evaluator and `aggregate_reports`, preserving helper/config/schema
semantics. Add a separate `meta_data_extraction.offline` package; runtime imports
must never reach it. Reuse the existing checked paths and exclusive run storage.

Minimal design: typed split and artifact declarations, content/version pinning,
one explicit record per publication group and variant, deterministic scoring,
safe completed replay and a comparison table. Do not pool chunks as papers or
invent missing correspondences. Preserve diagnostics/raw evaluator reports;
block metrics explicitly requiring new adjudication. No schema-wide average.

Baseline is direct execution of the unchanged inherited evaluator. Required
offline tests include exact synthetic parity, malformed/failed/unprocessed inputs,
missing/unknown usage, zero-support attributes, duplicate publication/derivative
leakage, changed frozen resources, runtime import separation, path safety and
repeat execution without replacing reports.

## E03 protocol frozen before scoring

The user's instruction to continue from M02 authorizes this M03 offline replay.
Selection uses existing corpus/condition metadata, before reading metric values
or gold/prediction contents. There are no released calibration predictions.

- Publication: **1804.07163**, the first sorted historical evaluation publication.
- Variants: **pan_gpt4o_full_0shot** and **pan_gpt4o_no_inst_0shot**. Same historical
  base model and zero-shot setting; the purpose is scorer reproducibility, not a
  new claim about prompt superiority from one publication.
- Inputs: `data/gold/1804.07163.json`; the two matching files under
  `data/experiment/<variant>/1804.07163.json`; matching released reports under
  `outputs/evaluation/<variant>/1804.07163_evaluation.json`.
- Split: `historical`, natural data, one publication analysis unit. Both variants
  and all declared versions/derivatives belong to that group. The inherited corpus
  has ten calibration and thirty historical evaluation papers. This replay does
  not establish an untouched future test set; future-held-out gold stays gated.
- Primary outcome: exact recursive equality of semantic evaluator report content
  to the corresponding released report, with only path provenance differences
  excluded. No rounding/tolerance or favorable subset of metrics.
- Freeze evaluator/helper versions **1.1.3**, schema versions **1.1.0**, raw code,
  config, schema, dependency and input hashes before evaluation. Store the actual
  frozen snapshots in the new run. A changed version/hash blocks execution.
- Budget: two offline record evaluations, **zero inference/network requests**.
  Do not execute result-regeneration scripts. Read gold only inside this offline
  scoring step; do not inspect contents in conversation or alter runtime based on
  their judgments. No runtime outputs will be tuned from these results.
- Output: fresh ignored `runs/agent/m03-20261007-historical-v1/`, with synthetic
  engineering cases reported separately. Cache replay is distinct from scoring.

Main trade-off: preserving the evaluator ensures historical comparability, but
its paper-specific identity helpers do not establish coverage on unseen papers.
Exact reproducibility is a separate claim from scientific validity or extraction
improvement. Alias/derivative declarations cannot prove that undeclared papers
are unrelated; curators must review future split membership before tuning.

## Implementation and worked example

Added `src/meta_data_extraction/offline/{manifest,scoring,runner}.py` and an empty
package initializer. This is an offline adapter, not a second evaluator.

- `manifest.py` binds artifact paths/hashes, variants and publication groups.
  Normalized DOI/arXiv aliases, explicit derivation lineage, shared paths and
  source hashes cannot cross splits or unrelated publication groups. An omitted
  variant becomes an explicit unprocessed observation. Future-held-out gold needs
  an explicit scoring flag before any gold read.
- `scoring.py` freezes evaluator/config/schema/code/dependency identity, verifies
  actual imported-module origins, calls inherited scoring unchanged, compares
  complete semantic reports, and exposes support alongside inherited aggregates.
  Unknown support stays unknown; a both-missing score of one is not positive
  evidence. No new schema-wide average or support-adjusted score is introduced.
- `runner.py` stages immutable input copies while preserving the gold filename
  stem used by historical identity helpers. It writes exclusively under the new
  run, keeps failed/unprocessed/blocked records, excludes fields requiring new
  adjudication (including dependent references), separates natural/synthetic
  results and resource totals by execution mode, and records harmed/improved/
  unchanged or unsupported per-publication field comparisons. Completed replay
  verifies source inputs, staged copies and completion hashes before returning.

For example, the M02 synthetic dataset refers to two distinct same-composition
samples. Its entity recovery, conditional composition attributes and dataset
references appear in separate rows. Missing DOI/funding support remains visible;
the inherited score is never presented as coverage of unreported facts. Both M02
exports can now be scored through this path without accessing gold at runtime.

Added three test modules and a network-denying offline conftest, one explicitly
synthetic expected-record fixture, and three `configs/agent/m03_*.json` files
(historical manifest, synthetic manifest, evaluator snapshot). The runtime import
guard now also forbids the offline package and checks nested runtime modules.
Updated progress, commands and data policy; no inherited implementation or schema
was edited, and no dependency or package-name change was needed.

## Checks and experiment outcome

- Main safe suite: **630 passed in 21.54s** (404 existing agent tests, 151 new offline
  tests and 75 inherited safe regressions).
- Additional inherited scorer/reference/safety cases: **41 passed, 6 deselected
  in 1.14s**. The six exclusions read natural gold/annotations outside the unit-test
  scope; E03 performs its own explicit two-record replay instead.
- Independent boundary/profile checks passed; the final runner unit tests mock
  scoring explicitly, while scorer tests use actual inherited code on synthetic
  files. The two CLI integration runs use actual inherited scoring.
- E03 historical replay: **2/2 exact semantic matches**, one historical
  publication, two variants, zero mismatches. A repeated command returned
  `cached_replay`, **zero new evaluations**, and zero model calls.
- Synthetic integration: both M02 canonical exports scored successfully against
  hand-authored fixture expectations; this is an engineering check, not expert
  gold or measured extraction performance.
- All 15 M00 audit hashes, the pre-M03 protected-file hashes and the five scoped
  historical input/report hashes remain unchanged. Final diff, ignore, whitespace
  and credential-pattern inspections are recorded in `../COMMANDS.md`.

Frozen evaluator ID:
`evaluator:cbe498a1a8ce3552f3f1bb93fa577656e6feda7b39d518c0ed01b61f8dad52ef`.
Exact commands are in [COMMANDS.md](../COMMANDS.md).

Artifacts under each run are `request.json`, `result.json`, `sidecar.json` and
hash-pinned staged `inputs/`. `result.json` contains metric rows, coverage,
per-publication comparisons, resource accounting and parity states; `sidecar.json`
preserves untouched evaluator reports. They remain ignored and are not copied
into released output locations or runtime source indexes.

| Run | Scope | Outcome |
|---|---|---|
| `m03-20261007-historical-v1` | 1804.07163, two GPT-4o zero-shot variants | 2 exact matches; 88 metric rows across both variants; 28 rows explicitly have zero support |
| `m03-20261007-synthetic-v1` | M02 invented publication, two mock variants | 2 scored; 88 metric rows; 48 rows explicitly have zero support |

Implementation readiness: **ready**. E03 status: **pilot**, with exact scorer
reproducibility established only for the frozen two-report subset. This is not a
model-quality comparison, evidence-support validation, novelty result or an
agentic improvement. Live inference requests in M03: **0**. The real `.env` was
not read. Historical token/latency metadata is preserved when present; unknown
request counts/costs remain null. No outcomes were fed into extraction changes.

## Remaining limits and next dependency

New-paper curated correspondences remain unreviewed. Such collections and their
dependent reference metrics are blocked from comparison until offline adjudication;
the raw frozen evaluator report can be retained as a diagnostic. Undeclared
preprints/duplicates still require curator review. More than one prediction per
publication-group/variant is rejected rather than averaging chunks as papers.
No future-held-out manifest or expert support judgments were invented.

The entire historical corpus was not rescored. Full inherited gold-reading tests,
result-regeneration scripts and inference were not run. Full environment sync
remains blocked by unavailable cached `hatchling`; the existing environment passed
with explicit checkout binding and no sync. No lint/type checker is configured.
POSIX storage publication remains untested on this Windows host. Path/hash controls
are application safeguards, not protection against an adversarial filesystem.

Stop after M03. Recommended next module: **M04 — PaNET terminology lookup and
conservative mapping**, starting with a pinned authentic resource and synthetic
lookup tests; real mapping accuracy needs reviewed mention-to-concept labels.
