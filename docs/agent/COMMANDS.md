# Agent commands and verification

Run from `D:\git\meta-data-extraction-agent` in PowerShell. The copied `.venv`
currently points its editable package at the sibling release checkout. Bind this
checkout explicitly for all commands below (no sibling checkout was edited):

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE = '1'
```

Use the existing project environment with no dependency resolution or network:

```powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent -q
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py -q
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.smoke --run-id m00-20261007-check
```

The smoke command writes `runs/agent/m00-20261007-check/smoke.json`. It checks
explicit synthetic input and inherited schema loading; it performs no extraction,
PDF parsing, evaluation or inference. A second invocation with the same run ID
must fail. Choose a new ID for a new run. Only the inherited selected tests use
mocked parser/model request objects; the smoke has no model mock or model result.

## Execution record

- Initial standard command runner could not start (`setup refresh had errors`).
  Local Node file/process tools were used for inspection and execution; edits used
  the patch tool. Independent reads were batched. No remote tools were needed.
- `git status --short`, `git branch --show-current`, `git rev-parse HEAD`, `rg
  --files` and focused `rg -n` searches established inventory and side effects.
  Git warned it could not read the global ignore file; repository-local ignore
  rules remain testable.
- `uv --version`: 0.8.22. Project interpreter: CPython 3.13.7.
- `uv run --frozen --offline python -m pytest tests/test_utils.py
  tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py
  tests/test_tracker.py tests/test_prompt_templates.py -q`: blocked by access to
  the default user uv cache before test collection.
- With `--cache-dir tmp/uv-cache --no-sync`, the same 35 tests initially passed,
  but import inspection revealed the sibling editable install. This preliminary
  result is not counted as validation of this checkout. The first M00 collection
  failed with three missing-`meta_data_extraction.agent` errors for that reason.
- `uv --cache-dir tmp/uv-cache sync --frozen --offline`: blocked because the new
  local cache lacks build dependency `hatchling`. No lock/dependency changes or
  network installation were made. Full environment synchronization is unverified.
- After setting `PYTHONPATH` above, the combined M00 and selected inherited command
  passed 118 tests before the final import-origin regression check was added.
  Final commands above passed 84 M00 tests in 0.93s and 35 inherited tests in 1.54s.
  The separate smoke command succeeded with zero model calls. Final baseline
  fingerprint, diff and ignore checks passed; see the module record.
- An initial package-version probe had a quoting error; a corrected probe read
  installed versions successfully. This was a command error, not a code failure.

## Deliberately skipped

The entire inherited suite is not a safe M00 default: several tests read final-test
gold, annotations, benchmark predictions or released reports. Only the six reviewed
test modules above were executed. Full PDF parsing, benchmark scoring/replay,
result regeneration and live inference were not run. No lint/type-check command
is configured in `pyproject.toml`; no pytest/tox/ruff/mypy/pre-commit/CI configuration
was found. There was no reason to install new checkers.

Never run `scripts/run_evaluation.py`, `compute_iaa.py`,
`compute_paper_results.py`, `build_paper_tables.py`, or `build_paper_figures.py`
with their release defaults for agent work. Do not use `run_experiment.py` as a
smoke test. A future authorized replay should call the inherited pure evaluator
from a separate offline harness and write returned reports to a fresh run.

## M01 — source and evidence inspection

Use the same PowerShell environment setup above. These commands were executed:

```powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/test_sections.py tests/test_utils.py -q
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_sources --config configs/agent/evidence_offline.yaml --run-id m01-20261007-check --query "80 K" --max-chars 160 --limit 1
```

Results: 177 tests passed in 2.55s (164 agent tests, including the original 84,
plus 13 inherited section/schema tests). This includes actual local conversion of
one generated synthetic PDF; the error-path PDF cases use explicit test doubles.
No benchmark PDF or gold record was opened, and no live model call was made.

The inspection wrote `runs/agent/m01-20261007-check/inspection.json`. It prepared
one synthetic Markdown source into 15 regions and selected Figure 2's exact
64-character caption for `80 K`. The artifact records hashes, offsets, parser
provenance, warnings, selections and zero model calls. It is not an extraction or
performance evaluation. Use a fresh run ID for every invocation.

`--max-chars` bounds each full-text selection and the combined targeted selection
(including separators). `--limit` bounds targeted passages; `--source-max-chars`
optionally truncates the stored representation with an explicit warning. Budgets
are Unicode code points, not model tokens. `complete` refers to selection of the
available representation; `incomplete_sources` separately exposes preparation
truncation, parser warnings and missing PDF image evidence.

The CLI exits 2 on configuration/path errors or failed preparation. For parse or
decode failures it saves explicit error records without an empty successful
extraction; an invalid config/missing config-selected input is rejected before
run creation. Default inputs are synthetic. Custom source lists must still use
explicit files within the M00 allowlisted roots.

## M02 — fixed extraction, isolated exports and replay

Use the same PowerShell source binding at the top of this document. Final executed
commands (all offline, without reading the real `.env`):

```powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py -q
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.run_extraction --config configs/agent/extraction_mock.yaml --response tests/agent/fixtures/m02_evidence_response.json --variant evidence --retain-raw
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.run_extraction --config configs/agent/extraction_mock.yaml --response tests/agent/fixtures/m02_evidence_response.json --variant evidence --retain-raw
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.run_extraction --config configs/agent/extraction_mock.yaml --response tests/agent/fixtures/m02_inherited_response.json --variant inherited --run-id m02-20261007-inherited-v1 --retain-raw
```

The final test command passed **439 tests in 16.64s** (404 agent + 35 inherited).
The first evidence command succeeded, the repeat returned `cached_replay`, and
the inherited command succeeded with evidence `not_requested`. All reported zero
new live requests. The canonical exports are identical. The evidence variant has
18 location-verified assertions, seven entity handles including the document and
four local-reference links; scientific support remains `not_assessed`.

Each final run stores `canonical.json`, `sidecar.json`, `raw-response.json`,
`request.json` and the checksummed completion `result.json` under its ignored
`runs/agent/<run_id>/`. `--discard-raw` is the explicit alternative. Unknown usage
is null; mock execution is distinct from live execution and replay. The CLI is
mock-only. The programmatic live path requires a trusted configured ChatAI endpoint
plus a `LiveRunSpec` matching its exact source, profile and rendered prompt.

Relevant intermediate checks, also executed offline:

- The six inherited modules passed 35 tests in 1.73s before M02 integration.
- Combined profile/provider/storage checks passed 123 tests in 5.26s before later
  hardening. Final component counts: profiles 32, providers 49, storage 48,
  extraction 89, runner 22, plus 164 pre-existing agent tests.
- The first combined full check passed 435 tests in 16.47s. Request hash formatting
  was then made identical between runner and storage; the affected runner command
  (`... python -m pytest tests/agent/test_run_extraction.py -q`) passed 22 in 3.61s.
  Four actual-SDK/local-transport tests were added before the final 439-test run.
- An early extraction robustness probe found an inherited large-integer parse
  exception; the new adapter now records an explicit parse failure and retains raw
  text according to policy. Tests cover oversized integers/deep nesting, empty
  valid records versus failed extraction, stale evidence/revisions, invalid IDs,
  timeout/429, missing credentials/usage, concurrency and interrupted export.
- Early delegated test issues (reserved pytest argument `request`, Windows GBK
  fixture reads, and platform-dependent deep-JSON behavior) were corrected. An
  exclusive-claim contention fixture now establishes the parent run directory
  first; unsafe concurrent path resolution remains a safe rejection, not a retry.
- Standard shell execution remains unavailable (`setup refresh had errors`), so
  local Node child-process execution launched these PowerShell/uv commands. Tests
  also used the existing `.venv/Scripts/python.exe -B` with explicit source binding
  during provider development. No dependencies or global environment were changed.

Final audit: the 15 baseline fingerprints and ten pre-M02 file hashes match;
`git diff --check` and protected tracked-path diff checks are clean, and Git ignores
`.env` and all demonstration runs. New files were inspected for whitespace and
credential assignments. Incidental Windows shell cache files under the literal
`%SystemDrive%` folder were preserved under ignored `tmp/m02-host-cache/`, after
checking both resolved paths remain within this workspace.

E02 is **not_run**, with no real model/endpoint performance measured. Full inherited
gold-reading tests, evaluator/release regeneration scripts, live inference and a
fresh environment sync were not run. Offline sync remains blocked by missing
cached `hatchling`; no lint/type checker is configured. The POSIX atomic-publication
branch remains untested on Windows. Proceed next only with authorized M03 work.

## M03 — frozen offline scoring and historical parity

Use the same explicit checkout binding at the top of this document. The inherited
evaluator remains unchanged; commands invoke the new offline package directly.

The main safe test command executed was:

```powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/offline tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py tests/test_alignment_entry_points.py tests/test_facility_staged_matching.py tests/test_dataset_name_evaluation_normalization.py tests/test_typographic_evaluation_normalization.py -q
```

Result: **630 passed in 21.54s** (404 agent, 151 offline, 75 inherited). A separate
inherited evaluator check executed:

```powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/test_evaluation.py tests/test_reference_resolution.py tests/test_evaluation_operational_safety.py -q -k "not test_report_has_evaluator_block and not test_sample_registry_alignment_drives_sample_references and not test_all_entity_matchers_record_correspondences_and_stages and not test_exact_unique_dataset_name_separates_identity_from_technique_scoring and not test_primary_iaa_metrics_are_symmetric_on_all_30_papers and not test_reversing_the_published_plan_preserves_all_field_scores"
```

Result: **41 passed, 6 deselected in 1.14s**. These exclusions prevent implicit
natural-gold/annotation inspection by unit tests. The scoped E03 replay below
does its own parity check against the unchanged released per-document reports.
Tests under `tests/offline` and `tests/agent` deny sockets and DNS; no live model
is needed. Runner unit tests mock scoring explicitly; scorer tests invoke actual
inherited scoring on synthetic records.

After the E03 protocol was written, the local helper command
`uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m03_prepare.py`
wrote two complete manifests, a synthetic expected fixture and a frozen evaluator
snapshot using `freeze_evaluator(Path.cwd())`. The helper used exclusive file
creation and file hashes, performed zero evaluations and made no model calls.
The complete specifications are saved as `configs/agent/m03_historical.json`,
`m03_synthetic.json` and `m03_evaluator.json`; no helper rerun is required.

The following integration commands then executed:

```powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.offline.runner --manifest configs/agent/m03_synthetic.json --evaluator configs/agent/m03_evaluator.json --run-id m03-20261007-synthetic-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.offline.runner --manifest configs/agent/m03_historical.json --evaluator configs/agent/m03_evaluator.json --run-id m03-20261007-historical-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.offline.runner --manifest configs/agent/m03_historical.json --evaluator configs/agent/m03_evaluator.json --run-id m03-20261007-historical-v1
```

Results: the two M02 synthetic exports scored; both historical reports matched
exactly; the repeated historical invocation returned `cached_replay` with zero
new evaluations. All commands reported **zero model calls**. The natural scope was
one historical publication (`1804.07163`) and two GPT-4o zero-shot variants selected
before score inspection. Known path metadata alone is excluded from parity.
Report values, numeric types, evaluator versions and schema provenance must match.

Each run's `result.json` completion envelope contains the metric row table, coverage,
per-publication field comparisons and resource accounting. `sidecar.json` holds
the original inherited reports; staged inputs retain the gold filename stem required
by historical identity matching. Both the original hashed inputs and staged copies
must still verify on completed replay. Changed code/resources/inputs require a new
explicit snapshot/specification and run ID; interrupted runs never automatically
rescore or overwrite files.

Default CLI access cannot open future-held-out gold. `--allow-final-gold` is only
for a separately authorized final offline scoring step; it was not used here.
The synthetic manifest deliberately references the completed M02 exports and its
invented expected fixture. Missing prerequisite artifacts produce explicit failures,
not invented records. Historical outputs are not promoted to runtime retrieval.

Other executed checks included 115 combined manifest/scorer/import-boundary tests
before final runner cases, 31 intermediate runner tests, and 116 final manifest/
runner tests. Independent review prompted overflow-JSON, lineage, sidecar,
publication-identity, missing-variant, mode-separated accounting and staged-replay
guards before the final suite and historical replay. An initial broad `rg --files`
inventory exceeded the process output buffer; focused metadata listings replaced
it. Standard shell execution remained unavailable, so commands ran through the
local Node child-process tool; no global environment or dependencies were changed.

Final audit checks: `git diff --check`; protected-directory `git diff --name-only`;
`git check-ignore` for `.env`, both M03 runs and temporary files; all 15 baseline
fingerprints, pre-M03 protected files and all five scoped historical input/report
hashes. New source/test/config/docs files were inspected for whitespace, merge
markers and credential patterns. Incidental host cache files under the literal
`%SystemDrive%` directory were preserved under ignored `tmp/m03-host-cache/` after
checking source/destination paths remained inside the workspace.

E03 status: **pilot**, demonstrating frozen-scorer reproducibility on 2/2 reports.
No model-quality improvement or scientific evidence-support accuracy is established.
No other historical publications, final-held-out gold or all-annotator comparison
were evaluated. No result-regeneration script, live inference or `.env` read ran.
Full environment sync remains blocked by missing cached `hatchling`; no lint/type
checker is configured. M03 stops here; M04 is the next recommended module.

## M04 — pinned PaNET lookup and unapproved proposals

Use the same PowerShell checkout binding at the top of this file. No dependencies,
lockfile or global environment changed. The final commands executed were:

```powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_panet_resources.py tests/agent/test_panet_lookup.py tests/agent/test_panet_mapping.py tests/agent/test_inspect_panet.py -q
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_panet --config configs/agent/panet_inspection.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_panet --config configs/agent/panet_inspection.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_panet --config configs/agent/panet_bound_inspection.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/offline tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py tests/test_alignment_entry_points.py tests/test_facility_staged_matching.py tests/test_dataset_name_evaluation_normalization.py tests/test_typographic_evaluation_normalization.py -q
```

Results: **168 M04 tests passed in 5.37s**; the final combined suite passed **798
tests in 50.64s** (572 agent, 151 offline, 75 inherited). Resource/lookup/mapping
unit tests use clearly synthetic namespaces. Runner unit tests explicitly mock the
resource loader to test control flow; the CLI commands use the actual pinned OWL.
Sockets/DNS are denied by the test fixtures, and no live provider was invoked.

The first CLI command produced three unapproved proposals, two unresolved contexts
and one no-match from six invented mentions. Its repeat returned `cached_replay`.
The record-bound command retained the existing M02 technique wording and returned
one unresolved `lexical_only` decision. Both run directories contain only
`request.json`, `result.json` and `sidecar.json`; no canonical replacement exists.
Every decision is unreviewed and has scientific support `not_assessed`.

Resource acquisition was separate from runtime. Direct Node/PowerShell download
routes were unavailable; a public GitHub read tool retrieved the immutable v1.1.0
repository blob. Its Git blob identity was checked against the official tree,
and runtime checks its SHA256. The newer v1.3.0 generated release asset was not
obtainable through that route. Exact provenance, attribution, resource terms and
the historical-version limitation are in [PANET_RESOURCE.md](PANET_RESOURCE.md).
The main source and all context annotations used in this module are explicitly
invented; there are no natural-publication mapping judgments.

Other executed checks and corrections:

- Resource tests passed 34; lookup tests passed 44; mapping tests passed 37; final
  runner tests passed 53. Intermediate combined checks passed 78 resource/lookup
  tests and 45 runner/import-boundary tests before added cache-forgery cases.
- A premature mapping/runner test invocation found the mapping test file had not
  yet been written and collected no tests. After the delegated file was ready, the
  complete M04 suite passed. An early runner fixture normalized newlines on Windows;
  it was corrected to write exact bytes before comparing hashes.
- `uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m04_probe.py`
  inspected only authentic scoped candidates for the invented expressions, with
  zero model calls. It confirmed the named concepts and the exact-before-overlap
  ranking used for conservative result-limit handling.
- The first full-suite invocation exceeded the Node tool's 30-second capture
  window and reset that tool session; its outcome was not counted. A repeat with a
  60-second tool window captured the 798 passing result, also stored in ignored
  `tmp/m04-final-tests.json`. No code change was made to obtain that passing result.
- Independent review prompted full cached-report/sidecar semantic checks, inherited
  parser/config dependency fingerprints, and record-sidecar single-read hashing.
  Tests explicitly reject forged metadata even if storage checksums are recomputed.

Final audit: `git diff --check`; protected-path diff inspection; Git ignore checks
for `.env`, all three M04 run directories and temporary files; the 15 M00 baseline
fingerprints and existing M02 runtime fingerprints; OWL SHA256/Git-blob identity;
new source/test/config/docs whitespace, conflict-marker and credential-pattern
inspection. Incidental shell cache files in the literal `%SystemDrive%` directory
were preserved under ignored `tmp/m04-host-cache/` after verifying both resolved
paths stayed within this workspace.

**Zero model requests** and no real `.env` read occurred. E04 is **blocked** by the
absence of expert-reviewed mention-to-concept labels; no natural mapping accuracy,
cost improvement or controller benefit was measured. The optional LLM selection
path, full OWL reasoning, newer resource upgrade and NeXus work remain unimplemented
in this module. Full environment sync still lacks cached `hatchling`; no configured
lint/type checker exists. No gold-reading inherited suite or regeneration script
ran. M04 stops here; M05 is next.

## M05 — pinned NXDL and review-gated profile guidance

Use the existing environment with this explicit PowerShell checkout binding:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE = '1'
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_nexus_resources.py tests/agent/test_nexus_registry.py tests/agent/test_nexus_units.py tests/agent/test_inspect_nexus.py -q
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_nexus --config configs/agent/nexus_inspection.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_nexus --config configs/agent/nexus_inspection.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/offline tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py tests/test_alignment_entry_points.py tests/test_facility_staged_matching.py tests/test_dataset_name_evaluation_normalization.py tests/test_typographic_evaluation_normalization.py -q
```

Results: **286 M05 tests passed in 14.10s** (resource 68, registry 64, units 94,
runner 60). The combined safe suite passed **1,084 tests in 52.70s** (858 agent,
151 offline, 75 inherited). The combined process ran with asynchronous output
capture to avoid the host tool's short blocking window; its exit-success output
is retained in ignored `tmp/m05-final-tests.json`. Tests use offline fixtures;
synthetic reviews and mocked authentic-loader runner cases are explicitly labelled.
No live model is required or called by the test suites.

Authentic inspection produced **four unresolved partial proposals, two scoped
no-matches, zero approved guidance entries and three exact definition lookups**.
All six mappings are unreviewed. Its invented quantity checks yielded two exact
conversions, one unavailable value and one incompatible-unit rejection. Repeat
execution returned `cached_replay`; SHA256 of every request/result/sidecar file
was identical before/after. The run at `runs/agent/m05-20261007-inspection-v1/`
contains only `request.json`, `result.json`, `sidecar.json`. No canonical output
was generated or replaced. The resource snapshot is
`nxdl:8a779d11868e2f94d45b0cc6d24285c0cf157c1d035ee1bf58b1bcee7e54bc3b`.

Other executed checks:

- Focused per-file commands used the same uv prefix: units 94 passed in 0.27s;
  registry final 64 passed in 3.38s; runner final 60 passed in 9.41s. Resource
  initially passed 58 in 1.36s, then 68 after parser-bound/constraint regressions.
  Joint resource/registry/runner verification passed 192 tests in 14.10s.
- `uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m05-resource-check.py`
  loaded and verified the authentic resource: 85 fields (NXsample 39, NXbeam 32,
  NXmonopd 14). This helper did not claim an inspection run.
- `uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m05_probe.py`
  checked actual definition addresses and six unreviewed mapping states before
  final parser hardening. Its earlier snapshot was superseded; final CLI artifacts
  use the snapshot above. This helper made no canonical or run-output writes.
- Independent review added unsupported-constraint notices, aggregate projection
  accounting, strict Boolean opt-in, non-text-equivalence refusal, and a successful
  artifact-size cap matching replay. Missing experimental data cannot be filled by
  standard requirements, even in tests with explicit required declarations.

Public resource acquisition used official GitHub read tools with an immutable
v2026.01 commit. Exact file URLs, byte/SHA256/Git-blob checks and preserved license
attribution are in [NEXUS_RESOURCE.md](NEXUS_RESOURCE.md). A guessed NXxpd path
returned 404; the official directory listing identified NXmonopd. One combined
file transfer exceeded the automatic tool review's 64000-byte input limit; smaller
writes of the same authorized files succeeded. A preliminary helper attempted
unavailable Node `process.env` and failed before starting Python; explicit
PowerShell environment binding resolved it. The standard shell tool itself failed
with `helper_unknown_error`, so local commands used Node child processes. These
are host-tool issues, not test failures or model attempts. No dependencies changed.

Final audit commands include `git diff --check`, protected-path
`git diff --name-only`, `git status --short`, `git check-ignore` and targeted
`rg` inventories. Byte hashing verified all 15 M00 baseline fingerprints and all
13 code-file pins in the completed M05 request. A pre-M05 inventory of 1,649 files
confirmed no removals and changes only to the three intended existing docs
(PROGRESS, COMMANDS, DATA_POLICY); inherited code, schemas, dependencies, released
assets and earlier agent/offline implementations remain byte-identical. New source,
tests, configs and docs were scanned for whitespace, conflict markers and credential
patterns. Resource byte/Git-blob pins and ignored artifacts were checked separately.
Incidental host cache output under literal `%SystemDrive%` was retained within
ignored `tmp/m05-host-cache/` after checking both paths stayed in this workspace.

**Zero live model requests** and no real `.env` read occurred. E05 and the actual
reviewed-profile gate remain **blocked**: relevant NXDL inheritance/defaults and
unhandled constraints need resolution, mapping entries need real human review,
and E05 needs independent labels plus a frozen comparison specification. Synthetic
engineering success is not empirical extraction validation. Full environment sync
still lacks cached `hatchling`; no lint/type checker is configured. No natural
gold evaluation, full extraction experiment or result-regeneration script ran.
Stop after M05; M06's independent record checks are the next coding module.


## M06 — record checks, diagnosis and bounded atomic patches

The existing uv environment, dependency lockfile and package name were preserved.
Commands used the explicit checkout binding shown below. The final checks ran:

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE = '1'
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_record_checks.py tests/agent/test_support_checks.py tests/agent/test_record_patches.py tests/agent/test_inspect_record.py -q
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_record --config configs/agent/record_inspection.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_record --config configs/agent/record_inspection.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_record --config configs/agent/record_structural_inspection.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/offline tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py tests/test_alignment_entry_points.py tests/test_facility_staged_matching.py tests/test_dataset_name_evaluation_normalization.py tests/test_typographic_evaluation_normalization.py -q
~~~

The focused suite passed **262 tests in 27.56s** (47 structural, 107 support,
61 patches, 47 runner). The combined safe suite passed **1,346 tests in 82.56s**
(1,120 agent, 151 offline, 75 inherited). The combined process used asynchronous
output capture; exit-success output is retained in tmp/m06-final-tests.json, with
focused output in tmp/m06-focused-tests.json. All model responses in new tests are
synthetic fixtures; no live inference is required or dispatched.

The first correction CLI made two MockProvider calls and applied exactly two scalar
changes: Dataset A's description changed from Measured at 500 K. to Measured at
300 K., and its sample_refs/0 changed from sample_2 to sample_1. Both specimens have
CaCO3 composition and remain distinct, with stable entity handles. The candidate
checker covered the changed description, sample attachment and facility attachment.
All judgments are mock_model_judgment, unreviewed. No other scalar changed, and the
original fixture bytes remain pinned and unchanged.

The repeated correction command returned cached_replay and zero new provider
requests. All five artifact-file SHA256 values were identical before/after. The
run is runs/agent/m06-20261007-patch-v1/. The structure-only command recorded the
sample_99 dangling reference, preserved the invalid candidate and made zero calls.
Its runs/agent/m06-20261007-dangling-v1/ contains request.json, result.json and
sidecar.json only; canonical.json was correctly absent. Both completed requests'
18 implementation-file hashes match the final source. CLI output and direct
artifact checks are captured in ignored tmp/m06-cli-results.json and
tmp/m06-artifact-audit.json. Overall coverage/completeness remains unsupported.

Other executed checks and implementation corrections:

- Per-file focused commands used the same uv prefix and python -m pytest path -q:
  structural 47 passed in 2.98s; support initially 104, then final 107 in 9.73s;
  patches final 61 in 5.69s; runner initially 44, then final 47 in 10.43s.
- Initial patch testing found one fixture used an invalid diagnosis name;
  missing_evidence corrected the fixture before the passing checks. Later cases
  cover source conflicts, reference reordering and required-support target bounds.
- Three added runner cases prove source conflicts and insufficient evidence skip
  the candidate-check call, preserve canonical input, and replay without dispatch.
- Review tightened pure prompt/replay/report profile integrity to reject a changed
  schema carrying an old profile ID, without requiring file IO at those boundaries.
  Hashes remain integrity checks, not signatures or scientific correctness proofs.
- uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m06_prepare.py
  created the invented source-bound record fixtures using the existing synthetic
  skeleton helper and M01/M02 profile/evidence logic. It used exclusive writes,
  generated no completed run, and called no model.
- Commands ran through local Node child processes because the normal shell tool
  remains unavailable in this environment. One large support-test write had a
  JavaScript escaping error before creating its file; a structured write resolved
  it. These host/write issues were not inference attempts or test failures.

Final audit: git diff --check; protected-path git diff --name-only; git status
--short; git check-ignore for .env, both M06 runs and temporary files; all 15 M00
baseline fingerprints; all 1,661 files from the pre-M06 hash inventory; fixture
pins and both completed implementation manifests; new-file whitespace, merge-marker
and credential-pattern scans. No existing file was removed. Only PROGRESS.md,
COMMANDS.md and DATA_POLICY.md changed among pre-existing files. Inherited code,
previous agent/offline implementations, schemas, dependencies and released assets
remain byte-identical. Incidental host cache output under the literal %SystemDrive%
directory was preserved under ignored tmp/m06-host-cache/ after checking both
resolved paths stayed inside this workspace; no unrelated files were deleted.

**Zero live model requests** and no real .env read occurred. E06 is **blocked**:
independent labelled real outputs and a frozen comparison protocol are missing.
The two-call synthetic demonstration proves plumbing and policy behavior, not
natural-paper diagnosis accuracy, harmful-patch rates, inference latency, costs or
controller benefit. M05's real reviewed-profile gate remains blocked; no standard
guidance was introduced. Full uv sync still lacks cached hatchling; no lint/type
checker is configured. No gold-reading inherited tests, full extraction experiment
or result-regeneration script ran. Stop after M06; M07 is the next coding module.


## M07 — bounded workflow and durable local execution

The required dependencies were added deliberately; the Python requirement,
distribution/import names and inherited scientific code were preserved:

~~~powershell
uv --cache-dir tmp/uv-cache add --no-sync 'langgraph==1.2.14' 'langgraph-checkpoint-sqlite==3.1.1'
uv --cache-dir tmp/uv-cache sync --frozen
uv --cache-dir tmp/uv-cache sync --frozen --offline
~~~

The first unprivileged add failed with socket-access error 10013 while fetching
PyPI metadata. The environment's automatic review approved the network-enabled
add and frozen sync commands, which then succeeded. No user confirmation was
needed. Frozen sync built the current project and corrected the old editable
pointer from the sibling release checkout without changing that other checkout.
The final ordinary offline sync audited 75 installed packages in 5ms successfully.
All previously locked package versions were retained except idna 3.11 → 3.20, required
by new httpx2's idna>=3.18 dependency. The lock adds 24 package entries (including a
platform-specific entry not installed here). Exact audit:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m07-dependency-audit.py
~~~

The helper compared the previous Git lock with the current TOML and reported no
removed packages. Installed signature inspection verified SqliteSaver(conn,serde=)
and explicit JsonPlusSerializer restrictions. Installed versions are LangGraph 1.2.14,
checkpoint 4.2.0, SQLite saver 3.1.1, LangChain core 1.6.7 and LangSmith 0.14.4. Official
source references are in [M07](modules/M07_workflow.md). External tracing is disabled
in code and tested with inherited tracing configuration; no cloud trace was sent.

Tests used this checkout binding and the existing uv environment:

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE = '1'
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_workflow_types.py tests/agent/test_workflow_policies.py tests/agent/test_workflow_tools.py tests/agent/test_workflow.py tests/agent/test_workflow_graph.py tests/agent/test_inspect_workflow.py -q -o cache_dir=tmp/pytest-cache
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/offline tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py tests/test_alignment_entry_points.py tests/test_facility_staged_matching.py tests/test_dataset_name_evaluation_normalization.py tests/test_typographic_evaluation_normalization.py -q -p no:cacheprovider
~~~

Results: **284 M07 tests passed in 47.07s** and **1,630 combined tests passed in 134.04s**
(1,404 agent, 151 offline, 75 inherited). The focused run had one permission warning
writing pytest's disposable cache; disabling only that cache produced the clean
final result. Actual exit-success outputs are retained in ignored
tmp/m07-focused-tests.json and tmp/m07-final-tests.json. The full suite used
asynchronous process capture; it did not exceed a blocking tool wait.

Focused worker commands used the same uv/pytest prefix: contracts+policies 95 passed
in 10.52s; tools 65 in 18.67s; transitions+graph 92 in 25.59s; CLI 32 in 12.23s. Earlier
passes were 92 contracts/policies before three observation-prompt cases, 56 tools
before unknown-usage cases (that command timed out during shutdown after passing
output), and 72 transitions/graph before the final integrity/accounting regressions.
One attempted test-file write had a JavaScript escaping error before creating the
file, then succeeded through a structured write. These were host/write issues,
not live inference or scientific experiment failures. The normal shell tool's
helper setup still failed; warm Node child processes and automatically approved
scoped command execution completed the checks.

The frozen synthetic protocol was written in the module record before these
actual runs. The following three commands each ran once, then each was repeated
once to verify completed replay:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_workflow --config configs/agent/workflow_fixed.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_workflow --config configs/agent/workflow_issue_aware.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_workflow --config configs/agent/workflow_generic.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_workflow --config configs/agent/workflow_generic.json --run-id m07-20261007-resume-v1 --max-steps 1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_workflow --config configs/agent/workflow_generic.json --run-id m07-20261007-resume-v1
~~~

All three variants finished in three actions and produced identical canonical
bytes, changing only Dataset A's 500 K description to 300 K and sample_2 to sample_1.
Both specimens and entity handles stayed distinct and stable. The fixed and
issue-aware variants each made two mock calls; generic made five, including three
mock policy selections. All share the exact frozen initial record/issues, tool
fingerprint and budgets. None used standards guidance or independent gold.

Repeats returned cached_replay, zero new calls, and identical hashes for all seven
files in each run, including both databases and the lock. The separate resume run
exited after one action/call with only request.json, checkpoints.sqlite,
dispatch.sqlite and workflow.lock. A second process made four more calls and
finished with the same canonical output and five cumulative calls as uninterrupted
generic execution. No allowance reset or hidden model dispatch occurred. Every
run's 27 combined implementation pins and all declared input pins match final files.
Capture files: tmp/m07-cli-runs.json, tmp/m07-cli-replays.json,
tmp/m07-cli-resume.json and tmp/m07-artifact-audit.json (all ignored).

Independent review found and fixed uncertain-call budget reuse, unaudited terminal
checkpoint shortcuts, rehashed completion/checkpoint inconsistencies, stale issue
resolution after record edits and counting predispatch blocks as live HTTP attempts.
Regression tests include genuine subprocess resume, incomplete dispatch claims,
concurrent-run exclusion, request/resource changes, corrupted state, pending reviews,
source-vs-definition separation, timeouts/cancellation and disabled tracing.
No arbitrary generated code or document-granted tool permissions are executable.

Final audit covers git diff --check, protected-directory diff/status, Git ignore
checks for .env/all four runs/tmp, pre-M07 hashes of 1,677 existing files, all 15 M00
baseline fingerprints (13 unchanged; dependency pair intentionally updated), new
source/test/config/fixture/doc content scans, final config/implementation pins and
exact corrected scalar paths. No files were removed. Only pyproject.toml, uv.lock,
PROGRESS.md, COMMANDS.md and DATA_POLICY.md changed among existing files. The literal
%SystemDrive% host-cache directory was absent at final inspection; no incidental
files required moving or deletion. No released asset or earlier-module
implementation changed.

**Zero live model requests** and no real .env read occurred. Public package metadata
and dependency downloads were separate authorized environment setup. Token totals
remain unknown for mock responses that omit usage; these are not cost or latency
measurements. E07 is **blocked** by missing independent labelled real outputs and
a frozen real comparison; fixed and issue-aware show no routing difference on this
one invented correction case. Coverage/completeness is unsupported; M05's actual
reviewed-guidance gate is still blocked. No gold-reading inherited tests, full
extraction experiment or regeneration script ran. No lint/type checker is configured.
The old frozen-sync blocker is resolved. Stop after M07; next is M08 human review.


## M08 — persistent explicit review (2026-10-07)

All Python commands used the existing uv environment, current checkout source, and
no bytecode writes. The common PowerShell prefix was:

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src'); $env:PYTHONDONTWRITEBYTECODE = '1';
~~~

Final safe regression command (exit 0, **1,795 passed in 206.57s**, no warnings):

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/offline tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py tests/test_alignment_entry_points.py tests/test_facility_staged_matching.py tests/test_dataset_name_evaluation_normalization.py tests/test_typographic_evaluation_normalization.py -q -p no:cacheprovider
~~~

The final collection contains 165 new M08 tests: 73 pure review, 61 persistence,
and 31 CLI. It also retains all 1,630 safe M07-era checks. Focused development runs
used test_review.py, test_review_store.py and test_review_cli.py; the first combined
store/CLI run passed 46 with three draft expectation mismatches (review status name,
returning the current head on old-response replay, and the single-writer lease).
Those tests were aligned with the declared contract. Independent review then added
unresolved-dependency, accounting, endpoint-context, compatibility and pre-BEGIN
path safeguards. The final combined run includes all these changes.

An early background full-suite attempt produced no retrievable result capture and
is not counted as verification. The final suite above ran in a tracked command
session and completed successfully; its result is saved in tmp/m08-final-tests.json.
Default sandbox shell setup failed before execution; scoped read/test commands used
the available reviewed command executor or warm local Node helper. A Windows process-
inventory diagnostic was denied, with no effect on checks. No dependency, lockfile
or environment update was needed. No lint or type-check command is configured.

Actual independent-process CLI diagnostic commands, each exit 0:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_workflow --config configs/agent/workflow_review.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.review_cli show --run-id m08-20261007-pending-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m08_prepare_simulation.py m08-20261007-pending-v1 edit
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.review_cli submit --run-id m08-20261007-pending-v1 --response runs/agent/m08-20261007-pending-v1/review-responses/simulated-edit.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.review_cli submit --run-id m08-20261007-pending-v1 --response runs/agent/m08-20261007-pending-v1/review-responses/simulated-edit.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.review_cli resume --run-id m08-20261007-pending-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.inspect_workflow --config configs/agent/workflow_review.json --run-id m08-20261007-unresolved-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m08_prepare_simulation.py m08-20261007-unresolved-v1 unresolved
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.review_cli submit --run-id m08-20261007-unresolved-v1 --response runs/agent/m08-20261007-unresolved-v1/review-responses/simulated-unresolved.json
~~~

The ignored helper creates explicitly simulated input JSON from the frozen invented
record/evidence; it disables network access and never claims human approval. The
first run used four mock policy calls to queue two cases, then one simulated edit
with three attestations corrected exactly two scalar leaves. Its duplicate command
returned replayed and retained one decision. Resume returned the identical manifest.
The second run used four mock policy calls, retained one simulated unresolved
response, and preserved its canonical record with zero active attestations. Review
processing used zero model calls; total live requests were zero. Unknown token usage
remains unknown. These are engineering diagnostics, not expert effort measurements.

Local SHA256 audits verified both export manifests, unchanged entity handles, exactly
two intended changed paths for the edit, zero for unresolved, and byte-identical
replay of all 12 correction-run files (including seven original M07 files). Captures:
tmp/m08-initial-hashes.json, m08-parent-pins.json, m08-before-replay.json,
m08-replay-audit.json, m08-artifact-audit.json and m08-final-audit.json.

Final checks include git diff --check plus a scan of new/changed text for whitespace,
conflict markers and credential-shaped values. The 1,698-file starting inventory
protects prior code/data; only the three progress/command/data-policy documents are
intentionally updated among those files. No previous implementation, benchmark asset,
AGENTS.md, pyproject.toml or uv.lock changed. No real .env read, natural-gold inspection,
result-regeneration script, full extraction experiment, live inference or external
trace/upload was performed. E08 remains blocked by missing actual review observations,
independent labels and its frozen assignment/budget; stop before M09.

## M09 — frozen synthetic study and offline reporting (2026-10-08)

All Python commands used the existing frozen uv environment and the common prefix:

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src'); $env:PYTHONDONTWRITEBYTECODE = '1';
~~~

Focused final checks, all exit 0:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_study_types.py tests/agent/test_study_runner.py -q -p no:cacheprovider
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_study_cli.py -q -p no:cacheprovider
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/offline/test_study.py -q -p no:cacheprovider
~~~

Results: **85 passed in 46.06s**, **65 passed in 3.80s**, and **82 passed in 28.53s**.
The final combined safe regression command exited 0 with **2,027 passed in
418.19s**, no warnings:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/offline tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py tests/test_alignment_entry_points.py tests/test_facility_staged_matching.py tests/test_dataset_name_evaluation_normalization.py tests/test_typographic_evaluation_normalization.py -q -p no:cacheprovider
~~~

This includes 232 new M09 checks plus all 1,795 previously passing safe checks.
The tracked session output is retained in tmp/m09-final-tests.json.
The first offline development run exposed Boolean call-cap coercion (42 pass,
one failure); strict prevalidation fixed it. Final regression cases also cover
reconstructed wrong-policy/cap children, invocation-mode accounting, failed-cell
unknown/observed resources, absent artifacts later appearing, altered sidecar state,
and injected extraction metadata that could affect inherited scoring eligibility.

Before execution, tmp/m09_prepare_fixture.py created the authored synthetic expected
record exclusively; tmp/m09_freeze_protocol.py froze the evaluator/protocol and
runtime specification. Both ran through the same uv prefix, exit 0. The new offline
protocol was placed at configs/offline/study_synthetic_evaluation.json, outside the
runtime CLI allowlist. Its defaults/digest did not change after freezing. Only the
explicit invented fixture is gold for this engineering diagnostic.

Actual CLI commands, all exit 0:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.study_cli dry-run --config configs/agent/study_synthetic.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.study_cli run --config configs/agent/study_synthetic.json --run-id m09-20261008-synthetic-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.offline.study --study-run-id m09-20261008-synthetic-v1 --protocol configs/offline/study_synthetic_evaluation.json --run-id m09-20261008-report-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.offline.study --study-run-id m09-20261008-synthetic-v1 --protocol configs/offline/study_synthetic_evaluation.json --run-id m09-20261008-reproduce-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m09_compare_reports.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m09_artifact_audit.py capture
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.study_cli replay --config configs/agent/study_synthetic.json --run-id m09-20261008-synthetic-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.offline.study --study-run-id m09-20261008-synthetic-v1 --protocol configs/offline/study_synthetic_evaluation.json --run-id m09-20261008-report-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m09_artifact_audit.py verify
~~~

Dry-run returned ready with six cells, one publication, caps two/five, one repeat,
21 reserved mock calls and zero live requests. Runtime made 15 mock calls: two each
for fixed and issue-aware at either cap, two for generic at cap two and five at cap
five. Five workflows finished; generic cap two exhausted its budget with two issues
unresolved. Every planned cell retained a valid canonical prediction. Token totals
remain unknown for all six cells; cost is not applicable to these mock providers.
Shared initial extraction and seed judgments are outside measured execution.

Primary datasets.attrs.sample_refs F1 was 1.0 for five corrected cells and 0.5 for
generic cap two, with two supported gold references. The fixed baseline and
issue-aware policy tie at both caps. One invented publication supports no uncertainty
interval or natural-paper/controller-effect claim. Entity recovery, conditional
attributes and references stay separate; no cross-field average was introduced.

Each fresh collection performed six offline evaluations. The independently staged
reproduction matched 264 metric rows, 176 per-field comparisons, four primary paired
comparisons and all 24,629 Markdown bytes. Runtime replay returned zero new calls;
collection replay returned zero new evaluations. SHA256 checks of all 89 files
across 13 run directories found no changed, missing or added artifacts. The helper
captures are tmp/m09-reproduction-audit.json, tmp/m09-before-replay.json and
tmp/m09-replay-audit.json. Actual study stdout is tmp/m09_actual_run.json.

Final repository checks used:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m09_final_audit.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m09_verify_text.py
git diff --check
git check-ignore .env tmp/m09-final-audit.json runs/agent/m09-20261008-synthetic-v1/request.json runs/agent/m09-20261008-report-v1/report.md runs/agent/m09-20261008-reproduce-v1/report.md
~~~

The 1,711-file starting inventory has no missing files or unexpected changes.
Only COMMANDS.md, DATA_POLICY.md and PROGRESS.md changed among existing files;
12 M09 files were added. Previous code/tests/configs, released assets, AGENTS.md,
pyproject.toml and uv.lock are byte-identical to the starting inventory. New/changed
text has no whitespace, conflict-marker or credential-shaped findings. Git ignore
checks passed; git diff --check exited 0 with existing LF-to-CRLF notices for the
already-modified .gitignore/pyproject.toml/uv.lock. Audit: tmp/m09-final-audit.json.

Normal shell and a cold Node helper failed before command execution because of
Windows sandbox setup refresh errors. Scoped local read/test commands therefore
used the available reviewed command executor. No automatic approval rejection or
user permission blocker remained. No dependency/environment update was needed;
no lint/type checker is configured. All actual diagnostic and final check commands
above completed successfully. CLI captures are tmp/m09-cli-checks.json.

No live inference, real .env read, natural-gold inspection, external tracing/upload,
inherited regeneration script or full extraction experiment ran. E09 is **blocked**
by missing approved natural publications/independent labels, reviewed guidance for
the main standards comparison, and its claim-specific protocol. The executor is
currently a shared-fixture repair-policy diagnostic, not full end-to-end extraction
or M08 review-export evaluation. No review-time or harmful-correction truth is
invented. The next dependency is an agreed M10 local pilot/handover scope; stop here.

## M10 — Local synthetic pilot and handover (2026-10-08)

All Python checks below ran from D:/git/meta-data-extraction-agent, with
PYTHONDONTWRITEBYTECODE=1. Repository test/build commands used:

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE = '1'
~~~

Focused final commands, all exit 0:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_pilot_bundle.py -q -p no:cacheprovider
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_pilot_core.py -q -p no:cacheprovider
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_pilot_http.py -q -p no:cacheprovider
~~~

Results: **42 passed in 90.63s**, **54 passed in 131.52s**, and **94 passed in
11.37s**, respectively. The final combined safe regression command passed all
**2,217 tests in 534.21s**, including 190 M10 checks and all 2,027 previously
passing safe checks:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/offline tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py tests/test_alignment_entry_points.py tests/test_facility_staged_matching.py tests/test_dataset_name_evaluation_normalization.py tests/test_typographic_evaluation_normalization.py -q -p no:cacheprovider
~~~

The first combined run returned 2,209 passes and three failures in 629.39s:
two Windows client connection resets on early rejection with unread input, and a
response completing before its structured log event. Logging before the response
and bounded early-error half-close/drain resolved these failures. Five additional
tests cover delayed/oversized input, event ordering, a silent sender deadline and
30 repeated early rejections. Captures: tmp/m10-first-combined-tests.json and
tmp/m10-final-tests.json. Earlier development checks also exposed overlong Windows
fixture paths (46 passes/six failures), now covered by shortened test roots and a
production preflight; missing handover-guide setup errors disappeared when the
guide was written. One intermediate draft parameter ID exceeded Windows path
limits and was replaced by an explicit short ID. These were development failures,
not baseline failures or skipped final tests.

Actual isolated builds, both exit 0, preserve v1 and stage the final fixes in v2:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.pilot_bundle build --root . --run-id m10-20261008-pilot-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.pilot_bundle build --root . --run-id m10-20261008-pilot-v2
~~~

Each bundle has 64 pinned assets. Final runtime:
runs/agent/m10-20261008-pilot-v2/runtime; manifest SHA256
dd8abc945115542bc3df987b32531f4899aa8f958da6586b9969c567c9047234.
Gold/evaluation modules and data, credentials, released results and previous raw
runs are excluded. The existing empty LICENSE is retained as unresolved_empty;
no new license or redistribution permission is inferred.

Clean setup commands:

~~~powershell
$env:UV_PROJECT_ENVIRONMENT = (Join-Path (Get-Location) 'runs/agent/m10-20261008-install-v1/venv')
uv --cache-dir tmp/uv-cache sync --project runs/agent/m10-20261008-pilot-v1/runtime --frozen --offline --no-dev
uv --cache-dir tmp/uv-cache sync --project runs/agent/m10-20261008-pilot-v1/runtime --frozen --no-dev
$env:UV_PROJECT_ENVIRONMENT = (Join-Path (Get-Location) 'runs/agent/m10-20261008-install-v2/venv')
uv --cache-dir tmp/uv-cache sync --project runs/agent/m10-20261008-pilot-v2/runtime --frozen --offline --no-dev
~~~

The first command failed because python-dotenv 1.2.2 was absent from the local
cache. The frozen online retry downloaded public locked packages, prepared 38
packages and installed 62; no inference endpoint was called. The final v2 clean
installation succeeded fully offline, building the preserved editable distribution
and installing the same 62 locked packages. Python remains 3.13.7 and neither
pyproject.toml nor uv.lock changed. Environments are outside the immutable runtime
bundles and inside ignored runs/agent. Captures: tmp/m10-setup-checks.json.

The actual HTTP diagnostic ran against each bundle in its own clean environment:

~~~powershell
$pilotRuntime = (Resolve-Path 'runs/agent/m10-20261008-pilot-v1/runtime').Path
$pilotPython = (Resolve-Path 'runs/agent/m10-20261008-install-v1/venv/Scripts/python.exe').Path
$env:PYTHONPATH = (Join-Path $pilotRuntime 'src')
& $pilotPython tmp/m10_http_diagnostic.py $pilotRuntime $pilotPython
$pilotRuntime = (Resolve-Path 'runs/agent/m10-20261008-pilot-v2/runtime').Path
$pilotPython = (Resolve-Path 'runs/agent/m10-20261008-install-v2/venv/Scripts/python.exe').Path
$env:PYTHONPATH = (Join-Path $pilotRuntime 'src')
& $pilotPython tmp/m10_http_diagnostic.py $pilotRuntime $pilotPython
& $pilotPython -m meta_data_extraction.agent.pilot_bundle verify --root $pilotRuntime
~~~

All exited 0. Each diagnostic made 24 real loopback HTTP requests across three
server processes, but only eight mock provider calls and zero live calls. Start
performed one mock call, restart/resume three more, and completed replay zero.
Status did not dispatch. Unauthorized and foreign-origin access were rejected.
An explicitly simulated unresolved decision retained the original canonical
record and zero active attestations. Another four-call run accepted a simulated
edit with three source-bound attestations and exactly two changed leaves:
/datasets/0/dataset_description (500 K to 300 K) and
/datasets/0/sample_refs/0 (sample_2 to sample_1). Specimens stayed separate.
Duplicate response submission replayed; stale review and stale export returned
409; current exports matched across restart. Logs excluded tokens, source/review
text and reviewer identifiers. All diagnostic servers stopped in finally blocks.
Final bundle verification returned verified=true with the same 64 assets/digest.
Results: tmp/m10-http-diagnostic.json and tmp/m10-http-diagnostic-v2.json; sanitized
logs use the corresponding -logs.jsonl filenames. The helper's second invocation
was configured to write v2 captures without replacing v1 evidence.

The temporary browser preparation helper, tmp/m10_browser_preview.py, created
one further pending fixture in v1 with four mock calls. The computer-use in-app
browser entry attempted http://127.0.0.1:54679 twice, with a tool reset between
attempts. Both failed before successful tab creation (Windows sandbox helper /
trusted Node startup errors). The stop marker ended the preview process cleanly;
no page interaction or screenshot was captured. The final bundled inline script
was extracted and compiled once using V8 new Function; syntax passed, but DOM
execution and rendered layout were not tested. Static HTML/CSP tests passed.
Get-Command checks found neither docker nor node. Docker build/run and browser
usability therefore remain untested. No container/public service was deployed.

Across actual M10 diagnostics there were **20 mock calls and zero live requests**:
eight v1, four browser-fixture preparation, eight final v2. Unit-test mocks are
separate. No real .env read, ChatAI request, paid inference, tracing, data upload,
natural-gold inspection, inherited regeneration script or full extraction run
occurred. The shared ten-request/rolling-hour ChatAI authorization remains unchanged
in AGENTS.md. API request counts are not model request counts; unknown token usage
is not recorded as zero or interpreted as a measured live cost/latency result.

Final repository checks:

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m10_final_audit.py
git diff --check
git check-ignore .env tmp/m10-final-tests.json runs/agent/m10-20261008-pilot-v2/runtime/pilot_bundle_manifest.json runs/agent/m10-20261008-install-v2/venv/pyvenv.cfg
~~~

The protected-file audit passed: 1,723 starting files, 1,735 final files, no missing
files, unexpected changes or text findings. Only COMMANDS.md, DATA_POLICY.md and
PROGRESS.md changed among existing files; 12 M10 files were added. All 64 bundled
source pins match the checkout. Earlier code/tests/configs, released assets,
AGENTS.md, pyproject.toml and uv.lock are byte-identical to the starting inventory.
The outcome is retained in tmp/m10-final-audit.json. Git ignore checks pass and
git diff --check exits 0 with
the pre-existing LF-to-CRLF notices for .gitignore, pyproject.toml and uv.lock.
Normal shell execution failed before running commands because of sandbox setup
refresh errors; scoped commands used the reviewed executor. No automatic approval
rejection occurred. No lint/type checker is configured.

Experiment E10 is **not_run**: no actual operator usability session, expert review
time or residual-error observations were measured. E09's named empirical gates
remain blocked. The next step is an operator trial following PILOT.md and an
agreed real-document scope. M11 is optional, requires a measured bottleneck and
explicit training/data/compute authorization, and was not started. Stop after M10.

## M10 follow-up — Trial prepared for later (2026-10-08)

The user chose to defer participation. This follow-up adds only the checkout-side
scripts/agent/start_pilot.py and docs/agent/PILOT_TRIAL.md, plus progress records.
The pinned PILOT.md, runtime modules, fixtures, schemas and dependency files remain
unchanged. Browser initialization was attempted with cua.getState after resetting
the helper; it failed before exposing a tab with the same Windows sandbox setup
refresh error. No browser action or screenshot occurred.

An initial PowerShell draft parsed and prepared m10-trial-ready-v1 successfully
through the managed shell, but its diagnostic in a fresh powershell.exe -NoProfile
-NonInteractive -File process failed because script execution was disabled. That
new draft was replaced by the Python launcher; no execution-policy or OS setting
changed. Final launcher checks ran with the existing local environment:

~~~powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe scripts/agent/start_pilot.py --help
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m10_trial_launcher_check.py
.\.venv\Scripts\python.exe scripts/agent/start_pilot.py --bundle-id m10-launch-check-v1 --prepare-only
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m10_prepare_trial_record.py
Get-Command Set-Clipboard
~~~

All final commands exited 0. The diagnostic passed eleven checks: collision
rejection; parent environment unchanged on success/failure; path traversal
rejection; missing resume rejection; occupied port with no token disclosure;
actual loopback startup; unauthenticated health 401; authenticated health 200 with
zero child runs; redacted logs; owned process tree stopped; all 65 runtime files
(64 assets plus manifest) byte-identical. The test used a public synthetic pilot
token through an environment variable, never the ChatAI key. It made two HTTP
requests and zero mock/live model calls. It stopped only its own launcher process
tree with taskkill /PID <created PID> /T /F, then confirmed the listener was closed.
This checks forced process cleanup; interactive Ctrl+C remains a later operator
task. Capture: tmp/m10-trial-launcher-check.json; sanitized output:
tmp/m10-trial-launcher-logs.jsonl. The test never invoked clipboard mode.

Fresh Python preparation produced a separate m10-launch-check-v1 bundle, confirming
the final launcher's build path. The later operator trial uses
runs/agent/m10-trial-ready-v1/runtime, with 64 assets and manifest SHA256
9cb0d2a6b213cc6d1d23417915ecaf3d8d5a03c433d7761eb98cfe87bca48ff5.
The outer directory contains trial-plan.json (protocol/launcher/source/runtime
pins, two runs, 16 maximum mock calls, zero live requests, 45-minute session cap)
and observations.md (blank human outcomes/times). Existing completed bundles and
artifacts were preserved. Neither prepared bundle contains an executed workflow.
Set-Clipboard exists; it was not called, and clipboard execution remains untested.

Later launch command, not left running by this follow-up:

~~~powershell
.\.venv\Scripts\python.exe scripts/agent/start_pilot.py --bundle-id m10-trial-ready-v1 --resume-bundle
~~~

Final preservation/ignore checks use:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/m10_trial_final_audit.py
git diff --check
git check-ignore runs/agent/m10-trial-ready-v1/observations.md runs/agent/m10-trial-ready-v1/trial-plan.json tmp/m10-trial-launcher-check.json
~~~

The audit passed: 1,735 starting files and 1,737 final files, two intended additions,
only COMMANDS.md, PROGRESS.md and the M10 module record changed among existing
files, and no missing files, unexpected edits or text findings. All 64 bundled
assets and the protocol/launcher/plan hashes match. No operator workflow has
started. Ignore checks and git diff --check passed, with only the earlier
LF-to-CRLF notices. Result: tmp/m10-trial-final-audit.json.

The full 2,217-test suite was not rerun: no runtime module/test/config/dependency
changed, and the scoped launcher diagnostic exercises the new behavior directly.
Human E10 remains not_run; browser execution, clipboard mode and Docker remain
untested. There were zero model requests or real .env reads in this follow-up.
No automatic approval rejection occurred. Stop within M10; the prepared operator
session is next, with no automatic fine-tuning or natural-document experiment.

## M10 fresh-publication integration (2026-10-09)

Scope and interfaces are in modules/M10_integration.md; operator instructions and
the actual architecture are in PUBLICATION_RUN.md. The starting checkout had
1,737 inventoried files and a 2,217-test safe baseline. Existing module changes
are limited to the explicit M08 publication proposal catalog and M07's disabled
automatic-patch budget preflight. No dependency, schema, evaluator, released
benchmark asset or AGENTS.md change was needed.

Commands used the existing environment, without syncing or updating dependencies:

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE = '1'
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_publication_checks.py tests/agent/test_publication_standards.py tests/agent/test_publication_service.py -q --basetemp tmp/pi1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_publication_types.py tests/agent/test_publication_checks.py -q -p no:cacheprovider --basetemp tmp/pc-types-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_publication_types.py -q -p no:cacheprovider --basetemp tmp/pc-types-v2
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_publication_live.py -q --basetemp tmp/pl4
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_publication_run.py -q --basetemp tmp/pr2
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_publication_service.py -q -p no:cacheprovider
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_workflow.py tests/agent/test_workflow_graph.py tests/agent/test_review_store.py -q -p no:cacheprovider
~~~

The initial combined focus run had 122 passes and one fixture assertion failure:
the checker's prompt bound selected 25 rather than 28 long excerpts. A shorter
explicit excerpt budget made the issue-slot test exercise its intended case.
Types/bridge then passed 177 checks. Adding strict numeric timeout cases brought
configuration checks to 123, all passing; the bridge has 66 passing checks.
Standards adapter checks passed 26 cases, and their affected regressions passed
338. Affected M06 checks passed 168 cases. The final service suite passed 33 in
8.96s and existing workflow/checkpoint/review regressions passed 153 in 104.79s.

The final runner suite passed 27 in 67.16s, including actual subprocess resumes
after extraction, checking and proposal generation. Its first run had 25 passes:
one assertion incorrectly expected an old issue summary after a record revision,
and another correctly rejected implementation changes made during concurrent
development. The subsequent stable-code run passed completely.

The simulated ChatAI transport suite passed 12 in 18.79s. It exercises the actual
adapter, frozen stage prompt scopes, both policies, failed-attempt accounting,
unknown-history quarantine, unknown proposal usage, and the shared ten-attempt
limit across multiple publications. SDK responses and clocks are invented and
network access is forbidden. These are offline tests, not live measurements.
Earlier drafts of this test lacked copied extraction files, used Windows' default
text encoding, or addressed a serialized record property incorrectly; all were
fixed before the final pass. No dispatch safeguard was bypassed.

The complete safe regression command is:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/offline tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py tests/test_alignment_entry_points.py tests/test_facility_staged_matching.py tests/test_dataset_name_evaluation_normalization.py tests/test_typographic_evaluation_normalization.py -q -p no:cacheprovider --basetemp tmp/pf1
~~~

Result: **2,504 passed in 695.59s (0:11:35)**, exit 0; all 2,217 prior checks and
287 new integration checks pass. Capture: tmp/publication-full-tests.log.
No lint/type tool is configured in the current pyproject; no such tool is claimed
as run. These are the previously
inspected safe inherited tests, not the inherited full extraction experiment or
result-regeneration scripts.

Actual cross-process command-line diagnostic:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/publication_diagnostic.py
~~~

The helper froze runs/agent/m10-integration-diagnostic-v1/protocol.json before
dispatch, with protocol SHA256
7abcfdebdd48281977b7cf3e57e8e01f634f4897fb26627c1ff62e2bef0519c8.
Its two invented-source runs budgeted at most eight mock calls and zero live
requests. It then passed 19 actual CLI processes: automatic pause after three
upstream calls and resume with two checks, status, persistent review, explicit
simulated unresolved submission, duplicate replay, stale submission/export
rejection, and the separate three-call manual proposal path. Exact argument
vectors, exit codes and local process timings are in commands.json; result.json
and artifact-hashes.json retain the outcomes and integrity evidence.

Automatic parent m10-integration-auto-v1 changes only Dataset A's description
from 500 K to 300 K and its sample reference from sample_2 to sample_1. Manual
parent m10-integration-manual-v1 retains the initial canonical record and exposes
the generated proposal for review. Completed replay/read/export preserved 37
automatic-run and 26 manual-run files byte-for-byte. Both remain pending review;
the simulated unresolved decision is not approval. The diagnostic used exactly
eight mock calls and zero live calls. Fixture authoring used one separate mock
extraction, and the earlier preserved pub-debug-v1 run used five mock calls.
Unit-test simulations are counted separately from these 14 local diagnostic calls.

Preservation and handoff checks:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/publication_final_audit.py
git diff --check
git check-ignore runs/agent/m10-integration-diagnostic-v1/result.json runs/agent/m10-integration-auto-v1/sidecar.json tmp/publication-full-tests.log
~~~

The audit compares the starting file inventory, protected assets, Python syntax,
text/credential patterns and the earlier completed/prepared pilot manifests. It
does not read the real .env. Output is tmp/publication-final-audit.json. Existing
workspace states remain bound to their old implementation; migration is explicit,
and unchanged isolated pilot bundles remain usable. Raw artifacts stay ignored.

Final audit passed: 1,737 starting files, 1,758 final files, 21 intended additions,
five intended existing-file edits, no missing files, no unexpected edits, and no
text/credential findings. All 1,488 protected files and the two prior 64-file
pilot manifests match. The deferred human trial remains unstarted. Python syntax,
git diff --check and ignore checks pass; Git retains its earlier LF/CRLF and
unavailable global-ignore warnings without a command failure.

Real ChatAI requests, natural-paper scoring, an observed human trial, browser
interaction and Docker were not run. Runtime accepts only explicitly supplied
source/standards inputs and does not inspect gold. The real key was not read,
logged or committed. The standing qwen3.8-27b authorization and persistent shared
ten-attempt/hour control remain unchanged. No cloud resources were created.
An earlier escalated tool review failed because its usage quota was unavailable;
normal workspace execution subsequently worked and all affected checks ran. No
approval rejection remains unresolved. Stop within M10; M11 is not started.

## M10 one-paper live pilot preparation (2026-10-09)

Scope and preregistration: [M10_real_paper_pilot.md](modules/M10_real_paper_pilot.md).
Only one calibration PDF, 1812.03956, is allowed. ChatAI endpoint and model are
frozen; maximum three calls, no retries, no automatic edits. The shared ledger
was absent and was initialized at 09:55:15 UTC. Its real one-hour unknown-history
quarantine ends no earlier than 10:55:15 UTC / 12:55:15 Berlin. No clock or history
was fabricated. The source, prompt, profile, evaluator, split and limits were
recorded before inference, under `runs/agent/m10-real-paper-prep-v2/`.

Commands used the existing frozen environment, without installing dependencies:

~~~powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE='1'
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/real_pilot_preflight.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/prepare_real_pilot.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/select_real_pilot.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/freeze_real_pilot_evaluator.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/prepare_real_pilot_v2.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/freeze_real_pilot_evaluator_v2.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tmp/test_real_pilot_job.py -q --basetemp tmp/jobtest-v1
~~~

Preflight saved a 1,758-file starting inventory and initialized persistent shared
accounting. The `.env` has `LLM_API_KEY`; its intended endpoint appears in a comment,
not a parsed endpoint variable. Safe inspection found that URL without displaying
the key. No credential value is recorded in the protocol or documentation.

The first source preparation failed the unchanged prompt-size assertion:
1804.07163 produced 111,989 bytes, above 100,000. Its exclusion/failure is retained
under prep-v1. Calibration source-length inspection, without gold values or model
outputs, selected 1812.03956; prep-v2's full prompt is 68,176 bytes. The v2 helper
wrote its artifacts successfully and then hit Windows console UnicodeEncodeError
while printing a title ligature. Subsequent JSON/spec/profile/hash validation
passed; the helper was not rerun over those artifacts. Evaluator freezing passed.
Read-only PDF inspection rendered the first page with existing PyMuPDF, with no
dependency changes. Runtime image/page-mapping limitations remain explicit.

The launcher passed **17 synthetic tests in 5.11s**. Tests use invented workspaces,
mocked subprocesses and clocks: no actual paper, credential, ledger mutation,
network or model call. They verify all-code seal changes, three-slot readiness,
exclusive claims, no retry after failure/timeout, changed collector rejection,
cancellation, wait deadline, missing ledger, environment isolation and suppression
of raw process logs. These are safety diagnostics, not live performance evidence.

The previous 2,504-test full safe suite remains the runtime baseline; runtime,
schemas, evaluator, dependencies and tests are unchanged by pilot preparation, so
that eleven-minute suite was not rerun. No inherited extraction experiment or
result-regeneration script was executed. Natural scoring and human review remain
unrun until explicitly recorded by the job's result artifacts.

Collector preflight and queued-job launch:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tmp/test_collect_real_pilot.py -q --basetemp tmp/pc1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/run_real_pilot_job.py seal
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/run_real_pilot_job.py check
& .\tmp\launch_real_pilot.ps1
~~~

Collector tests passed **4 in 6.74s**, covering an actual synthetic manual workflow
through collection/report/replay, failed extraction, an incomplete-parent scoring
gate and the actual frozen evaluator. Five fixture-only mock invocations, zero
live requests and no natural gold values were used. The strict evaluator snapshot
required JSON-mode deserialization; this experiment-helper fix is covered. The
collector SHA256 is
`457ea8a557927efa7414888cbd6a77b186efc83713c2ad1c27f3daeea6d0bd2a`.

Sealing passed with 93 frozen files and zero model calls. Readiness verification
passed its integrity checks and correctly returned `waiting` /
`request_history_unknown`, with ten free slots and zero calls. The launcher then
started hidden local process 12548 at 10:19:08 UTC, with network-sandbox escalation
approved for this already-authorized bounded job. It does not create a scheduled
task, cloud resource or external tracing session. The single-use claim and
`job-events.jsonl` record execution; `job-finished.json` records completion or an
explicit stop. Local `report.json` and `report.md` are created only after collection.

The queued pipeline's exact command is the existing interpreter followed by:

~~~powershell
-m meta_data_extraction.agent.publication_cli run --root D:\git\meta-data-extraction-agent --spec runs/agent/m10-real-paper-prep-v2/publication-spec.json --chatai-endpoint https://chat-ai.academiccloud.de/v1 --load-dotenv
~~~

After completed runtime verification, the job invokes
`python tmp/collect_real_pilot.py --prep runs/agent/m10-real-paper-prep-v2`
as a separate offline process. These commands are queued, not reported as already
completed measurements. They use the frozen run and score IDs and do not regenerate
released artifacts. A pre-dispatch cancellation can be requested by creating
`runs/agent/m10-real-paper-prep-v2/cancel.request`; a claimed/interrupted job never
automatically retries. Keep this workspace's sealed inputs and environment unchanged
while the job is pending; a mismatch stops execution.

Preservation checks:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/audit_real_pilot.py
git diff --check
git check-ignore runs/agent/m10-real-paper-prep-v2/protocol.json tmp/run_real_pilot_job.py tmp/collect_real_pilot.py .env
~~~

Audit: 1,758 starting files and 1,759 final files; only COMMANDS.md and PROGRESS.md
changed, with the new M10 real-paper experiment record added. No missing files,
unexpected changes or credential findings. All 1,478 files under the five protected
data/result trees, and all code/tests/configuration/schema/environment pins, match
the starting inventory. The selected ignored PDF is separately hash-bound by the
protocol and seal. Python helper syntax and ignore checks pass. Git retains its
pre-existing LF/CRLF and unavailable global-ignore warnings; checks return exit 0.

## M10 provider-limit verification (2026-10-09)

The user requested checking ChatAI's limits and adapting the local cap. Reviewed
the official [SAIA API Limits documentation](https://docs.hpc.gwdg.de/services/ai-services/saia/index.html#api-limits).
It exposes account limits and remaining quota through response headers. Published
example values were not treated as the configured key's actual quota.

Before dispatch, created the old pilot's `cancel.request` with the reason for the
quota check. The existing worker exited normally at 10:25:55 UTC with
`operator_cancelled_before_dispatch`, `no_automatic_retries=true`, and zero model
requests. Its original preparation, seal and claim remain immutable. Starting
inventory for this follow-up: `tmp/chatai-limits-starting-hashes.json`, 1,759 files.

Executed once, with network-sandbox escalation approved for this explicit check:

~~~powershell
.\.venv\Scripts\python.exe tmp/check_chatai_quota.py
~~~

The helper loads LLM_API_KEY privately and sends one GET, without a body, to
`https://chat-ai.academiccloud.de/v1/chat/completions`. It disables redirects and
retries, has a 20-second timeout, never requests generation, and records only
allowlisted numeric quota headers and safe status. It does not retain a response
body, credential or arbitrary HTTP headers. Request and result artifacts are in
`runs/agent/chatai-quota-check-20261009-v1/`.

Observed result: **HTTP 401**, no quota headers, 188.09ms local HTTP latency,
one administrative API attempt and **zero model calls**. This is authentication
failure, not an extraction performance measurement; provider-side quota charging
for the rejected request is unknown. The helper's `observed` status means an HTTP
response was received, not that authentication succeeded. No retry occurred.

The actual quota is unverified. The existing ten-request/hour cap and run budget
remain unchanged, and the pilot is blocked pending a valid credential/endpoint
and successful quota verification. AGENTS.md records this status and the rule for
adapting limits from observed headers. No production limiter, schema, dependency,
evaluator, released result or ledger history was changed. No regression suite was
rerun because runtime code is unchanged. A fresh job version is required before
resuming; the cancelled claim is not overwritten or automatically retried.

Final read-only inventory/credential/syntax audit passed (artifact:
`tmp/chatai-limits-audit.json`): 1,759 files before/after, exactly AGENTS.md,
COMMANDS.md, PROGRESS.md and the M10 pilot note changed; no added/missing files,
unexpected edits or secret matches. Shared model-attempt ledger remains zero;
the administrative HTTP attempt is recorded separately. The worker's actual
claim PID 23040 is no longer running. `git diff --check` and ignore checks passed
with the pre-existing Git warnings. No new model invocation or repeated quota
probe was made during verification.

## M10 endpoint documentation follow-up (2026-10-09)

The user questioned whether the endpoint caused the rejection. The official
[SAIA documentation](https://docs.hpc.gwdg.de/services/ai-services/saia/index.html#api-limits)
uses different hostnames for its inference and quota examples. The earlier quota
GET substituted the inference host. Its 401 did not establish that the key itself
was invalid. The existing Bearer header matches the documented authentication.

A private local formatting check reported only booleans: key present; no leading,
trailing or embedded whitespace; no pre-existing Bearer prefix; no difference
from disabling dotenv interpolation. It did not print the key or modify `.env`.

One corrected administrative request, separately authorized through network
sandbox escalation, reused the prior bounded helper with a fresh destination:

~~~powershell
.\.venv\Scripts\python.exe tmp/check_chatai_quota_v2.py
~~~

Exact method/URL: GET `https://saia.gwdg.de/v1/chat/completions`, no body, Bearer
authentication, no redirects, no retries, 20-second timeout. Result: **HTTP 401**,
no numeric quota headers, 185.36ms local HTTP latency, one administrative request,
zero model calls. Artifacts: `runs/agent/chatai-quota-check-20261009-v2/`.
No response body, arbitrary headers or credential values are retained. Across
both checks there are two administrative attempts and zero model invocations;
provider quota charging remains unknown. No additional requests were issued.

The documented inference base remains `https://chat-ai.academiccloud.de/v1`;
there is no evidence warranting an inference endpoint change. Authentication or
access remains unresolved, account quota is unverified, the 10/hour cap stays
in place, and the old pilot remains cancelled. The blocked inference work has
not restarted. Production code and dependencies are unchanged; no runtime
regression tests were rerun for this documentation/administrative check.

The endpoint follow-up audit (`tmp/chatai-endpoint-audit.json`) and
`git diff --check` passed: only the four intended guidance/progress documents
changed, no missing/added inventory files or credential matches, helper syntax
valid, shared model-attempt count still zero. Ignore checks cover `.env`, the
helper and local quota artifacts. Existing Git warnings remain unchanged.

## M10 internal offline performance assessment (2026-10-09)

The user requested evaluation by the assistant while ChatAI is unavailable.
Scope/result: [M10_self_assessment.md](modules/M10_self_assessment.md). No live
service or network request, no credential read, no natural gold/evaluator output
read, and no runtime implementation change occurred. The cancelled pilot stayed
cancelled; the ten-request/hour cap stayed unchanged.

Current focused test command:

~~~powershell
$testBase=Join-Path (Get-Location) 'tmp/performance-evidence-pytest-20261009-v1'
if (Test-Path -LiteralPath $testBase) { throw 'Refuse existing pytest base' }
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE='1'
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_publication_checks.py tests/agent/test_publication_run.py tests/agent/test_publication_live.py tests/agent/test_study_runner.py tests/offline/test_study.py -q --basetemp $testBase
~~~

Result: **230 passed in 76.82s**, exit 0. The live-adapter tests use invented SDK
responses/clocks, not a service. An initial invocation using the default Windows
temporary directory hit fixture-setup permission errors and was interrupted;
a `-x --tb=short` diagnostic confirmed inaccessible
`C:\Users\huzha\AppData\Local\Temp\pytest-of-huzha`, before product execution.
The fresh workspace basetemp above resolved the environment issue. The prior
2,504-test full safe suite was not rerun because runtime code is unchanged.

Saved-artifact verification:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/audit_saved_performance.py
~~~

Exit 0. Network and mock/live provider dispatch are actively forbidden by the
audit. Thirteen M09 completion manifests, 264 reproduced metric rows, 176 field
comparisons, four primary comparisons and 24,629 identical Markdown bytes verify.
All 63 M10 saved hash pins verify; fresh read-only publication status reports
zero new calls. All 152 audited files remain byte-identical. Original counts
(fifteen M09 mock calls, eight M10 integration mock calls) are historical and
were not re-executed here. Summary: `tmp/performance-saved-artifact-audit-20261009.json`.

Fresh source-only diagnostic:

~~~powershell
.\.venv\Scripts\python.exe tmp/freeze_self_assessment.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/run_self_retrieval.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/audit_self_retrieval.py
~~~

The first helper freezes ten questions, four conditions, source/code pins and a
1,759-file starting inventory before retrieval. Source-only inspection of the
original PDF and its frozen prepared text authored exact reference phrases and
actual PDF page references; the PDF pages were rendered/visually inspected using
the installed PyMuPDF. All 36 source excerpts and nineteen reference alternatives
verify against the original/prepared text. The reference file was created before
retrieval and remains hash-bound by `execution-lock.json`. Eight questions with
fourteen components are answerable; two stay outside that denominator, with
missing source information explicitly recorded. No benchmark gold was used.

Forty deterministic retrieval/prefix selections ran with zero model/network
requests. Lexical 1,500/6,000-character conditions yielded respectively 2/8 and
3/8 complete exact-reference captures; prefix controls yielded 0/8 and 1/8.
Component counts are 4/14, 6/14, 0/14 and 3/14. Independent arithmetic/pin checks
passed and all 331 repeated evidence-reference instances resolve. This is an
internal phrase-capture diagnostic on one calibration publication, not extraction
accuracy, production-query performance or independent semantic truth. No runtime
fix or reference tuning followed the observed results.

Artifacts: `runs/agent/m10-self-assessment-v1/` and
`runs/agent/m10-self-source-audit-v1/`. The source, ignored run artifacts and
released results remain unchanged by reporting. Full commands and provenance
for the source-only note are retained beside its scope and validation artifacts.

Final assembly and preservation checks:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/finalize_self_assessment.py
git diff --check
git check-ignore runs/agent/m10-self-assessment-v1/retrieval-results.json runs/agent/m10-self-source-audit-v1/evidence-note.json tmp/run_self_retrieval.py
~~~

All passed. Final inventory: 1,759 starting files, 1,760 final; one new assessment
record and three intended existing-document edits, no missing or unexpected files.
All 1,478 protected data/result files and baseline source/tests/configuration/
schema/AGENTS/environment files match. No credential-pattern findings; the actual
credential was not read. Helper syntax and frozen diagnostic input pins verify.
The shared model ledger remains at zero and the prior pilot remains cancelled.

## M10 evidence follow-up — 2026-10-09

Scope: offline runtime fixes for PDF review proposals and selectable initial
support retrieval. This supersedes the earlier statement that every PDF blocks
proposal generation; automatic PDF edits remain blocked. No M11 work, provider
retry, credential read, natural gold inspection or dependency change.

All invocations used `PYTHONPATH` set to the checkout's `src` and
`PYTHONDONTWRITEBYTECODE=1`, with frozen/offline/no-sync uv. Each basetemp was new.

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_retrieval_focused.py tests/agent/test_retrieval.py -q --basetemp tmp/pytest-retrieval-focused-20261009-b
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_publication_checks.py tests/agent/test_publication_run.py -q --basetemp tmp/evidence-bridge-pytest-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_publication_pdf_proposals.py -q --basetemp tmp/pdf-proposals-pytest-v1
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_publication_retrieval.py tests/agent/test_retrieval_focused.py tests/agent/test_retrieval.py -q --basetemp tmp/pytest-pubretr-v2-b
~~~

Results respectively: **51 passed in 0.37s**, **93 passed in 54.66s**,
**14 passed in 14.31s**, and **68 passed in 3.78s**. The last command includes the
51 retrieval tests; these commands cover 175 distinct tests, including 65 new
cases. During development one new retrieval test's expected newline/separator
count was corrected before the passing run; no production defect was involved.
All provider results were authored mocks; even auto-requested PDF cases perform
only three mock calls, persist an unreviewed proposal, and leave canonical output
unchanged. Direct atomic application still rejects incomplete source reports.

A synthetic 100,002-character repetitive paragraph took 0.214s for focused
retrieval, selecting 800 characters. This checks local implementation cost only,
not provider latency or a statistical performance estimate.

A fresh frozen protocol compares `lexical_v1` and `focused_v2` at character caps
1,500 and 6,000, top-three regions, with the same ten questions and unchanged
internal references from the earlier assessment. Generic implementation and
synthetic checks preceded the replay; no tuning followed the results.

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/run_evidence_followup.py
git diff --check
git check-ignore runs/agent/m10-evidence-followup-v1/retrieval-results.json tmp/run_evidence_followup.py
~~~

All exited 0. Complete reference-phrase capture changed from 2/8 to 4/8 at
1,500 characters and 3/8 to 4/8 at 6,000; components changed from 4/14 to 7/14
and 6/14 to 7/14. Two questions remain unanswerable and outside the denominators.
This is one-paper development replay, not extraction accuracy or held-out
validation. See `modules/M10_evidence_followup.md` for method, scope and trade-offs.
Existing Git warnings about inaccessible global ignore configuration and line
ending conversion persist; the commands succeeded.

Independent replay audit verified 18 frozen hashes, 40 metric cells and 111
evidence instances (41 distinct links). All 20 legacy selections match the old run
exactly. At 1,500 characters q09/q10 improve with three component gains; at 6,000
q09 improves with one component gain. No primary component or complete question
regressed at either cap. Partial context for unanswerable q08 remains excluded.
The audit is `runs/agent/m10-evidence-followup-v1/retrieval-audit.json`.

Full safe regression command (log captured under the fresh run):

~~~powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE='1'
if (Test-Path tmp/ef-full-v1) { throw 'basetemp exists' }
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent tests/offline tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py tests/test_alignment_entry_points.py tests/test_facility_staged_matching.py tests/test_dataset_name_evaluation_normalization.py tests/test_typographic_evaluation_normalization.py -q -p no:cacheprovider --basetemp tmp/ef-full-v1 *> runs/agent/m10-evidence-followup-v1/full-suite.log
$testExit=$LASTEXITCODE
Get-Content runs/agent/m10-evidence-followup-v1/full-suite.log -Tail 8
exit $testExit
~~~

Result: **2,569 passed in 699.09s (0:11:39)**, exit 0. All 2,504 previous tests and
65 additions passed. Default tests enforce offline behavior; mocked live-adapter
tests do not send real requests. No lint or type checker is configured in
`pyproject.toml`; no such check is claimed. Released result regeneration, broad
live extraction and human usability trials were not run.

Final artifact and preservation check:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/finalize_evidence_followup.py
git diff --check
~~~

The full hash comparison found exactly four runtime edits, three existing-document
edits, three new test files and one module record. Final inventory is 1,764 files,
with no missing files. All 1,478 protected files, configuration/schema/environment
files, unchanged strict application guards and 35 frozen prior artifacts verify.
The 1,760-file baseline was reconstructed from the prior audited inventory plus
its three documentation edits and new report because the retrieval worker had
already edited its owned files when the new inventory was recorded. This method
and previous audit hash are explicit in `inventory-note.json`.

No credential-pattern findings; actual credentials were never read. Protocol pins
and helper/runtime/test syntax verify. The shared ledger remains at zero model
attempts and the previous pilot remains cancelled. New `assessment.json` and
`final-audit.json` preserve these outcomes under `runs/agent/m10-evidence-followup-v1/`.
`assessment.json` combines the observed evidence; `final-audit.json` retains the
preservation result under the new assessment run. Existing Git warnings are unchanged.

## M10 source-first reference readiness — 2026-10-09

The user confirmed ChatAI access is still not corrected. No authentication retry,
model call or live job ran. Added only an offline reference packet/validator and
synthetic tests; existing runtime, evaluator, schemas and dependencies remain fixed.

Focused commands used `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1` and fresh workspace
basetemps:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/offline/test_reference_types.py -q --basetemp tmp/ref-types-v5 --tb=short
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/offline/test_reference_review.py -q --basetemp tmp/reference-review-io-pytest-v4
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/offline/test_reference_omissions.py -q --basetemp tmp/ref-omissions-v2
~~~

Results: **51 passed in 0.53s**, **32 passed in 15.95s**, and **2 passed in 1.21s**.
These 85 new cases include blank/partial/complete declared reviews, source/evidence/
revision checks, missingness, immutable IO, excerpt generation and omissions through
the unchanged scorer. An invented two-entity reference with one correct predicted
entity yields TP=1, FN=1, FP=0 and entity F1=2/3 for samples and datasets separately.

Development checks exposed and fixed the new packet's incorrect CRLF prompt hash
assumption and nonfinite-value serialization risk. Independent review caught the
subset-schema/zero-count ambiguity; both packet and spec now require the full
canonical pair. Test-only funding-field, newline-decoding and absent aggregate
precision/recall-key expectations were corrected. The evaluator was not modified.

Source-only natural preparation and verification:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/prepare_reference_readiness.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.offline.reference_review prepare --spec runs/agent/m10-reference-readiness-v1/packet-spec.json
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/verify_reference_readiness.py
~~~

All exited 0. The protocol was frozen before preparation. Packet
`m10-reference-181203956-v1` remains `awaiting_human_reference`. All 78 prepared
source regions resolve, the 22,929-character bundle exactly matches the earlier
prepared source, and unavailable-image/page-mapping warnings remain intact.
All six checklist collections are unreviewed; canonical record, reviewer and
evidence remain unfilled. No prior gold, predictions or assistant answers were
included. Original PDF and natural source artifacts remain local and ignored.

`m10-reference-blank-check-v1` is explicitly an engineering check of the empty
template, not a human submission. It returns `incomplete`, all six declared entity
counts null, zero annotations and no canonical/gold export. Packet and receipt
replays preserve bytes and mtimes. Verification forbids sockets/DNS, records zero
model requests, and writes `runs/agent/m10-reference-readiness-v1/packet-audit.json`.

Combined safe regression:

~~~powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE='1'
if (Test-Path tmp/ref-regression-v1) { throw 'basetemp exists' }
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/offline tests/agent/test_storage.py tests/agent/test_sources.py tests/agent/test_smoke.py tests/agent/test_record_checks.py tests/agent/test_profiles.py tests/agent/test_paths.py tests/agent/test_evidence.py tests/agent/test_config.py -q -p no:cacheprovider --basetemp tmp/ref-regression-v1 *> runs/agent/m10-reference-readiness-v1/regression.log
$testExit=$LASTEXITCODE
Get-Content runs/agent/m10-reference-readiness-v1/regression.log -Tail 7
exit $testExit
~~~

Result: **591 passed in 117.18s**, exit 0. Includes 85 new cases, all existing
offline tests and the reused adapter/import-boundary checks. The previous full
runtime suite's 2,569-pass baseline is unchanged; no new runtime code justified
repeating the whole suite. No configured lint/type tooling, inference, result
regeneration, natural gold scoring or human usability session was run.

Final preservation and syntax checks:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/finalize_reference_readiness.py
git diff --check
git check-ignore runs/agent/m10-reference-181203956-v1/packet.json runs/agent/m10-reference-readiness-v1/final-audit.json tmp/finalize_reference_readiness.py
~~~

Audit: 1,764 initial / 1,770 final files; three existing-document edits, two new
offline implementation files, three new tests and one module record; no missing
or unexpected changes. All 1,478 protected files, 47 previous artifacts, existing
runtime/scorer/schema/config/environment/guidance files, packet/protocol pins and
blank receipt remain unchanged. No credential-pattern findings; no credential
read. Shared model accounting remains zero and the previous pilot remains
cancelled. Existing Git ignore/line-ending warnings persist without failing checks.

## M10 comprehensive component evaluation (2026-10-09)

Unchanged-runtime offline comparisons, authored failure probes and one previously
inspected natural retrieval development set. No natural gold, credentials, auth
attempt or live inference. Report: `modules/M10_component_evaluation.md`.
Artifacts: `runs/agent/m10-component-eval-v1/`. The initial read-only inline Python
inventory recorded 1,770 workspace files and 16,978 earlier run files. Each helper
then froze its cases and code/resource hashes before diagnostic execution.

Environment and exact experiment commands:

~~~powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE='1'
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/evaluate_component_retrieval.py freeze
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/evaluate_component_retrieval.py run
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/component_checks_20261009.py freeze runs/agent/m10-component-eval-v1/checks-evaluation
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/component_checks_20261009.py run runs/agent/m10-component-eval-v1/checks-evaluation
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/verify_component_checks_20261009.py
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python runs/agent/m10-component-eval-v1/standards-evaluation/evaluate_standards.py freeze
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python runs/agent/m10-component-eval-v1/standards-evaluation/evaluate_standards.py run
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -B runs/agent/m10-component-eval-v1/orchestration-evaluation/evaluate.py freeze
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -B runs/agent/m10-component-eval-v1/orchestration-evaluation/evaluate.py run
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -B runs/agent/m10-component-eval-v1/orchestration-evaluation/audit.py
~~~

Preparation failures: retrieval's first freeze exited 1 on an unmatched closing
parenthesis; standards' first freeze exited 1 because its invented workspace lacked
root markers. Both were corrected before protocol creation or experimental
outputs. All frozen diagnostic passes exited 0 once, without post-result tuning.
Orchestration also ran syntax checks and two actual child Python processes for
resume; component command records retain those invocations and read-only audits.

Results: 40 old natural selections reproduced exactly; ten synthetic retrieval
cases; 27 extraction/support/patch cases using 31 mock calls; orchestration's
22 rows plus replay/resume using 39 mock calls. Standards: 50 known resource
strings, 18 gates, six mapping requests, three definitions, 14 quantity cases plus
one inconsistent-input probe and three literal mention probes. Zero live calls.
No result-regeneration scripts, evaluator changes or independent labels.

Fresh non-overlapping regression commands; each temporary directory was checked
absent before use:

~~~powershell
if (Test-Path tmp/component-regression-v1) { throw 'basetemp exists' }
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_sources.py tests/agent/test_evidence.py tests/agent/test_retrieval.py tests/agent/test_retrieval_focused.py tests/agent/test_extraction.py tests/agent/test_record_checks.py tests/agent/test_support_checks.py tests/agent/test_record_patches.py tests/agent/test_panet_lookup.py tests/agent/test_panet_mapping.py tests/agent/test_nexus_registry.py tests/agent/test_nexus_units.py tests/agent/test_workflow.py tests/agent/test_workflow_policies.py tests/agent/test_workflow_graph.py tests/agent/test_review.py tests/agent/test_review_store.py tests/agent/test_review_cli.py tests/agent/test_providers.py tests/agent/test_pilot_core.py tests/agent/test_pilot_http.py tests/agent/test_pilot_bundle.py tests/agent/test_publication_checks.py tests/agent/test_publication_run.py tests/agent/test_publication_pdf_proposals.py tests/agent/test_paths.py tests/agent/test_storage.py tests/offline/test_reference_omissions.py -q -p no:cacheprovider --basetemp tmp/component-regression-v1 *> runs/agent/m10-component-eval-v1/regression.log
$testExit=$LASTEXITCODE
Get-Content runs/agent/m10-component-eval-v1/regression.log -Tail 10
exit $testExit
~~~

Result: **1,400 passed in 299.52s**, exit 0. Additional provider/service/resource
boundaries, with no repeated test files:

~~~powershell
if (Test-Path tmp/component-boundaries-v1) { throw 'basetemp exists' }
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent/test_publication_live.py tests/agent/test_publication_service.py tests/agent/test_panet_resources.py tests/agent/test_nexus_resources.py -q -p no:cacheprovider --basetemp tmp/component-boundaries-v1 *> runs/agent/m10-component-eval-v1/boundary-regression.log
$testExit=$LASTEXITCODE
Get-Content runs/agent/m10-component-eval-v1/boundary-regression.log -Tail 8
exit $testExit
~~~

Result: **147 passed in 15.82s**, exit 0; total **1,547 passed**. Tests named
publication-live use fake SDK responses/clocks. HTTP tests permit only ephemeral
loopback connections. No external network, browser rendering, Docker or human
trial ran. No configured lint/type tool. Product code is unchanged, so the full
suite was not repeated. Mock tests are not extraction performance observations.

Independent read-only software cross-audits recomputed retrieval and checking
counts, preserved as `checks-evaluation/retrieval-cross-audit.json` and
`standards-evaluation/checks-cross-audit.json`. These are not independent
scientific judgments. Component audits verify frozen inputs, saved evidence,
three idempotent patch replays, three direct/persistent state parity pairs and
two actual cross-process resumes.

Final checks:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python tmp/finalize_component_evaluation.py
git diff --check
git check-ignore runs/agent/m10-component-eval-v1/retrieval/results.json tmp/evaluate_component_retrieval.py
~~~

`final-audit.json` records every baseline file, all 1,478 protected files, all
16,978 previous run files, unchanged shared quota ledger/cancelled job, frozen
helper pins, result counts, syntax and credential-pattern checks. Intended edits
are three documentation records and ignored run/helpers only. Existing Git
ignore/line-ending warnings do not establish a new failure.
