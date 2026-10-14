# M00 — Repository audit, protected baseline, and run isolation

Date: 2026-10-07. Scope: M00 only.

## Question and design

What already works, and how can additive development avoid overwriting or
contaminating the released benchmark?

Reuse the inherited strict YAML reader and schema utilities. Preserve the
parser, provider, extractor, validator, tracker, schemas, and evaluator unchanged.
Add an offline-only configuration, a fresh-run directory allocator and exclusive
JSON writer, and a synthetic configuration/input smoke command. This command is
not an extractor or evaluator.

Inputs: a strict config under `configs/agent/`, a portable run ID, explicit files
under `data/raw/` or `tests/agent/fixtures/`, and named inherited schemas.
Output: `runs/agent/<run_id>/smoke.json`, reporting input/schema checks only.
Reject unknown config keys, live mode, directories/globs, offline benchmark
inputs, path traversal, links, reused run IDs, and existing output files.

Example: `m00-smoke` allocates `runs/agent/m00-smoke/`; `../outputs` fails before
writing. `data/gold/example.json` cannot be selected as a source.

Trade-off: fixed input roots and one fixed output root are intentionally less
flexible than arbitrary paths plus a denylist. They provide a small auditable
boundary. New sources can be staged in ignored `data/raw/`; authorized offline
evaluation belongs in a separate future harness. No release-write override is
provided by the runtime.

## Baseline and required verification

Initial HEAD: `74b8101f9572527101ea5c73abaddaf66aaba1e5`, branch `hu`.
Initial changes: untracked user-provided `AGENTS.md`, `CODEX_MODULE_PLAN.md` only.
No agent implementation or module records existed.

Required checks: selected inherited parser/schema/prompt/request/tracker tests;
path rejection and exclusive creation/writes; strict source/config selection;
runtime import boundary; no-network synthetic smoke; unchanged protected tracked
assets and baseline hashes. Final commands/outcomes are recorded below and
in `../COMMANDS.md`.

## Status

- Implementation: ready
- Offline tests: pass (84 M00 tests and 35 inherited regressions)
- Integration: pass (synthetic input/config/schema/run-directory smoke)
- Empirical validation: not_run (E00 requires no new inference)
- Next dependency: M01 source preparation and evidence retrieval, after M00.

## Changed files

- `.gitignore`: ignore `/runs/agent/`.
- `src/meta_data_extraction/agent/__init__.py`, `paths.py`, `config.py`,
  `smoke.py`: additive path/config boundary and offline smoke entry point.
- `configs/agent/offline.yaml`: explicit synthetic source and inherited schemas.
- `tests/agent/conftest.py`, `test_paths.py`, `test_config.py`, `test_smoke.py`,
  `fixtures/synthetic_source.md`: network-denying tests and invented fixture.
- `docs/agent/REPO_AUDIT.md`, `DATA_POLICY.md`, `COMMANDS.md`, `PROGRESS.md`,
  `modules/M00_baseline.md`: inventory, fingerprints, boundaries and handoff.

No existing Python, schema, benchmark configuration, data or result file changed.
No dependencies added. The user-supplied planning files remain unmodified.

## Executed checks and outcomes

All final Python commands used PowerShell `PYTHONPATH` set to this checkout's
`src` and `PYTHONDONTWRITEBYTECODE=1`; see `../COMMANDS.md` for setup and earlier
failed attempts.

```text
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/agent -q
84 passed in 0.93s

uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m pytest tests/test_utils.py tests/test_sections.py tests/test_parse_output.py tests/test_llm_request.py tests/test_tracker.py tests/test_prompt_templates.py -q
35 passed in 1.54s

uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.smoke --run-id m00-20261007-check
Created runs/agent/m00-20261007-check/smoke.json; checks_passed; model_calls=0

git diff --check
Passed (Git emitted a line-ending advisory for .gitignore).

git diff --exit-code HEAD -- data outputs paper schemas configs pyproject.toml uv.lock src/meta_data_extraction/pipeline.py src/meta_data_extraction/components src/meta_data_extraction/evaluation.py src/meta_data_extraction/alignment.py src/meta_data_extraction/normalization.py
Passed: no tracked protected/inherited changes.

git check-ignore -v runs/agent/m00-20261007-check/smoke.json
Matched .gitignore rule /runs/agent/.
```

All 15 raw-byte baseline fingerprints in `REPO_AUDIT.md` were recomputed and
matched. Git status/diff and new files were inspected for unintended assets and
credentials. Incidental Windows cache files created during local tool execution
were preserved under ignored `tmp/m00-host-cache/` after verifying both move
paths were within this workspace. No cleanup of copied benchmark assets occurred.

The synthetic smoke resolves one invented input, hashes its bytes, reuses both
inherited schema loaders, and exclusively writes a diagnostic. It emits no
canonical extraction. For example, a run named `outputs` still resolves to
`runs/agent/outputs/`; `../outputs` is rejected. A second allocation or artifact
write fails. Tests also verify rejection of junctions/hard links and a run
directory replaced with a link before writing.

## Limits and decision

Keep this small foundation. M00 checks application-level path selection, not
adversarial filesystem races or mislabeled copies of gold placed in source
directories. Inherited tools do not acquire these guards automatically. No
release-write override or offline evaluation harness was added.

Full frozen environment sync is blocked by unavailable cached `hatchling` and
the default cache permission issue. The existing `.venv` has a stale sibling
editable path; explicit source binding made the final tests valid and is checked
by a regression test. Full inherited suite, PDF conversion, benchmark replay,
live inference, expert review and scientific comparisons remain unrun. No
configured lint/type checker was found. Licensing remains unresolved.

E00 status: `not_run` for empirical validation; no experiment, split, budget,
model judgment or performance claim was fabricated. Existing 660 predictions
are independently replayable in a later authorized offline scoring harness.

Recommended next module: M01, source preparation and evidence retrieval. Reuse
the inspected parser/document types behind explicit source selection and add
source hashes, stable regions and resolvable evidence addresses. Do not implement
M02 providers/controller behavior until its own authorization.
