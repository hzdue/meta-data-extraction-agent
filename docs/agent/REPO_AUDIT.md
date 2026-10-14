# M00 repository audit

Observed 2026-10-07 in `D:\git\meta-data-extraction-agent`.

## Starting state

- HEAD: `74b8101f9572527101ea5c73abaddaf66aaba1e5`; branch: `hu`.
  HEAD matches the planning snapshot; the current branch is not snapshot `main`.
- Initial Git status: only untracked user-supplied `AGENTS.md` and
  `CODEX_MODULE_PLAN.md`. They were preserved, not rewritten or staged.
- There was no `agent/` implementation, agent configuration, agent test suite or
  `docs/agent/` record. Existing `docs/` contained annotation guidelines.
- Distribution `meta-data-extraction` 0.1.0, import `meta_data_extraction`, Python
  requirement `>=3.13`. These remain unchanged. No dependency or lockfile edit.
- Root `LICENSE` is zero bytes. Attribution remains intact; licensing and
  redistribution permissions remain unresolved.

## Actual architecture and reuse

| Implementation | Existing behavior and reuse decision |
|---|---|
| `base/{parser,llm,extractor,validator,tracker}.py` | Abstract component contracts; `LLMResponse` includes content, token counts and finish reason. Retain for adapters. |
| `data_models.py` | Pydantic Document/Section/Figure/Table, Provenance and ExtractionResult. Document has text and optional source path but no source hash/stable evidence address. Retain; evidence contracts are M01+. |
| `app_config.py` | Pydantic extraction configuration; live provider and tracker defaults. Do not use it as the new offline run configuration. |
| `pipeline.py` | `Pipeline.from_config` loads `.env`, instantiates PDF parser, live client, extractor, validator and tracker; `run` calls them in order. Preserve this inherited baseline. |
| `components/parser.py` | PyMuPDF4LLM PDF-to-Markdown plus deterministic heading splitting. Reuse through later source adapter; no page evidence invented. |
| `components/llm.py` | OpenAI-compatible chat-completions adapter, environment credential, configurable endpoint/token parameter/reasoning setting. `generate` calls external or local inference. Keep unchanged; later run authorization must precede use. |
| `components/extractor.py` | Schema prompts, optional released few-shot examples, character/token truncation warnings, JSON/fence/fallback parsing, raw output and usage/provenance. Failed parse can return `{}` plus warnings. Later adapter must expose failure explicitly. |
| `components/validator.py` | `SimpleValidator` drops unknown top-level fields. It does not validate nested structure, semantic support or completeness. |
| `components/tracker.py` | Local JSON writer, sanitizes config name, creates directories and replaces existing files. No run isolation and document ID is not a safe arbitrary path component. Keep historical behavior; do not use directly for new agent writes. |
| `utils.py` | YAML/AppConfig loading, schema merge/version/skeleton, zero/few-shot prompt rendering. M00 reuses `load_merged_schema` and `get_schema_refs` with validated schema paths. Few-shot loading is not invoked. |
| `config_validation.py` | Generic duplicate-key-rejecting YAML mapping reader. Reused by M00 without loading any evaluator config. |
| `evaluation.py` | Offline `evaluate(gold_path, pred_path, plan_path=None, identity_gold_path=None)` returns a report; `aggregate_reports` aggregates. It loads records and invokes alignment; no report write in `evaluate` itself. Reuse later only in an isolated offline harness. |
| `alignment.py`, `normalization.py` | Entity/reference matching and evaluator normalizers. Alignment loads `configs/alignment_helper.yaml`, including reviewed paper-specific forms. Entire path is offline-only; new-paper aliases/identity judgments require explicit review, not runtime reuse. |

No LangGraph implementation or dependency exists. M00 does not add orchestration,
extraction, standards lookup, evidence retrieval or a second provider stack.

## Schemas and configuration

`schemas/` contains `generic.json`, `photon_and_neutron.json`, their two
`*_no_instructions.json` variants, and two prompt templates. These are custom
field-definition dictionaries (`type: list[...]`, namespaces, instructions), not
ready-to-use JSON Schema documents. The full schemas report version 1.1.0; prompt
versions are 1.1.0 (zero-shot) and 1.2.0 (few-shot). Do not guess a replacement
canonical Pydantic schema or silently append evidence/standardization fields.

`configs/` initially contains 22 experiment YAML files, `evaluation_metrics.yaml`
and `alignment_helper.yaml`. Both evaluator configuration versions are 1.1.3.
Hosted settings include endpoint placeholders; local/institutional endpoints are
still live inference and not authorized by their presence. No credential values
were inspected. M00 adds only `configs/agent/offline.yaml`, with unknown keys and
live modes rejected.

## Available assets (inventory, not scoring)

| Location | Observed availability | Role |
|---|---:|---|
| `data/corpus.json` | 40 paper entries: 10 calibration, 30 evaluation | Offline split/annotation-pair manifest |
| `data/gold/` | 40 JSON files | Protected gold records; contents not inspected |
| `data/annotations/` | 60 JSON files in 3 annotator directories | Protected independent annotations; contents not inspected |
| `data/experiment/` | 660 JSON files across 22 condition directories | Protected released predictions; no new inference needed for an authorized replay |
| `data/few_shot/` | 3 JSON exemplars | Calibration text plus expected answers; excluded from new runtime sources |
| `data/raw/` | 40 local PDFs under paper-ID directories | Ignored source files; existence does not authorize inference or uploads |
| `data/processed/` | Absent | Historical ignored tracker default |
| `outputs/evaluation/` | 660 JSON reports | Protected released evaluator outputs |
| `outputs/iaa/` | 31 JSON files | 30 comparisons and summary |
| `outputs/paper/` | 1 file | Protected aggregated results |
| `paper/` | 26 files | Manuscript, styles, tables, figures and bibliography; protected |

Counts used directory listings only. The corpus manifest was read to establish
split membership/counts; no final-test gold/annotation/prediction contents were
read or used to change the runtime. M00 makes no claim about prediction quality.

## Side effects and external calls

| Entry point | Write or external behavior |
|---|---|
| `Pipeline.from_config` / `OpenAICompatibleLLM.generate` | Reads `.env`/configured credential; `generate` makes a chat-completions call. No M00 invocation. |
| `PdfParser.parse` | Reads supplied PDF through PyMuPDF4LLM. Only mocked conversion and heading logic tested; local PDF contents not parsed. |
| `LocalTracker.log_run` | Writes `<output_dir>/<sanitized_config>/<document_id>.json`, replacing files. |
| `scripts/run_experiment.py` | Defaults to evaluation corpus; reads `data/raw`; creates configured tracker output and `status.md`; can run live inference. Not a smoke command. |
| `scripts/run_evaluation.py` | Deletes/replaces reports in `outputs/evaluation/<condition>/`. |
| `scripts/compute_iaa.py` | Deletes/replaces comparisons and summary in `outputs/iaa/`. |
| `scripts/compute_paper_results.py` | Replaces `outputs/paper/results.json`. |
| `scripts/build_paper_tables.py` | Replaces six numerical tables in `paper/tables/`. |
| `scripts/build_paper_figures.py` | Writes `paper/figures/few_shot.pdf`. |

`arxiv` is an installed dependency; no active download call was found in the
inspected source/scripts. No script listed above was executed for M00.

## Tests and environment

Initial `tests/` has 27 test modules plus `__init__.py` and
`report_expectations.py`. Coverage includes schema/prompts, parser/output parsing,
provider requests, tracker, corpus/few-shot, runner behavior, alignment identities
and permutations, reference resolution, evaluation safety, aggregation and paper
rendering. Several evaluator/identity tests read final-test gold and are excluded
from the M00 baseline; the full suite was not run.

The six selected modules (`test_utils`, `test_sections`, `test_parse_output`,
`test_llm_request`, `test_tracker`, `test_prompt_templates`) contain 35 safe tests:
schema/prompt reads, synthetic/mocked inputs, request construction without API
calls and tracker writes confined to pytest temporary directories.

Environment: uv 0.8.22, CPython 3.13.7; installed Pydantic 2.13.2, PyYAML 6.0.3,
OpenAI 2.32.0, PyMuPDF4LLM 1.27.2.2, SciPy 1.18.0 and pytest 9.0.3. No configured
lint/type checker or CI workflow was found. The project `.venv` has a stale
editable install pointing at a sibling release checkout; the verified commands
set `PYTHONPATH` to this checkout and disable bytecode writes. A test now checks
the imported package origin. No other checkout was edited.

Default uv cache access is denied in this environment. Frozen offline sync using
`tmp/uv-cache` lacks cached `hatchling`; full sync remains blocked. Existing
dependencies support offline tests with `--no-sync`. See [commands](COMMANDS.md)
for exact invocations and failed attempts, and [M00 record](modules/M00_baseline.md)
for final results.

## Baseline fingerprints

SHA-256 of raw working-tree bytes, captured before implementation. CRLF/LF
conversion may change these across checkouts; compare using the same checkout
bytes. Git HEAD plus these schema/prompt/evaluator/config/dependency hashes fix
the inspected baseline, not a new evaluator or scientific result.

| Path | SHA-256 |
|---|---|
| `pyproject.toml` | `57ab3a7a8e41524ffb4c43069a1e32ae500c8a31495bfffc1573901991d21c24` |
| `uv.lock` | `5aae35c8ec94aa78854dab769ad45570668dc55c6a8ac23277e89b7f5618d22a` |
| `data/corpus.json` | `0af897afeb96f48b0dbdd81cbbbaac4bf035471cc5624f0f07d58d5fc5171816` |
| `schemas/generic.json` | `7cc7a487c4a7133459d387d45f8b158657c0ffd9824b77df84bdf25416ada258` |
| `schemas/generic_no_instructions.json` | `7c531e6900c74b7dce509fca0a98fac06285862a97337c59fd4425845b04713d` |
| `schemas/photon_and_neutron.json` | `e075c4929e41c8d1971c763f33f58dee24d7baae2132de516e37f202eef3fe4a` |
| `schemas/photon_and_neutron_no_instructions.json` | `8514bfd0967a5d9889328ba64b828f14038ffed2c44ace5efa997ed44e0aeb47` |
| `schemas/prompt_template.md` | `86cadac42e2b3c61ff3efec1c704537c54015c0f0c5df604820f0e6ade341b59` |
| `schemas/prompt_template_fewshot.md` | `a82d1331d8caa6b5f949ffc273a8f457262d5f60de98d48b4231d281a5ad3971` |
| `src/meta_data_extraction/evaluation.py` | `3bcb9f4d1de8eee1a850f1ea63cd83c70d0f70552cb7df77765fd0dcc366a5a9` |
| `src/meta_data_extraction/alignment.py` | `1f76750410a94172a93e0d50f555502b3082654cae37e58a09d273bc118abaf6` |
| `src/meta_data_extraction/normalization.py` | `cf7ca16b418da1a19990987c59f29d915bc1c3a70d4ba24b01bf4bbcdb58194d` |
| `src/meta_data_extraction/config_validation.py` | `f1264e8d47592c8708733d88d2450002566cb90827c75b6e677608b3e4620dbc` |
| `configs/evaluation_metrics.yaml` | `705f0e6813e2f740d81ad10f0be16ac58a1ccf976e55a8f55d4956f257b53792` |
| `configs/alignment_helper.yaml` | `71ae08c3e31f790f76885a9341f5f7516b97ea7f36f3ec13215d39626410b241` |
