# M02 — Typed provider boundary and fixed extraction

Date: 2026-10-07. Scope: M02 only; no controller loop or standards lookup.

## Question and minimal design

Can the fixed workflow produce canonical metadata with resolvable evidence and
explicit failures before adding adaptive decisions? Example: two same-composition
samples remain distinct local IDs; a dataset's sample reference and its quoted
caption are stored against stable internal handles and a record revision.

Reuse M01 DocumentBundle/EvidenceRef, M00 path isolation, inherited schema/prompt
loaders, LLMExtractor and OpenAICompatibleLLM. Preserve the historical modules.
Freeze the actual custom schema and prompt bytes in an internal Profile. Validate
declared structural types/IDs/references without inventing a canonical Pydantic
replacement or importing evaluation knowledge.

The inherited-prompt variant calls the unchanged extractor through a capturing
provider adapter. The separate evidence variant requests a canonical `output`
plus evidence proposals. Both use one fixed provider call, the same source text,
and explicit parse/truncation/usage states. Evidence resolution verifies quotation
locations; it is not a scientific support judgment. Uncited assertions remain
explicitly unlinked, never implicitly supported.

New contracts: immutable Profile, ProviderResult, RecordSnapshot, entity handles,
assertion/evidence sidecar and run manifest. Store raw output locally under an
explicit retention setting, separate from canonical JSON. Invalid/truncated output
must never become a successful empty extraction. Export a completed run atomically
and idempotently; replay must not call the provider again.

Live ChatAI calls remain subject to the recorded standing approval. Use a shared
SQLite ledger with durable pre-dispatch reservations, at most ten attempts per
rolling hour across runs/workers, and bounded run budgets. Disable SDK retries.
New/unknown history starts with a conservative one-hour wait; do not invent prior
usage. Freeze endpoint/model/source/profile/prompt scope before any live call.

Trade-off: strict new-path diagnostics can reject an inherited output that the
historical top-level-key filter accepted. Keep both raw output and compatibility
behavior visible rather than changing schemas or silently repairing facts.

## Verification plan and readiness

Offline cases: compatibility prompt/output, strict empty canonical record versus
invalid/non-object output, truncation, deterministic mock, missing/unknown usage,
missing credentials, timeout/429 with no retry, shared persistent budgets, invalid
evidence, stale revisions, duplicate/local IDs, reference consistency, immutable
profile bytes, safe repeated export and no duplicate inference on rerun.

## Starting architecture and implementation

M00 already isolated explicit inputs and run directories. M01 supplied immutable
source/region snapshots and lexical retrieval. The inherited `LLMExtractor` used
the existing prompt and custom schema dictionaries, made one `BaseLLM` call, and
parsed JSON with a possible empty-object fallback. `SimpleValidator` filtered
unknown top-level keys. `OpenAICompatibleLLM` already built chat requests and
converted usage. There was no agent fixed-run envelope, persistent request ledger,
or evidence-producing extraction/export path. No LangGraph controller is needed
for this fixed stage.

Added production files under `src/meta_data_extraction/agent/`:

- `profiles.py`: frozen inherited resource snapshots; structural validation,
  canonical local-ID/reference checks and explicit validation limitations.
- `providers.py`: deterministic mock and the inherited ChatAI adapter, typed
  errors/usage, frozen live scope, no automatic retries or redirects, shared SQLite
  ledger. Both wall and monotonic clocks must allow an hour before allowance is
  restored. New history waits an hour; pending attempts remain charged.
- `contracts.py`: immutable canonical snapshots, revision-bound handles,
  reference links and assertion/evidence status, without new canonical fields.
- `extraction.py`: one fixed call, compatibility or evidence prompt, explicit
  parse/schema/truncation states and exact evidence resolution. Repairs remain
  visible. Unknown finish reasons or unavailable evidence require review.
- `storage.py`: exclusive execution claims, raw retention, checksummed exports,
  atomic completion marker and safe replay; interrupted claims never auto-rerun.
- `run_extraction.py`: explicit one-source orchestration, current-source/profile
  and evidence revalidation on cache replay, code/dependency provenance and an
  offline fixture CLI. Live programmatic entry requires the configured ChatAI
  endpoint and an exact source/profile/prompt specification.

Added five corresponding test modules, shared invented fixture helpers, three
synthetic fixtures, and `configs/agent/extraction_mock.yaml`. Narrowed the existing
runtime-import test to permit only the intended provider/SDK adapter route while
keeping evaluation, alignment, normalization, pipeline and script imports blocked.
Updated progress, commands and data-policy records. `AGENTS.md` already contains
the user's standing ChatAI approval and rolling limit; its contents are unchanged
in M02. No inherited implementation, schema, package identity or dependency changed.

## Worked example and outcomes

The invented source reports distinct Sample A and Sample B, both CaCO3, and a
dataset referring to both. Their `sample_1`/`sample_2` canonical identities remain
separate; changing list order preserves their internal handles. Dataset references
resolve to those handles, while evidence assertions bind to the record revision.
Idless funding handles use content plus duplicate occurrence: content changes
create a new handle, with no attempted cross-version identity inference.

Two offline CLI runs export identical canonical records. The evidence variant has
18 assertions with resolvable quotations, seven handles including the document,
and four reference links. The inherited variant explicitly says evidence was not
requested. A repeat of the evidence command returns `cached_replay`, with zero new
calls. Raw text is retained locally because these runs explicitly select that
policy. Reported token usage remains null, and mock cost is not applicable.
`succeeded` means structural completion and valid locations, not scientific support:
all assertions retain `support_status: not_assessed`.

Final demonstration runs:

- `runs/agent/m02-20261007-evidence-v1/`
- `runs/agent/m02-20261007-inherited-v1/`

An earlier `m02-20261007-evidence-check` development run is preserved separately;
subsequent code changes intentionally invalidate its request fingerprint. All
artifacts remain ignored, separate from released outputs.

## Checks, limits and handoff

The final checkout-bound combined offline command passed **439 tests** in 16.64s:
404 agent and 35 inherited regressions. An earlier combined run passed 435; after
aligning nested/storage request hash formatting, the affected runner suite passed
22 again. Four further tests exercise the installed OpenAI SDK against local
`httpx.MockTransport`, confirming one send for 429, redirect and timeout, plus
unknown usage on success. See `../COMMANDS.md` for exact commands. Agent tests deny
sockets/DNS. Concurrency tests use synthetic local ledgers, threads and processes,
never the live service.

All 15 M00 audit fingerprints remain unchanged. The ten files hashed before M02,
including AGENTS, the lockfile, inherited adapters and M01 source/path code, remain
unchanged. Released directories were neither regenerated nor edited. Git diff,
ignore and new-file inspections are recorded with the final verification.

Implementation: **ready**. Offline tests/integration: **pass**.
Empirical E02: **not_run**; live requests this module: **0**. No live ChatAI endpoint
or publication comparison manifest was frozen, and the real `.env` was not read.
No extraction quality, cost, latency, evidence support, or agentic improvement is
claimed. Fresh live accounting requires the documented one-hour history quarantine.

Remaining limits: full `uv sync` remains blocked by missing offline `hatchling` in
the local cache; the inherited full suite is intentionally skipped because it
opens benchmark/gold artifacts. No configured lint/type checker exists. POSIX
completion publication was not exercised on this Windows host. Accounting covers
this checkout's adapter; unrelated clients/other checkouts must share coordinated
limits. Ledger schema/clock regression and abandoned reservations require explicit
reconciliation, with no automatic reset. No adversarial filesystem protection or
cross-process review workflow is claimed. Boolean unknowns are unresolved in the
inherited schema and are never silently converted to false.

Next module: **M03 — Offline experiment harness and frozen evaluation**. It should
isolate the inherited scorer, freeze a publication split and evaluator versions,
and replay an explicitly approved subset without altering released results.
Stop here; M03 and adaptive control are not implemented in this turn.
