# M10 follow-up - Opper one-paper GPT-4o pilot

Date: 2026-11-03. Scope: prepare a bounded launcher only; do not run live
inference in this implementation turn.

## Question, interface and minimal design

Can the existing publication workflow run one calibration paper through the
guarded Opper provider with an inspectable token budget and no accidental live
dispatch? The fixed inputs are `data/raw/1812.03956/paper.pdf` at SHA-256
`158984e801673f2ce019d9fd793684183bbc288e1b27856039aa37b6c9845bce` and model
`openai/gpt-4o-2024-08-06` at `https://api.opper.ai/v3/compat`.

The launcher has four explicit operations: offline `prepare`, authenticated
non-inference `catalog`, explicit `run`, and read-only `report`. Preparation
freezes the publication spec, launcher hash, protocol, and token plan in a fresh
ignored prep directory and initializes Opper's shared ledger without reading a
credential. Catalog checking sends no prompt or document, has no retry or
redirect, records a separate administrative attempt, and retains only allowlisted
model and numeric quota metadata. Run requires a successful catalog result,
sufficient advertised context/output limits, explicit live and retention
acknowledgements, available ledger history, and the local `.env` key before it
creates the publication parent. Report never dispatches.

Baseline: reuse M01 PDF preparation, M02 prompt/provider boundaries, and the M10
publication runner. Use `issue_aware`, at most three calls, 8,192 output tokens
per call, 100,000 prompt bytes, a 240-second request timeout, hash-only raw
retention, no automatic patches, no standards enrichment, and no evaluator.
Exact provider usage after a call is authoritative. Preflight reports exact
prompt characters/bytes, a labelled characters/4 heuristic, a byte-based
conservative planning bound, and maximum reserved input/output capacity; these
are not tokenizer measurements.

## Worked example and trade-off

For the pinned paper, the first rendered prompt is 68,033 characters and 68,176
UTF-8 bytes. The rough characters/4 estimate is 17,009 input tokens; the
conservative byte-plus-overhead planning bound is 68,304. Later prompts depend on
model output, so the run reserves at most 100,128 input-token equivalents and
8,192 output tokens per request, for three calls (300,384 input-token equivalents
and 24,576 output tokens across the run). The conservative input figures are
capacity bounds, not expected billable usage.

The main trade-off is deliberate over-reservation: a 100,000-byte prompt limit
and byte-as-token bound reduce context-overflow risk but substantially overstate
likely English token use. A successful model-catalog check therefore also has to
advertise at least 108,320 context tokens and 8,192 output tokens. Actual
`input_tokens` and `output_tokens` returned by Opper replace estimates; absent
usage remains unknown rather than zero.

## Tests and outcome

Offline tests cover immutable preparation, exact source/model/budgets, labelled
token estimates, one redacted catalog attempt, advertised limit gates, live and
retention confirmations, credential and ledger gates, and known/unknown usage
reporting. The focused launcher suite passed 8 tests. The combined provider,
publication types, extraction, support checks, workflow, and workflow-tools suite
passed 436 tests with one known Windows concurrency case deselected; its exact
isolated test had already passed.

No credential was read and no catalog or model request was sent. Experiment
status is `not_run`; implementation readiness is distinct from empirical
validation. The next M10 dependency is the operator's retention/tracing review,
one authenticated catalog check, ledger readiness, and then the explicitly
confirmed one-paper run. Do not start M11.

## Operator-requested initial-history override

The operator subsequently requested immediate execution and reported no known
successful Opper inference; the local Opper ledger independently contains zero
attempt reservations. Add an explicit run-scoped override for untracked external
history rather than disabling the rolling-hour cap. The override must be
exclusive, bound to the current ledger identity and run ID, retained without
credentials, and accepted only while local attempt count is zero. It acknowledges
unknown external history; it does not assert a verified zero. Normal per-run and
rolling request/output limits remain enforced, and ChatAI behavior is unchanged.

Offline override verification passed 20 focused tests, with 50 unrelated cases
deselected. The complete launcher/provider files passed 69 tests with the known
Windows process-concurrency case deselected. No model or catalog request was made
by these tests.

The first operator execution exposed a child-scope defect: the acknowledgement
covered the publication parent, but extraction dispatch is owned by its
deterministic `pub-...-extract` child. The child therefore stopped with
`request_history_unknown`. The Opper ledger contained zero reservations, so no
inference HTTP request was made and token usage remained unknown. The failed
parent remains immutable.

The corrected launcher creates acknowledgements for the parent readiness check
and exactly four model-capable child owners: `extract`, `check`, `proposal`, and
`provider`. Tests now verify each child independently passes readiness while the
fresh-ledger window is active. The focused 20-test selection and all 9 launcher
tests pass offline. A fresh `v2` run is required; the completed failed `v1` is not
rewritten or retried.

## Live pilot outcome

The fresh `m10-opper-4o-181203956-v2` run completed on 2026-11-03 with status
`pending_review`. Two guarded Opper requests completed successfully: 18,290 input
and 4,108 output tokens for extraction, followed by 21,693 input and 874 output
tokens for checking. Total measured usage is 39,983 input and 4,982 output tokens;
usage is fully known, with no uncertain dispatches. This is below the three-call
and 24,576 reserved-output-token limits.

The run produced an immutable canonical record and evidence/review sidecar. The
record contains one sample, one facility, and one dataset; the sidecar contains
43 extraction assertions and three workflow issues. It remains `pending_review`:
no automatic patch, expert approval, entity-recovery assessment, or evaluator was
applied. The runner's generic `empirical_status` field remains `not_run`; at the
module level this bounded operational measurement is a `pilot`, not validation.

Using the catalog's prompt/completion numeric prices as USD per token gives a
mechanical estimate of USD 0.0999575 input plus USD 0.04982 output, USD 0.1497775
total. The launcher did not interpret or persist this as billed cost, and Opper
billing/logs remain authoritative. The read-only `report` replay returned the
same two-call/token totals without dispatch.
