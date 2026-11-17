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

## Claude Opus 5.5 route follow-up (2026-11-17)

Question: can the same bounded one-paper interface select Opper's exact
`aws/claude-opus-5-5` route without replacing the historical GPT-4o default or
allowing model drift between stages? The minimal additive design exposes
`--model` on `prepare`, `catalog`, `run`, and `report`. Preparation freezes the
selected provider-prefixed ID in both the publication specification and protocol;
each later operation must repeat it and fails before network access on mismatch.

The baseline remains the prior pinned paper, issue-aware three-call workflow,
8,192-token per-call output reservation, hash-only raw retention, no retries,
no automatic patches, and no evaluator. A worked example is fresh run
`m10-opper-opus55-181203956-v1`: offline preparation froze the Claude AWS route
and the unchanged 68,176-byte first prompt. It reported
`request_history_unknown`, zero network/model requests, and therefore requires
the normal one-hour readiness wait unless the operator explicitly accepts
untracked external history under the existing run-scoped gate.

The main trade-off is an explicit repeated CLI value instead of inferring the
route from mutable local state. It is slightly more verbose, but makes operator
intent inspectable and rejects a different catalog or inference model before
dispatch. The focused offline suite passed 10 tests, including a synthetic
Claude catalog and a wrong-model no-network failure. Live experiment status is
`not_run`; no credential was read and neither the catalog nor inference endpoint
was contacted. The next dependency is the operator's authenticated catalog check,
retention review, and bounded live command after ledger readiness.

The first authenticated Claude catalog check returned HTTP 200 with the exact
model, 1,000,000-token context, 128,000-token output limit, EU region/residency,
and no prompt/document/model request. The following run command stopped before
inference with `successful_catalog_check_required`: one execution gate still
compared the saved route with the historical default constant. The gate now
compares with the explicitly repeated model, and the Claude test exercises the
full catalog-to-execute boundary with a fake publication runner. The focused
suite still passes 10/10; live inference remains `not_run`.

The operator then explicitly acknowledged untracked initial history and ran the
fresh `v1` scope. One request completed normally (`finish_reason=stop`) with
26,173 input and 2,053 output tokens, no truncation, and fully known usage. The
strict parser rejected the response as `invalid_json_object`; therefore the
parent ended `extraction_failed`, with no canonical record and no checker or
controller call. Hash-only retention preserved response SHA-256
`5ed049cd938e3ababc5dd3d5a474ed2302ef48616f6a966747d80695f6cd6522`, not the
response text, so the exact JSON defect cannot be reconstructed. At catalog
prices, mechanical usage multiplication is about USD 0.1603272, not a bill.

This is a failed live `pilot`, not a validated extraction. The immutable run must
not be retried or overwritten. A defensible next attempt needs a fresh ID and a
frozen output-control change, preferably Opper JSON-object/structured output;
retaining the next raw response locally would make another parse failure
diagnosable. That method change and any second paid call require a new scope.

At the operator's request, fresh `v2` freezes those controls explicitly:
`response_format=json_object` is part of `OpperRunSpec` and is propagated to the
OpenAI-compatible request, while `raw_retention=retain` is part of the publication
specification. Both values must be repeated at every launcher stage, so neither
can drift after preparation. Historical Opper scopes still default to text and
hash-only retention. The new request-shape, provider, publication-type, and
launcher tests passed offline (212 passed, one known Windows concurrency case
deselected). `m10-opper-opus55-181203956-v2` was prepared with zero network/model
requests; live status remains `not_run` for `v2`.

The operator then explicitly requested removal of the initial one-hour Opper
wait. `OpperRequestLedger` now skips only that new-ledger quarantine; the shared
ten-attempt rolling-hour limit, per-run request/output budgets, durable attempt
reservations, and all ChatAI behavior remain unchanged. Existing local attempts
are still counted, and unrelated external use remains outside local observation.
Offline regression passed 212 tests with the known Windows concurrency case
deselected. A read-only check of the real ledger reported nine remaining attempts
and no readiness error; it made no request.

The live `v2` extraction then completed one provider request but exhausted its
entire 8,192-token output reservation. The provider reported
`finish_reason=length`; extraction recorded `response_truncated=true` and did not
attempt parsing. Retained raw content begins with a Markdown JSON fence despite
the requested JSON-object response format, has unmatched object/array delimiters,
and ends mid-value. This establishes that the route did not enforce the requested
format and that truncation, not schema validation, caused the terminal failure.
Usage was 26,173 input / 8,192 output tokens; no downstream stage ran.

Fresh `v3` makes one controlled change: the per-call output cap is explicitly
frozen at 16,384 tokens and repeated at every stage. JSON-object request and local
raw retention remain unchanged. The launcher now accepts a bounded
`--max-output-tokens` value and uses the prepared value for catalog gates and
reported reservations. Offline regression passed 212 tests with one known
Windows concurrency deselection. Preparation recorded 49,152 maximum reserved
output tokens across three calls, `ledger_readiness=ready`, and zero requests.

## Claude Opus 5.5 v3 live outcome

`m10-opper-opus55-181203956-v3` completed on 2026-11-17 with two guarded calls
and status `pending_review`. Extraction used 26,173 input / 2,047 output tokens;
checking used 35,975 input / 2,562 output tokens. Both stopped normally. Total
usage is fully known at 62,148 input / 4,609 output tokens. Catalog-price
multiplication gives USD 0.3748492 for this run, not a billing receipt.

The extraction JSON and canonical schema are valid. The immutable record contains
three affiliations, five authors, one funding item, one facility, two samples,
and two datasets; it emits 59 assertions. Both proposed evidence links were
invalid, leaving all 59 assertions unlinked, and PDF images remain unavailable.
These facts require review and do not establish entity recovery or correctness.

The checker returned valid JSON and twelve judgments, but reversed the order of
`/datasets/0/measurement_technique` and
`/datasets/1/dataset_description`. The strict request-order binding rejected the
entire automated report as `invalid_support_response`; no judgment or patch was
accepted. Three review cases remain for checker failure, 47 assertions outside
the twelve-target budget, and incomplete source content. Automatic patching was
disabled. Operational status is a successful live `pilot`; scientific validity
remains unvalidated and pending independent review.

## Claude Fable 5.2 preparation (2026-11-17)

At the operator's request, fresh run `m10-opper-fable52-181203956-v1` freezes
`anthropic/claude-fable-5-2` for the same paper and workflow. It retains raw
responses locally, requests JSON-object output, permits at most three calls, and
reserves 16,384 output tokens per call. Offline preparation reported ready and
made zero network/model requests.

Opper currently lists only the Anthropic US route for this exact model, with
abuse-monitoring retention, no training, and catalog prices of USD 8/M input and
USD 40/M output. The conservative whole-run token-plan multiplication is USD
4.369152; this is a planning ceiling, not expected usage or a bill. Live status
is `not_run`; executing the command acknowledges the US route, retention posture,
and frozen resource cap.

## Claude Fable 5.2 v1 outcome and v2 effort control

The operator completed `v1` catalog and live execution. Its sole extraction call
used 26,173 input and the full 16,384 output tokens, ending at
`finish_reason=length`. The retained visible response is empty and hashes to the
empty-string SHA-256, so extraction correctly recorded truncation without a parse
attempt. No canonical record, evidence checking, proposal, or automatic patch was
created. Catalog-price multiplication is USD 0.864744, not a billing receipt.

The route defaults to `high` adaptive-thinking effort, and all 16,384 output
tokens were therefore consumed without visible content. Opper's live OpenAPI
schema exposes `reasoning_effort` on the OpenAI-compatible chat request; its live
catalog entry for this exact model advertises `low|medium|high|xhigh|max`. The
launcher now treats that value as a typed, frozen run parameter and passes it to
the existing provider adapter.

Fresh `m10-opper-fable52-181203956-v2` changes one factor to
`reasoning_effort=low`. Model, document and hash, JSON-object request, retained raw
output, 16,384-token per-call cap, three-call cap, no retries, and disabled
automatic patching remain unchanged. Offline preparation reported ready and made
zero requests. The deterministic affected suite passed 213 tests with two
concurrency cases deselected; the thread case passed alone but was timing-sensitive
inside the complete affected suite. Experiment status for `v2` is `not_run`.
