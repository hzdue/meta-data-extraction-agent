# M10 follow-up - Opper 30-paper evaluation batch

Date: 2026-11-03. Scope: prepare a bounded, resumable batch launcher; do not
dispatch inference while implementing or testing it.

## Question and boundaries

Can the existing guarded publication workflow process the 30 publications marked
`evaluation` in the frozen `data/corpus.json` manifest through
`openai/gpt-4o-2024-08-06`, while preserving source isolation, request/token
accounting, and immutable per-paper outcomes?

Runtime may read only the corpus's split metadata, the 30 explicitly pinned PDFs,
ordinary schemas/prompts, and its own ignored run artifacts. It must not read
annotation files, gold records, evaluator configuration, alignment helpers,
historical predictions, or paper-result outputs. Extraction and later offline
evaluation remain separate operations. No evaluator is part of this launcher.

## Interface and minimal design

Add a frozen source-only config and a Python launcher with `prepare`, `preflight`,
`catalog`, `run-available`, and `status` operations. `prepare` verifies the exact
corpus hash, evaluation membership, and every PDF hash without credentials or
network. `preflight` renders each first prompt in a credential-free subprocess
with a per-paper timeout, retaining only sizes, hashes, and source warnings. Any
unavailable prompt, or one exceeding either the 130,000-byte or 119,000-token
guard, makes the 30-paper batch ineligible rather than silently dropping or
truncating a publication.

`catalog` performs one separately recorded authenticated `GET /models`, without a
prompt, document, redirect, or retry. Live execution requires explicit live,
retention, and catalog-derived cost-ceiling acknowledgement. Papers run
sequentially with the existing issue-aware controller, hash-only raw retention,
no automatic patches, no PaNET/NeXus enrichment, and at most three requests /
8,192 output tokens per paper. Each paper has an immutable run ID and result.

The shared Opper ledger retains its ten-attempt rolling-hour cap. Before beginning
or resuming a publication, `run-available` requires at least three available
global slots, so a worst-case paper cannot be knowingly started with an
insufficient hourly allowance. It stops with a resumable waiting status instead
of sleeping, resetting accounting, running workers concurrently, or bypassing
unknown history. The one-paper initial-history override is not generalized to
this batch.

Hard batch capacity is 90 requests, 10,710,000 guarded input tokens, and 737,280
reserved output tokens. These are capacity bounds, not
expected/billed usage. Exact provider usage is aggregated when present; unknown
remains unknown. Catalog prices may produce a labelled maximum planning estimate,
but not a billing claim.

The guarded input limit is 119,000 tokens per call, including a conservative
16-token chat-message allowance. Prompt bodies are counted with the GPT-4o
`o200k_base` encoding. The independent UTF-8 byte limit is 130,000 and remains a
second fail-closed bound.

## Baseline and required tests

Reuse the M01 PDF preparation, M02/Opper guarded adapter and ledger, and M10
publication runner. The one-paper pilot is the operational baseline: two requests,
39,983 input and 4,982 output tokens, pending review, with no automatic patch.
Do not infer 30-paper quality or exact cost from that single calibration paper.

Offline tests must cover exact 30-paper selection without gold access, source and
config tamper rejection, partial/idempotent preflight, timeout/oversize failures,
catalog redaction, cost acknowledgement, three-slot start gating, sequential
resume, immutable failed papers, aggregate known/unknown usage, and zero network
under default tests. Experiment status before execution: `not_run`.

## Implementation and outcome

The frozen configuration pins the corpus manifest, exact ordered set of 30
evaluation publication IDs, each source PDF hash, endpoint, model, controller
budgets, prompt/schema resources, and launcher protocol. The runner implements
the five operations above.

An initial byte-only diagnostic (`m10-e30-gpt4o-v1`) showed that 13 papers crossed
the former 100,000-byte limit. It is preserved as an unused diagnostic. The
revised `m10-e30-gpt4o-v2` preflight used exact `o200k_base` tokenization plus the
independent 130,000-byte bound. All 30 papers parsed and were eligible. Their
first prompts totalled 2,781,509 UTF-8 bytes and 824,184 exact body tokens; the
largest first prompt was 42,931 tokens. Preflight made zero network/model
requests, read no credential, and did not inspect gold.

At the prices observed in the pilot catalog, the hard planning ceiling is USD
34.1478. Live execution remains sequential and resumable, requires explicit
retention and cost-ceiling acknowledgement, and reserves three shared rolling-hour
slots before starting a paper. No batch catalog request or inference request has
been made. Experiment status: `not_run`.

Focused offline checks passed 145 launcher/publication tests and 5 targeted
provider/token/ledger tests. The broader selected agent suite had 452 passes, 2
intentional concurrency deselections, and 23 pre-existing failures caused by the
repository's frozen `publication_source.md` fixture hash mismatch; those failures
occur before provider construction and were not altered here.
