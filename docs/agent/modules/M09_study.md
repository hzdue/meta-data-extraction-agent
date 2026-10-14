# M09 — Controlled study configuration, execution and reporting

Status: synthetic study tooling ready. Natural end-to-end experiment E09 is blocked;
stop before M10.

## Question and starting architecture

What improves quality at a given resource cap, and where does adaptation fail to
help? M03 already provides explicit publication splits, a frozen inherited scorer,
support-aware field summaries and safe offline runs. M07 already executes bounded
fixed, issue-aware and generic selectors over the same synthetic source/record and
tools. M08 provides durable, explicitly scoped review with immutable revision
exports. M09 reuses these implementations without changing their historical behavior.

## Minimal design

A runtime study specification names supplied synthetic publications, workflow
config pins, policy variants, common call caps, repeats, total budget, primary
outcome, denominator policy and an opaque offline-protocol digest. Preflight
verifies all runtime inputs before dispatch, checks equal source/tool access and
estimates the worst-case request budget. Fresh immutable study manifests govern
deterministic child run IDs and preserve every planned cell, including failed,
budget-exhausted or missing results. Replay verifies frozen inputs and artifacts;
original usage remains separate from new replay consumption.

A separate offline protocol freezes the evaluator and explicit synthetic gold.
Collection invokes the unchanged M03 runner by cap/repeat, consumes stored raw
canonical predictions and generates a new JSON/Markdown report under runs/agent.
It retains per-field support and missingness, failed documents, harmed/improved
publication pairs and unknown resource quantities. Repeats are first paired within
the same cap and collapsed within publication; they never increase the independent
publication count. No schema-wide average is introduced.

The current executor is a synthetic repair-policy diagnostic using shared initial
records and standards disabled. It is not the complete A1/A2/A3 experiment or an
end-to-end extraction measurement. Missing reviewed NeXus guidance and independent
natural labels remain blockers. Human time, source-support error, mapping accuracy
and harmful-correction truth stay unavailable without independent observations.

## Frozen engineering protocol before execution

One invented publication: the existing M06 source (SHA256
a7f2c455d3f1cb10a47c6e3865bbc4e0bd6e288ce8ff6bae1b72fab8eb619996),
its original record and explicitly synthetic expected two-scalar correction.
Compare fixed, issue_aware and generic_llm selectors with common caps of two and
five mock model calls, one deterministic repeat: six planned cells, maximum 21
mock calls, zero live requests. The parent source preparation, initial record and
precomputed issue judgments are shared; their offline interpretation is disclosed,
not claimed as live extraction/checker latency. Each cell keeps the same other
workflow limits and tools.

Primary diagnostic: inherited datasets.attrs.sample_refs F1, with support and the
last valid canonical prediction scored even if processing exhausts its budget.
Every planned cell remains in processing/quality denominators; missing predictions
are unprocessed, not successful empty records. Effect threshold zero is descriptive
for this engineering fixture, not a scientist-approved confirmatory threshold.
There is only one independent publication, so no uncertainty interval or natural
paper performance claim is supported. Freeze a fresh evaluator snapshot because
the historical M03 snapshot predates the deliberate M07 dependency update.

## Required checks

Strict runtime/offline separation and forbidden path tests; all input pins checked
before any dispatch; same evidence/tool access and caps; publication-level identity
and repeat checks; all failures/missing cells visible; unknown costs remain unknown;
immutable manifest/child replay; stored predictions reproduce scorer metrics;
deterministic publication aggregation and honest one-publication uncertainty;
safe report locations and unchanged released assets. Run focused tests and the
established safe combined regression suite. No real .env, live inference, final-test
gold, inherited regeneration scripts, deployment or external tracing is needed.

## Implemented interfaces and scope

`agent/study_types.py` defines the strict synthetic-only StudySpec;
`agent/study_runner.py` provides dry_run, run_study and read-only completed replay;
`agent/study_cli.py` exposes dry-run/run/replay. The separate `offline/study.py`
defines StudyEvaluationProtocol, collect_study, publication-paired aggregation and
the Markdown/JSON report. New tests are test_study_types.py, test_study_runner.py,
test_study_cli.py and offline/test_study.py. No previous implementation was edited.

The reusable runtime config is configs/agent/study_synthetic.json. The offline
protocol is configs/offline/study_synthetic_evaluation.json; only its opaque digest
enters runtime. The new authored expectation is tests/offline/fixtures/m09_expected.json.
Its two intended differences from the frozen original record were specified before
execution: Dataset A's description 500 K to 300 K, and sample_2 to sample_1. Entity
identities, the other specimen and all other scalar leaves remain unchanged.

Preflight froze specification digest
0f4c05646a32d1fd67f82c46aeb2e2b8f6b5262d5287f9cae678994e52dfef14
and protocol digest
f5b2d2d0b96d74bf17da85c6dc9777a682cae65b927bac5087fdb40b53a70a3e.
The evaluator is
evaluator:c7706e69b8ae7e9b739b799da246ccf8c64779798f3e4c23defe9d333ccd8dda;
the historical M03 snapshot remains unchanged.

## Outcomes and trade-off

The actual study runs under runs/agent/m09-20261008-synthetic-v1. All six cells
exported valid records, consuming 15 mock calls against 21 reserved, with zero
live requests. Fixed and issue-aware each used two calls at either cap; generic
used two at cap two and five at cap five. Generic at cap two stopped with two
unresolved issues and its original record. The other five cells finished with the
same two-scalar correction. The common fixture initialization is outside these
counts; token totals are unknown, and mock latency is not live performance.

The main trade-off is strict reproducibility versus flexibility: all inputs, code,
dependencies, limits and comparisons are frozen, and incomplete/changed runs fail
closed. A changed method needs a fresh run. This keeps replay auditable but does
not provide automatic migration or recovery of an interrupted parent study.
Last-valid-record scoring retains useful partial results while exposing unresolved
work separately; it never equates valid JSON or completed artifacts with support.
For this invented case, issue-aware routing has no observed advantage over fixed.
The simplest adequate fixed policy remains the baseline.

E09 natural-paper pilot/confirmatory work remains **blocked** by its missing approved
development set/independent labels, reviewed standards needed for the main
comparison, and a frozen claim-specific protocol. This executor does not integrate
full initial extraction, approved standards guidance or selected M08 review exports.
No real-review duration, natural error label, provider nondeterminism, authenticated
expert approval or complete-schema coverage was measured. Independent gold/support/
mapping/harm labels must not be replaced by these scripted expectations.

The frozen primary endpoint has two supported gold references in this fixture:

| Policy | Cap 2 F1 / calls | Cap 5 F1 / calls |
|---|---|---|
| fixed | 1.0 / 2 | 1.0 / 2 |
| issue_aware | 1.0 / 2 | 1.0 / 2 |
| generic_llm | 0.5 / 2 (two unresolved) | 1.0 / 5 |

The generic-minus-fixed paired difference is -0.5 at cap two and zero at cap five;
issue-aware-minus-fixed is zero at both caps. All have one independent publication,
so no confidence interval is reported. No new error taxonomy, independent support
judgment or harmful-correction truth is inferred from these metric differences.

Offline collection at runs/agent/m09-20261008-report-v1 performed six evaluations.
Fresh collection at runs/agent/m09-20261008-reproduce-v1 performed another six using
the same saved predictions, reproducing all 264 metric rows, 176 per-field paired
comparisons, four primary comparisons and the 24,629-byte Markdown report exactly.
Runtime replay made zero new calls; collection replay made zero new evaluations.
All 89 files across the study, six workflow children, two collections and four
scoring runs remained byte-identical after replay. These runs are ignored artifacts.

Focused checks passed: 85 runtime/specification, 65 CLI and 82 offline collection
tests. Independent review added regressions for misleading invocation/failed-call
accounting, wrong child policy/cap and injected sidecar metadata. Completed child
exports are bound to their committed state and frozen source/profile before M03.
The combined safe regression passed **2,027 tests in 418.19s**, including all 232
new M09 checks. Exact commands and the final repository audit are recorded in
[COMMANDS.md](../COMMANDS.md).

Next recommended module: **M10**, after agreeing a small local pilot and handover
scope. Resolve E09's empirical prerequisites before making performance claims;
M10 packaging or simulated review does not satisfy those gates. Stop after M09.
