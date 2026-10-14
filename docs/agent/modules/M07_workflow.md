# M07 — LangGraph and bounded intervention policies

Date: 2026-10-07. Scope: M07 only; stop before M08. Implementation ready; E07 blocked.

## Question, reuse and baseline

Does diagnosis change the next action usefully under the same tools and budgets?
This module establishes testable execution mechanics; E07 needs a frozen real-output
comparison and independent correctness labels before any performance conclusion.

M00–M06 already provide safe run paths, supplied-source retrieval, frozen profiles,
provider accounting, immutable record revisions, reviewed standards gates, support
checks and atomic patches. Reuse these ordinary functions. Add one small LangGraph
wrapper, not a replacement extraction stack. The fixed route is the baseline;
transparent issue-aware routing and a generic model selector share the same action
validator, tool context, budgets and persisted state.

## Interfaces and minimum behavior

Actions: retrieve supplied source evidence, consult an approved definition, propose
an allowlisted supplied local correction, request review, or finish. Arguments are
typed and bounded. A supplied proposal catalog makes local edits reviewable; M07
selects from it rather than generating unrestricted patches. M06 independently
checks old/new assertions and affected references before applying any correction.

The controller owns total/per-issue action limits, model-call limits, a no-progress
stop, elapsed-time/cancellation checks and explicit unknown token usage. Source
passages and approved definitions remain distinct; definitions cannot establish an
experimental fact. Actual M05 guidance remains empty/blocked. Pending review is a
revision-bound state, never approval; M08 owns human edits and approval/resume.

State has an explicit version. SQLite LangGraph checkpoints store JSON-compatible
state with a restricted serializer, and a separate dispatch journal claims each
step before calls. A repeated completed step reuses its result; an incomplete claim
stops with uncertain call accounting and never automatically repeats dispatch.
Tracing is disabled even with inherited tracing configuration. Run artifacts stay
under fresh ignored runs/agent paths; no gold/evaluator files enter runtime.

Cancellation/timeouts are checked at action/call boundaries; an in-flight provider
uses its own bounded request timeout and is not forcefully killed. Patch application
is skipped after cancellation or deadline expiry. Raw model text follows explicit
hash-only retention for the synthetic M07 run; hashes/observations remain recorded.

## Checks and E07 boundary

Tests cover each action and policy, invalid/model-invented tools and IDs, repeated
retrieval without new evidence, exhausted budgets, unknown usage, failures,
timeouts/cancellation, definition/evidence separation, pending review, restart and
idempotence, interrupted dispatch claims, unsafe paths and tracing suppression.
Synthetic judgments/reviews are labelled and establish no scientific accuracy.

E07 compares fixed, generic model and issue-aware repair policies from the same
frozen record/issue state, with the same evidence/tools/budgets. No independent
labelled real-output comparison is available, so empirical validation is blocked.
No live inference or real .env read is needed; standing ChatAI authorization and
shared rolling-hour accounting remain unchanged.

## Frozen synthetic integration protocol (before execution)

This is an engineering diagnostic, not E07 empirical validation. The three
workflow_fixed.json, workflow_issue_aware.json and workflow_generic.json configs
pin the same invented M06 source/record and frozen mock issue judgments. The
publication lineage is synthetic M06; no natural training/calibration/test papers
or gold records enter the run. Source SHA256 is
a7f2c455d3f1cb10a47c6e3865bbc4e0bd6e288ce8ff6bae1b72fab8eb619996.

Vary only the policy path; source access, supplied correction catalog, checker
fixtures and action budgets are shared. Each run allows 12 actions, five per issue,
eight provider invocations, two consecutive no-progress actions and a persisted
120-second deadline. Unknown token usage is explicitly allowed under the call
bound, never reported as zero total usage. Fixed/issue-aware selectors are ordinary
functions; generic selection uses a three-response synthetic script through the
same provider boundary. All new model dispatches are mocks.

Primary engineering outcome: bounded execution changes only the intended Dataset A
description and sample attachment, retains stable specimen identities, records all
observable calls/actions and finishes with both supplied issues resolved. Compare
canonical outputs and call counts, not diagnosis accuracy. Evaluator/metric version:
not applicable, no scoring invoked. Seed interpretation is cached fixture input,
excluded from new-call counts; this is not an end-to-end latency measurement.

Then repeat completed runs to check zero new dispatches and unchanged export bytes.
Run generic once with max_steps=1 in a separate run, exit the process, and resume
from SQLite; the cumulative provider count must match its uninterrupted run. Tests
separately cover unresolved/pending review and synthetic approved standards. Real
standards guidance remains unavailable, never substituted for publication evidence.

## Outcomes

Implemented six ordinary modules/adapters: workflow_types.py, workflow_policies.py,
workflow_tools.py, workflow.py, workflow_graph.py and inspect_workflow.py. Added six
test modules, three frozen run configs and five invented response fixtures. The
inherited extraction, M01–M06 code, evaluator, schemas and released assets did not
change. Domain selection/checking/correction logic remains outside LangGraph.

Pinned direct dependencies are LangGraph 1.2.14 and langgraph-checkpoint-sqlite 3.1.1;
resolved checkpoint is 4.2.0. Installed signatures were inspected. Frozen sync now
succeeds, including offline, and fixes the stale editable-package pointer to this
checkout. The lock retains all earlier package versions except idna 3.11 → 3.20,
required by the new httpx2 dependency's idna>=3.18 bound. Python>=3.13 and the
meta-data-extraction distribution/import names are unchanged. Dependencies are local
to the uv environment; no global install or other-checkout edit occurred.

Verification: **284 M07 tests passed in 47.07s** (95 contracts/policies, 65 tools,
46 transitions, 46 graph/persistence, 32 runner). A nonfunctional pytest-cache
permission warning prompted disabling that cache for the final suite. The full
safe suite passed **1,630 tests in 134.04s**, without warnings: 1,404 agent, 151
offline and 75 inherited. Exact commands are in [COMMANDS.md](../COMMANDS.md).

The fixed, issue-aware and generic runs each finished in three actions (retrieve,
propose/check/apply, finish). Fixed and issue-aware used two mock checker calls;
generic used the same two plus three mock policy selections. All canonical exports
are byte-identical: only the 500 K description becomes 300 K and sample_2 becomes
sample_1. Same-composition specimens and entity handles remain distinct. These
results establish no routing advantage: fixed and issue-aware take the same route
on this deliberately simple fixture. Ambiguity-to-review differences are tested
separately with invented cases, not presented as scientific performance.

Runs are runs/agent/m07-20261007-fixed-v1/, issue-aware-v1/ and generic-v1/ with the
same m07-20261007- prefix. Repeating each command made zero new dispatches and
preserved all seven files, including SQLite databases, byte-for-byte. The separate
m07-20261007-resume-v1 run checkpointed after one generic call and one action with
no canonical/result export, then a new process used four further calls to finish.
Its cumulative five calls and canonical result match uninterrupted execution.
All four runs share the exact initial state, tool-scope fingerprint and budget;
all 27 combined implementation pins and all input pins match the final files.

Independent review added protections against uncertain-budget reuse, corrupted
checkpoint/completion states, stale issue resolution after edits, and counting
predispatch blocks as live HTTP attempts. Pending reviews are explicit and cannot
become approvals. Interrupted steps retain uncertain possible attempts; the normal
ChatAI persistent ledger still owns the shared rolling-hour dispatch allowance.
Known token subtotals are flagged incomplete when any observation lacks usage.

### Main trade-off and limits

A separate dispatch journal makes restart behavior conservative: completed work
can replay, but an unfinished claim stops for explicit recovery/new-run handling.
It cannot tell whether an unrecorded remote response succeeded. Correction choices
come from the supplied frozen catalog; unrestricted patch discovery is not added.
Timeout/cancellation is cooperative between bounded calls, and no running HTTP
request is forcefully killed. Real human review/edit/resume is M08, not simulated
as expert approval here.

E07 status: **blocked**, not a pilot accuracy result. No independent labelled
real-output set or frozen real comparison protocol is available. No live Qwen call,
real .env read, natural gold evaluation, external tracing, full extraction experiment
or result regeneration ran. M05's genuine reviewed-profile gate remains blocked;
no real standards guidance entered these runs. Coverage/completeness remains
unsupported. No lint/type checker is configured. The former full-sync blocker is
resolved. Stop after M07; next is **M08 — persistent human review and explicit
approval**, using these revision-bound cases and checkpoint foundations.

## Official API references checked

The implementation uses StateGraph with an explicit thread_id, synchronous SQLite
checkpointing and static interruption after each completed step. Serializer module
allowlists and pickle fallback are explicitly restricted. A fresh context plus
tracing_context(enabled=False) suppresses inherited and environment-based tracing.
These interfaces were checked against the installed versions and official sources:
[graph API](https://docs.langchain.com/oss/python/langgraph/graph-api),
[persistence](https://docs.langchain.com/oss/python/langgraph/persistence),
[interrupt semantics](https://docs.langchain.com/oss/python/langgraph/interrupts),
[SQLite saver](https://reference.langchain.com/python/langgraph.checkpoint.sqlite/SqliteSaver),
[tracing controls](https://docs.langchain.com/langsmith/trace-without-env-vars),
[LangGraph release metadata](https://pypi.org/pypi/langgraph/1.2.14/json), and
[SQLite saver release metadata](https://pypi.org/pypi/langgraph-checkpoint-sqlite/3.1.1/json).

