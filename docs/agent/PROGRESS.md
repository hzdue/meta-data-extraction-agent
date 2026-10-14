# Agent module progress

Updated: 2026-10-09. M00–M10 are implemented within their recorded scope, including
the fresh-publication M10 integration. Stop within M10; empirical gates remain separate.

| Module | Implementation | Offline tests | Empirical validation | Integration | Next action |
|---|---|---|---|---|---|
| M00 | ready | pass (84 M00 + 35 inherited) | not_run | pass (synthetic smoke, zero model calls) | M01 source preparation and evidence retrieval |
| M01 | ready | pass (164 agent + 13 inherited) | not_run | pass (synthetic Markdown inspection and generated PDF) | M02 typed extraction/provider boundary |
| M02 | ready | pass (404 agent + 35 inherited) | not_run | pass (two synthetic fixed variants and verified replay) | M03 offline experiment harness and frozen evaluation |
| M03 | ready | pass (630 main + 41 additional; 6 excluded) | pilot (E03: 2/2 exact historical report parity, one publication) | pass (M02 exports scored; historical replay verifies without rescoring) | M04 PaNET terminology lookup and conservative mapping |
| M04 | ready | pass (168 new; 798 combined) | blocked (E04 needs reviewed mention-to-concept labels) | pass (authentic PaNET, invented mentions, record binding and replay) | M05 NeXus definitions and a reviewed profile registry |
| M05 | ready (reviewed-profile gate blocked) | pass (286 new; 1,084 combined) | blocked (E05 needs resolved/reviewed mappings and independent labels) | pass (authentic NXDL inspection and byte-preserving replay) | M06 independent record checks; resolve/review NeXus before guidance use |
| M06 | ready | pass (262 new; 1,346 combined) | blocked (E06 needs independent real-output labels and a frozen comparison) | pass (two synthetic corrections, invalid-record retention and zero-dispatch replay) | M07 bounded intervention policies; standards guidance remains gated |
| M07 | ready | pass (284 new; 1,630 combined) | blocked (E07 needs independently labelled real outputs and a frozen comparison) | pass (three synthetic policies, real SQLite restart/resume, zero-dispatch replay) | M08 persistent human review and explicit approval |
| M08 | ready | pass (165 new; 1,795 combined) | blocked (E08 needs real review observations, independent residual-error labels and a frozen comparison) | pass (same-run SQLite review, separate-process simulated edit/unresolved responses, immutable replay and crash recovery) | M09 controlled end-to-end study and ablations; freeze protocol and resolve required empirical gates |
| M09 | ready (synthetic study tooling; natural end-to-end path gated) | pass (232 new; 2,027 combined) | blocked (E09 needs approved natural publications/independent labels, reviewed guidance and a claim-specific protocol) | pass (six frozen cells, isolated scoring, independent stored-prediction reproduction and immutable replay) | M10 agreed local pilot/handover; resolve empirical gates before performance claims |
| M10 | ready (fresh-publication CLI/JSON API; earlier synthetic browser pilot remains separate) | pass (287 integration checks; 2,504 combined safe tests) | not_run (E10 needs an observed operator session) | pass (actual extraction-to-review subprocess diagnostic, guarded corrections, manual proposals, replay); live service, browser and Docker untested | Deferred operator trial, then a frozen natural-publication comparison with independent labels; M11 remains optional |

See [M00 record](modules/M00_baseline.md), [M01 record](modules/M01_evidence.md),
[M02 record](modules/M02_extraction.md), [M03 record](modules/M03_evaluation.md),
[M04 record](modules/M04_panet.md), [M05 record](modules/M05_nexus.md),
[M06 record](modules/M06_checks.md), [M07 record](modules/M07_workflow.md),
[M08 record](modules/M08_review.md), [M09 record](modules/M09_study.md),
[M10 record](modules/M10_pilot.md), [pilot operator guide](PILOT.md),
[M10 integration record](modules/M10_integration.md),
[one-paper pilot protocol](modules/M10_real_paper_pilot.md),
[internal offline performance assessment](modules/M10_self_assessment.md),
[fresh-publication operator guide and diagram](PUBLICATION_RUN.md),
[prepared operator trial](PILOT_TRIAL.md),
[repository audit](REPO_AUDIT.md),
[data policy](DATA_POLICY.md), and [commands](COMMANDS.md).

Implementation readiness means local safeguards and offline checks work; it is
not a measured extraction result or evidence for the controller hypothesis.

Protected scientific assets and the 13 schema/prompt/evaluator/config baseline
fingerprints are unchanged. M07 deliberately updates pyproject.toml and uv.lock
for pinned LangGraph/SQLite dependencies. Frozen environment sync now succeeds,
including offline, and the editable package points to this checkout. Earlier
module notes below preserve their historical environment limitations.

M01 adds immutable source/evidence snapshots and supplied-corpus lexical retrieval.
Its synthetic inspection is under `runs/agent/m01-20261007-check/`, with no model
calls. E01 evidence-recall comparisons remain `not_run`: no reviewed question/
evidence set or comparison protocol was supplied. ChatAI approval remains in
`AGENTS.md`; M01 has no live provider execution path.

M02 adds frozen schema/prompt profiles, typed mock/live providers, canonical and
evidence envelopes, a single-call fixed path and immutable exports with safe replay.
The two synthetic demonstration runs are `runs/agent/m02-20261007-evidence-v1/`
and `runs/agent/m02-20261007-inherited-v1/`. Their canonical JSON is identical;
evidence locations resolve, with scientific support explicitly unassessed.

Final M02 command: 439 tests passed in 16.64s. The configured ChatAI adapter has a
persistent shared ten-request/hour limit, no automatic retries/redirects and a
one-hour quarantine for unknown history. Tests include the actual installed SDK
with a local simulated transport. No real `.env` read or live request occurred.
E02 remains `not_run`: no live endpoint/publication comparison manifest was frozen.
Runtime does not import or consume offline gold/evaluator helpers. No scoring
harness or controller was added during M02.

M03 adds a separate offline package with publication/split declarations, frozen
evaluator/input hashes, safe scoring/replay and supported per-field comparisons.
The explicit historical replay is `runs/agent/m03-20261007-historical-v1/`: both
released GPT-4o zero-shot variants for 1804.07163 match their released reports
exactly, excluding only path metadata. Repeating the command performs zero new
evaluations. The synthetic M02 export integration is separate under
`runs/agent/m03-20261007-synthetic-v1/`.

M03 used zero inference requests and did not change runtime behavior based on gold.
The 630-test main suite and 41 additional inherited checks pass. Unknown costs and
request counts remain unknown; zero-support and blocked fields remain visible.
E03 is a small scorer-reproducibility pilot, not model/agent performance validation.
No future-held-out set, expert correspondences or mapping judgments were invented.

M04 adds local, versioned PaNET loading, family-scoped lexical candidates and
unapproved mapping sidecars. The authentic v1.1.0 resource is pinned by commit and
byte hash; its limitations, attribution and terms are in [PANET_RESOURCE.md](PANET_RESOURCE.md).
It is a historical snapshot, not the latest release or a full OWL classification.

The invented six-mention inspection produced three unapproved proposals, two
unresolved contexts and one no-match. The existing M02 canonical-technique example
stayed unresolved within the selected powder/small-angle families, with its record
revision bound and source wording unchanged. Verified replay succeeded. The final
suite passed 798 tests, with zero model calls and no real `.env` read. E04 remains
blocked by missing reviewed terminology labels; these engineering checks establish
no natural-paper mapping accuracy or agentic performance result.

M05 adds a pinned NXDL declaration loader, a content-bound mapping/review registry,
explicit temperature conversion policy and offline inspection CLI. The authentic
v2026.01 subset has 85 fields across NXsample, NXbeam and NXmonopd; provenance and
limits are in [NEXUS_RESOURCE.md](NEXUS_RESOURCE.md). It is not a complete NXDL
processor or NeXus compliance assessment.

The inspection at `runs/agent/m05-20261007-inspection-v1/` contains four unresolved
partial proposals and two scoped no-matches, all unreviewed. Default approved
guidance is empty. Three definition lookups and four invented arithmetic/missingness
checks completed; cached replay preserved every artifact byte. Canonical records,
source expressions, extraction prompts and previous module implementations did not
change. The final safe suite passed 1,084 tests (286 new M05 tests).

M05 used zero model calls and did not read the real .env or natural gold. The
reviewed-profile gate remains blocked by unresolved relevant NXDL inheritance,
defaults/constraints and missing human mapping reviews; E05 also needs independent
labels and a frozen comparison manifest. Unit-test review records are synthetic.
M06 can begin with independent checks, while standards-guided runtime use remains
disabled pending those dependencies.


M06 adds deterministic structure/reference reports, bounded source-support and
attachment judgments, conflict diagnosis, and a separate atomic patch executor.
It reuses the inherited validator and the M01/M02 evidence, identity, provider and
run-storage boundaries. Invalid candidates preserve unexpected content without
becoming successful empty records. Coverage remains explicitly unsupported;
automated judgments remain distinct from expert approval and independent gold.

The invented correction run at runs/agent/m06-20261007-patch-v1/ changes Dataset A's
preparation-derived 500 K description to its 300 K acquisition temperature and
sample_2 to sample_1. Both same-composition specimens and all entity handles remain
separate and unchanged. Two fixed mock calls check the original/candidate; replay
makes zero new calls and preserves all artifact bytes. A separate sample_99 run at
runs/agent/m06-20261007-dangling-v1/ records the dangling reference without a checker
call or canonical export. Exactly two scalar leaves changed in the correction run.

The final safe suite passed 1,346 tests, including 262 new M06 tests. No live calls,
real .env reads, natural gold inspection or released-artifact changes occurred.
Patches default to disabled and remain narrowly allowlisted; conflicting evidence,
stale revisions, structural failures and incomplete support block application.
Automated support judgments can still be wrong. E06 is blocked by missing independent labels and a frozen comparison
specification, so no diagnosis accuracy or agentic performance claim is made.
M05's reviewed-profile gate remains blocked; no NeXus/PaNET guidance enters M06.
Next is M07's fixed and bounded issue-aware workflow; stop before implementing it.


M07 adds one bounded workflow over shared ordinary functions. Fixed, transparent
issue-aware and generic model selectors use the same validated tool interface,
source access, proposal catalog and limits. SQLite checkpoints plus a dispatch
journal and process lease make restart/replay explicit; incomplete dispatch stops
with uncertain accounting. Pending review remains pending, and record edits reopen
stale issue resolutions. Definitions never become publication evidence.

The three frozen synthetic runs under runs/agent/m07-20261007-{fixed,issue-aware,
generic}-v1/ each performed retrieve/propose/finish and made the same two intended
canonical changes. Fixed and issue-aware used two mock calls each; generic used
five. Replays made zero calls and preserved all files byte-for-byte. A fourth
resume-v1 run exited after one action/call and resumed in another process with four
more calls, matching the uninterrupted generic result. This fixture shows no
advantage for issue-aware routing; no natural-paper performance result is claimed.

The final safe suite passed 1,630 tests (284 new M07 tests). Zero live requests,
real .env reads or released-asset edits occurred. M07's required dependency update
adds LangGraph 1.2.14 and SQLite saver 3.1.1, preserving all old locked package versions
except the required idna upgrade. Frozen online and offline sync now pass; the
previous missing-hatchling/stale-editable environment limitations are resolved.
E07 remains blocked by missing independent labels and a frozen real-output protocol;
M05 reviewed guidance remains blocked, and completeness remains unsupported.
Stop after M07. M08's real persistent review, explicit approval and safe edit/resume
flow is the next module.


M08 adds explicit approve/edit/unresolved responses and a durable SQLite review
stage inside a pending M07 run. The original workflow checkpoints/exports remain
immutable. Reviews bind exact state/revision, values, entities, semantic roles and
source passages; dataset decisions also cover their sample/facility attachments.
Edits reuse the M06 scalar candidate builder, recheck structure and invalidate
stale approvals. Unresolved dependencies reopen affected approvals. Duplicate
responses cannot create a second decision or rewind the current export.

The synthetic correction at runs/agent/m08-20261007-pending-v1/ changes only
Dataset A's 500 K description to 300 K and sample_2 to sample_1. A second process
submits one explicitly simulated decision with three attestations; replay/resume
preserves all 12 artifact files and all entity handles. The separate unresolved-v1
run keeps its original record and no active approval assertions. Each parent used
four mock calls; review used zero calls, and all live request counts remain zero.

The final safe suite passed 1,795 tests, including 165 new M08 tests. No prior
production module, dependency, schema, prompt, benchmark asset or AGENTS.md changed.
No real .env read, live inference, natural-gold inspection or real reviewer approval
occurred. Reviewer identity remains self-declared, reviewed exports must be selected
explicitly, coverage remains unsupported, and incompatible states require a
separately implemented migration or a fresh run. E08 effort/error claims remain
blocked by missing real observations and independent labels; M05 guidance remains
gated. Stop after M08. M09's frozen controlled study is the next module.

M09 adds a strict synthetic-only study specification, no-write dry-run, frozen
parent/child execution, explicit read-only replay and a separate offline collector.
It reuses M07's policies and M03's unchanged scorer. Runtime contains only the
offline protocol's digest; gold and evaluator inputs remain offline. Policy/cap,
source/profile/revision, sidecar and resource checks bind collected predictions to
the frozen cell. Missing and failed cells remain visible; unknown usage stays unknown.

The actual study at runs/agent/m09-20261008-synthetic-v1 contains one invented
publication, three policies and caps of two/five calls: six cells and 15 mock calls
against a maximum of 21, zero live requests. Fixed and issue-aware use two calls
and reach sample-reference F1 1.0 at both caps. Generic reaches 0.5 with two
unresolved issues at cap two, and 1.0 with five calls at cap five. The fixture
supports no advantage for issue-aware over fixed, no publication-level uncertainty
interval and no natural-paper performance claim.

The report and independent reproduction are runs/agent/m09-20261008-report-v1
and runs/agent/m09-20261008-reproduce-v1. Each performs six offline evaluations;
264 metric rows, 176 per-field comparisons, four primary comparisons and Markdown
bytes match exactly. Replay makes zero new calls/evaluations and preserves all 89
artifact files. Mapping/support/harmful-correction truth and observed review time
remain unavailable. No prior implementation, dependency or released asset was
changed; the ChatAI authorization and shared ten-request/hour limit remain intact.
The final safe regression suite passed 2,027 tests in 418.19s, including 232 new M09
checks and all 1,795 previously passing safe checks.

This is a repair-policy engineering diagnostic with shared precomputed initialization,
not the full standards-enabled A2/A3 or end-to-end extraction experiment. E09 stays
blocked by its named empirical prerequisites. Stop after M09; the next module is
M10's agreed small local pilot and handover, without implying empirical validation.

M10 adds a single-user, loopback-only authenticated HTTP API and a small review
page over unchanged M07 workflows and M08 ReviewStore. It packages 64 explicitly
allowlisted runtime assets into a fresh isolated bundle, excluding offline
evaluation/gold, credentials and previous runs. Only the bundle's runs/agent tree
is mutable; versioned exports remain bound to the exact review state. The fixed
invented source and mock responses make recovery reproducible but do not support
document upload or live provider selection. Reviewer identities remain self-declared.

The final bundle is runs/agent/m10-20261008-pilot-v2/runtime. A separate clean
environment installed 62 unchanged locked packages offline. An actual diagnostic
passed 24 HTTP requests across three server processes: interrupted workflow resume,
simulated unresolved/edit decisions, stale-state rejection, duplicate replay and
identical exports after restart. The edit changes only Dataset A's preparation-derived
500 K description to 300 K and sample_2 to sample_1; specimen identities are preserved.
This final diagnostic used eight mock calls. Across both preserved diagnostic
bundles and one prepared browser fixture, M10 used 20 mock calls and zero live calls.

The final safe suite passed 2,217 tests in 534.21s, including 190 M10 tests. The first
combined run exposed early-response close and log-order races on Windows; bounded
draining and log-before-response fixes passed five additional regressions and the
full rerun. Prior code, tests, schemas, prompts, dependencies, benchmark assets and
AGENTS.md remain unchanged. No real .env read or released-result generation occurred.

The UI script compiled and static/CSP checks passed, but the browser helper failed
to initialize; no rendered interaction or usability result is claimed. Docker is
absent, so the provided container recipe is untested. E10 remains not_run and E09's
natural-paper/independent-label/reviewed-guidance gates remain blocked. Stop after
M10. The next practical step is an operator trial following PILOT.md, then an agreed
real-document scope. Optional M11 needs a measured bottleneck and explicit
training/data/compute authorization; it is not the automatic next step.

M10 follow-up (2026-10-08): the user chose to prepare the operator trial for later.
The new Python launcher reuses the existing verified bundle/server and environment;
PILOT_TRIAL.md defines the synthetic task checklist and blank observations. The
prepared bundle m10-trial-ready-v1 has its frozen trial-plan.json and observations.md
beside runtime/. Eleven launcher checks plus fresh preparation passed, including
actual local authenticated health and process cleanup. Zero new mock/live model
calls occurred, and the service is stopped. Browser initialization is still blocked;
clipboard execution, Docker and the human session remain untested. The runtime
package and prior results are unchanged. Continue with the prepared human trial
when the operator is available; E10 stays not_run and M11 was not started.

M10 integration follow-up (2026-10-09): the former demonstration supplied its
record, issues and candidate correction from fixtures. The new publication runner
starts with a source/hash and provider scope, performs evidence extraction,
selects bounded assertion checks, generates a constrained proposal when warranted,
and enters the existing persistent controller and M08 review. Six additive runtime
modules, six test modules, one configuration and six synthetic fixtures implement
this path. Existing workflow/review changes are limited to the manual-policy
budget preflight and a versioned, content-addressed proposal catalog.

Optional PaNET lookup produces unreviewed, revision-bound candidates for actual
extracted technique expressions. Only approved, resolved NeXus registry guidance
enters controller tools. These definitions remain distinct from publication
evidence. The fresh integration exposes fixed and issue-aware policies; the
generic model-selected comparison remains separately available through M07/M09.

The frozen diagnostic at runs/agent/m10-integration-diagnostic-v1 passed 19 CLI
processes with eight mock calls and zero live requests. Automatic extraction,
checking and proposal generation pause after three calls; another process uses
two fresh checks and corrects only Dataset A's description (500 K to 300 K) and
sample attachment (sample_2 to sample_1). The manual run uses three calls and
leaves its generated proposal available without changing canonical data. The
fixture's twelve checked targets leave thirteen emitted assertions unchecked,
so review remains pending. No entity-recovery or completeness result is implied.

An explicitly simulated unresolved review applies once, duplicate submission
replays, stale states are rejected, and 37 automatic plus 26 manual artifacts are
byte-identical after completed replay. Review makes no model calls. The ChatAI
adapter and shared ledger also pass offline transport checks with invented SDK
responses and clocks, including ten-request enforcement across multiple runs,
failed attempts, unknown history and unknown proposal usage. This is not live
service performance evidence. No real .env was read and no real model request ran.

Implementation readiness remains distinct from empirical validation: natural
extraction quality, a comparison with extraction alone, harmful changes, actual
review effort and usability remain unmeasured. NeXus expert mapping review remains
gated. The new interface is a local CLI/authenticated JSON API; it adds no document
upload or browser review page. Earlier isolated pilot bundles and the deferred
human-trial preparation remain unchanged. Docker and browser interaction remain
untested. Use fresh run IDs for this implementation; old workspace checkpoints
need explicit compatibility migration. Stop within M10; the next work is the
prepared operator session and a scoped natural-paper comparison, not automatic
fine-tuning.

The final safe suite passed 2,504 tests in 695.59 seconds: the previous 2,217 plus
287 integration checks. The final inventory contains 1,758 files, with 21 intended
additions, five intended existing-file edits, no missing files or text/credential
findings, and 1,488 protected files unchanged. Both prior 64-file pilot bundles
still match their own frozen manifests. No schema, prompt, evaluator, dependency,
AGENTS.md or released-result changes occurred. Exact commands and remaining
validation limits are recorded in COMMANDS.md.

M10 real-paper preparation (2026-10-09): the user authorized the small live
evaluation. The configured ChatAI endpoint was found in a comment in `.env`;
the key remains private. A frozen one-calibration-publication protocol selects
1812.03956, using the unchanged evidence prompt and qwen3.8-27b with at most three
calls and no automatic edits or retries. Historical candidate 1804.07163 was
excluded before inference because its prompt exceeded the unchanged byte cap.
Runtime and offline scoring use separate fresh run directories. No final-test
gold is authorized; calibration gold values remain deferred until outputs freeze.

The new shared ledger requires a full hour of known history before any dispatch:
earliest 2026-10-09 10:55:15 UTC / 12:55:15 Berlin. The sealed one-shot helper waits
for this condition and three free slots, while every call still uses the existing
shared ten-attempt rolling-hour limit. Preparation has made zero live requests.
The launcher passed seventeen offline synthetic safety checks. Live execution
and measurement are not yet complete; see the pilot record and local job status.

The prepared PDF retains unavailable-image/page-mapping warnings, which prevent
automatic proposal generation. Initial/final canonical records should therefore
remain identical under the manual policy. Dataset recovery interpretation also
requires independent alignment review; raw diagnostics and eligible metrics stay
separate. This pilot cannot establish extraction improvement. The runtime,
evaluator, schemas and dependencies are unchanged; no new module is opened.

At the 10:20 UTC handoff, the one-paper job is queued in hidden local process
12548, verified alive with an exclusive claim and a `waiting_for_accounting`
event. It has zero dispatched attempts. The earliest eligible time is still
10:55:15 UTC; the computer must stay running. Four collector checks also passed,
including synthetic success/report/replay and failure handling. The inputs are
sealed across 93 files. Experiment status is `not_run` at handoff, not a live
performance result; `runs/agent/m10-real-paper-prep-v2/job-finished.json` and the
eventual report are authoritative for subsequent execution. No automatic retry
or expansion beyond the declared paper is permitted.

M10 quota follow-up (2026-10-09, supersedes the queued status above): the user
requested checking ChatAI's actual limits and adapting the local cap. The old
job stopped at 10:25:55 UTC with `operator_cancelled_before_dispatch`; zero model
requests were made. One separately recorded, authenticated, non-inference GET to
the configured ChatAI endpoint returned HTTP 401 without rate-limit headers.
Account limits therefore remain unverified; the 10/hour cap is unchanged pending
a valid local credential/endpoint and successful check. There was no retry.
The protocol, cancelled claim, ledger and released assets remain intact. AGENTS.md
now records the verification rule and blocker. Actual status: `blocked`, with
no natural-paper prediction or score. Next: verify the corrected credential,
adapt to observed limits, then prepare a fresh bounded pilot job within M10.

Endpoint follow-up (2026-10-09): reviewed the exact official quota URL after the
user questioned the earlier hostname. One corrected GET to
`https://saia.gwdg.de/v1/chat/completions` also returned HTTP 401 with no quota
headers. Bearer authentication matches the documentation; private local checks
found no whitespace, repeated Bearer prefix or dotenv interpolation change in
the key. The earlier response did not establish universal key invalidity.
Total administrative requests: two; model calls: zero. No further retry was made.
The 10/hour cap and cancelled pilot remain unchanged; authentication/access and
account quota verification remain blocked. No runtime code changed.

M10 internal assessment (2026-10-09): at the user's request, completed an offline
evaluation without ChatAI. Current focused verification passed 230 tests in
76.82s. Saved M09/M10 manifests, metrics and 152 artifact files verify unchanged;
issue-aware and fixed remain tied on the one authored repair fixture, with two
mock calls and sample-reference F1 1.0. That is not natural extraction evidence.

A frozen source-only diagnostic on calibration paper 1812.03956 tested ten
questions; two are unanswerable from the supplied main text. Lexical top-three
retrieval captured all required exact reference phrases for 2/8 answerable
questions at 1,500 characters and 3/8 at 6,000, compared with prefix baselines
0/8 and 1/8. Fourteen reference components yielded 4/14, 6/14, 0/14 and 3/14.
Forty selections and 331 repeated evidence-reference instances resolve correctly.
These are assistant-derived phrase-capture measurements, not extraction accuracy
or independent semantic recall; one question retrieves useful alternative
wording that the strict frozen metric does not credit. Production assertion-based
queries and extraction seed links were not tested by these handwritten questions.

Code inspection confirms a major practical ceiling: all current PDF inputs carry
an unavailable-images warning, producing a needs-review support report and thus
no eligible automatic proposal. The checker also examines only emitted assertions
(12/25 in the integration fixture), and patches cannot create missing entities.
No runtime, prompt, schema, evaluator, limit or dependency changed in this turn;
zero network requests and zero natural-gold access. The internal diagnostic is a
`pilot`; end-to-end natural extraction gain and E10 human usability remain unrun.
Next recommended scope is M10 PDF evidence coverage and omitted-entity evaluation,
followed by an authenticated frozen extraction/control comparison. Do not start M11.

M10 evidence follow-up (2026-10-09): continued the user's requested implementation
with two bounded runtime changes. A verified PDF support check limited only by
unavailable images can now produce text-grounded, unreviewed proposals for human
review. It retains `needs_review` and all warnings; automatic application remains
blocked even when requested. Synthetic actual-PDF integration takes three mock
calls, persists a proposal in durable review, preserves the canonical record and
replays without calls. The direct atomic apply gate also rejects these reports.

Added selectable `focused_v2` retrieval for initial support planning: generic term
weights and windows around matches, with exact offsets and explicit omissions.
`lexical_v1` remains the default and unchanged; extraction seed priority and later
controller retrieval are unchanged. Version-2 support plans pin the strategy and
reject old plans without migration. Changed code requires fresh run IDs; previous
experiments and the cancelled pilot remain preserved.

The frozen one-paper development replay improves complete exact-phrase capture
from 2/8 to 4/8 at 1,500 characters and 3/8 to 4/8 at 6,000. Components improve
4/14 to 7/14 and 6/14 to 7/14, with no losses in this narrow metric. An independent
audit verified all 40 selections, 111 evidence instances and exact reproduction of
20 old lexical selections. This is not held-out validation or extraction accuracy:
the questions and internal assistant labels were previously inspected, and the
production queries/seed evidence were not benchmarked. No tuning followed results.

Focused checks passed 175 distinct tests, including 65 new cases. Experiment
status is `pilot` for the retrieval diagnostic; live extraction benefit remains
`blocked`, and human usability remains unrun. Zero service calls, unchanged
ten-attempt/hour accounting, no natural gold inspection or environment/schema/
evaluator changes. See `modules/M10_evidence_followup.md` for the example, trade-off
and exact limits. Next stays in M10: authenticated fresh comparison plus independent
entity/attribute/reference review, including omitted entities. Do not start M11.

Final verification for this follow-up: **2,569 passed in 699.09s**, exit 0, in the
full safe suite (2,504 prior checks plus 65 additions). The 1,764-file inventory
contains exactly seven intended edits and four additions; no missing files.
All 1,478 protected result/data files and 35 frozen earlier-run artifacts match.
The new run contains the assessment, independent retrieval audit and final audit.

M10 reference readiness (2026-10-09): user confirmed ChatAI access is still not
corrected; live evaluation remains blocked with no new authentication/inference
attempt. Implemented an offline source-first reference packet and annotation
validator, reusing source/evidence/profile/structural checks and immutable storage.
The existing scorer already counts omitted gold entities and is unchanged. Runtime
review still concerns emitted assertions; the new offline packet covers all six
entity collections before a reviewer sees predictions.

Prepared `runs/agent/m10-reference-181203956-v1/`: original PDF, exact available
text, frozen schema/regions/warnings, instructions and a blank submission template.
No existing gold, predictions, model proposals or assistant reference answers enter
it. Both complete canonical schemas are required; missing scope cannot become zero
entities. An excerpt CLI supplies exact evidence objects. Supplied references get
separate immutable validation receipts, with null/uncertain/missing work explicit.
Self-declared reviewer identity or independence never establishes expert gold,
whole-source coverage or approved scoring alignment.

Verification: **591 passed in 117.18s**, including 85 new tests. Synthetic omissions
produce one false negative and F1 2/3 for two gold entities/one correct prediction,
confirming the inherited evaluator's existing behavior. All 78 natural source
regions resolve; the bundle matches the prior 22,929-character snapshot exactly.
The blank-form engineering receipt stays incomplete, six entity counts unknown,
zero human annotations and no canonical/gold export; immutable replays pass.

Preservation: 1,770 files, exactly three documentation edits and six additions,
no missing files; all 1,478 protected files and 47 earlier artifacts unchanged.
Existing runtime/scorer, schemas, configs, dependencies and ten-attempt/hour
accounting are unchanged. No credentials read, model/network calls or natural
gold inspection. See `modules/M10_reference_readiness.md` and the new run's
`packet-audit.json` / `final-audit.json`.

Status: packet complete, natural annotation/comparison `not_run`; service access
and independent labels/alignment remain blocking inputs. Next within M10 is actual
source-first review, then a frozen reference/correspondence adapter and authenticated
comparison preserving initial, pre-review and human-reviewed outputs separately.
Do not begin M11 or treat preparation as extraction validation.

M10 comprehensive component evaluation (2026-10-09): completed the requested
offline assessment across source preparation, extraction/validation/evidence,
retrieval, PaNET, NeXus/quantities, support/patches, controllers, persistence,
review, provider accounting, service/packaging and offline evaluation. Report:
`modules/M10_component_evaluation.md`; fresh artifacts:
`runs/agent/m10-component-eval-v1/`. Product code and experiment definitions stayed
unchanged; no natural gold, credentials or live service were used.

Natural retrieval development replay reproduced all 40 earlier selections:
focused versus legacy gives 4/8 versus 2/8 complete questions at 1,500 characters,
and 4/8 versus 3/8 at 6,000. Its local search is about 14 times slower (roughly
16 ms versus 1.2 ms), and ten authored stress cases reveal one gain, four losses,
five ties. Losses include `In` treated as a stopword and clipped distant negation,
sample attachment and second facts. Legacy remains default; these results do not
establish natural extraction gain.

In nine authored patch interventions, explicit model-judgment acceptance yields
two beneficial changes and one harmful change; default policy makes no changes.
The harmful case supplies consistently false checker judgments, allowing 300 K
to become unsupported 900 K. Description correctness remains 4/9, while sample
references improve 8/9 to 9/9; do not pool them into a schema-wide metric.
Valid location/structure checks also pass wrong values, wrong attachments and an
omitted entity. These are possible failure demonstrations, not natural error rates.

Fixed and issue-aware controllers both repair the two supplied errors in two
calls; generic needs five and repairs neither at cap two. A wrong conflict
diagnosis makes issue-aware defer both corrections. Persistence matches direct
states, adds about 221 ms median paired local overhead and avoids repeated calls
on replay/two cross-process resumes. PaNET resource targets improve 29/50 to
50/50 versus exact labels, but publication integration remains a post-workflow
sidecar; NeXus still has zero approved guidance. Human benefit is unmeasured.

Fresh checks: 1,400 passed in 299.52s plus 147 distinct boundary tests in 15.82s,
1,547 total. Frozen diagnostics used 70 scripted calls, zero live calls; independent
software audits recomputed key results. Final preservation evidence is in the run's
`final-audit.json`: initial 1,770 workspace files, 1,478 protected files and 16,978
previous run artifacts. Only evaluation documentation/ignored artifacts changed.
Status is offline `pilot`, natural extraction validation `blocked`; no overall
accuracy-gain claim. Next remains M10 independent source-first annotation/reviewed
correspondences, then a fresh authenticated comparison once access is corrected.
Do not start M11, retry the rejected credential, or tune frozen methods to these
evaluation cases without a new explicitly versioned development study.
