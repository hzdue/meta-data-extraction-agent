# M10 — Local pilot, packaging and handover

Status: implemented and offline/API checks pass within the single-user local
synthetic pilot scope. Browser/container checks remain untested; E10 human
usability/effort evidence is not_run. Stop after M10.

## Question, starting architecture and baseline

Can another person run, inspect and recover the workflow without notebook repair?
M07 already supplies bounded, persistent mock workflows; M08 supplies explicit
revision-bound review, SQLite transactions and immutable canonical/sidecar exports.
M09 supplies isolated offline study scoring. Existing CLIs are the baseline. Reuse
these implementations without changing canonical semantics or empirical claims.

## Scope and interfaces

Default pilot scope: one operator, a loopback-only authenticated HTTP API and a small
review page, using the existing invented M06/M08 source, initial record and mock
policy responses. An optional interface preference was requested while independent
implementation proceeds. There is no document upload, live inference configuration,
public/cloud deployment, external export destination or authenticated expert role.

The pilot backend provides health, bounded start/resume, status, review, explicit
response submission and version-bound export. It delegates to inspect_workflow and
ReviewStore. Restart, stale revision and exact response replay retain their existing
semantics. The local export adapter returns unchanged canonical JSON together with
its review sidecar and manifest; pending or simulated review is never approval.

The transport requires a token from META_AGENT_PILOT_TOKEN for API requests, binds
only loopback and checks Host/Origin and request limits. Tokens, source/review text
and request bodies are excluded from structured logs. The page uses no external
assets and keeps its token only in memory. This is a local prototype, not a
production or multi-user service.

Packaging copies only explicitly allowlisted runtime code, schemas, prompt,
configuration and invented fixtures to a fresh ignored runs/agent/<id>/runtime.
Gold/evaluation code and data, released outputs, paper files, credentials and
previous raw runs must be absent. A manifest binds immutable assets; only that
bundle's runs/agent tree is mutable. Preserve package name, Python requirement and
lockfile. Provide a non-root container recipe for the isolated bundle, but Docker
is absent in this environment: a build/run cannot be claimed without execution.

## Checks and diagnostic before execution

Unit/integration checks cover authentication, same-origin access, input size/path/
JSON bounds, secret redaction, no provider/network egress, stale/concurrent review,
idempotent exports and persistent restart. Validate the packaging allowlist and
transitive imports from an isolated process. Attempt an environment-scoped clean
installation with the unchanged lockfile, distinguishing missing cache/platform
tools from code failures. Run the established safe regression suite.

The actual engineering diagnostic will start the invented pending-review workflow,
inspect source evidence, submit an explicitly simulated response in another
request/process, reject a stale response, replay the original submission and
download the bound canonical/sidecar export. This measures API/recovery mechanics,
not expert time or natural-paper extraction quality. No real .env or live model
request is needed. E09's natural labels/reviewed-guidance gates remain unresolved.

## Implementation and operator handover

New runtime code is pilot_bundle.py, pilot_core.py and pilot_http.py, with
pilot_ui.html. Packaging is frozen by configs/agent/pilot_bundle.json and
deploy/agent/Dockerfile plus its .dockerignore. Each Python component has a matching
test_pilot_* module. [PILOT.md](../PILOT.md) provides exact setup, token handling,
API routes, evidence/review operation, restart recovery and export instructions.
The package name, Python requirement, lockfile and earlier modules are unchanged.

The facade serializes operations with the existing OS lease, permits eight scoped
runs, checks a 128 MiB/4,096-file budget and preflights Windows export path length.
The transport bounds request bodies to 128 KiB, headers to 16 KiB and responses to
2 MB, with strict JSON and five-second request reads. Early-error shutdown drains
uninterpreted input for at most 151,552 bytes/250 ms; it cannot parse or dispatch
an unauthorized request. Structured events precede complete responses so clients
cannot observe completion before the audit event is recorded.

The main trade-off is a reproducible, small local demo versus a general extraction
service. Fixed pinned fixtures keep evidence, recovery and cost boundaries
reviewable, but this endpoint does not accept new documents or live model settings.
The guided unresolved action is usable without constructing JSON; approve/edit
still needs an explicit JSON response with source-bound attestations. This is a
deliberate prototype limit, not a finished curator interface or expert identity
system. A bundle change requires a new bundle/run rather than silent migration.

## Observed outcomes and remaining gates

The final bundle is runs/agent/m10-20261008-pilot-v2/runtime, with 64 pinned assets
and manifest SHA256
dd8abc945115542bc3df987b32531f4899aa8f958da6586b9969c567c9047234.
It excludes offline/evaluation modules and data, credentials and previous runs.
The clean environment at runs/agent/m10-20261008-install-v2/venv installed all 62
locked packages offline. The first attempt needed missing public package downloads
to populate the local cache; no dependency or lockfile changed.

The final independent-process diagnostic passed 24 loopback HTTP requests across
three server processes. A one-step start used one mock policy call; a new process
resumed with three more. Status and completed replay used zero new calls. An
explicitly simulated unresolved response produced one decision, left canonical
JSON unchanged and retained zero active attestations. A second run used four mock
calls and one simulated edit with three source-bound attestations, correcting only
Dataset A's 500 K description to 300 K and sample_2 to sample_1. Specimen identities
and all other scalar leaves stayed unchanged. Duplicate submission returned replayed;
stale review and export requests returned 409. Export was identical after restart,
and logs omitted tokens, source text and reviewer response content.

The final diagnostic used eight mock calls and zero live requests. The preserved
first bundle used another eight, plus four to prepare a browser fixture: 20 mock
invocations across actual M10 diagnostics, no ChatAI requests. Unit-test fixture
calls are separate engineering checks. Unknown usage stays unknown; no live
latency/cost or reviewer-time inference is made.

Focused checks passed: 42 packaging, 54 core and 94 final HTTP/UI tests. The first
combined suite exposed two Windows early-close resets and a response/log timing
race (2,209 pass, three fail); transport fixes and five targeted regressions address
them. The final combined suite passed 2,217 tests in 534.21s, including all 2,027
previously passing safe checks and 190 M10 checks. Exact commands and the
protected-file audit are in [COMMANDS.md](../COMMANDS.md).

Docker is absent, so image build/run and container recovery remain **untested**.
The computer-use browser helper failed during initialization, again after reset;
no successful rendered-page interaction occurred. The static UI and CSP checks
passed, and the inline script compiled in V8, but browser execution, visual layout
and interactive usability remain **untested**. The temporary preview was stopped.
No public deployment, real .env read, live inference or external export occurred.

Experiment E10 is **not_run**: no actual user usability session or correction-time
observations were collected. E09's independent-label, reviewed-guidance and natural
comparison gates remain blocked. The next practical step is an operator trial of
this local pilot and an agreed scope for approved real-document work. M11 is
optional and requires a measured bottleneck plus explicit training/data/compute
authorization; do not start it automatically. Stop after M10.

## Follow-up: prepare the operator trial for later (2026-10-08)

The user selected preparation for a later session. Added the development-checkout
launcher scripts/agent/start_pilot.py and [PILOT_TRIAL.md](../PILOT_TRIAL.md), a
frozen synthetic task protocol with blank observation fields. The launcher uses
the existing .venv and unchanged packaging/server entry points. It prepares a
fresh bundle or explicitly verifies an existing one, scopes child environments,
and keeps a temporary pilot token out of arguments, logs and files. Default
serving copies that token to the clipboard; --prepare-only performs no clipboard
write or server/model dispatch. The clipboard branch remains untested.

The prepared trial bundle is runs/agent/m10-trial-ready-v1/runtime, manifest
9cb0d2a6b213cc6d1d23417915ecaf3d8d5a03c433d7761eb98cfe87bca48ff5.
Its outer directory holds trial-plan.json and observations.md, with no human
observations filled in. The plan pins the protocol, launcher, runtime and invented
source. It budgets two child runs, at most 16 mock calls, zero live requests and
45 minutes for the later session. Core tasks cover evidence inspection, unresolved
review, a stale second tab, process restart and export equality. The optional edit
remains a guided manual JSON task; no accuracy or speed comparison is implied.

A PowerShell launcher draft could not run under a fresh shell's existing script
execution policy. The final Python launcher avoids a .ps1 file and makes no Windows
policy change. Eleven diagnostic checks passed, covering bundle collision/path
rejection, parent environment preservation, missing-bundle and occupied-port
errors, actual loopback startup, health 401/200, redacted logs, owned-process cleanup
and byte-identical bundle files. A separate fresh Python preparation also passed.
No child workflow or model request ran; the temporary listener was stopped.

Browser tool initialization still failed before creating a tab. Browser rendering,
clipboard copying, Docker and human interaction remain untested. E10 stays not_run,
M11 remains unopened, and no empirical gate was relaxed. The existing 2,217-test
baseline remains the last full suite; this follow-up changed no runtime package,
schema, fixture or dependency, so only launcher/handover checks were run.

## Follow-up: connect fresh extraction to review (2026-10-09)

The user requested completion of the integration after discussing its remaining
gaps. [M10_integration.md](M10_integration.md) records the added fresh-publication
CLI/JSON API, automatic source checks and proposal generation, standards adapters,
and durable review handoff. [PUBLICATION_RUN.md](../PUBLICATION_RUN.md) is its
operator guide. This extends M10; it does not change the earlier frozen pilot
bundles, claim a human usability result, or open optional M11.
