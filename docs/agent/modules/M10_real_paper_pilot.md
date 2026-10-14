# M10 follow-up — one real-paper development pilot

Latest status (2026-10-09): **blocked**. The user's subsequent provider-limit
request supersedes the queued handoff below. The job was cancelled before any
inference at 10:25:55 UTC. One non-inference quota check returned HTTP 401 and no
quota headers. A valid local credential/endpoint and successful verification are
needed before adapting limits and preparing a fresh pilot job. Original sealed
artifacts and ledger history are preserved; do not restart the cancelled claim.

Endpoint follow-up: a second, separately recorded administrative check used the
exact documented `saia.gwdg.de` quota URL and also received HTTP 401 without quota
headers. There have been two non-inference API requests and zero model calls.
Authentication/access remains unresolved; neither response proves the key is
invalid for every route. No additional retry, limit change or pilot restart ran.

Opened 2026-10-09 on the user's request to proceed with the small real-paper
evaluation. This stays within M10; M11 is not started. Live evaluation is not yet
complete at the 10:20 UTC handoff. The frozen protocol and job status are under
`runs/agent/m10-real-paper-prep-v2/`; progress below distinguishes preparation,
queued execution and observed results.

Handoff status: **queued; experiment not_run**. Hidden local process 12548 was
launched at 10:19:08 UTC and verified alive with its exclusive claim. Its event at
10:19:25 UTC is `waiting_for_accounting` / `request_history_unknown`, ten free
slots, zero dispatched attempts. The first eligible time is 10:55:15 UTC. The
job will attempt the sealed runtime once and then collect offline results;
`job-finished.json` distinguishes completion from a stop, and `report.json` /
`report.md` exist only after successful collection. The computer must remain
running for this local process to finish. This is not a completed live result.

Question: can the existing fresh-publication runner extract a real calibration
paper, perform bounded evidence checks and preserve a reviewable result using
the authorized ChatAI service? Compare the saved initial extraction with the
final manual-policy canonical record, and score both through the existing offline
harness. This measures baseline extraction and diagnostic operation; automatic
corrections are disabled, so equal canonical records imply zero metric change
by design. This is not a test of automatic correction benefit.

Starting architecture: reuse M01 PDF preparation, M02 evidence extraction and
ChatAI adapter, M06 support checks, the M07 controller, M08 persistent review,
M10 publication CLI and M03 isolated evaluator. The prior safe suite passed
2,504 tests. No runtime, schema, prompt, dependency or evaluator implementation
changes are part of this experiment.

The declared publication is `1812.03956`, from the existing calibration split.
Its source SHA256 is
`158984e801673f2ce019d9fd793684183bbc288e1b27856039aa37b6c9845bce`.
The initial historical candidate `1804.07163` failed the unchanged 100,000-byte
prompt limit at 111,989 bytes, before inference. Its failed preflight is preserved
under `runs/agent/m10-real-paper-prep-v1/`. The replacement was selected as the
shortest calibration PDF by plain-text length, with arXiv ID as the tie-break,
before inspecting model outputs or gold values. The full evidence prompt is
68,176 bytes for 11 pages and 22,929 prepared text characters. This convenience
selection is not representative of the corpus.

The endpoint `https://chat-ai.academiccloud.de/v1` is present in a comment in the
local `.env`; the credential variable is `LLM_API_KEY`. Its value is never printed,
logged or committed. The exact approved model is `qwen3.8-27b`. The frozen run
allows one publication, at most three calls, 8,192 output tokens per call,
100,000 prompt bytes per call, a 240-second provider timeout, twelve selected
assertions and no inference retries. PaNET and NeXus guidance are disabled for
this diagnostic; no unreviewed mapping is treated as approved guidance.

The real shared ledger did not exist at preparation. Its creation at
2026-10-09 09:55:15 UTC starts the existing full-hour unknown-history quarantine.
Earliest possible dispatch is 10:55:15 UTC (12:55:15 Europe/Berlin). Neither clock
nor ledger history is backdated. The queued helper requires three available
slots before launching; each actual attempt is still atomically reserved under
the shared ten-attempt rolling-hour cap. Failures consume the allowance too.

Ignored experiment helpers prepare, seal, wait, run once and collect results.
The seal pins source, prepared representation, profile, prompt, runtime code,
collector, evaluator and relevant environment versions. A persistent exclusive
claim prevents automatic retries after interruption. Input changes, cancellation,
an accounting error or a two-hour wait deadline stop the job. Only safe process
metadata and log hashes are retained by the launcher. Raw model outputs follow
the selected local retention policy inside ignored run directories.

Runtime parent: `runs/agent/m10-real-181203956-v1/`. After completed runtime
artifacts are verified, a separate collector invokes the existing offline scorer
in `runs/agent/m10-real-181203956-score-v1/`. It alone may read the declared
calibration gold values. Preflight has only hashed gold bytes. No final-test gold
is authorized. The evaluator was frozen before inference as
`evaluator:c7706e69b8ae7e9b739b799da246ccf8c64779798f3e4c23defe9d333ccd8dda`.
All source derivatives and both prediction variants remain one publication.

Worked example and trade-off: this PDF's prepared source reports
`images_unavailable` and `page_mapping_unavailable`. Those warnings remain in the
runtime evidence rather than being removed to enable a correction. Incomplete
source coverage prevents a successful support-report gate and therefore prevents
patch generation. The expected useful output is an extraction with bounded
diagnostics and unresolved review, typically two calls, within the three-call
cap. This conservative behavior protects corrections but limits what this PDF
pilot can establish about the controller.

Primary outcome: dataset gold-entity recovery with the unchanged inherited
precision/recall/F1 and denominators. Its interpretation remains blocked until
alignment coverage is independently reviewed. The manifest explicitly uses
`alignment_status=unreviewed`: raw scorer diagnostics are retained, while sample,
dataset and funding collections and dependent dataset references are excluded
from eligible comparisons. Eligible conditional attributes, references and their
support remain separate; there is no schema-wide average.

Report calls, failures, known tokens, provider latency, unknown usage/cost,
unselected assertions and pending review. Shared initial extraction is counted
once rather than summed across comparison variants. Same-model diagnostics are
not independent support/harm labels. No real human approval, measured review
time, confidence interval or general improvement claim is supported.

Required checks: input integrity, single execution, shared accounting readiness,
no inference retries, post-runtime collector integrity, explicit failed variants,
unknown-resource accounting and unchanged released assets. The one-shot launcher
passed seventeen synthetic tests using invented workspaces, clocks and processes;
those tests neither modify the real ledger nor call a model. Exact commands and
subsequent status are recorded in [COMMANDS.md](../COMMANDS.md).

Four additional collector tests passed, including a complete synthetic workflow
through report/replay, failed extraction and an incomplete-parent scoring gate.
They used five fixture-only mock calls and zero live requests. A strict evaluator
JSON-deserialization issue was fixed only in the ignored collector helper.
Sealing verifies 93 files; readiness verification correctly waits without making
an attempt. Final preservation checks found only the two documentation updates
and this new record, with all baseline code and protected results unchanged.

Next within M10: inspect the saved pilot report, resolve calibration alignment
coverage independently, and conduct the deferred human review session. A further
frozen correction comparison needs an appropriate evidence/label scope. Do not
expand this one-paper run or tune the frozen method after reading its gold scores.
