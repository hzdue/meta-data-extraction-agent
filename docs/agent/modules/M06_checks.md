# M06 — Record checks, issue diagnosis and supported patches

Date: 2026-10-07. Scope: M06 only; stop before M07. Implementation ready; E06 empirical validation blocked.

## Question and reusable starting architecture

Can checks distinguish malformed representation, unsupported assertions, wrong
attachments and genuine ambiguity enough to produce useful issues and bounded
local corrections? This module tests mechanics; scientific accuracy needs E06.

M00–M05 already provide source/evidence hashes, immutable record revisions and
entity handles, frozen inherited profiles, structural validation, the bounded
provider adapter, exclusive run storage, offline evaluation and standards sidecars.
Reuse those implementations. There is no revision-bound support checker or atomic
patch executor to reuse. Keep inherited extraction/evaluation and canonical schemas
unchanged. The M05 reviewed-profile gate remains blocked; no NeXus guidance enters
M06 prompts or evidence.

## Interfaces, minimal design and baseline

Inputs: exact frozen profile; explicit source bundle; candidate canonical record;
bounded assertion paths and resolvable EvidenceRefs; provider; optional typed patch
and an explicit local patch policy. Outputs: three separate layers of checks,
observable provider accounting, model-judgment sidecars and an atomic patch receipt.

1. Structural/reference checks reuse validate_record. Invalid records retain all
   fields and explicit diagnostics; dangling references cannot become valid empty
   snapshots. The baseline is this deterministic structure-only report.
2. Evidence-sensitive checks include assertions, source excerpts and linked entity
   context. Supported/contradicted/insufficient-evidence verdicts are model judgments,
   labelled mock or live, never expert gold. Wrong attachment and role confusion
   require context, not only matching scalar text. Conflicts stay unresolved.
3. Coverage is explicitly unsupported in M06. Checking emitted assertions does not
   prove complete sample/dataset recovery or absence of missing gold entities.

Use one bounded checker call per revision and the existing provider boundary.
Timeouts, malformed/truncated output, missing/unknown evidence, unknown usage and
stale source/revisions cannot become successful empty reports. Preserve raw output
only under an explicit retain/hash-only policy; store concise observable judgments,
not chain-of-thought.

Patches are separate from checking: build a candidate without mutation, then apply
only under an explicit policy with old contradictory and new supportive judgments.
Allow at most four replacements of dataset description, measurement technique or
individual sample references. Preserve local IDs/entity handles and array topology;
forbid sample identity edits, entity creation/deletion/merge/split and arbitrary JSON
surgery. Recheck the full canonical structure and all sample/facility attachments
of touched datasets. Invalid, harmful, unsupported or stale proposals retain the
original state. New evidence binds the candidate revision; old support is never
silently rebound. A content-bound receipt makes completed application idempotent.

The CLI uses explicitly synthetic sources and MockProvider fixtures only, with
fresh ignored runs and verified completed replay. Pure checker APIs retain the
existing exact-scoped ChatAI boundary; no live experiment is needed here.

## Tests and E06 boundary

Required offline cases: preparation temperature misused as measurement temperature;
same-composition specimens with different treatments; valid swapped and dangling
sample references; correctly missing optional values; evidence failure; conflicting
diagnoses; harmful correction; stale source/record evidence; atomic multi-operation
rollback, stable identities, safe storage and idempotent replay. Synthetic model
responses exercise contracts and policies, not measured diagnosis accuracy.

E06 compares structural-only, generic source-support and identity/relationship-aware
checks on frozen real outputs. It needs independently labelled random and difficult
subsets, explicit sampling treatment and a frozen manifest; none is supplied.
Report E06 blocked rather than treating checker self-agreement as independent truth.
No final-test gold, result-regeneration scripts, live calls or real .env access.
Standing ChatAI model/ten-request rolling-hour authorization remains unchanged.

## Outcomes

Added four ordinary Python modules: record_checks.py for preserved structural
reports, support_checks.py for one-call evidence/attachment checks and conflict
diagnosis, record_patches.py for candidate preparation and atomic application, and
inspect_record.py for a fixture-only offline run/replay command. The runner reuses
M01 parsing, M02 profiles/provider contracts and RunStore. Two configs and five
invented fixture files make the behavior reproducible; no second extraction stack,
LangGraph dependency, standard-guidance gate bypass or evaluator change was added.

The checker reports each selected assertion's exact value, semantic role, entity
handle, relationship targets, source/revision hashes and supplied evidence. One
call accepts at most 32 targets with eight evidence links each; prompts/responses
are bounded. Conflicting judgments are retained for human review, not majority
voting. Malformed responses, provider failures, truncation, unknown finish reasons
and source-availability warnings remain explicit. Unknown usage stays unknown.

Patch proposals are supplied inputs, not automatically generated by this module.
The policy defaults to disabled. Accepted edits require complete before/after
checker execution, an old contradicted assertion, and a supported candidate plus
all affected references. Source conflicts cannot authorize an edit. Rejection is
atomic; stable entity identities and the original snapshot remain intact. Accepted
edits invalidate the previous revision's assertions and emit new evidence bindings
plus an unreviewed automated support sidecar. Hashes establish content consistency,
not authenticity, expert approval or scientific truth.

### Worked synthetic example and trade-off

The invented source says Sample A and Sample B both have composition CaCO3, but
have different treatments. Sample A is prepared at 500 K and measured as Dataset A
at 300 K. The initial record says Dataset A was measured at 500 K and attaches it
to Sample B. The structural baseline accepts both errors: the description is text
and sample_2 exists. Two fixed mock judgments flag preparation/acquisition confusion
and wrong_sample; candidate checks support 300 K, sample_1 and the unchanged facility
attachment. Only those two scalar leaves change, preserving the two samples and
all other content. A separate sample_99 fixture is structurally invalid and is
retained without a canonical export or checker request.

This conservative patch surface is easy to audit and cannot silently merge or
split specimens, but cannot fix missing entities/values, sample identity fields,
arbitrary attributes or reorder relationships. Requiring supporting automated
judgments reduces some mechanically unsafe changes; it cannot guarantee a model
judgment is scientifically correct. Only selected paths are checked, including in
the demonstration; other emitted fields remain unassessed.

### Verification and limits

All **262 M06 tests passed in 27.56s**: 47 structure, 107 support/diagnosis,
61 patch and 47 runner checks. The combined safe suite passed **1,346 tests in
82.56s** (1,120 agent, 151 offline, 75 inherited). Exact commands and intermediate
outcomes are in [commands](../COMMANDS.md).

The frozen correction run is runs/agent/m06-20261007-patch-v1/: two mock requests,
zero live requests, two intended scalar changes, stable entity handles and a new
revision. It exports canonical.json, sidecar.json, raw-response.json, request.json
and result.json. The repeated command performed zero new provider dispatches and
preserved SHA256 of every artifact. The structure-only run at
runs/agent/m06-20261007-dangling-v1/ retained sample_99 and its dangling-reference
issue, used no model and emitted no canonical.json. Both completed runs bind all
18 implementation-file hashes; all frozen source/record/response pins still match.

All 15 M00 baseline fingerprints remain unchanged. The pre-M06 inventory confirms
no removed files; only PROGRESS.md, COMMANDS.md and DATA_POLICY.md change among
existing files. New code/tests/configs/fixtures and this module record are additive.
No inherited implementation, lockfile, schema, released benchmark or result was
modified. No real .env read, live call, natural gold evaluation, result regeneration
or external resource download occurred. Full uv sync remains blocked by unavailable
cached hatchling; tests use the existing environment and explicit checkout binding.
No lint/type checker is configured.
E06 status: **blocked**. No independent real-output labels or frozen comparison
manifest are available. Live inference, natural-paper diagnosis accuracy, harmful
patch rates and cost/latency performance remain untested; no performance claim is
made from fixed mock responses. Coverage/completeness is explicitly unsupported.
The M05 reviewed-profile gate remains blocked and default guidance remains empty.

Next module: **M07 — LangGraph and bounded intervention policies**, beginning with
fixed and rule-based routes over these same components and budgets. Stop here; M07
and the later persistent human review workflow are not implemented in M06.
