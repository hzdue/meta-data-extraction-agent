# M10 integration completion — fresh publications

Opened 2026-10-09 after the user explicitly requested completion of the implementation.
This extends M10's integration scope; it does not open optional M11 training.

Question: can one explicitly supplied publication pass through actual extraction,
bounded evidence checks, the existing controller, persistent review and export
without a prebuilt record, precomputed diagnoses or a fixture correction catalog?

Baseline: the existing M10 synthetic pilot and its 2,217 passing safe regression
checks. Reuse M01 source preparation, M02 extraction/provider boundaries, M04 PaNET,
M05 reviewed NeXus registry, M06 checks/patch executor, M07 workflow and M08 review.
Keep the inherited package/schema/evaluator and existing synthetic entry points.

Minimal design: a typed publication specification freezes one source/hash/split,
profile, provider, policy, standards and budgets. A parent manifest and deterministic
child runs isolate extraction, checking, proposal generation and workflow. Each
model dispatch requires a persisted scope/claim before execution; partial calls
never silently retry. Live stages share the existing persistent ten-attempt/hour
ledger. Runtime reads no gold/evaluation inputs. Actual extracted assertions drive
retrieval and issues; at most one bounded model proposal uses the existing scalar
patch allowlist. Corrections still need fresh before/after evidence checks.

Deterministic fixed and issue-aware policies are the initial live integration;
the existing generic-selector experiment remains separately available. Automatic
patches are explicit opt-in. PaNET proposals remain unapproved sidecars. Only
approved/resolved NeXus guidance may reach the controller; absence stays visible.
Incomplete sources, unselected assertions and coverage limits remain explicit.

Required checks: source/config tampering; malformed/failed/truncated extraction;
automatic target/relation selection; no unsafe/identity edits; standards gates;
parent/child budgets and live shared accounting; persisted restart and zero-call
replay; dynamic proposals visible in M08; local API path with declared documents;
stale/duplicate review and export; preserved released assets and no secrets.
Use authored offline responses by default and distinguish mock checks from any
separately frozen live diagnostic. No performance improvement is assumed.

Implemented interfaces:

- `publication_types.PublicationSpec` freezes one explicit source, its split,
  canonical profile, policy, optional standards, retention and budgets. Runtime
  configuration rejects precomputed records, issues and proposal catalogs.
- `publication_run.run_publication` and `read_publication` compose the existing
  extraction, checks, controller and durable review stage. Completed stages resume
  across processes; completed parents verify inputs and artifacts without calls.
- `publication_checks` selects actual emitted assertions and evidence, translates
  support outcomes into issues, and parses at most four allowlisted scalar edits.
- `publication_standards` produces unreviewed PaNET sidecars for the current record
  and supplies only approved, resolved NeXus registry guidance to the controller.
- `publication_cli` and `PublicationService` expose run/status/review/submit/export
  plus the existing authenticated loopback transport for one configured run.

The inherited package, canonical schema and evaluator remain unchanged. The old
synthetic pilot is still a separate frozen demonstration; its packaged runtime
does not automatically acquire these new modules. The fresh-publication interface
is a CLI/JSON API, with no document upload or new browser review interface.

Two small existing-module changes complete the handoff. M08 accepts the new
versioned, content-addressed proposal catalog. M07 no longer reserves correction
checker calls when automatic patching is disabled, so a three-call manual run can
retain its proposal for review. Unknown usage after proposal generation also
disables further automatic calls when the configured policy is `stop`. NeXus
registry fields match full JSON pointer patterns, including collection identity.

Worked example: the authored extraction response reports Dataset A at 500 K and
attached to sample_2. Source checking identifies preparation/acquisition confusion
and the wrong specimen. A generated proposal changes only the description to
300 K and the reference to sample_1. Two fresh checks guard the edit. Sample IDs
and other records remain intact. The twelve-target budget leaves thirteen emitted
assertions unchecked; their paths remain in the support plan and review stays
pending. This is an engineering fixture, not independent correction truth.

Trade-off: one combined scalar proposal and bounded evidence selection keep calls
and recovery simple. This can leave correctable assertions or omitted entities
unresolved. The runner records those limits; it does not infer entity recovery or
completeness from validating emitted fields. The same model's checks are not
independent gold.

Validation: all 2,504 safe tests passed in 695.59 seconds, including 287 new
integration checks. The actual CLI diagnostic passed 19 separate processes using
eight mock calls and zero live requests; completed replay preserved all 63 checked
pipeline/review files. Full commands and results are recorded in
[COMMANDS.md](../COMMANDS.md). The transport tests exercise the real adapter and
shared ledger with invented SDK responses and clocks; no real ChatAI request or
real `.env` read is involved. Protected assets and both prior pilot bundles pass
the final preservation audit. Implementation status: ready within this scope.

Experiment status: `not_run` for a fresh-publication accuracy comparison and human
operator effort. Independent labels, observed review, approved NeXus mappings and
an appropriate frozen natural-paper protocol remain prerequisites for those
claims. No approval or successful experiment was invented. Browser interaction
and Docker remain untested.

See [PUBLICATION_RUN.md](../PUBLICATION_RUN.md) for the architecture and commands.
Use fresh run IDs after implementation changes; old workspace checkpoints require
an explicit compatibility migration, while earlier isolated pilot bundles remain
unchanged. Stop within M10. Next: the deferred operator trial and a scoped natural
publication comparison; optional M11 is not started.
