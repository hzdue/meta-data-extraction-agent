# M08 — Persistent human review and explicit approval

Status: implementation ready; restart, stale/duplicate and export recovery gates pass. Real review experiment E08 is blocked; stop before M09.

## Question and starting architecture

Can a user resolve a precise metadata question and resume the same run safely after a restart? M07 already has typed revision-bound pending cases, a bounded workflow, SQLite checkpoints/dispatch accounting, and immutable terminal exports. It cannot accept reviewer responses. M06 supplies narrow scalar candidate construction, full structural/reference checks, and source-bound evidence validation. These implementations will be reused without changing their historical behavior.

## Minimal design and interfaces

An additive versioned review session lives under the existing pending run's review/ directory. Original M07 exports and checkpoints remain immutable; the review stage resumes locally without model dispatch or a budget reset. SQLite transactions commit responses, revisions and idempotency keys together. Responses bind both the exact record revision and review state, and explicitly choose approve, edit or unresolved. Reviewer identity is self-declared; synthetic responses never become human or expert approval.

Review edits reuse the bounded M06 candidate builder, validate the full canonical record, resolve supplied evidence, and require fresh explicit support attestations for each selected assertion and all touched dataset relationships. Revisions invalidate prior approval bindings and reopen unselected cases. The record's original/revised versions and decisions remain available. Immutable per-state exports are derived from the committed database, with recovery for incomplete publication. Unknown state/code versions fail closed; no implicit migrations.

CLI views show affected entities, exact disputed values, source passages, uncertainty, available definitions and candidate corrections. No UI or new provider dependency is needed. Approval is scoped to reviewed assertions and is not a claim of completeness or authenticated expertise.

## Frozen engineering baseline and checks

Comparison: pending-only M07 baseline versus the new review transaction/resume behavior. Split: existing invented M06 publication/source and records only, with explicitly simulated responses. Source SHA256: a7f2c455d3f1cb10a47c6e3865bbc4e0bd6e288ce8ff6bae1b72fab8eb619996. No natural gold, real reviewer identity, model inference, .env read, or paid/live request is planned. Primary outcomes are restart equality, stale/duplicate safety, validated atomic edits, unresolved retention, and recoverable exports, not extraction accuracy. No evaluator or metric changes.

Required checks: independent-process pause/load/response/export; old revision/state rejection; duplicate response no duplicate effect; invalid edits/evidence/relationships rejected; unresolved retained; commit/export interruption recovery; incompatible versions and changed pins rejected; safe path and immutable M07/released-asset checks. Run focused offline tests and the established safe combined suite.

E08 review-time/error/prioritization comparison remains blocked by missing actual review observations, independent residual-error labels, and a frozen matched/counterbalanced assignment and budget. Simulated engineering responses do not measure expert effort.

## Outcomes

Added four production modules: review_types.py (immutable contracts), review.py (pure transitions), review_store.py (SQLite persistence and revision exports), and review_cli.py (show/submit/export/resume). Added three test modules, one pinned workflow config and four synthetic policy fixtures. No prior production module, package name, dependency, lockfile, canonical schema or scientific asset changed.

Final safe suite: 1,795 passed in 206.57 seconds, including 165 new M08 tests (73 pure review, 61 persistence, 31 CLI). The checks include separate processes, simulated provenance, invalid/stale edits, repeated responses, closed-store and competing-writer rejection, SQLite auxiliary-path checks before transactions, parent accounting/corruption checks, before/after-commit faults, six export crash points, and immutable historical exports. No lint or type checker is configured. Exact commands and infrastructure notes are in ../COMMANDS.md.

The fresh correction run runs/agent/m08-20261007-pending-v1/ contains one simulated decision, three fresh assertion/relationship attestations and two intended scalar changes. Its review export is review/exports/6b4b4c4b96ffc27960a13921eb52cd1e008e73d244a30ec2ed56556fdec68120/. Duplicate submit and resume preserved all 12 files byte-for-byte, including all seven original M07 files, and retained one decision. A separate run runs/agent/m08-20261007-unresolved-v1/ retained one simulated unresolved response, zero active attestations and the unchanged canonical record. Export hashes and stable entity handles were verified for both.

Each parent workflow used four mock policy calls; each review stage used zero model calls. There were zero live requests, no real .env reads, no natural-gold inspection, no real human approval, and no measured reviewer effort. Original token usage remains explicitly unknown. Synthetic responses are never independent gold or evidence of performance.

## Worked example and trade-off

The invented Dataset A record initially attached Sample B and described its measurement as 500 K. The targeted view shows both the current Sample B (prepared at 700 K) and proposed Sample A (prepared at 500 K), alongside the source statement that Dataset A measured Sample A at 300 K. A simulated edit replaces only the description with 300 K and the sample reference with sample_1. Three explicit attestations cover the description, sample attachment and facility attachment. Both specimens and all entity handles survive unchanged.

The review stage deliberately does not mutate the original M07 terminal snapshot or restart model execution. Its current SQLite state and returned revision-export manifest identify the reviewed result; the parent's top-level canonical.json and pending_review status remain historical. This preserves reproducibility and budgets, but consumers must explicitly select the reviewed export. Identity labels are self-declared, not authenticated expertise. Completed means the queued cases were answered, not that every publication assertion is supported or the extraction is complete.

Approval is conservative: editing invalidates all old-revision assertions and reopens unselected cases. Leaving an attachment unresolved removes its current attestation and reopens dependent approvals. An approval cannot silently override a separately unresolved dependency. These safeguards can increase review work; no effort-saving claim is made.

## Operation, compatibility and limits

Use review_cli show with an existing pending run ID, then explicitly supply a ReviewResponse JSON to submit. The file must be under that run's review-responses/ directory or the synthetic fixture directory. It identifies idempotency_key, expected_state_id, expected_revision, case_ids, outcome, provenance, reviewer_id and rationale. Edits add a bounded PatchProposal and exact ReviewAttestation entries (path, canonical value_json, semantic role and resolvable evidence). Missing attestations or unsupported changes are rejected. An unresolved response supplies no approval attestations. Never label a generated response user_supplied or claim expert identity on the user's behalf.

submit commits once and exports afterward. If export is interrupted, resume or export completes it from the committed database without another decision or provider call. Repeating the exact response replays the current head; reusing its key for a different response is rejected. A private unfinished staging file may be rebuilt from committed bytes, while completed files cannot be replaced.

State, implementation, profile, parent-artifact, dependency and Python-version mismatches fail closed. There is no automatic migration or operator force switch. To inspect an old session, restore its pinned compatible checkout/environment; to adopt changed inputs/code, create a fresh workflow run and review session, retaining the old run. Migrating existing decisions requires a separately designed, tested operation and is not implemented here. A corrupt/missing initial review database is not silently rebuilt. An unresolved case stays closed in that revision; a genuine new edit can rebase it, but there is no general reopen command.

No UI, authenticated multi-user service, semantic truth verifier, broad entity editing, automatic post-review model continuation, or empirical queue comparison was implemented. Source-linked reviewer assertions can still be wrong. E08 remains blocked by missing real review observations and independent residual-error labels; reviewed standards guidance remains subject to the unresolved M05 gate.

Next recommended module: M09, controlled end-to-end study and ablations. First freeze its comparison, publication split, review protocol, resources, evaluator and budgets and resolve the necessary independent-label/standards gates. This M08 work does not authorize or claim that study.

## Planned local CLI diagnostic

Use the pinned configs/agent/workflow_review.json with the existing M06 source and correction. Four synthetic policy responses retrieve and queue the two issues, consuming four mock calls and zero live requests. Budgets: 12 actions, 5 per issue, 8 mock-call slots, 120 seconds for the workflow; review has at most 64 responses, 32 cases, 4 scalar edits and 32 attestations per response. No model call occurs during review; original accounting remains frozen.

Fresh runs m08-20261007-pending-v1 and m08-20261007-unresolved-v1 will demonstrate a clearly simulated paired correction and a separately unresolved response. Process A queues/shows pending cases and exits. A later process submits an explicit simulated JSON response; a further process replays it and resumes export. Compare current canonical bytes and stable entity IDs with the intended M06 candidate, count decisions, and hash every original M07 file before/after. A repeated response must not create a second decision. These diagnostics do not score an evaluator or measure reviewer time.
