# M10 source-first reference readiness — 2026-10-09

Question: can an independent reviewer inventory the supplied publication before
seeing predictions, with explicit coverage and exact evidence, so later scoring
can detect omitted entities rather than checking only emitted assertions?

Scope: continue M10 offline. The user confirmed ChatAI access is not corrected;
live evaluation stays blocked, with no new authentication/inference attempt.
Reuse DocumentBundle/EvidenceRef, the frozen canonical Profile, structural/local-ID
checks, safe run storage and the existing offline evaluator. No runtime, evaluator,
schema, model budget or released data changes. Baseline: 2,569 safe tests passed.

Minimal design: an offline-only source-first packet and submission validator.
The packet contains the complete available prepared text, source regions/warnings,
canonical field instructions, and blank reference/coverage forms for all six entity
collections. It contains no existing gold, predictions, alignment aliases, model
proposals or earlier assistant answers. A separately supplied reference is checked
for packet/source/profile bindings, canonical structure, local IDs, every entity's
identity evidence, scalar/reference evidence and explicit collection coverage.
Missing or uncertain work stays incomplete; self-declared reviewer provenance is
never authenticated expertise or independent gold. No automatic scoring promotion.

Implementation interfaces: `offline.reference_types` for pure typed validation and
`offline.reference_review` for packet preparation, immutable validation receipts
and local CLI. New source/annotation artifacts stay in ignored fresh run directories.
Use the existing source allowlist; reject final-held-out scope in this preparation
interface. Keep future alignment adjudication and scoring as separate explicit work.

Required checks: source-only read boundaries; blank versus completed/partial forms;
duplicate IDs, dangling relationships, stale/forged evidence and binding changes;
coverage of each collection and every entity identity; null/unresolved distinctions;
fresh paths, tamper detection and idempotent replay; human/synthetic/assistant
provenance; inherited scoring still counts omitted entities as false negatives.
Use invented fixtures, then prepare only the approved calibration paper 1812.03956.

Status at start: implementation `not_run`; natural annotation and live comparison
`blocked`. This packet cannot establish extraction gain, expert participation,
whole-PDF coverage or alignment validity. The inherited scorer currently cannot
accept new references from runs or certify new-paper correspondence; do not relabel
them as historical gold. Next after this preparation is actual independent review
and an explicitly frozen reference/alignment adapter, within M10, not M11.

## Implementation and use

Added `src/meta_data_extraction/offline/reference_types.py` and
`reference_review.py`, with three synthetic test modules. Runtime and existing
scoring code are unchanged. Both complete canonical schemas are required so
collections absent from an ablated/subset profile cannot be mistaken for zero
entities. Source/schema/implementation hashes bind the packet; submissions are
separate immutable receipts. Completed replay verifies originals and exports;
changed inputs, invalid evidence or interrupted writes fail explicitly.

The actual packet is `runs/agent/m10-reference-181203956-v1/`:

- `instructions.md`: source-first authoring steps and limits.
- `original.pdf`, `source.txt`, `packet.json`: the exact original, available text,
  source hash, region offsets and unavailable-image/page-mapping warnings.
- `schema.json`: unchanged field meanings, granularity and local-ID instructions.
- `submission-template.json`: no canonical record, reviewer or evidence; six
  explicitly unreviewed collection checklists. Blank counts are unknown, not zero.

No existing gold, predictions, alignment aliases, earlier assistant reference
answers or correction suggestions enter this packet. The paper has previously
been inspected by the assistant; a clean packet does not turn it into a held-out
test or establish a new reviewer's independence.

Preparation specification and protocol:
`runs/agent/m10-reference-readiness-v1/{packet-spec.json,protocol.json}`.
Protocol SHA-256: `4d86a39bbaef66aac8c0fe02b3958a8b6bcba1e7289da902e9a7b2f1585e8b09`.

~~~powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE='1'
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.offline.reference_review prepare --spec runs/agent/m10-reference-readiness-v1/packet-spec.json
~~~

The reviewer copies the blank form to a new ignored run file and authors the
reference before seeing predictions. Annotate each entity identity, local ID,
scalar and relationship using exact source evidence. Nulls and empty fields need
explicit missingness review; unresolved or unavailable information stays visible.
A local ID's evidence establishes the entity it denotes, not that the paper
literally prints the assigned ID. Review notes are concise observable explanations,
not private reasoning traces. Do not coerce unknown Booleans to false to satisfy
a schema; retain unresolved schema/missingness issues.

To construct an exact EvidenceRef without manually copying hashes, use the
following command with a region ID from `packet.json`; optional `--start` and
`--end` select a subspan at original text offsets:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.offline.reference_review excerpt --packet-run m10-reference-181203956-v1 --region-id REGION_ID
~~~

After actual authoring, submit the new file and its SHA-256 with a fresh receipt
run ID. These are placeholders, not an existing completed review:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.offline.reference_review submit --packet-run m10-reference-181203956-v1 --submission runs/agent/YOUR_REVIEW/response.json --sha256 RESPONSE_SHA256 --run-id YOUR_FRESH_RECEIPT
~~~

An unfilled draft yields `incomplete`. Even a fully annotated, self-declared human
review yields only `available_text_review_recorded`: its independence, semantic
correctness and whole-source coverage remain unestablished, and scoring remains
blocked for reference/alignment adjudication. Actual observed review time may be
recorded; missing time stays null. No reviewer declaration is invented or supplied
automatically. Generated excerpts establish locations, not support judgments.

Worked example: the source/reference includes two distinct specimens but a
prediction emits only one, with correct attributes. Two new synthetic scorer
checks confirm that the unchanged evaluator reports one true positive, one false
negative and entity F1 2/3 (for samples and, separately, datasets). This catches
omission even when checking the emitted record alone would miss it. It is software
verification, not measured extraction performance on the natural paper.

Trade-off: a complete source-first inventory is more reviewer work than validating
selected model assertions, but it supplies the denominator needed for entity
recovery. The packet intentionally captures drafts without claiming expert truth;
human review and correspondence adjudication remain separate gates. Future reports
must preserve initial extraction, pre-review controller output and human-reviewed
export separately, so human assistance is not called autonomous controller gain.

## Outcomes and handoff

Implementation checks passed: **591 tests in 117.18s**, including all 85 additions
and the complete existing offline suite plus reused storage/source/schema/path/
evidence and runtime-import-boundary checks. The last full runtime regression
baseline remains 2,569 passing tests; it was not repeated because runtime is
unchanged. No lint/type checker is configured. Development fixes and exact commands
are recorded in `../COMMANDS.md`.

The natural packet contains the original 573,552-byte PDF and the same 22,929
prepared characters as the prior source snapshot. All 78 region references resolve;
images and page mapping remain unavailable in prepared evidence. Source-first
forms contain no reviewer, canonical record or judgments. The engineering blank
receipt under `m10-reference-blank-check-v1` is `incomplete`, with six unknown
entity counts and no canonical export; it is not a human review. Packet and receipt
replays preserve bytes and modification times. `packet-audit.json` records these
checks under the preparation run.

Final preservation audit: 1,764 initial files, 1,770 final; three intended document
edits and six additions, no missing files. All 1,478 protected files and 47 prior
assessment/pilot/quota artifacts remain unchanged, as do existing runtime/offline
scoring code, schemas, configs, AGENTS.md and dependencies. No credentials were
read; zero model requests and no network. The old cancelled job and shared ledger
remain intact. `final-audit.json` records this check.

Experiment status: **`not_run`** for natural annotation/performance comparison;
packet preparation is complete and `awaiting_human_reference`. No new extraction
performance result is claimed. Live authentication/quota, actual independent human
annotations, reviewed correspondences and human usability remain blocked/unrun.

Next action: have a reviewer who has not seen the predictions author the supplied
source-first reference. After actual annotations are frozen, implement the narrowly
scoped offline reference/correspondence adapter without changing metric definitions,
and prepare the fresh live comparison only after authentication and quota checks
succeed. Do not mark annotations historical, silently unblock alignment, restart
the cancelled job or advance to M11.
