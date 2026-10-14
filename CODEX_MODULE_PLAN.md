# Module-by-module design and experiment plan
## meta-data-extraction-agent

Prepared: 7 October 2026. Place this file in the repository root alongside `AGENTS.md`.

**Use:** authorize one module per Codex task. For that module, design, implement, test, and run only authorized experiments. Do not execute the entire roadmap in one task. M00 is the starting point when the checkout has not yet been assessed.

**Outcome:** a usable local/pilot service, a reproducible fixed baseline, and a controlled investigation of standards-assisted, issue-aware correction. A working fixed workflow is a worthwhile result even if adaptivity does not improve it.

## 1. Starting point: this is not an empty repository

The GitHub repository inspected for this plan was `hzdue/meta-data-extraction-agent`, branch `main`, at commit:

```text
74b8101f9572527101ea5c73abaddaf66aaba1e5
```

Read-only inspection covered the root listing, README, dependency file, source/schema listings, pipeline, extractor, and validator. The repository was not cloned, executed, or changed while preparing these documents. Codex must verify the current local state rather than assume it still matches this snapshot.

Observed starting assets:

| Existing asset | What was observed | Development implication |
|---|---|---|
| `pyproject.toml`, `uv.lock` | Python >=3.13; distribution `meta-data-extraction`; Pydantic, OpenAI-compatible client, PDF parsing and benchmark dependencies. LangGraph is not listed in the inspected dependency file. | Preserve the working environment/package identity. Add only necessary compatible dependencies, updating the lock deliberately. |
| `src/meta_data_extraction/pipeline.py` | Existing parser -> extractor -> validator -> tracker pipeline. | Keep as an inherited baseline; wrap reusable components rather than replacing it wholesale. |
| `components/extractor.py` | Schema prompting, character truncation warning, JSON parsing with fallback, provenance. Parse failure can return an empty object with warnings. | Preserve historical behavior but give the new runtime explicit failure/truncation states. Never report failed parsing as a successful empty extraction. |
| `components/validator.py` | `SimpleValidator` filters unknown top-level keys; it is not a complete structural or semantic checker. | Add explicit validation in the new path and preserve the original output for diagnostics. |
| `schemas/` | `generic.json`, `photon_and_neutron.json`, instruction-removed variants, prompt templates. | Canonical schema and task instructions are authoritative. New evidence/standardization fields belong in sidecars. |
| `alignment.py`, `evaluation.py`, `normalization.py` and evaluation configs | Existing benchmark evaluation implementation. | Offline reuse only; do not leak gold-specific alignment knowledge into runtime. Inspect details before reuse. |
| README, `data/`, `outputs/`, `paper/`, scripts | The repository retains benchmark/release assets. README describes scripts that replace evaluation and paper outputs. | Do not delete or rewrite these assets. Use new run directories and isolated adapters. |
| `LICENSE` | Root listing reports an empty file. | Record redistribution/licensing uncertainty. Do not choose a license or infer permission. This is not a legal assessment. |

These observations are not an audit of every file or a claim that every README command currently succeeds. Snapshot references are in Section 12.

### Proposed additive layout

Use existing equivalent implementations when present. This is a destination map, not an instruction to create every file as an empty scaffold.

```text
AGENTS.md
CODEX_MODULE_PLAN.md
src/meta_data_extraction/agent/
    contracts.py          # typed state, records, issues, evidence, actions
    evidence.py           # document regions and source-only retrieval
    providers.py          # mock + adapter to the existing model client
    extraction.py         # fixed extraction and supported local updates
    profiles.py           # versioned approved profiles; canonical schema refs
    standards/            # PaNET, NXDL, mapping registry
    checking.py           # structural and source-support checks
    policies.py           # fixed, generic, and issue-aware action policies
    workflow.py           # thin LangGraph orchestration
    review.py             # versioned review cases and responses
    storage.py            # record/sidecar persistence and export
    cli.py                # local run/status/review/resume commands
configs/agent/
tests/agent/
    fixtures/             # clearly synthetic, small, redistributable cases
    unit/
    integration/
    contracts/
docs/agent/
    REPO_AUDIT.md
    ARCHITECTURE.md
    PROGRESS.md
    DATA_POLICY.md
    RESOURCE_REGISTRY.md
    COMMANDS.md
    modules/Mxx_<name>.md
    experiments/E##_protocol.md
    experiments/E##_report.md
scripts/agent/            # isolated offline evaluation / study commands
runs/agent/<run_id>/      # ignored outputs, checkpoints, predictions and logs
```

The offline harness may import inherited evaluation code. Runtime modules must not import it or read its gold-specific configuration. Do not use a repository-wide retrieval index.

## 2. Product, research question, and fixed boundaries

### Product

Input: an approved publication source set and a selected, versioned metadata profile.

Output: canonical metadata plus evidence, standardization mappings, explicit issues, and review history. The workflow supports a bounded follow-up action, a persistent pause for review, and an inspectable export.

The first release is not a universal ETL platform, a self-modifying schema designer, or a complete NeXus experimental-data exporter. Configuration is expert-reviewed and separate from extraction.

### Primary research question

> Given the same extractor, source access, standards tools, and resource budget, does explicit diagnosis of a problem help a controller choose a more effective next action than generic reflection or a fixed correction sequence?

Candidate issue types are representation error, insufficient source evidence, questionable entity/reference attachment, unresolved identity, possible omission, and specification ambiguity. These are hypotheses about a useful representation, not guaranteed true labels obtainable from the current benchmark.

### Supporting questions

- Does PaNET lookup improve justified technique normalization, including knowing when a more specific concept is unsupported?
- Do selected NeXus definitions improve interpretation of existing fields without increasing unsupported values or harmful repairs?
- Can evidence-grounded review cases reduce correction effort while preserving final quality?

Measure these separately. Do not claim that the existence of the architecture, LangGraph use, or ontology lookup establishes novelty. The initial task is not a broad literature review; verify the closest primary work when a paper claim is proposed.

### Non-negotiable distinctions

1. A standard defines meaning and representation; a publication supplies evidence of facts.
2. Profile validity, model-assessed support, extraction completeness, and measured benchmark quality are different outcomes.
3. An emitted record can be correct while other records are missing. Coverage checking is not solved by checking emitted values.
4. An unresolved assertion is not automatically false. `not_reported`, `not_found`, `not_applicable`, and parser/unavailable-evidence failures must not be conflated. Preserve the canonical schema's conventions; place finer diagnostic statuses in sidecars.
5. Local-ID renaming does not change identity. Matching composition alone does not establish sample identity. All accepted mappings and patches must preserve reference consistency.
6. More tool calls or a stronger model are not evidence that the controller is better. Count their cost and isolate the changed factor.

## 3. How Codex should execute a module

For each module, complete one small, coherent change. The task should produce code and evidence, not just an architecture proposal.

Use this cycle:

**Inspect -> specify -> implement -> test -> experiment if authorized -> decide -> hand off.**

A module has separate readiness fields:

```yaml
module: M00
implementation: not_started  # not_started | in_progress | ready | blocked
offline_tests: not_run       # not_run | pass | fail | blocked
empirical_validation: not_run # not_run | blocked | pilot | validated
integration: not_run         # not_run | pass | fail | blocked
next_action: null
```

Passing synthetic integration tests can make the implementation ready; it does not make the scientific claim validated. A blocked live run should not prevent implementing and testing unrelated offline behavior. Do not substitute fabricated model or expert results.

### Module record template

Create/update `docs/agent/modules/Mxx_<name>.md` with:

- The question and one concrete example.
- Existing functions reused, changed, or intentionally not reused.
- Input/output contract and failure states.
- Minimum design and one plausible alternative; explain the chosen trade-off.
- Exact files changed and executed checks, including failures/skips.
- The comparison, data split, permissions, costs, and observed results of any experiment.
- Remaining risks, a keep/change/defer decision, and the next dependency.

Keep records concise. A design diagram, website, or slide deck is not required. Add only documentation that supports use, maintenance, or evaluation.

### Authorization boundary

Default: local code, offline fixtures, and safe tests. Public documentation may be read under normal permissions. Live inference, private-data transfer, paid jobs, external tracing, fine-tuning, deployment, and catalog writes need separate scoped approval. Implement their configuration without invoking them.

A run authorization should specify allowed endpoints, models, documents/split, maximum calls/tokens/concurrency, and a monetary cap when pricing is known. If cost is unavailable, keep it null and require usable hard resource limits. Report the limit and stop reason. A credential alone is not permission.

## 4. Interfaces to establish early

Use Pydantic/internal types where helpful; do not replace the inherited canonical schema with a guessed Pydantic equivalent. Inspect whether its schema representation can be validated directly before choosing a validator.

| Contract | Minimum information |
|---|---|
| `DocumentBundle` | source ID/hash, parser/version/options, immutable text/regions, supplied-source boundary, missing regions/truncation warnings |
| `EvidenceRef` | source hash, region ID, resolvable offsets/quotation, actual page reference only if available |
| `Profile` | canonical schema hashes, task definitions, identity/missingness rules, approved mappings, profile/resource versions |
| `RecordSnapshot` | canonical output, stable internal entity handles/local IDs, revision ID, original extraction warnings |
| `AssertionEvidence` | record revision, entity handle plus field/relation address, evidence refs, source expression, normalization history, support status |
| `Issue` | category, affected record/assertion, observations/evidence, status, attempted actions, scope; distinguish rule-detected from model-assessed |
| `Action` | allowlisted action kind, issue ID, bounded arguments, expected record revision, validated budget request |
| `Patch` | approved operation, expected revision, supporting evidence, before/after audit information; no arbitrary code execution |
| `ReviewCase/Response` | unique IDs, record revision, question/candidates/evidence, actual actor and timestamp, proposed/accepted/unresolved decision |
| `RunManifest` | code/config/prompt/model/source/profile/resource/evaluator hashes, dataset role, budget, usage, errors, cache/live flags |

Persist source text outside the graph state when practical. Evidence addresses must survive list reordering: use stable entity handles and versioned snapshots rather than relying solely on mutable array indices.

Keep the original extraction response under the approved retention policy. Model-reported confidence is an uncalibrated signal unless separately calibrated and evaluated.

## 5. Module roadmap

### M00 — Repository audit, protected baseline, and run isolation

**Question:** What works already, and how can new development avoid breaking or contaminating the benchmark?

**Reuse/read:** README, dependency/lock files, schemas, pipeline/components, test configuration, safe test entry points, evaluation configuration and data manifests. Read evaluation code for interface design, not runtime knowledge.

**Implement:**

- Record HEAD/working-tree state and relevant schema/evaluator hashes in `REPO_AUDIT.md`. Reconcile with the snapshot above; do not reset the checkout to it.
- Inventory modules, tests, config paths, actual data availability, and permissions. Identify current write paths and external service calls.
- Create `PROGRESS.md`, `DATA_POLICY.md`, and safe run-directory/config conventions. Add a minimal offline configuration and tests proving paths cannot point to protected release locations without explicit authorization.
- Establish runtime/evaluation boundaries and add an import/configuration test that runtime cannot load gold, `alignment_helper.yaml`, or paper-result outputs.
- Record the inherited environment and safe test commands. Install/sync within the project environment if permitted; record unavailable packages instead of silently changing versions.

**Verification:** run selected safe inherited tests and a synthetic no-network smoke check; confirm no tracked benchmark/result files changed. Do not run result-regeneration scripts at default paths.

**Experiment E00:** none involving new inference. Document the reproducible baseline and any independently replayable existing predictions without claiming live performance.

**Gate:** observed starting state, run isolation, test results, and exact commands are recorded. No bulk cleanup, package rename, or new agent features. This is the recommended first Codex assignment.

### M01 — Source preparation and evidence retrieval

**Question:** Can an assertion be linked back to real, stable source regions, and can targeted inspection retrieve relevant evidence?

**Inputs:** supplied text/Markdown; approved PDFs only when parser dependencies/data permit. **Outputs:** `DocumentBundle`, `EvidenceRef`, and retrieval candidates.

**Reuse:** inspect `components/parser.py`, inherited document types, and parsing utilities. Preserve the inherited parser path for historical comparisons.

**Implement:** deterministic region IDs based on source hash and representation; section/caption/table boundaries where available; paragraph retrieval using a simple lexical method; warnings for unavailable content. Provide both full-text selection and targeted region retrieval. Do not download related publications automatically or invent page mappings from plain text.

**Offline cases:** two same-composition samples; a measurement described only in a caption; duplicate captions; empty/broken parser output; Unicode units; a document edit causing hash invalidation; injected instructions inside source text. Treat source content only as data.

**Experiment E01 (after authorization/labels):** compare full-text selection, fixed chunks, and section-aware lexical retrieval on a small reviewed set of questions/needed passages. Report evidence recall at a token budget, lookup cost, and missing-evidence cases. Gold-evidence injection is a labeled diagnostic upper-bound condition, never a production policy.

**Gate:** every evidence link resolves; hashing/offset tests pass; retrieval is restricted to the approved source set; lack of page-image access is explicit. A runnable text-only module is acceptable when PDF inspection is blocked.

### M02 — Typed state, provider boundary, and fixed end-to-end extraction

**Question:** Can we produce the inherited canonical output with traceable evidence and reliable failure handling before adding adaptive decisions?

**Inputs:** `DocumentBundle`, fixed `Profile`, provider/config. **Outputs:** `RecordSnapshot`, evidence sidecar, run manifest and raw-response status.

**Reuse:** `components/extractor.py`, `components/llm.py`, `base/llm.py`, schema/prompt loaders, and existing record types. Keep a compatibility adapter to the inherited pipeline; do not import gold-dependent utilities.

**Implement:** a deterministic mock provider plus a real-provider adapter (not invoked by tests); JSON/schema parse status; usage accounting; stable entity handles and local-reference assembly; evidence-reference resolution; isolated export. Record all repairs/retries and truncation rather than silently returning success. No autonomous loop yet.

**Baseline distinction:** preserve an inherited-prompt compatibility run. A new evidence-producing prompt is a separate variant because evidence requirements can themselves affect extraction cost/quality.

**Tests:** compatible canonical output, invalid/non-object/truncated responses, missing credentials, timeout/429 without uncontrolled retries, missing usage as unknown, wrong evidence references, empty valid extraction versus failed extraction, safe reruns without duplicate exports.

**Experiment E02:** authorized development smoke test, initially one paper, then a small manifest-selected set. Compare inherited extraction and the evidence-linked fixed path with equal document access/model settings. Treat this as an engineering/prompt comparison, not an agentic result.

**Gate:** independent fixed run works, exported metadata remains schema-compatible, and all failure states are inspectable. Mock success does not imply a working live endpoint or scientifically correct evidence.

### M03 — Offline experiment harness and frozen evaluation

**Question:** Can module improvements be measured without changing the target or contaminating runtime?

**Inputs:** exported predictions and separately supplied gold. **Outputs:** per-document metrics, support counts, failure/cost summaries, immutable run manifests.

**Reuse:** inherited evaluation/alignment and scoring plan behind an isolated adapter. Inspect signature/config dependencies rather than guessing CLI arguments. Do not change their semantics as part of this module.

**Implement:** an explicit document-split manifest, safe output directory, deterministic replay, and a result table comparing run variants. Keep the scorer in a separate offline execution path. Record alignment-helper version and limitations for unseen papers; if an evaluator needs new adjudications, label the metric blocked rather than guess correspondence.

**Data policy:** ten calibration papers can inform development. The historical thirty-paper evaluation set can support historical comparison, but once its failures guide the new method it is no longer an untouched confirmatory set. Prepare a future held-out manifest before tuning. Keep every derivative of a publication in the same split; check preprints/versions and duplicates.

**Experiment E03:** replay a small approved subset of existing predictions into a temporary run directory and verify parity with the frozen evaluator where reference outputs exist. This is scorer reproducibility, not evidence that the new extractor improved.

**Gate:** a single new run can be scored without overwriting released results; runtime has no gold/helper access; failures and low-support fields remain visible. No newly invented schema-wide metric.

### M04 — PaNET terminology lookup and conservative mapping

**Question:** Does controlled technique lookup improve justified normalization without over-specific or unsupported tags?

**Inputs:** source mention/context, approved ontology snapshot. **Outputs:** ranked candidates and a mapping decision with evidence and unresolved/no-match states, stored outside the canonical output unless already permitted by the schema.

**Implement:** versioned local concept loading, labels/available synonyms/hierarchy, lexical candidate retrieval, optional LLM selection from retrieved IDs, source provenance and cache invalidation. Start with one or two technique families. Use authentic resources; record redistribution terms. Clearly synthetic namespaces are allowed only in tests.

**Tests:** known label/synonym; broader versus more-specific ambiguity; obsolete/unknown ID; technique mentioned in related work but not used in the current study; no appropriate concept; unavailable resource/version mismatch. A valid ID alone does not prove a correct assignment.

**Experiment E04:** no normalization versus lexical lookup versus lookup plus contextual selection, holding extraction fixed. Obtain expert-reviewed mention-to-concept labels; current extraction gold may not contain these labels. Report candidate recall, exact justified mapping accuracy, over-specific errors, unresolved fraction, and cost. Do not silently count a broad parent concept as equivalent to a specific label; report hierarchy-aware results separately if defined.

**Gate:** only retrieved, versioned authentic IDs can be accepted in real runs; uncertain mappings remain unresolved; natural and synthetic test performance remain separate. Mapping approval cannot be fabricated by Codex.

### M05 — NeXus definitions and a reviewed profile registry

**Question:** Do selected community definitions clarify existing field meanings rather than merely encourage fuller-looking output?

**Inputs:** inherited profile, NXDL snapshot, selected concepts/mappings. **Outputs:** relevant definitions/constraints and reviewed mapping entries.

**Implement:** a small NXDL loader or existing-library adapter; definition retrieval; mapping entries containing source definition, class/field context, units/type where stated, mapping scope, status and reviewer/version. Resolve necessary includes/inheritance or mark them unresolved; do not handwave them away. Base-class possibilities and application requirements are different. Class/field references are not asserted to be universal HDF5 paths.

Use exact/equivalent/partial/no-match distinctions where useful. Keep proposed mappings unreviewed. A publication-oriented profile is not a complete application-compliant NeXus file. Do not add required scientific values because a standard expects them.

**Tests:** preparation versus acquisition temperature; same quantity used for different roles; incompatible units; unavailable required experimental data; definition context/version changes; no suitable mapping. Preserve precision and raw expressions in deterministic conversions; conversion policy must be explicit.

**Experiment E05:** existing field instructions versus the same instructions plus selected verified definitions, with the extractor/evidence and PaNET configuration fixed. Measure interpretation errors, source support, harmful corrections and cost. New fields outside the inherited schema belong in a separately approved extension experiment, not the main comparison.

**Gate:** a small useful reviewed profile is enough. No automatic schema generation, broad ontology engineering, or full NeXus export is needed to continue.

### M06 — Record checks, issue diagnosis, and supported patches

**Question:** Can checks distinguish a malformed representation, unsupported assertion, wrong attachment, and genuine task ambiguity well enough to guide action?

**Inputs:** profile, record revision, source evidence, optional standards context. **Outputs:** typed issues and proposed local patches, not a single quality score.

**Implement three separate check layers:**

1. Deterministic structural/reference checks using the actual canonical profile. Report rather than silently discard unexpected content in the new path.
2. Evidence-sensitive checks of assertions, including the linked entities. Return supported/contradicted/insufficient evidence with references and a short explanation. Label them model judgments, not expert judgments.
3. Limited coverage signals, such as an unrepresented sample label or measurement mention. Explicitly distinguish a possible omission from a proved missing gold entity. Label coverage unsupported when it is not implemented.

Keep the checker separate from the patch executor. Patches require an expected revision, typed operations, and supporting evidence; rejected/stale patches leave the old state intact. Recheck affected relationships after updates. Entity merges/splits require explicit policy or human approval, not arbitrary JSON surgery.

**Tests:** preparation temperature wrongly used as measurement temperature; two same-composition specimens with different treatment; valid but swapped sample reference; dangling reference; correctly missing optional value; evidence lookup failure; conflicting issue diagnoses; harmful correction; record/evidence revision mismatch.

**Experiment E06:** structural checks alone, generic source-support checking, and identity/relationship-aware checking on frozen real outputs. Obtain independent labels for a small random sample plus an explicitly oversampled difficult subset; report each separately or use appropriate sampling weights. Include controlled reference swaps as diagnostics. Report per-category error detection, false alarms, unresolved cases and verifier cost. Audit apparently accepted cases too.

**Gate:** actionable issue reports and atomic patches work; no quality claim is based on the checker agreeing with itself. Document which causes can actually be distinguished with available evidence and which remain hypotheses.

### M07 — LangGraph and bounded intervention policies

**Question:** Does diagnosis change the next action in a useful way, rather than just add more calls?

**Reuse:** all prior modules. Keep the inherited pipeline intact. LangGraph coordinates state transitions; ordinary modules still run and test independently.

**Implement:**

- A fixed route and a transparent rule-based issue-aware policy first. Add a generic LLM policy behind the same action interface for comparison, not as an extra framework.
- Permitted actions: retrieve source evidence, consult an approved definition, propose a local update, request review, finish unresolved/finished. Every action has validated arguments and a budget.
- Total/per-issue limits, no-progress detection, unknown-cost handling, timeout/cancellation and persistent state. A review action creates a pending review case; do not treat it as approval. M08 supplies the full external review flow.
- Record actual inspected evidence, short action rationale, changes, failures, and calls. Use the installed supported checkpoint APIs and verify official documentation [O2–O4].

**Tests:** deterministic mock cases traverse every action; invalid tool request rejected; repeated retrieval without new evidence stops; action budget exhausted with unresolved issues retained; a standards-definition passage cannot be used as experimental evidence; retries do not duplicate records. No external tool permissions originate from a document.

**Experiment E07:** start competing controllers from the same frozen initial extraction and issue state, with identical tools and budgets. Compare fixed correction order, generic LLM reflection, and rule-based issue-aware routing. The proposed controller may later use an LLM to choose within the allowed action set; add it only as a separate variant. Evaluate final correctness offline, harmful patches, useful/ineffective actions, unresolved issues and total cost. Ground-truth best-action information is diagnostic only and never a runtime input.

Report this as a repair-policy study; measure full end-to-end runs separately because shared initial states/cached extraction do not represent real latency savings.

**Gate:** bounded graph works on synthetic cases; fixed and adaptive methods share components; pending review is explicit. No claim that a rule-based policy is learned, optimal, or a new method merely because it uses LangGraph.

### M08 — Persistent human review and explicit approval

**Question:** Can a user resolve a precise metadata question efficiently, and can the same run resume safely after a restart?

**Inputs:** pending review case and record revision. **Outputs:** approved/edited/unresolved response, updated revision, rechecked export.

**Implement:** CLI first, then a small UI only when authorized. Show affected entities, disputed assertion, passages, relevant definitions, candidate corrections, and the actual uncertainty. Avoid requiring review of the whole JSON for every issue.

Use a file-backed SQLite or other supported durable checkpointer for the single-user MVP. If an interrupted node restarts, repeated pre-interrupt code must be safe. Apply review edits with optimistic revision checks and an idempotency key. Validate and recheck edits before accepting them; preserve original and revised records. A reviewer can explicitly leave a case unresolved. Reopen/migrate incompatible state only through a documented operation.

**Tests:** process A pauses and exits; process B loads the same run and resumes; stale response rejected; duplicate response has no duplicate side effect; invalid edit rejected; unresolved response retained; crash during patch/export recoverable. Use simulated reviewer responses only for these engineering tests, marked as simulated.

**Experiment E08:** compare a confidence/default review queue with evidence-and-relationship issue prioritization under a fixed review budget. Measure actual review time, final residual error, reviewer corrections, and unresolved cases. Use matched or counterbalanced assignment to reduce memory/order effects. Actual expert identity/approval is never synthesized by Codex. A perfect-review simulation is a separate, explicitly optimistic analysis.

**Gate:** restart/resume and stale/duplicate safety pass. Real effort-savings claims remain blocked until real review observations exist.

### M09 — Controlled end-to-end study and ablations

**Question:** What actually improves quality, at what cost, and in which cases does adaptivity fail to help?

**Prerequisites:** stable module interfaces and fixed scorer. Finish E04–E08 only to the degree needed to justify the final comparison; do not run every possible factorial combination.

**Primary design:** use the comparison matrix in Section 7. Freeze selected profile, parser representation, extractor/model configuration, standards snapshots, controller prompts, review rules, split manifest and evaluator before the final run. Declare the primary outcome and any acceptable trade-off thresholds before examining held-out results.

**Implement:** reusable experiment configs, run authorization and dry-run estimate, immutable manifests, result collection, publication-level summaries, and a report generator that does not modify inherited paper outputs.

**Tests:** prediction/scorer separation; excluded gold/helper paths not loaded at runtime; all attempted documents accounted for; missing or failed runs visible; deterministic aggregation on fixtures; metrics reproduce from stored raw predictions and manifests.

**Experiment E09:** compare the narrow selected variants at several practical resource caps. Start with a small approved development pilot, then a frozen confirmatory set if data are available. Record repeats/seeds and provider nondeterminism. Do not assume hundreds of records give hundreds of independent documents.

Report entity recovery, conditional attributes and references with support, mapping correctness, source-support errors, harmful corrections, latency/calls/tokens, review effort and unresolved fraction. Include failed/truncated runs and denominator policies. Use paired publication-level uncertainty estimates where appropriate. A new end-to-end measure requires a predeclared counting policy, not an ad hoc sum of existing scores.

**Decision:** keep the simplest adequate variant. A stronger fixed workflow, no measurable standards benefit, or an inconclusive small-sample result is valid. Record limitations rather than manufacture a positive research narrative.

**Gate:** complete reproducibility record and honest claims. “Validated” requires the named independent evidence; otherwise label the study pilot or blocked.

### M10 — Usable pilot, packaging, and handover

**Question:** Can another person operate the service and inspect/recover its failures without the developer repairing a notebook?

**Implement only the agreed pilot scope:** a small API/review UI, containerized service, user documentation, safe storage, health/status endpoints, structured logs, and an approved export adapter. One deployment target is enough. Azure is an option, not a scientific requirement.

Keep gold, paper files, benchmark outputs, credentials and raw local experiments out of the runtime image using an explicit packaging allowlist or `.dockerignore`. Runtime service permissions should prevent access to evaluation data, not rely solely on prompts. Do not expose unauthenticated document upload, enable external tracing, or publish a catalog record without the agreed controls.

**Tests:** clean install/container build, mock end-to-end API path, state recovery, input-size/path checks, secret redaction, no unexpected egress, concurrent/stale update handling appropriate to declared single-user/multi-user scope. A single-user SQLite prototype must not be described as a production-scale service.

**Experiment E10:** user pilot on approved documents, observing usability, failure recovery and actual correction time. Small-user pilot evidence is not commercial validation or enterprise operating experience.

**Gate:** demonstrable workflow, exact setup/run/review/evaluation instructions, limitations, architecture and decision records, and a concise case study. Public deployment/cloud resources/costs require explicit approval.

### M11 — Optional specialized fine-tuning; not on the critical path

Only open this module after a measured bottleneck and explicit training/data/compute approval. Candidate modules are schema-conditioned extraction, relationship verification, or issue classification; do not train an all-purpose agent merely to add fine-tuning to the project.

Retrieve versioned NeXus/PaNET definitions rather than relying on model weights as the only standards source. Human corrections require review and a defined scope before becoming training labels. Do not automatically train from whatever users edit.

Compare the same base model with and without tuning, using identical retrieved definitions and inference budgets; include a stronger prompted reference if feasible. Split by document and test an unseen technique/profile when claiming transfer. Report quality, calibration/abstention behavior where relevant, latency/cost and regressions. Synthetic errors can supplement, not replace, natural held-out cases. Do not train on final-test corrections and then report them as held-out.

Keep the module only if it provides a defensible quality/efficiency gain. Otherwise retain prompting or a simpler classifier. No reinforcement learning, multi-agent training, or broad training-data campaign is required by this plan.

## 6. Staging and scope discipline

Recommended order:

```text
M00 -> M01 -> M02 -> M03 -> M04 -> M05 -> M06 -> M07 -> M08 -> M09 -> M10
                                                                  M11 optional
```

This is a development sequence, not a schedule guarantee. Existing implementations may already satisfy parts of a module; verify rather than rebuild. M04/M05 resource preparation can proceed independently after M02/M03, but keep user-authorized task scope explicit.

There are three useful deliverables before the full roadmap is complete:

- **After M02/M03:** fixed, evidence-linked extraction and isolated scoring.
- **After M07/M08:** an adaptive, resumable local demonstration with human review.
- **After M09/M10:** measured comparison and a usable pilot/handover.

For a January 2027 handover target, protect the latter deliverables by cutting optional fine-tuning, automatic schema design, broad multimodal parsing, full NeXus export and platform integrations before cutting tests, evidence integrity, evaluation isolation, or restart safety. This is a scope priority, not a commitment to finish every module by that date.

## 7. Final experiment matrix

Use these as named design alternatives, not a mandatory large benchmark grid. Module-level experiments determine which variants warrant the final study.

| Variant | Initial extraction/checking | Standards | Intervention policy | Purpose |
|---|---|---|---|---|
| H0 | inherited benchmark-compatible path | inherited task instructions only | inherited behavior | historical/compatibility reference; not necessarily identical evidence burden |
| A0 | new fixed evidence-linked path | off | fixed sequence | controlled new baseline |
| A1 | same as A0 | approved PaNET + NeXus | same fixed sequence | standards effect |
| A2 | same components as A1 | same resources | generic LLM tool selection/reflection | strong adaptive baseline |
| A3 | same components as A1 | same resources | explicit issue-aware rule controller | effect of diagnosis and targeted routing |
| A4 (optional) | same components as A1 | same resources | issue-aware model chooses among typed actions | whether a learned/model-directed choice adds value |

Use A2 versus A3/A4 for the main controller claim. Keep diagnosis/checker model access equal where possible; if diagnosis is the intervention, count its computation and state that difference explicitly. Keep system/tool access and permitted source set the same. The baseline must not be denied captions or standards tools available to the proposal.

Ablate PaNET and NeXus separately in their module studies when necessary; do not infer which resource helped from a combined arm alone. Profile approval is a separate human action, not a hidden improvement to the task definition.

Possible primary endpoint: dataset/sample recovery at a predeclared computational cap with precision and harmful-correction guardrails. Possible review endpoint: residual relationship errors after a specified expert-time budget. Choose one primary endpoint for a paper; retain the others as diagnostics. Exact thresholds require domain/stakeholder judgment before the confirmatory run.

Every comparison reports both quality and resources. Do not force simple methods to waste calls merely to match usage; compare under common caps and show actual consumption. Reused cached initial extractions are appropriate for a repair-policy experiment but do not demonstrate live end-to-end latency reductions.

## 8. Data, fixtures, and measurement protocol

### Data roles

| Role | Permitted use |
|---|---|
| Small synthetic fixtures | Engineering tests, predictable control-flow tests, known transformations. Never headline empirical extraction results. |
| Approved development papers | Prompt/rule development, standards-mapping proposals, diagnostic labels and small live pilots. |
| Frozen natural model outputs | Checker/controller diagnosis experiments; keep provenance and parent-publication split. |
| Controlled variants of verified records | Relationship-swap, ID/order invariance, omission/duplication and unit-role diagnostics. Keep all variants with the parent document. |
| Historical evaluation set | Historical comparison with disclosed tuning exposure. Do not relabel it untouched. |
| New sealed test set | Confirmatory comparison after freezing method and evaluator. Any intervention labels/test corrections remain outside training. |

A practical pilot can begin with approximately 5–10 development papers and several dozen reviewed assertions/relationships, if available and approved. These are suggested starting sizes, not sufficient sample-size guarantees. Select final sample size using document variability, annotation effort, and the intended claim.

Existing gold may lack PaNET concept labels, NeXus mapping judgments, source spans, issue causes, and intervention outcomes. Inventory these gaps. Codex may prepare annotation cases and provisional suggestions, but only genuine independent annotations/adjudications are labeled as such.

### Minimum fixture family

- Correct record, no repair needed.
- Same composition but two distinct samples.
- Valid local reference pointing to the wrong sample.
- Additional measurement stated only in a caption.
- Preparation temperature wrongly assigned to acquisition.
- A genuinely unreported optional quantity.
- Ambiguous technique requiring a broader PaNET concept or abstention.
- Syntactically invalid/truncated output.
- Missing/unavailable standards resource.
- Duplicate/stale human review.
- A source passage containing instruction-like text that must not control tools.

Do not make the runtime checker read fixture expected answers. A deterministic mock provider may encode scripted behaviors for a labeled test, but these are not learned capabilities or independent evaluation results.

### Run manifest template

Create the actual schema during M02/M03. This template describes content, not an already implemented command/API:

```yaml
run_id: <unique>
variant: <H0|A0|A1|A2|A3|A4>
code_sha: <actual>
working_tree_patch_hash: <null-or-hash>
config_hash: <actual>
data_manifest_hash: <actual>
data_role: <synthetic|development|historical|sealed_test>
source_hashes: {}
parser_version_and_options: {}
profile_hash: <actual>
resource_versions: {panet: null, nexus: null}
prompt_hashes: {}
provider: <mock-or-approved-endpoint-alias>
model_id: <actual>
model_parameters: {}
evaluator_sha_and_config_hashes: {} # offline only
budget:
  max_calls: <approved integer>
  max_input_tokens: <approved integer>
  max_output_tokens: <approved integer>
  max_actions_per_issue: <approved integer>
  max_cost: null
usage:
  calls: <observed>
  tokens: <observed-or-null>
  cost: null
  latency_ms: <observed-or-null>
cache_mode: <off|record|replay>
failures: []
review_mode: <none|simulated|human>
```

Record model routing, verification, embedding/retrieval inference and retries in total usage. Keep missing usage explicit. Cache replay must not be counted as freshly observed model behavior or used to imply repeat independence.

### Interpretation rules

- Predeclare aggregation, null/missing rules and failed-document handling.
- Report conditional scores with their eligible entity/reference support. Increased recovery can change the composition of that subset.
- Report partial failures, refusals, truncation, empty outputs, and unprocessed documents. Do not silently exclude hard cases.
- Separate natural versus synthetic results and before versus after human review.
- Analyze at the publication level; paired confidence intervals/bootstrap are suitable when assumptions and sample size permit. Describe small-sample uncertainty rather than declaring general superiority.
- Standards compliance is not source support; judge agreement is not expert truth; valid references are not correct attachments.
- Runtime methods never receive source-specific gold aliases or “number of gold datasets still missing.”

## 9. Progress and report templates

Create `docs/agent/PROGRESS.md` with one row per module. Initial status is `not_started` unless verified existing code satisfies it. Do not prefill success from this plan.

```text
Module | Implementation | Offline tests | Empirical validation | Integration | Evidence/report | Next step
M00    | ...            | ...           | ...                  | ...         | ...             | ...
```

An experiment protocol must record hypothesis, comparison, dataset/selection method, split, evaluator, budgets, authorization, primary outcome, failure handling, and stopping rules before the run.

An experiment report must record actual config/commit, commands, completed/failed cases, results, uncertainty, resource consumption, representative errors, limitations and a keep/change/defer recommendation. When blocked, state exactly what is missing and which conclusions cannot be drawn. Do not replace observed results with an illustrative table of invented numbers.

Use append-only run identifiers for artifacts. Concise aggregate reports may be committed after a privacy/license check; raw documents and responses are not automatically publishable.

## 10. Acceptance checklist for any module handoff

1. The requested behavior is implemented, or the precise independent portion/blocker is recorded.
2. Contracts and failure states are explicit; integration does not require duplicating the inherited pipeline.
3. Appropriate offline tests run, and pre-existing failures are distinguished from regressions.
4. Canonical schemas/released outputs remain unchanged unless separately authorized.
5. No runtime access to gold/evaluation helpers or unapproved documents/resources.
6. Evidence and resource versions resolve; synthetic/mock status is unambiguous.
7. Expensive/external operations remained within approved scope.
8. Exact commands, changed files, results, unresolved limitations and next module are documented.
9. A plain-language explanation shows how one example moves through the module and why the design was chosen.
10. The report does not confuse code completion with improved extraction quality.

## 11. Prompts for the user to give Codex

### First task: M00 only

```text
Read AGENTS.md and the M00 section of CODEX_MODULE_PLAN.md. The repository
already exists: inspect the current checkout and preserve all user work.
Implement M00 only. Establish the protected baseline, safe new run paths,
runtime/evaluation separation, and progress records; run the relevant safe
offline checks. Do not recreate the repo, delete copied benchmark assets,
rename the package, or run paid inference/result-regeneration scripts.
Explain the actual starting architecture and recommend the next module.
Stop after M00, with evidence of completed checks and any blockers.
```

### Subsequent module task

```text
Read AGENTS.md, the current PROGRESS.md, and the Mxx section of
CODEX_MODULE_PLAN.md. Implement Mxx only, reusing existing components.
Write the concise design/experiment note, implement the smallest complete
behavior, and run relevant offline tests. Do not stop at a proposal.
Live inference is not authorized by this prompt. Prepare the bounded run
configuration when needed; mark empirical validation blocked/not_run.
Update the module record and progress. Explain a worked example, the
trade-off, actual test results, and the next dependency. Stop after Mxx.
```

### Separately authorizing a small experiment

```text
Run experiment E## as specified in its reviewed protocol. Use only the
approved endpoint/model, document manifest, and limits recorded in that
protocol. Freeze the named code/config/evaluator before the run; write new
artifacts to a unique run directory. Do not change prompts or thresholds
mid-comparison, access sealed test material beyond the protocol, or exceed
the limits. Report failures and unknown usage honestly. Stop with the
reproducible report; do not proceed to another experiment automatically.
```

The last prompt becomes operative only after actual endpoint/model/data/budget values and permissions are recorded. A placeholders-only protocol is not approval.

## 12. Verified starting references and implementation documentation

This plan proposes a design; it does not claim to have tested framework APIs or project code. These primary sources were consulted on 7 October 2026. Recheck the installed version before implementation and record compatibility decisions.

### Repository observations

- [R1] Inspected snapshot: https://github.com/hzdue/meta-data-extraction-agent/tree/74b8101f9572527101ea5c73abaddaf66aaba1e5
- [R2] README (setup, release layout, scripts with replacing outputs): https://github.com/hzdue/meta-data-extraction-agent/blob/74b8101f9572527101ea5c73abaddaf66aaba1e5/README.md
- [R3] Dependencies: https://github.com/hzdue/meta-data-extraction-agent/blob/74b8101f9572527101ea5c73abaddaf66aaba1e5/pyproject.toml
- [R4] Pipeline: https://github.com/hzdue/meta-data-extraction-agent/blob/74b8101f9572527101ea5c73abaddaf66aaba1e5/src/meta_data_extraction/pipeline.py
- [R5] Extractor: https://github.com/hzdue/meta-data-extraction-agent/blob/74b8101f9572527101ea5c73abaddaf66aaba1e5/src/meta_data_extraction/components/extractor.py
- [R6] Existing validator: https://github.com/hzdue/meta-data-extraction-agent/blob/74b8101f9572527101ea5c73abaddaf66aaba1e5/src/meta_data_extraction/components/validator.py

### Official documentation

- [O1] Codex project guidance: https://developers.openai.com/codex/guides/agents-md (currently redirects to the official ChatGPT Learn documentation).
- [O2] LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- [O3] LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- [O4] LangGraph durable execution documentation entry point: https://docs.langchain.com/oss/python/langgraph/durable-execution (currently redirects to persistence; use the installed-version guidance).
- [O5] NeXus class definitions: https://manual.nexusformat.org/classes/index.html
- [O6] NXDL: https://manual.nexusformat.org/nxdl.html

Resource starting point for M04, to inspect and version during implementation rather than treat as an already approved local snapshot: https://github.com/pan-ontologies/PaNET

Do not cite the prior conversation or this design plan as evidence that a technique is state of the art or a contribution is novel. A publication claim requires a separate verified comparison with primary literature.
