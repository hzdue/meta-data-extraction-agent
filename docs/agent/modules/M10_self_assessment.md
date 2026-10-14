# M10 internal offline performance assessment

Completed 2026-10-09 at the user's request to evaluate the system without the
unavailable ChatAI service. Status: **pilot**, restricted to an internal retrieval
diagnostic, current offline checks and saved synthetic experiment verification.
Natural end-to-end extraction/correction performance remains **not_run/blocked**.
No new module or runtime behavior was implemented.

## Assessment

The workflow's tested integrity and recovery behavior is well supported. There is
still no demonstrated natural-paper extraction benefit from the agentic layer.
The saved synthetic comparison shows no benefit of issue-aware routing over the
simpler fixed policy. A real-paper diagnostic exposes brittle passage selection,
and code inspection establishes that the current global PDF warning blocks
automatic correction proposals. These limits matter more than adding another
controller or a more elaborate agent architecture.

## Scope, baseline and method

Reuse M01's prepared source and SourceCorpus, the existing M09 study/report and
M10 integration artifacts, and existing offline regression tests. The baseline
safe full-suite result is 2,504 passes from the earlier integration; this task ran
230 focused tests and read-only artifact checks, not another full-suite run.
The retrieval comparison uses unchanged lexical top-three selection versus a
simple full-text prefix at two predeclared character caps, 1,500 and 6,000.
This isolates evidence selection rather than provider or controller behavior.

The source is the previously selected calibration paper 1812.03956, the shortest
calibration PDF by plain-text length. It has 11 pages, 22,929 prepared characters
and 78 source regions. The preparation remains partial, with
`page_mapping_unavailable` and `images_unavailable`. Its SHA256 is
`158984e801673f2ce019d9fd793684183bbc288e1b27856039aa37b6c9845bce`.

Before retrieval, an assistant inspected the original PDF, including rendered
pages, and recorded 22 source findings with 36 verified excerpts. A frozen list
of ten natural-language questions and source-only reference phrases defines the
diagnostic. Eight questions have sufficient reported information, with fourteen
required components. Two are unanswerable as posed from the supplied main paper:
absolute energies on the primary/additional qXRR beamlines, and the deposition
site plus complete specimen identity for the annealing multilayer. Their partial
context is retained but excluded from the answerable-question denominator.

These references are assistant-derived internal judgments, not independent
expert gold. No benchmark gold, annotations, historical natural predictions or
natural evaluator reports were read. Source-informed questions are not a random
or held-out set, and production queries derived from emitted assertions and seed
evidence links are not being benchmarked. No Codex-generated extraction was
substituted for qwen output or labelled as a live system result.

The frozen protocol, references, execution lock, full selections and results are
under `runs/agent/m10-self-assessment-v1/`. Source inspection artifacts are under
`runs/agent/m10-self-source-audit-v1/`. Protocol SHA256:
`2c99e37b440accec6e96b2d1fe2a292059a443a6ae6a7a75fb02cd9dbc8bb7ff`.
Reference SHA256:
`9d157e1eff21621c93194a1f1cad53bd2eb45d76628c0647d1ecb3ab2f82385d`.
References and executable were locked before retrieving; neither changed after
results. All four conditions use the same exact prepared source.

## Observed retrieval results

The primary criterion is strict: every required reference component must have an
accepted exact phrase wholly present in a returned passage. It measures known
reference-phrase capture, not whether an LLM could answer correctly.

| Selection | Character cap | Questions with all reference phrases | Reference components found | Mean characters returned |
|---|---:|---:|---:|---:|
| Lexical, top 3 | 1,500 | 2/8 | 4/14 | 1,500 |
| Lexical, top 3 | 6,000 | 3/8 | 6/14 | 3,020 |
| Prefix | 1,500 | 0/8 | 0/14 | 1,500 |
| Prefix | 6,000 | 1/8 | 3/14 | 6,000 |

All forty selections passed evidence resolution; an independent arithmetic/code
audit verified 331 returned reference instances against source hash, region and
offset. These include repeated passages and are not independent observations.
Every 1,500-character lexical selection clipped a passage; none at 6,000 did.
Increasing the cap helped one complete question but left ranking/top-three limits.
No confidence interval is justified for this one-publication diagnostic.

Worked example: for the annealing ramp-rate question, the 6,000-character search
found the repeat series down to 34.5 K/s but missed the illustrated experiment's
maximum 90.9 K/s. It returned only 3,806 characters because geometry and image
processing occupied the other passage slots. For the detector/distance question,
the larger-cap results likewise omitted Pilatus 300k and approximately 1.8 m.
More character allowance alone did not recover those specific reported facts.

The metric is deliberately limited. The exposure/framing question retrieves a
figure caption stating 9 Hz and a conclusion stating 100 ms time resolution
during annealing, but its 1,500-character score is 0/2 because the frozen phrases
are different. That is potentially useful alternate context, although resolution
is not automatically identical to integration time. Scores were not changed
after inspection. Therefore 2/8 must not be presented as 25% extraction accuracy
or general evidence recall. Phrase hits also do not prove correct attachments.

## Verified controller and reliability evidence

The M09 comparison is one invented publication with two supported gold sample
references, six policy/budget cells and fifteen original mock calls. Its primary
metric is sample-reference F1, not overall extraction quality:

| Policy | Cap 2: F1 / mock calls | Cap 5: F1 / mock calls |
|---|---:|---:|
| Fixed | 1.0 / 2 | 1.0 / 2 |
| Issue-aware | 1.0 / 2 | 1.0 / 2 |
| Generic model selector | 0.5 / 2 | 1.0 / 5 |

Fixed and issue-aware make the same two authored corrections. Generic leaves two
issues unresolved at the lower cap. This supports retaining fixed as the simplest
adequate baseline; it does not establish a natural-paper advantage for any policy.
The audit verifies thirteen completion manifests and exact original/reproduction
parity for 264 metric rows, 176 field comparisons, four primary comparisons and
24,629 Markdown bytes. It did not regenerate released benchmark results.

M10's saved integration retains five automatic-path and three manual-path mock
calls, pending review, safe replay and the two expected scalar edits. All 63
saved integration hash pins verified. Its checker selected **12 of 25 emitted
assertions**, leaving thirteen unselected; entity recovery and missingness are
explicitly unassessed. Across the saved-artifact audit, 152 files were unchanged.
Current focused verification passed **230 tests in 76.82 seconds**. The tests
include invented SDK responses and clocks, not live service measurements.

## Design limits confirmed in code

1. Every current PDF bundle requires the unavailable-images warning. Support
   checking consequently returns `needs_review`, while proposal eligibility
   requires `succeeded`. This globally blocks automatic proposals for PDF inputs,
   including assertions grounded entirely in available text. The warning is
   scientifically honest; weakening it merely to enable edits is not a solution.
2. Checking covers selected assertions already emitted. It does not search for
   absent samples/datasets or establish completeness of null fields. Successful
   checks cannot imply complete metadata recovery.
3. Automatic patches are limited to existing dataset descriptions, measurement
   techniques and individual sample references, at most four operations. The
   mechanism cannot add omitted entities or repair general sample/facility
   metadata. That narrow scope prevents unsafe identity edits but caps benefit.
4. Retrieval ranks query-word overlap and clips region prefixes. It has no
   semantic matching, and ranking does not prove a correct experimental relation.
   The strict source-only scope protects the gold boundary but needs more useful
   passage selection and explicit coverage measurements.

These are code-inspected constraints, distinct from the measured retrieval
results. No runtime fix, new prompt, changed evaluator, or relaxed warning was
introduced in this evaluation. Prior cancelled live pilot artifacts remain intact.

## Remaining limits and next work

This turn made zero network/API requests and accessed no credential or natural
gold. Provider invocation counts in saved reports are historical; fixture calls
inside tests are synthetic. Real model latency, token usage/cost, extraction
precision/recall, harmful corrections and human review effort remain unmeasured.
Internal assistant judgments and arithmetic cross-checks are not expert validation.

Recommended next work stays within M10: resolve PDF evidence coverage at the
assertion/region level, preserve unresolved visual evidence for review, and add
explicit omitted-entity evaluation. Test any resulting change with a new frozen
protocol rather than tuning this diagnostic in place. Keep the fixed controller
as baseline; after authentication and independent reference review, run the
bounded extraction-alone versus controller comparison. M11 is not justified.

Exact commands, setup limitations and preservation checks are in
[COMMANDS.md](../COMMANDS.md). No claim of general extraction improvement is made.
