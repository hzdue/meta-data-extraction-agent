# M10 evidence follow-up — 2026-10-09

Question: can the publication bridge expose useful, unreviewed PDF corrections
without weakening the edit gate, and can generic query-centred retrieval retain
more relevant text within the same source and character limits?

Scope: reuse SourceCorpus, SupportPlan, the existing support checker, scalar
proposal parser, workflow and durable review. No canonical schema, evaluator,
released asset, dependency, provider authorization or automatic apply gate changes.
This is a continuation of M10, not a new roadmap module.

Design: selectable `focused_v2` retrieval uses supplied-corpus term weights and
bounded windows around matches; default `lexical_v1` remains available unchanged.
Record the selection in a version-2 support plan and the frozen publication spec.
An otherwise complete, normally terminated PDF check with only unavailable images
may generate a proposal against resolved text. Keep its `needs_review` status and
warnings, and require human review. Truncated text, parser warnings, incomplete
responses, conflicts, and stale bindings remain ineligible. Automatic application
still requires complete successful before/after support reports. Preserve old run
artifacts; implementation pins require fresh runs, with no implicit state migration.

Baseline: 230 focused tests passed in the preceding self-assessment; the previous
full safe suite passed 2,504 tests. Lexical retrieval captured all frozen reference
phrases for 2/8 and 3/8 answerable questions at 1,500 and 6,000 characters on one
calibration paper. These internal labels and handwritten questions do not measure
extraction accuracy. All PDFs previously blocked proposal generation.

Required checks: unchanged legacy selection; generic long-tail, stopword, Unicode,
numeric, negation and budget/offset cases; PDF review-only proposal, durable replay,
unchanged canonical output and call bounds; negative source/response/binding cases;
existing safe agent/offline and affected inherited regression tests. A fresh frozen
development replay will compare both retrieval strategies with the old labels and
budgets, reporting harmed as well as improved cases. No tuning after its results.

Experiment status was `not_run` at design time. Execution results follow; no live
calls, network, natural gold, expert approval or human-time measurement occurred.

## Implemented behavior

`publication_checks` now distinguishes a completed check limited by unavailable
PDF images from a failed checker. Only a verified, normally completed report whose
sole diagnostic is `source_incomplete` and whose warnings are exactly unavailable
images/page mapping can yield text-based proposals. Its status stays `needs_review`.
Other source limits and failures remain blocking. The proposal prompt explicitly
states that missing images may invalidate the suggestion and human review is needed.

`publication_run` disables automatic patches before reserving correction-checker
calls when the source is incomplete. The existing scalar patch executor and its
successful-before/after-report requirement are unchanged. Example: an invented PDF
says acquisition was at 300 K while an authored extraction says 500 K. The runner
saves a proposed correction, retains the 500 K canonical record, and exposes the
proposal in durable review. Manual and automatic-requested cases each use three
mock calls, zero live calls, and produce no approval or correction-checker stages.

`retrieval.SourceCorpus.search(strategy="focused_v2")` uses a fixed generic
stopword list, supplied-region inverse document frequency, and one contiguous
window of at most 800 characters per matching region. It retains original source
hashes, region IDs and offsets. Scores are word overlap, not support judgments.
The legacy strategy remains the default. `PublicationSpec.retrieval_strategy`
selects the initial support-planning method and participates in run fingerprints;
`SupportPlan` version 2 records it. Existing extraction seed links retain priority;
subsequent controller retrieval actions still use the legacy method.

Trade-off: a window can recover matches near a region's end but can omit qualifiers
or separate measurements elsewhere in that region. Three windows can use at most
2,404 characters including separators, even when the cap is larger. Human review
remains necessary for PDF proposals, and lexical matching cannot establish that a
scalar belongs to the right sample or dataset. No entity-creation behavior was added.

## Development replay

Fresh artifacts: `runs/agent/m10-evidence-followup-v1/`. The protocol, unchanged
earlier references, executable, source and method were pinned before execution.
Protocol SHA-256: `61c3aaa0c1632b57c92c00c43f83157a80201f5950601fb122a6b23042d9baf6`.
Reference SHA-256: `9d157e1eff21621c93194a1f1cad53bd2eb45d76628c0647d1ecb3ab2f82385d`.
The retrieval implementer used generic synthetic cases without access to this
paper's questions, labels or prior selections. This is still development replay:
the parent had seen the prior results, and neither the paper nor labels are new.

| Strategy / character cap | Complete question phrases | Required components | Mean characters used (all 10 queries) |
|---|---:|---:|---:|
| lexical_v1 / 1,500 | 2/8 | 4/14 | 1,500.0 |
| focused_v2 / 1,500 | 4/8 | 7/14 | 1,449.4 |
| lexical_v1 / 6,000 | 3/8 | 6/14 | 3,020.0 |
| focused_v2 / 6,000 | 4/8 | 7/14 | 1,750.6 |

Ten questions on one calibration publication include two unanswerable questions
outside the denominator. Forty selections executed without a model. The outcome
is exact reference-phrase capture, not extraction accuracy, independent evidence
recall or a controller comparison. Production assertion-generated queries and seed
links were not benchmarked. No strategy or label was adjusted after these results.

Independent arithmetic and binding checks passed: all 18 frozen hashes, 40 metric
cells and 111 evidence instances (41 distinct links) verify. All 20 legacy
selections exactly match the preceding frozen run. At 1,500 characters, complete
questions q09 (ramp rates) and q10 (static time resolution) improve, with three
component gains including q07's acquisition rate. At 6,000, only q09 and its
maximum-rate component improve. Neither cap loses a primary component or complete
question. q08 gains partial deposition context but remains unanswerable/excluded.
This does not establish that all retrieved text is semantically sufficient or
that changing extraction outputs would be harmless.

## Verification and remaining gate

Added 65 synthetic checks: 34 retrieval, 17 publication strategy/version/replay,
and 14 PDF proposal checks. Focused commands passed 175 distinct tests, including
existing bridge, publication-run and retrieval regression cases. An independent
code review found no route from the relaxed proposal guard into automatic PDF
application. The strict executor, support checker, workflow tools, review logic
and provider adapter remain unchanged. Exact commands are in `../COMMANDS.md`.

Final full safe suite: **2,569 passed in 699.09s (11:39)**, exit 0, including all
2,504 previous checks and 65 additions. No lint/type command is configured in
`pyproject.toml`. Final inventory: 1,760 starting files, 1,764 final, seven intended
edits and four additions, no missing files. All 1,478 protected data/result files
and 35 prior pilot/assessment/quota artifacts remain byte-identical. Configuration,
schemas, AGENTS.md, environment and strict application guards are unchanged.
The ledger still records zero model attempts. `assessment.json`,
`retrieval-audit.json` and `final-audit.json` retain the local evidence.

Experimental status: `pilot` for the internal retrieval replay; natural extraction
gain remains `blocked` by unresolved ChatAI authentication/quota verification and
the absence of a frozen independently reviewed comparison. The ten-attempt rolling
hour limit remains intact. No new authentication attempt or inference was made.
Human browser usability was prepared for later and remains unrun. A model checking
its own output is not independent gold; these fixtures do not measure expert time.

Next recommended work stays in M10: resolve service access, prepare a fresh bounded
real-paper comparison using independently reviewed entity/attribute/reference
labels, then measure baseline extraction versus reviewed controller output and
report improvements, harms, unresolved records and review effort separately.
Include omitted gold entities; emitted-assertion checks alone cannot measure recall.
Do not restart the cancelled job, reuse old fingerprints, or begin M11.
