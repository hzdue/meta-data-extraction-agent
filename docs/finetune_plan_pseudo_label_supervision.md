# Pseudo-label supervision fine-tuning plan

## Status

**Length-filtered pilot is ready for Stage A; training has not started.** The
deterministic 508/57/30 paper split is frozen, and all 600 source PDFs have been
parsed and cached. Four pseudo-train teacher labels failed the current schema and
were excluded without repair, leaving 504 eligible train and 57 validation D
records. The approved resource filter retains 442 complete records at or below
32,660 tokens and excludes 62 longer records without truncation. Both worst-case
training probes and the complete validation loss/generation probes pass on the
local RTX 5070 Ti. No pseudo-label model training has started. This plan does not
change the completed 40-paper experiment or its model-selection decision.

## Objective

Test whether broad zero-shot supervision from a 31B teacher improves a 4B
scientific metadata extractor beyond fine-tuning on eight human-annotated
papers.

The primary candidate is a two-stage recipe:

1. train one LoRA adapter on complete-JSON pseudo-labels;
2. continue the same adapter on the existing human-gold D examples, with a new
   optimizer.

This is a teacher-student distillation experiment. The 595 predictions are
pseudo-labels, not human ground truth.

## Verified starting point

- Source corpus: 600 photon and neutron papers, with 595 predictions and 5
  missing predictions.
- Teacher: `google/gemma-4-31B-it`, revision
  `842da3794eaa0b77d5f08bae87a17459d91ff475`, zero-shot, temperature 0.
- The supplied `generic.json` and `photon_and_neutron.json` are byte-identical
  to the current repository schemas.
- None of the 600 papers overlaps with the existing 40-paper benchmark.
- All 595 available predictions parse as JSON and pass a basic structural audit:
  the expected top-level collections are present, IDs are unique within their
  collections, both dataset reference fields are present, and no references
  dangle.
- The five missing predictions must be excluded, never converted into empty
  targets.

These checks establish format compatibility, not scientific correctness.

## Fixed experimental design

### Tasks

Use only complete-output **D** supervision in the initial pseudo-label study:

```text
paper text + current extraction prompt + current schemas
    -> complete teacher JSON
```

Do not initially derive E or R examples from the teacher predictions. The
completed experiments showed that complete-output exposure is important, and
projecting noisy labels into auxiliary tasks would add another variable.

### Conditions

| Condition | Pseudo-label exposure | Human-gold exposure | Purpose |
|---|---:|---:|---|
| `base` | 0 | 0 | Unadapted reference |
| `gold_d24` | 0 | D24 | Existing gold-only control |
| `pseudo_length_filtered` | One fixed pass over 442 papers | 0 | Measure teacher imitation within the validated memory envelope |
| `pseudo_length_filtered_then_gold_d24` | The same fixed pass | D24 | Primary candidate |

`gold_d24` and `pseudo_length_filtered_then_gold_d24` must receive the same final
gold schedule, learning rate and six optimizer updates. Their only intended
difference is the pseudo-label stage.

### Seeds

- Pilot: seed 42.
- Full exploratory comparison: seeds 42, 123 and 456, only after the pilot
  passes data, memory and generation checks.

### Corpus roles

The preferred paper-quality split is:

| Split | Target size if all records pass | Purpose |
|---|---:|---|
| Pseudo training | 508 | Stage-A adapter optimization |
| Pseudo validation | 57 | Imitation loss and format diagnostics |
| Future independent test | 30 | Human annotation; never use pseudo-labels |

Select the 30 future-test papers before training, using only manifest metadata.
Stratify all splits by subject group and paper-length bin. Keep the future-test
paper IDs hidden from training preparation and ignore their existing predictions.

After formal validation, the approved hardware filter applies only to pseudo
training and retains every complete sequence at or below 32,660 tokens. Keep the
57-paper pseudo-validation set unchanged. The 442-example training pass uses 110
four-example updates and one final two-example update; never move a reserved
future-test or validation paper to fill that accumulation window.

If no new human annotation will be produced, use a 536/59 pseudo train/validation
split and evaluate on the existing thirty papers only as reused development
data. That version cannot support a new independent-test claim.

## Proposed repository layout

Use the existing repository structure; do not create a new top-level folder.

```text
configs/
  finetune_pseudo_prepare.yaml
  finetune_pseudo_only.yaml
  finetune_pseudo_then_gold.yaml
  finetune_pseudo_evaluation.yaml

data/
  share_600_papers/                     # unchanged source package
  finetune/
    pseudo_label_supervision/
      splits.json
      prepared/
        train_pseudo_d.jsonl
        validation_pseudo_d.jsonl
        future_test_manifest.jsonl
        preparation_manifest.json
        length_report.json
        length_filtered_training_manifest.json
        quality_report.md
        text/

outputs/
  finetune/
    pseudo_label_supervision/
      pseudo_length_filtered_seed<seed>/
      pseudo_length_filtered_then_gold_d24_seed<seed>/
      evaluation/
```

All generated artifacts must retain paper, prediction, schema, prompt, tokenizer
and code hashes.

## Step 1 — Source-freezing layer skipped

**Decision:** Skipped by user direction on 2026-11-17. The preparation workflow
uses the colleague-provided `manifest.jsonl` directly and records its hash in the
split manifest, but does not add a second source inventory or re-fingerprint every
source artifact before splitting.

The split step still checks the expected 600/595/5 counts and zero overlap with
the existing benchmark. The source package remains unmodified.

## Step 2 — Freeze the pseudo and future-test splits

**Status (2026-11-17): Complete.** The frozen manifest contains 508 pseudo-train,
57 pseudo-validation and 30 future-test papers, plus five papers excluded from
supervision because their teacher predictions are missing. It has zero overlap
with the existing 40-paper benchmark.

**Purpose:** Prevent leakage and post-hoc selection.

**What to do:**

- Select the future-test papers using only paper ID, subject group, date and page
  count.
- Balance subject groups and paper-length bins across the remaining pseudo train
  and validation sets.
- Use a fixed declared split seed.
- Save paper IDs, stratification fields, selection rules and a manifest hash in
  `data/finetune/pseudo_label_supervision/splits.json`.
- Reject any later attempt to prepare a future-test paper as training data.

**Done when:** The split is deterministic, disjoint, hashed and reviewed before
any training metric is produced.

## Step 3 — Parse and cache the papers

**Status (2026-11-17): Complete.** All 600 PDFs have complete Markdown and
metadata cache pairs. The final audit found one content-producing parser
identity, no truncation and no prediction-file reads. Fifteen papers have pages
flagged by the heuristic word-presence check for later review; these are flags,
not automatic exclusions.

**Purpose:** Use the same document representation as the existing extractor.

**What to do:**

- Parse every eligible PDF with the existing production parser.
- Cache the extracted Markdown and parser metadata under the new prepared-data
  directory.
- Record parser identity, timeout/failure status, text length and source PDF hash.
- Do not copy text from an undocumented teacher preprocessing pipeline.
- Exclude parser failures explicitly; do not create empty documents.

**Done when:** Each eligible pseudo example has reproducible cached text or a
recorded exclusion reason.

## Step 4 — Validate labels and create D records

**Status (2026-11-17): Complete for the provisional pilot.** Of 565 candidate
supervised predictions, 561 passed complete nested shape, type, allowed-value,
cardinality, ID and reference validation. Four pseudo-train labels contained a
funding entry with a null required grant ID and were excluded without changing
their values. The prepared sets contain 504 train and 57 validation D records.
A 28-paper stratified Codex-assisted review found no additional exclusion that
was clearly warranted; quality flags remain attached to the affected records.
Independent domain-expert review is still recommended before a paper-quality
claim.

**Purpose:** Turn teacher predictions into auditable supervised examples without
silently repairing them.

**What to do:**

- Validate every prediction against both current schemas.
- Check required collections, field types, ID uniqueness and reference integrity.
- Flag suspiciously empty or extreme entity inventories for review.
- Manually inspect a stratified sample of 20–30 train/validation predictions,
  covering all subject groups and length bins.
- Render the existing D prompt with the current schema and cached paper text.
- Use the teacher JSON as the assistant target. Canonicalize JSON whitespace only;
  do not alter field values.
- Store provenance and token counts in every JSONL record.

**Done when:** All prepared targets pass formal validation, reviewed quality
issues are documented, and every record can be traced to its source prediction.

## Step 5 — Audit context length and hardware feasibility

**Status (2026-11-18): Complete for the approved length-filtered pilot.** The
read-only pseudo-data loader verifies the frozen split, exclusions,
artifact and per-record provenance, future-test isolation, prompt reconstruction,
teacher-target schema validity and the pinned tokenizer identity. It first
reconstructs all 504 eligible answer-only train rows, then verifies and applies
the separately frozen 32,660-token manifest to retain 442. All 57 validation rows
remain unchanged, with EOS supervision and no truncation.

The shared validator now supports the full 57-paper pseudo-validation set with an
equal-paper mean, while preserving the existing two-paper gold diagnostics. The
`fixed_endpoint` policy makes intermediate checkpoint loss and generation results
diagnostic only and requires the declared final update for a completed run.
Step-zero validation also records whether it disables the adapter (fresh stage A)
or evaluates an already-loaded adapter (the later stage-B boundary).

The isolated probe runner freezes the four required shape cases and uses the
production trainer without changing NF4, BF16 compute, activation offload,
MLP chunking, gradient checkpointing, batch size, accumulation or context. Plan
the cases with:

```bash
uv run --group finetune-cuda python scripts/smoke_pseudo_finetune.py \
  --case plan --offline
```

Each GPU case must use its own persistent `--output-dir`.

The original 504-record population did not fit this GPU: the 63,524-token
longest-prompt and 62,306-token longest-target backward passes OOMed even with
`PYTORCH_ALLOC_CONF=expandable_segments:True`. The approved derived manifest
therefore excludes 62 complete records above 32,660 tokens (12.30% of records,
19.30% of input tokens and 14.46% of supervised tokens) without changing or
truncating any retained example. It records all included/excluded IDs, lengths
and the source JSONL hash.

On the NVIDIA GeForce RTX 5070 Ti (15.54 GiB usable, native BF16), the filtered
32,530-token longest-prompt case passed at 12.95 GB peak reserved VRAM and 100.5
seconds of measured CUDA time. The distinct 32,028-token longest-target case
passed at 12.81 GB and 96.3 seconds. Both completed a real four-example
accumulation window, produced finite loss and 248 nonzero LoRA gradients, changed
the adapter, and left all frozen weights unchanged.

The unchanged 57-paper validation loss probe passed at 10.07 GB peak reserved
VRAM in 274 seconds, with step-zero equal-paper loss 0.0451447. Full-budget
generation passed all 57 papers at 11.01 GB in 5,161 seconds: every output ended
with EOS, none hit the 16,384-token cap, 56/57 were valid JSON and 44/57 passed
the full schema. These format results are baseline diagnostics, not a selection
gate or human-quality result. Reports and MLflow records are preserved under
`outputs/finetune/pseudo_label_supervision/probes/`.

Observed token throughput plus five planned validation rounds gives a rough
Stage-A wall-time estimate of 10–11 hours on this machine; generation dominates.

The source length audit used
Qwen/Qwen3.5-4B tokenizer revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, the complete user-plus-assistant
chat sequences for all 508 pseudo-train and 57 pseudo-validation papers fit the
65,536-token project limit without truncation. The maximum is 63,524 tokens;
68 examples exceed 32,768 tokens and three use at least 90% of the project
limit. Future-test predictions were not read. Exact per-paper counts are stored
in `data/finetune/pseudo_label_supervision/prepared/length_report.json`.

**Purpose:** Avoid truncation and confirm that the training schedule fits the
available GPU.

**What to do:**

- Tokenize with the pinned Qwen tokenizer and disabled thinking.
- Require the full prompt, full target and EOS to fit within 65,536 tokens.
- Do not silently truncate paper text or target JSON.
- Exclude over-limit training records with documented reasons. Keep their counts
  and subject/length distribution in the report.
- Run forward/backward memory probes on the shortest, median,
  longest-fitting-prompt and longest-target examples.
- Use the existing NF4, BF16 compute, activation-offload, MLP-chunking and
  gradient-checkpointing policy.
- Run a small timing sample before estimating total server time.

**Done when:** Every included record fits the declared context and the worst-case
probe completes on one native-BF16 CUDA GPU without changing the shared training
configuration.

## Step 6 — Train the pseudo-only adapter

**Preparation status (2026-11-18): Ready to launch; training not started.** The
reviewed Stage-A contract consumes 442 unique D records once in 111 optimizer
updates (110 four-example windows plus a final two-example window), uses fixed
endpoint selection, and records diagnostics at updates 0, 32, 64, 96 and 111.
Its CPU dry run and all required RTX 5070 Ti probes pass.

**Purpose:** Measure what Qwen learns from broad teacher supervision alone.

**What to do:**

- Start a fresh LoRA adapter from the pinned Qwen3.5-4B base revision.
- Reuse the established QLoRA configuration: rank 16, alpha 32, dropout 0.05,
  learning rate `1e-4`, microbatch 1, gradient accumulation 4, constant schedule,
  no packing and answer-only loss.
- Train for one fixed pass over the 442 retained pseudo-training records. This is
  111 optimizer updates, with the last update consuming the final two records.
- Save diagnostic checkpoints near 25%, 50%, 75% and 100% exposure.
- Measure pseudo-validation target-token loss and generated structural validity
  at step zero and each checkpoint.
- Use the fixed one-pass endpoint for the primary experiment. Intermediate
  checkpoints describe the learning curve and are not selected using the reused
  thirty-paper evaluation.

**Done when:** The endpoint reloads onto a fresh quantized base, exposure counts
match the frozen split, and all checkpoints contain complete provenance and
optimizer state.

## Step 7 — Calibrate on human-gold D24

**Preparation status (2026-11-18): Stage-boundary runner complete; training not
started.** The runner requires the completed, hash-verified Stage-A update-111
adapter, reloads only its LoRA weights as trainable, and creates a new Stage-B
trainer at optimizer update zero. Optimizer, scheduler and exposure state are not
restored from Stage A. The GoldD24 seed-42 schedule hash exactly matches the
existing gold-only control, and the two conditions retain identical learning-rate,
precision and six-update schedules.

**Purpose:** Correct teacher-specific errors while preserving broad domain
adaptation.

**What to do:**

- Reload the pseudo-only endpoint as a trainable adapter on the same frozen base.
- Reset the optimizer and scheduler at the stage boundary; retain the learned
  LoRA weights.
- Train on the eight existing human-gold D records, each presented three times:
  D24, six optimizer updates.
- Use the same gold schedule, learning rate and precision policy as the
  `gold_d24` control.
- Save the pseudo-only boundary identity and the final calibrated checkpoint.
- Validate on the two existing human validation papers for diagnostics, but use
  the fixed D24 endpoint rather than post-hoc step selection.

**Done when:** The final endpoint proves exact pseudo-stage provenance, D24 gold
exposure, a fresh stage-B optimizer and successful adapter reload.

## Step 8 — Run the exploratory comparison

**Purpose:** Determine whether pseudo-label pretraining adds value beyond D24.

**What to do:**

- Evaluate `base`, `gold_d24`, `pseudo_length_filtered` and
  `pseudo_length_filtered_then_gold_d24` with the ordinary one-call D prompt and
  identical decoding settings.
- Report full-schema validity, generic metadata, facilities/samples/datasets F1,
  both reference F1 scores, JSON validity, failures, latency and peak VRAM.
- Report results by subject group and paper-length bin where sample sizes permit.
- Use paired per-paper differences against `gold_d24` and show every seed.
- Treat the existing thirty-paper corpus as reused development data. It may
  shortlist the recipe but cannot confirm a new winner.

**Done when:** All declared models, papers and failures appear in a reproducible
comparison with frozen checkpoint and evaluator hashes.

## Step 9 — Promotion rule and multi-seed confirmation

**Purpose:** Avoid promoting a method from a single favorable run.

After the seed-42 feasibility run, proceed to all three seeds only if
`pseudo_length_filtered_then_gold_d24`:

- preserves required reference fields and full-schema generation;
- stays within 0.03 of `gold_d24` on every structural F1 field;
- improves at least two of the five structural fields by at least 0.02; and
- has no new systematic generation or schema failure.

For the three-seed result, require the main gains to have the same direction in
at least two seeds, improve more papers than they worsen, and not be dominated by
one paper. These are exploratory shortlist rules, not independent-test evidence.

**Done when:** The method is either stopped with a recorded reason or shortlisted
without changing thresholds after viewing results.

## Step 10 — Independent human evaluation

**Purpose:** Make a defensible final claim.

**What to do:**

- Human-annotate the 30 papers reserved before training.
- Apply the same schema, annotation guidance and quality checks as the existing
  benchmark.
- Freeze the final adapter choice, prompts, evaluator and analysis before opening
  the new gold labels.
- Evaluate `gold_d24` and at most one shortlisted pseudo-label recipe once.
- Report paired uncertainty and all failures; do not revise the model afterward.

**Done when:** A previously untouched, human-annotated test set has been evaluated
once with the complete analysis frozen.

## Interpretation rules

- Improvement from `pseudo_length_filtered` measures imitation and broad domain exposure;
  it does not validate the teacher labels.
- Improvement only after gold calibration supports pseudo-label pretraining plus
  human correction as a combined recipe.
- Strong pseudo-validation loss with weak human metrics indicates successful
  teacher imitation but poor label quality or teacher-student bias transfer.
- A pseudo-only advantage that disappears after calibration may indicate
  catastrophic forgetting or an overly aggressive gold stage; investigate only
  after the primary fixed recipe is reported.
- Do not claim a 595-paper gold training set, because the targets are generated.
- Do not use the five missing predictions as negative or empty examples.
- Do not introduce E/R mixtures, loss weighting, curriculum variants or
  checkpoint sweeps until the primary D-only recipe is complete.

## Primary deliverables

- Frozen pseudo/future-test split manifest.
- Preparation manifest, length audit and quality report.
- Frozen length-filter manifest with included/excluded paper IDs and source hash.
- Pseudo-only and pseudo-then-gold configs and run contracts.
- Per-seed exposure, token, runtime and memory summaries.
- Reproducible exploratory evaluation on reused development data.
- Final independent-test report if new human annotations are produced.

## Final decision

The default experiment is **one fixed pass of D-only pseudo-label supervision,
followed by a reset-optimizer GoldD24 calibration stage**. The existing D24 model
is the primary control. Until a new independent human test is completed, any
advantage from the 600-paper corpus remains exploratory.
