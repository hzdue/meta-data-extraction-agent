# Qwen3.5-4B Fine-Tuning Pilot

## Objective and fixed design

Compare JSON-only and structure-aware fine-tuning for PaN metadata extraction.
Prepare data and scripts locally; run training on the institute's Linux server.

| Split | Papers | Use |
|---|---:|---|
| Training | 8 calibration papers | Fine-tuning |
| Validation | 2 calibration papers | Checkpoint selection |
| Test | 30 evaluation papers | Final comparison only |

Evaluate the selected eight-paper checkpoints directly. No refitting on all ten
papers is needed for this pilot.

| ID | Method | Training tasks |
|---|---|---|
| M0 | Base model | None |
| M1 | JSON-only | Complete JSON |
| M2 | Structure-aware | Complete JSON + entities with attributes + references |

All methods use the same `Qwen/Qwen3.5-4B` revision, parsed text, full extraction
schema, tokenizer, and **one-call full-JSON inference**, with thinking disabled.
Both adapters start independently from the original checkpoint. Gold entities
are never supplied during validation or test inference.

## Proposed repository structure

```text
configs/
  finetune_json_only.yaml
  finetune_structure_aware.yaml
data/finetune/
  splits.json
  prepared/
    text/
    train_json_only.jsonl
    train_structure_aware.jsonl
    validation.jsonl
    length_report.json
scripts/
  prepare_finetune_data.py
  run_finetune.py
  evaluate_finetune.py
src/meta_data_extraction/finetune/ # Reusable preparation/training helpers as needed
tests/test_finetune_*.py          # Data, masking, and integration checks
outputs/finetune/<run_id>/        # Adapters, logs, predictions, and metrics
docs/finetune_plan.md             # Workflow and exact execution commands
pyproject.toml                   # Existing uv project; add a finetune dependency group
uv.lock                          # Shared dependency lockfile
```

Extend the existing repository and uv project. Use `finetune_*.yaml` names so new
configs are excluded from the benchmark runner's `pan_*.yaml` discovery. Reuse
source PDFs, gold records, parser, prompt builder, and evaluator; preserve existing
benchmark files. Keep run outputs on persistent server storage at the configured
output path. Implement one numbered step at a time.

## Step 1 — Audit data and freeze the split (local)

**Purpose:** Establish reproducible training, validation, and test membership.

**What to do:**

- Read `data/corpus.json`; verify PDFs in `data/raw/<id>/paper.pdf` and gold
  records in `data/gold/<id>.json`. Check duplicate papers/versions across splits.
- Select and freeze eight training and two validation papers from the calibration
  pool, considering technique and length. Keep all thirty evaluation papers held
  out. Reuse `data/finetune/splits.json` on subsequent runs rather than regenerating
  the split.
- Record source checksums and schema versions. Inspect
  `docs/annotation_guidelines.md` and `configs/evaluation_metrics.yaml`.

**Done when:** `data/finetune/splits.json` contains disjoint 8/2/30 sets, all inputs
exist, and any annotation/schema inconsistencies are documented and resolved.

## Step 2 — Prepare targets and audit lengths (local)

**Purpose:** Create both training datasets from the same annotations.

**What to do:**

- Implement `scripts/prepare_finetune_data.py` to write derived artifacts under
  `data/finetune/prepared/`. Persist Markdown from the existing `PdfParser`, with
  parser version and text hashes. Use identical text and canonical JSON
  serialization in both methods.
- Generate these three task types:

| Task | Input | Target |
|---|---|---|
| D: complete JSON | Paper + schema + instructions | Entire gold JSON |
| E: entities | Paper + schema + task instructions | Facilities, samples, and datasets with IDs and all non-reference attributes |
| R: references | Paper + schema + task instructions + gold E inventory | Dataset ID, reference field, and ordered target IDs |

- E excludes `sample_refs` and `facility_refs`; R predicts those fields,
  including empty lists. Its inventory input therefore contains no reference
  answers. Bibliographic and document-overview fields are supervised through D.
- Preserve local IDs, values, nulls, and ordering. Test deterministic projection,
  valid reference endpoints, and reconstruction of the three domain collections
  from E + R. New span annotations or reasoning traces are unnecessary.
- Export prompt/completion JSONL with paper ID, task, and provenance: **8 D records**
  for JSON-only; **24 D/E/R records** for structure-aware; **2 D records** for
  validation. Every derived example stays in its source paper's split.
- Count complete rendered sequences using the pinned **4B tokenizer and chat
  template**, including R's inventory, targets, and special tokens. Earlier length
  estimates are provisional. Treat 65,536 as a candidate cap; confirm actual
  lengths and GPU feasibility. Reject overflow before trainer truncation.
- Check inference input length plus reserved output tokens, including test inputs,
  without consulting test gold. Apply a documented shared context policy if needed.

**Done when:** The three JSONL files have counts 8/24/2, target checks pass, and
`data/finetune/prepared/length_report.json` confirms that full examples fit the
configured limits.

## Step 3 — Build one configurable trainer (local)

**Purpose:** Support both methods with the same implementation.

**What to do:**

- Add a `finetune` dependency group to the existing `pyproject.toml` for PyTorch,
  Transformers, PEFT, TRL, and MLflow, and resolve `uv.lock`. Verify Qwen3.5 and
  Linux/CUDA compatibility while retaining the project's Python requirement.
  Use `uv sync --locked --group finetune` and run scripts through
  `uv run --locked --group finetune python scripts/<script>.py`.
- Implement `scripts/run_finetune.py` using the two `configs/finetune_*.yaml` files,
  with CPU dry runs, checkpoint save/resume, and adapter reload. Start with BF16
  LoRA, rank 16, alpha 32, dropout 0.05, learning rate `1e-4`, microbatch 1,
  accumulation 4, gradient
  checkpointing, and packing disabled. Verify targeted language-model modules;
  keep vision frozen and unused.
- Keep optimizer, schedule, effective batch size, seed 42, finite `max_steps`,
  and evaluation interval identical across methods. Set the step cap and interval
  after the hardware check, before comparative runs.
- Use a fixed **D:E:R = 1:1:1 example ratio** for M2. M1 uses D only. Equal updates
  do not imply equal supervised tokens, compute, or full-JSON exposure; log each.
- Compute loss on target tokens and the end marker only; mask all inputs and
  padding. Test actual labels after tokenization and collation.
  [TRL completion-only training](https://huggingface.co/docs/trl/sft_trainer#train-on-completion-only)

**Done when:** Both configurations use the same trainer, local tests pass, and
uv setup and launch commands are documented in this plan.

## Step 4 — Implement validation and checkpoint selection (local)

**Purpose:** Select training duration without using test performance.

**What to do:**

- Evaluate only the two complete-JSON validation examples for both methods.
  Compute mean target-token loss per paper, then average the two paper scores.
- Evaluate at step zero and at common step intervals. Save checkpoints, select
  each method's minimum validation loss, and break ties in favor of the earlier
  checkpoint. If step zero wins, record the unadapted selection explicitly.
- Record generated entity/reference F1 as diagnostics; do not use these to
  override the declared loss-based selection rule.
- Run to the bounded step cap initially; defer automatic early stopping.
  Preserve the selected checkpoint and its exact configuration.

**Done when:** Selection is reproducible, includes the untuned baseline, and
requires no test scores.

## Step 5 — Integrate MLflow (local setup, server storage)

**Purpose:** Keep configurations, selected checkpoints, and results traceable.

**What to do:**

- Use one experiment, `pan-qwen35-4b-finetuning`, with initial run names
  `base`, `json_only_seed42`, and `structure_aware_seed42`.
- Configure an explicit persistent tracking backend and artifact location, or use
  the institute's existing service. Keep metadata and artifacts outside temporary
  GPU-job directories. [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- Log the following for each run:

| Category | Record |
|---|---|
| Configuration | Method, seed, resolved YAML, LoRA/optimizer settings, task ratio, context and decoding limits |
| Provenance | Model/tokenizer/parser revisions, code version, uv.lock hash, runtime versions, split and data hashes |
| Training | Loss curves, learning rate, updates, task counts, supervised and total tokens, GPU time, peak VRAM |
| Selection | Best step/loss, token exposure at that step, adapter artifact or persistent location and checksum |
| Evaluation | Aggregate metrics, per-paper results, raw predictions, parse failures, evaluator version |

- Have `scripts/evaluate_finetune.py` attach results to the corresponding MLflow
  run ID. Distinguish selected-checkpoint statistics from final-step totals.
  Preserve adapters without duplicating the base model or uploading source PDFs.

**Done when:** Runs are identifiable, selected models are recoverable, and later
evaluation results can be attached to the correct run.

## Step 6 — Smoke-test and run training (server)

**Purpose:** Verify hardware feasibility, then produce the selected models.

**What to do:**

- Record actual GPU count/VRAM, CUDA versions, and storage locations; verify
  transferred data hashes and synchronize dependencies using the committed
  `uv.lock` and `uv sync --locked --group finetune`.
- Test training on a small calibration subset, then forward/backward/optimizer
  steps on the longest example. Check gradients, memory, save/resume, adapter
  reload, and full-JSON generation. Never assume BF16 at 64K fits one A100.
- Resolve resource problems using a supported common configuration. Freeze both
  configs, the shared step cap, validation interval, and decoding settings.
- Start fresh M1/M2 runs on the eight training papers; select checkpoints using
  Step 4. Keep the two validation papers out of training.

**Done when:** Both runs complete with reproducible checkpoint selection, intact
inputs, persistent checkpoints, and MLflow records.

## Step 7 — Evaluate and compare (server/local)

**Purpose:** Measure extraction quality on the thirty held-out papers.

**What to do:**

- Generate complete JSON with M0 and the selected M1/M2 models, using identical
  prompts and decoding. Reuse old baseline predictions only if provenance matches.
- Write the existing `ExtractionResult` format and call the current evaluator
  functions through `scripts/evaluate_finetune.py`. The benchmark's existing
  runner assumes a fixed set of conditions; preserve that workflow.
- Report entity F1 for facilities/samples/datasets, conditional attribute scores,
  dataset-to-sample and dataset-to-facility reference F1, generic metadata scores,
  JSON/schema validity, and generation failures. Keep all thirty papers in the
  accounting and report training cost alongside quality.
- Attach predictions, per-paper scores, and a short M0/M1/M2 comparison to MLflow.
  Do not change checkpoints or training settings using these test results.

**Done when:** All three models have complete test accounting and reproducible
metrics from saved predictions.

## Interpretation and later work

This is a preliminary comparison: two validation papers, one training seed, and
unequal task/token exposure limit conclusions. Disclose the benchmark's prior use.
An M2 improvement motivates further experiments; it does not establish that task
decomposition caused the gain.

Defer matched-budget controls, repeated seeds, cross-validation, refitting on all
ten papers, separate attribute tasks, staged inference, and second-domain transfer.
When additional annotations arrive, freeze larger development and independent test
sets before further method selection.

## Requesting implementation

> Read `docs/finetune_plan.md` and implement Step N only. Reuse completed artifacts,
> run the checks in “Done when,” and report changed files, results, and exact next
> commands. Finish this step before proceeding. GPU execution belongs to the server
> steps and requires an available institute allocation.
