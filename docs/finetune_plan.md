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
  evaluation_metrics.yaml       # Shared scoring rules
  alignment_helper.yaml         # Shared entity-alignment rules
  finetune_prepare.yaml
  finetune_json_only.yaml
  finetune_structure_aware.yaml
  finetune_tracking.yaml         # Shared MLflow settings; server paths configured later
data/finetune/
  splits.json
  audit.md
  prepared/
    text/
    train_json_only.jsonl
    train_structure_aware.jsonl
    validation.jsonl
    length_report.json
    preparation_manifest.json
    quality_review.md
scripts/
  audit_finetune_data.py
  prepare_finetune_data.py
  smoke_finetune.py              # Isolated Step 6 GPU feasibility probes
  run_finetune.py
  evaluate_finetune.py
src/meta_data_extraction/finetune/ # Reusable preparation/training helpers as needed
tests/test_finetune_*.py          # Data, masking, and integration checks
outputs/finetune/<run_id>/        # Adapters, logs, predictions, and metrics
docs/finetune_plan.md             # Workflow and exact execution commands
pyproject.toml                   # Existing uv project; add a finetune dependency group
uv.lock                          # Shared dependency lockfile
```

Extend the existing repository and uv project. Use the explicit `finetune_*.yaml`
configs for preparation and training, and retain the shared scoring/alignment
configs. The old `configs/pan_*.yaml` benchmark configs were intentionally deleted;
they are not required for this pilot and should not be restored for fine-tuning.
Reuse source PDFs, gold records, parser, prompt builder, and evaluator; preserve
remaining benchmark data and results. Keep run outputs on persistent server
storage at the configured output path. Implement one numbered step at a time.

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

**Status: complete.** Membership and source fingerprints are frozen in
[`splits.json`](../data/finetune/splits.json); checks and the calibration inventory
are in [`audit.md`](../data/finetune/audit.md).

- Validation: **2503.18475**, the longest calibration paper (65 pages), at the
  user's request; and **2010.02965**, a moderate-length spectroscopy paper with
  six datasets and rich sample references. All seven technique strata remain
  represented among the eight training papers. The thirty test papers are unchanged.
- Length ranking uses complete PDF plain text, not model tokens. Step 2 must
  establish exact lengths; the long validation paper must still fit validation
  loss and generation. Inspect the recoverable PDF warnings recorded in the audit.
- Text-source checksums normalize line endings for Windows/Linux portability;
  PDF checksums cover raw bytes. Reuse the frozen split; do not regenerate it.

Verify the artifacts without rewriting them:

```powershell
uv run --no-sync python scripts/audit_finetune_data.py --verify
```

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

**Status: complete (local preparation and length audit).** The unchanged parser
processed all 40 PDFs. Training/validation JSONL counts are **8/24/2**; test papers
are used only for inference-input checks, not target construction. D records are
identical between the two training datasets. E + R reconstruction checks pass.

Configuration: [`configs/finetune_prepare.yaml`](../configs/finetune_prepare.yaml).
The official [Qwen3.5-4B tokenizer](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
is pinned to revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; thinking is disabled.
Only CPU preparation dependencies were added in the `finetune-prepare` uv group.
No model weights were loaded and no training was run.

| Measured sequence | Tokens |
|---|---:|
| Longest training example: 2104.10884, D | 32,660 |
| Validation: 2010.02965, D | 28,053 |
| Validation: 2503.18475, D | 51,268 |
| Largest inference prompt + 16,384-token output reservation | 63,718 |

All fit the provisional **65,536-token cap** without truncation. These counts
include the actual chat template, full schema, targets where applicable, and
special tokens. GPU feasibility remains Step 6; do not infer it from token counts.

Artifacts and checks:

- [`length_report.json`](../data/finetune/prepared/length_report.json): per-example
  and inference counts, overflows and quality flags.
- [`preparation_manifest.json`](../data/finetune/prepared/preparation_manifest.json):
  parser/tokenizer revisions, source/text/dataset hashes and configuration.
- [`quality_review.md`](../data/finetune/prepared/quality_review.md): targeted PDF
  inspection, including title-order and duplicated-caption defects retained from
  the shared parser. Successful parsing does not mean perfect transcription.
- Markdown, per-paper cache metadata and JSONL payloads are included in Git so
  the server can obtain them by pulling this repository. Cached text and existing
  outputs are checked, never silently
  overwritten; changed inputs/settings require an explicit new output directory.
- JSONL uses conversational `prompt` and `completion` arrays plus paper/task IDs,
  provenance and token counts. For Step 3, reuse `encode_record()` in
  `finetune/tokenization.py`: supervise only the JSON target and EOS, not the
  empty thinking prefix or the template's trailing newline. Actual training
  collation/masking tests remain Step 3.

Original preparation commands (CPU only; require the source PDFs, which remain
untracked; first run downloads tokenizer files):

```powershell
uv sync --locked --group finetune-prepare
uv run --locked --group finetune-prepare python scripts/prepare_finetune_data.py
```

Resume with `--stage parse` or `--stage build`; use `--offline` once the tokenizer
is cached. The preparation manifest preserves the original environment, including
the old `pyproject.toml` and `uv.lock` hashes. Strict preparation replay
(`--stage verify`) requires that original environment. After Step 3's dependency
changes, use the trainer's **dry runs below** to verify transferred data: they
check preparation code, schema, split, text, payload hashes and tokenization, and
explicitly report dependency-file changes. No source PDFs are needed for those
dry runs. Do not overwrite the historical manifest to make replay pass.

Run the preparation tests (including prompt/projection checks) and the frozen-input
audit. The older audit/corpus/prompt test modules are absent from the current checkout:

```powershell
uv run --locked --group finetune python -m pytest tests/test_finetune_preparation.py -q -p no:cacheprovider
uv run --locked --group finetune python scripts/audit_finetune_data.py --verify
```

Verification on 2026-10-27: **63 focused tests passed**, frozen-input verification
passed, and offline artifact replay passed. The broader suite recorded 406 passes,
2 skips, 2 failures and 3 setup errors. The failures/errors came from legacy
benchmark-reproduction tests requiring the intentionally deleted `pan_*.yaml`
configs, not from fine-tuning preparation. These are historical results; the legacy
test modules are absent from the current checkout. Continue testing shared
extraction/evaluation components when changes affect them.

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

**Done when:** Both configurations use the same trainer, fine-tuning and applicable
shared-component tests pass, and uv setup and launch commands are documented here.

**Status: complete (local implementation and verification, 2026-10-27).**
Both offline CPU dry runs passed, as did **41 preparation/training tests** and the
frozen-source audit. Tiny-model tests verified answer-loss equivalence, unchanged
base/vision weights, exact interrupted-versus-uninterrupted adapter weights for
both methods, and identical predictions after adapter reload. Prepared artifacts
were not modified. No 4B weights were downloaded and no real GPU training ran.

- [`run_finetune.py`](../scripts/run_finetune.py) uses
  [`finetune_json_only.yaml`](../configs/finetune_json_only.yaml) or
  [`finetune_structure_aware.yaml`](../configs/finetune_structure_aware.yaml).
  They differ only in method and output directory. Step 6 now sets `max_steps=18`
  to the user-approved common training budget; an explicit positive override is
  available for server smoke runs. The initial scheduler is constant.
- All **248 language attention/MLP projections** are LoRA targets, covering both
  full and linear attention. Vision, embeddings and the output head stay frozen.
  The actual pinned architecture is checked on meta tensors without loading weights.
- Precomputed JSON-plus-EOS labels are preserved through collation; prompt,
  empty-thinking prefix, trailing newline and padding are masked. No packing,
  retemplating or truncation is applied. Loss is target-token weighted across
  each gradient-accumulation window. TRL's fused loss avoids materializing logits
  for every paper token. See [TRL SFT](https://huggingface.co/docs/trl/sft_trainer).
- Each shuffled pass visits every example once, giving M2 D:E:R = 1:1:1 per
  complete pass. A partial final pass may differ; actual task counts, supervised
  tokens and total tokens are saved with checkpoints and the final summary.
- Checkpoints contain adapter, optimizer, scheduler, random state, trainer state
  and exposure counts. Resume requires the same resolved configuration, data,
  training code and runtime, and the latest complete checkpoint. Nonempty output directories
  cannot be reused for fresh runs. `reload_adapter()` checks the pinned base
  identity before loading saved adapter weights onto a fresh base model.
- `run_manifest.json` records data/model/configuration/runtime provenance;
  `training_summary.json` records final training statistics. `adapter/` holds the
  final-step adapter; Step 4's `selection.json` identifies the model selected for
  evaluation. Step 5 adds explicit MLflow logging; automatic library integrations
  remain disabled to avoid duplicate runs.
  Generated `outputs/finetune/` files are ignored by Git.

Set up and check both datasets locally or after pulling onto the server:

```powershell
uv sync --locked --group finetune
uv run --locked --group finetune python scripts/run_finetune.py --config configs/finetune_json_only.yaml --dry-run
uv run --locked --group finetune python scripts/run_finetune.py --config configs/finetune_structure_aware.yaml --dry-run
uv run --locked --group finetune python -m pytest tests -q -p no:cacheprovider
```

Add `--offline` to dry runs when the pinned tokenizer and model config are cached.
Dry runs read all artifacts and tokenize the selected training set plus validation;
they neither load 4B weights nor train. Unit tests train a tiny randomly initialized
Qwen3.5 fixture on synthetic token sequences only.

The lockfile retains Python **>=3.13** and pins Transformers 5.21.0/tokenizers
0.23.3 to preserve preparation identity. It resolves PyTorch 2.10.0, PEFT 0.21.2,
TRL 1.16.0 and Accelerate 1.15.0. Windows uses CPU PyTorch; Linux x86_64 uses
PyPI's CUDA 12.8 build. Linux dependency resolution was checked without installing
on a server. Server driver compatibility, native BF16, and long-sequence memory
remain Step 6. See [uv/PyTorch configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).
This initial trainer supports one process and one visible GPU; multi-GPU training
would require an explicitly tested extension.

Example **server-only two-update smoke run**, once a GPU is allocated and Step 5's
persistent MLflow settings are configured:

```bash
CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT=outputs/finetune/venv-gpu uv run --locked --group finetune-cuda python scripts/run_finetune.py --config configs/finetune_json_only.yaml --train --smoke --max-steps 2 --output-dir outputs/finetune/json_only_smoke
```

Use the structure-aware config and a separate output directory for its smoke run.
For an interrupted longer run, repeat its exact command with
`--resume-from-checkpoint <output-dir>/checkpoint-<step>`. The step cap must be
unchanged and greater than that checkpoint's step. Choose output paths on persistent
server storage. Do not launch the main comparison until Steps 4–6 are ready.

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

**Status: complete (local implementation and verification, 2026-10-27).**
All **66 preparation/training/validation tests passed**, both offline dry runs
passed, and every frozen source checksum was verified. Tests cover equal-paper
loss, step-zero selection, earliest ties, diagnostic failures, prompt-only
generation, preserved training randomness, exact resumed weights/history for both
methods, and recovery when a later checkpoint save is incomplete. Prepared data,
schemas and the shared evaluator are unchanged. No 4B weights were downloaded and
no real GPU training or paper validation ran locally.

**How it runs:**

- Both training configs now set `validation_steps: 6` and `save_steps: 6`; these
  must match. This is the user-approved interval for the pilot.
  Validation runs before the first update, at that interval, and at the
  final step even if it falls between intervals. There is no automatic early stop.
- Only the two frozen validation D examples are used. Each paper is processed
  separately with no gradients: average negative log-probability over its gold
  JSON and EOS, then average the two paper losses equally. Prompt, thinking
  scaffold, trailing template newline and padding are excluded.
- Step zero explicitly disables the LoRA adapter to measure the original model.
  Validation restores training mode and random states afterward, and does not
  increment training exposure counters. Nonfinite loss fails the run rather than
  silently dropping a paper or choosing a checkpoint from an invalid score.
- At every validation point, generate one complete JSON per paper from the D
  prompt alone, using greedy decoding and the fixed **16,384-token** output budget.
  Save raw responses, output-limit flags, parse/scoring failures, and entity F1.
  Reuse the shared evaluator for reference F1: it is conditional on aligned
  datasets; `n_pairs` is recorded and F1 is null when no dataset pairs align.
  Invalid output gets a failure record and null diagnostic scores, never exclusion
  from validation loss. These diagnostics do not affect model selection.

**Outputs under the run directory:**

| Artifact | Purpose |
|---|---|
| `validation_history.json` | Per-step/per-paper losses, generation diagnostics, and training exposure |
| `validation/step-<n>/` | Raw generations, parsed predictions, validation gold copies, evaluator reports |
| `checkpoint-<n>/validation_state.json` | Validation history saved alongside adapter/optimizer/random state |
| `selection.json` | Lowest-loss step, earliest tie, selected exposure, base revision, adapter path/checksum |

If step zero wins, selection has `kind: base` and no adapter path. Otherwise,
`checkpoint` names the retained selected checkpoint. `adapter/` remains the last
step's export; downstream evaluation must use `selection.json`. No checkpoint is
deleted. Resuming restores checkpoint-local history, avoids repeating step zero,
and rejects changed validation data/settings or an incomplete history. A later
partial checkpoint save does not prevent resuming the latest complete checkpoint.

**Local verification commands:**

```powershell
uv run --locked --group finetune python -m pytest tests/test_finetune_validation.py tests/test_finetune_training.py tests/test_finetune_preparation.py -q -p no:cacheprovider
uv run --locked --group finetune python scripts/run_finetune.py --config configs/finetune_json_only.yaml --dry-run --offline
uv run --locked --group finetune python scripts/run_finetune.py --config configs/finetune_structure_aware.yaml --dry-run --offline
```

Omit `--offline` for the first tokenizer/config download on a new machine. Dry runs
also verify validation membership, answer boundaries, and generation reservation;
they load no model weights. Actual Qwen3.5-4B validation and memory checks run on
the server in Step 6 using the existing `--train` command. Tiny-model tests here
check implementation behavior, not scientific extraction performance.

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

**Status: complete (local implementation and verification, 2026-10-28).**
All **76 tests passed**, including MLflow run identity, resume, metric replay,
selected-adapter preservation and later evaluation attachment. Both offline dry
runs passed without a tracking connection, and frozen source checksums were
verified. Tiny-model runs with and without MLflow logging produced identical
weights and validation histories. Prepared data, schemas and the shared evaluator are
unchanged. No full model weights were downloaded, institute service accessed or
real GPU training run. Persistent server storage remains to be configured before
Step 6.

**Configuration:** Both method configs reference
[`finetune_tracking.yaml`](../configs/finetune_tracking.yaml). Set `tracking_uri`
and `artifact_location` there, or override them with `MLFLOW_TRACKING_URI` and
`FINETUNE_MLFLOW_ARTIFACT_LOCATION`. The checked-in locations are unset: dry runs
work without connecting to MLflow, while real training requires an explicitly
configured backend. Use an existing HTTP(S) tracking service, or an absolute
SQLite database path plus an explicit artifact location on persistent storage.
An existing experiment's artifact location must agree with the configuration.

**What is recorded:**

- `mlflow_run.json` in the output directory holds the run ID, backend, artifact
  URI and execution-attempt number. Resuming verifies the same contract and
  continues that run. Successful runs finish as `FINISHED`; interrupted/failed
  attempts use `KILLED`/`FAILED`. Tracking failures are surfaced, not silently ignored.
- MLflow receives the resolved YAML, code commit/source hashes, model/tokenizer
  revisions, parser/schema versions, frozen split/data hashes, runtime versions and
  optimizer settings. Model loading starts after the run identity is saved.
- Loss curves are uploaded at step zero/checkpoint commits. Training loss still
  has one point per optimizer update; uploads include only committed updates.
  Replays check existing metric history, avoid duplicates and reject conflicting
  values at the same step. Validation logging preserves the regular training-loss
  log, so both curves remain complete at validation steps.
- Validation losses, F1/failure diagnostics, task counts and token exposure are
  recorded. `selected/*` statistics describe the chosen checkpoint; `exposure/*`
  describes training progress. Timing/memory metrics are separated by execution
  attempt, including resumed work. CUDA-event elapsed time is measured for
  training, validation and optimizer phases; it is not a kernel-utilization
  measurement. CPU checks omit CUDA timing/VRAM values. Abrupt process termination
  can leave resource measurements only through the last logged checkpoint.
- Selection metadata and the selected adapter's config/weights are uploaded by
  default, with its checksum and persistent local path. Set
  `upload_selected_adapter: false` to retain only the path/checksum. If step zero
  wins, the run records the pinned base model without uploading adapter weights.
  Optimizer checkpoints remain on persistent server storage; base weights and
  source PDFs are not uploaded. Validation generations and evaluator reports are
  saved as diagnostic artifacts.

`tracking.RunTracker.start(..., method="base")` supports an evaluation-only base
run. Step 7 will call `tracking.attach_evaluation(output_dir, metrics, artifacts,
evaluator_metadata)` for base/M1/M2: it reads the saved run ID, attaches results
under `test/*` and `evaluation/`, and rejects changed reattachments. M1/M2 must be
complete first. This step supplies that interface; it does not implement Step 7's
test-generation/evaluation runner or create production runs before execution.

**Server setup example** (replace `/persistent/pan-qwen35` with an allocated,
existing persistent directory):

```bash
export MLFLOW_TRACKING_URI=sqlite:////persistent/pan-qwen35/mlflow.db
export FINETUNE_MLFLOW_ARTIFACT_LOCATION=file:///persistent/pan-qwen35/mlflow-artifacts
uv sync --locked --group finetune
```

For an institute tracking service, set `MLFLOW_TRACKING_URI` to its URL and leave
the artifact location unset to use the service's experiment storage. Keep training
output/checkpoint directories on persistent storage as well. Do not create a new
tracking database in each temporary GPU-job directory.

View a configured SQLite-backed experiment using:

```bash
uv run --locked --group finetune mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI" --host 127.0.0.1 --port 5000
```

**Local verification:**

```powershell
uv run --locked --group finetune python -m pytest tests -q -p no:cacheprovider
```

The tracking tests use isolated temporary SQLite stores and
synthetic Qwen models. No institute service, production experiment, full model
weights or GPU execution is needed. MLflow was already included in the locked
`finetune` dependency group; no dependency changes were required.

## Step 6 — Smoke-test and run training (server)

**Purpose:** Verify hardware feasibility, then produce the selected models.

**What to do:**

- Record actual GPU count/VRAM, CUDA versions, and storage locations; verify
  transferred data hashes and synchronize dependencies using the committed
  `uv.lock` and the `finetune-cuda` dependency group on this Linux GPU host.
- Test training on a small **training-only** subset, then forward/backward/optimizer
  steps on the longest training example across D/E/R. Check gradients, memory,
  save/resume, and adapter reload. Separately test forward-only complete-JSON loss
  and generation on both validation papers, including the longest held-out paper.
  Never train on validation examples, or assume BF16 at 64K fits one A100.
- Resolve resource problems using a supported common configuration. Freeze both
  configs, the shared step cap, validation interval, and decoding settings.
- Start fresh M1/M2 runs on the eight training papers; select checkpoints using
  Step 4. Keep the two validation papers out of training.

**Done when:** Both runs complete with reproducible checkpoint selection, intact
inputs, persistent checkpoints, and MLflow records.

**Status: complete (2026-10-28).**
The selected machine is the local RTX 5070 Ti (16 GB), with outputs and MLflow
under `outputs/finetune/`. BF16 plus CPU activation offload, MLP chunking and
optimized linear attention passed the shortest example but ran out of memory on
the longest D example. The common fallback is NF4 double-quantized QLoRA with
BF16 computation, CPU activation offload and 2,048-token MLP chunks. Rank,
learning rate, accumulation, loss weighting, full context and splits remain
unchanged. QLoRA passed short and longest D/E/R training, both validation losses,
and full-budget generation on both validation papers. Integrated two-update runs
for both methods stopped at checkpoint 1, resumed in fresh processes to
checkpoint 2, and reproduced both validation losses after adapter reload.

Both fresh 18-update comparisons completed. JSON-only validation loss was
0.075139 / 0.057769 / 0.074680 / 0.085207 at steps 0/6/12/18 and selected
`outputs/finetune/json_only_seed42/checkpoint-6`. Structure-aware loss was
0.075139 / 0.060714 / 0.061765 / 0.068738 and selected
`outputs/finetune/structure_aware_seed42/checkpoint-6`. Fresh NF4 base plus
selected-adapter reloads reproduced every per-paper selected loss exactly.

Use the locked `finetune-cuda` group in the separate GPU environment described in
[the hardware report](finetune_step6_report.md). Validation and subsequent test
inference must use the same NF4 base for M0, M1 and M2; precision is persisted in
checkpoint/selection metadata. The CPU test suite passed 85 tests.

The user approved 18 optimizer updates per method and validation/checkpoints every
6 updates. Both MLflow runs are `FINISHED`, selected adapter artifacts and
precision contracts are uploaded, and local checkpoints are complete. An external
turn interruption killed an earlier structure-aware attempt before checkpoint 6;
it is marked `KILLED` and preserved separately, while the reported comparison is
the subsequent fresh run. No prepared artifacts or test inputs were modified.

**Staged checks:** [`scripts/smoke_finetune.py`](../scripts/smoke_finetune.py)
reuses the existing data verification, trainer, validation and MLflow helpers.
`--case plan` selects the shortest D and longest D/E/R examples without loading
weights. Each GPU case requires a new output directory, records its actual
hardware/configuration and fails with a nonzero exit code and saved report.
Training probes use one complete gradient-accumulation window, require finite
nonzero adapter gradients, compare frozen-weight hashes and save an adapter plus
optimizer checkpoint after a successful update. They do not certify resume or
checkpoint selection; those still require an integrated training smoke run.

Local tracking configuration (run from the repository root; these paths are
approved for this machine, and must be replaced when moving to another server):

```bash
export CUDA_VISIBLE_DEVICES=0
export MLFLOW_TRACKING_URI="sqlite:///$PWD/outputs/finetune/mlflow.db"
export FINETUNE_MLFLOW_ARTIFACT_LOCATION="file://$PWD/outputs/finetune/mlflow-artifacts"
UV_PROJECT_ENVIRONMENT=outputs/finetune/venv-gpu uv sync --locked --group finetune-cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/outputs/finetune/triton-cache"
outputs/finetune/venv-gpu/bin/python scripts/smoke_finetune.py --config configs/finetune_structure_aware.yaml --case plan --offline
```

On hardware/settings that can accommodate the full examples, start with:

```bash
outputs/finetune/venv-gpu/bin/python scripts/smoke_finetune.py --config configs/finetune_json_only.yaml --case short --offline --output-dir outputs/finetune/smoke_new_short
```

After that succeeds, run `longest_D`, `longest_E`, `longest_R`, `validation_loss`
and `validation_generation` as separate cases with the structure-aware config and
distinct output directories. The validation cases use only the two held-out D
examples and never call the optimizer. Generation uses the full fixed output
budget; parse failures remain diagnostics rather than fabricated scores.
`smoke_report.json` records success/failure, active phase/paper, memory and timing;
failures also save `smoke_traceback.txt`. MLflow runs carry `finetune.phase=smoke`.
A passed individual case does not mean Step 6 is complete.

## Step 7 — Evaluate and compare (server/local)

**Purpose:** Measure extraction quality on the thirty held-out papers.

**What to do:**

- Generate complete JSON with M0 and the selected M1/M2 models, using identical
  prompts and decoding. Reuse old baseline predictions only if provenance matches.
- Write the existing `ExtractionResult` format and call the current evaluator
  functions through `scripts/evaluate_finetune.py`, using the shared
  `configs/evaluation_metrics.yaml` and `configs/alignment_helper.yaml`. Do not
  invoke the legacy `scripts/run_evaluation.py` runner: its 22-condition config
  discovery is unrelated to the M0/M1/M2 comparison. No `pan_*.yaml` files are needed.
- Report entity F1 for facilities/samples/datasets, conditional attribute scores,
  dataset-to-sample and dataset-to-facility reference F1, generic metadata scores,
  JSON/schema validity, and generation failures. Keep all thirty papers in the
  accounting and report training cost alongside quality.
- Attach predictions, per-paper scores, and a short M0/M1/M2 comparison to MLflow.
  Do not change checkpoints or training settings using these test results.

**Done when:** All three models have complete test accounting and reproducible
metrics from saved predictions.

**Status: complete (2026-10-29).**
[`evaluate_finetune.py`](../scripts/evaluate_finetune.py) now builds the thirty
audited D prompts without opening test gold, loads M0 or the selected M1/M2
checkpoint with the common NF4 policy, greedily generates one paper at a time,
and commits resumable `ExtractionResult` predictions. The runner strictly checks
JSON, the merged field schemas and local references, then calls the existing
evaluator and alignment helper. Primary corpus metrics always include all thirty
papers: invalid or unscorable generations are explicitly reported and evaluated
as empty predictions rather than omitted. A secondary aggregate covers only valid
predictions accepted by the evaluator.

Configuration is frozen in
[`finetune_evaluation.yaml`](../configs/finetune_evaluation.yaml). Outputs go to
`outputs/finetune/evaluation/{m0,m1,m2}` with per-paper raw responses,
`ExtractionResult` predictions, reports, generation/scoring JSONL, full field
metrics, and `comparison.json`/`comparison.md`. The comparison includes JSON and
schema validity, facilities/samples/datasets F1, both reference F1 scores, generic
metadata, generation resources, and full/selected training exposure. MLflow
attachment is deliberately allowed only for a complete three-condition result;
it reuses the finished M1/M2 runs and creates one evaluation-only M0 run.

Run from the repository root in the Step 6 GPU environment. The process is
resumable at paper boundaries and loads the three conditions sequentially:

```bash
source outputs/finetune/local_env.sh
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/outputs/finetune/triton-cache"
outputs/finetune/venv-gpu/bin/python scripts/evaluate_finetune.py \
  --condition all --offline --attach-mlflow
```

To generate/score conditions as separate jobs, omit `--attach-mlflow` and use
`--condition m0`, `m1`, or `m2`. After all three finish, build the comparison and
attach it without loading model weights:

```bash
outputs/finetune/venv-gpu/bin/python scripts/evaluate_finetune.py --condition all --compare-only
outputs/finetune/venv-gpu/bin/python scripts/evaluate_finetune.py \
  --condition all --score-only --offline --attach-mlflow
```

`--score-only` never loads a model and requires complete cached generation for
all selected conditions. The full CPU regression suite passes **96 tests**, including
all thirty frozen gold schema/reference checks, test-prompt reconstruction with
test-gold reads blocked, all-paper failure accounting, comparison consistency,
and existing training/tracking checks.

The production run completed all **90 generations** (30 papers × M0/M1/M2), with
no generation exception, output-limit failure or scoring fallback. All three
conditions produced valid JSON for every paper. M0 and M1 were full-schema valid
on 28/30 papers; M2 was full-schema valid on 0/30 because it systematically
omitted `sample_refs` and `facility_refs` from every predicted dataset.

M1 is the strongest usable adapter: versus M0, samples F1 rose from 0.640 to
0.765, datasets from 0.541 to 0.800, sample-reference F1 from 0.743 to 0.795,
facility-reference F1 from 0.700 to 0.734, and the generic macro from 0.897 to
0.909; facilities declined from 0.753 to 0.718. M2 reached 0.752 samples F1,
0.790 datasets F1 and 0.923 generic macro, but both reference F1 scores were zero
because of the missing fields, so it does not satisfy one-call full-schema
extraction. Mean generation latency was 77.5 seconds for M0 and about 143 seconds
for each adapter.

All three MLflow runs are `FINISHED` and contain 62 test metrics plus prediction,
report, evaluator and comparison artifacts. The complete audit and interpretation
are in [`finetune_step7_report.md`](finetune_step7_report.md). Neither selected
checkpoint nor any training setting was changed after evaluation.

## Step 8 — Controlled D/E/R ablations (future independent-test study)

**Purpose:** Determine whether M2's missing reference fields came from reduced
complete-JSON exposure, interference from E or R, or the shape of the original R
target. This is a new experiment, not a retrospective reinterpretation of Step 7.

### Frozen design

Use seeds **42, 123 and 456** for every condition. An exposure is one training
example consumed once; four exposures equal one optimizer update. The matrix is
frozen in [`finetune_ablation.yaml`](../configs/finetune_ablation.yaml):

| Condition | Family | D | E | R | P | Examples | Updates | Question |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `d_only` | shared control | 24 | 0 | 0 | 0 | 24 | 6 | D-only reference point |
| `matched_d_de` | matched D | 24 | 8 | 0 | 0 | 32 | 8 | Effect of E at fixed D exposure |
| `matched_d_dr` | matched D | 24 | 0 | 8 | 0 | 32 | Effect of original R at fixed D exposure |
| `matched_d_der` | matched D | 24 | 8 | 8 | 0 | 40 | E+R at fixed D exposure |
| `matched_compute_de` | fixed update budget | 12 | 12 | 0 | 0 | 24 | D/E allocation under six updates |
| `matched_compute_dr` | fixed update budget | 12 | 0 | 12 | 0 | D/R allocation under six updates |
| `matched_compute_der` | fixed update budget | 8 | 8 | 8 | 0 | Reproduce M2 exposure under six updates |
| `matched_d_drp` | R redesign | 24 | 0 | 0 | 8 | 32 | Final-shaped reference supervision |

`P` is a reference-only patch with the same `datasets[].{id,sample_refs,
facility_refs}` nesting used by the final D output. It is deterministically
derived from each frozen training R record and its entity inventory; it does not
read new gold, invent entities, or change reference answers.

The `matched_compute_*` family matches example slots and optimizer updates, not
literal FLOPs: D/E/R completions have different lengths. Actual total and
supervised token exposure, wall time and memory must therefore be reported with
every run. The matched-D family is the primary control for whether M2 simply saw
too little complete JSON; the fixed-update family measures allocation under the
original six-update budget.

### Scheduling and checkpoint selection

- Each task draws from all eight training papers before repeating a paper.
  Remainders and the final order are deterministic for the declared seed. The
  saved run contract records exact task counts, per-paper exposure and a schedule
  checksum.
- Training uses the Step 6 model, NF4/LoRA, batch, accumulation, optimizer and
  decoding settings. Only the declared schedule, seed, endpoint and output path
  vary. Endpoints are exactly 6, 8 or 10 optimizer updates as shown above.
- Step 6 is retained as a recoverable checkpoint for the 8- and 10-update runs,
  but it is not a selectable scientific endpoint. Selection compares only the
  unadapted step-zero model with the condition's fixed-exposure endpoint.
- The predeclared lexicographic validation rule first maximizes the number of
  schema-valid validation papers, then mean conditional F1 across `sample_refs`
  and `facility_refs`, then minimizes equal-paper D target-token NLL, with the
  earlier step as the final tie-break. This prevents a low-loss checkpoint that
  omits required reference fields from winning.

The existing validation set has only two papers, so selection remains noisy.
Report every seed and condition; do not select or discard seeds after seeing
results. Treat schema validity, entity/reference F1 and per-paper failures as
primary outcomes, with training tokens, time and memory as cost outcomes.

### Data and evaluation gate

The thirty Step 7 papers have now influenced method choice and are development
data for this follow-up. They must **not** be reused to make a new confirmatory
claim. Before launching the full matrix, freeze a larger development split if
available and set `independent_test_split` to a new untouched test manifest.
The ablation configuration has `allow_current_test_evaluation: false`, and the
runner deliberately provides no path to the current Step 7 evaluator.

Evaluate the new independent test exactly once after the analysis rule and set of
conditions are fixed. Compare seed-level results and paired per-paper differences;
report mean, spread and bootstrap confidence intervals rather than only the best
run. If no new independent test can be obtained, label the matrix exploratory and
do not present it as confirmatory evidence.

### Implementation and commands

**Status: exploratory 24-run matrix and descriptive analysis complete (2026-10-30).**

[`run_finetune_ablation.py`](../scripts/run_finetune_ablation.py) supports three
explicit modes. `--plan` validates and prints the whole frozen matrix without
loading a tokenizer or model. `--dry-run` verifies one condition's frozen data,
P transformation, tokenization, exact schedule, context limits and LoRA target
architecture without loading weights or using a GPU. `--train` is the only mode
that trains and retains the Step 6 resumability and MLflow contracts. Pass
`--resume outputs/finetune/ablations/<condition>_seed<seed>/checkpoint-<step>`
with `--train` to resume the latest complete checkpoint. Ablation
runs are written under `outputs/finetune/ablations/<condition>_seed<seed>`.
The complete CPU regression suite passes **113 tests**, including exact exposure,
all-paper coverage, P-target equivalence, endpoint-only selection and the existing
preparation/training/evaluation controls.

The first run, `matched_d_drp` seed 42, completed all eight updates in 1,700.7
seconds and is recorded in MLflow as run
`c16facaeae17465396b06fd3af5cb86a`. Its exact exposure was 24 D plus 8 P examples.
The fixed endpoint was selected over the base model: equal-paper validation NLL
fell from 0.07514 at step 0 to 0.05693 at step 8, both validation outputs were
valid JSON, and mean conditional sample/facility-reference F1 was 1.0. One of the
two endpoint outputs was fully schema valid; the other used `null` for two
`bibliographic.funding[].grant_id` fields, unrelated to the redesigned reference
fields. The selected adapter is `checkpoint-8` under
`outputs/finetune/ablations/matched_d_drp_seed42`. This one-seed validation result
is a successful execution check, not evidence that P is superior. No Step 7 test
paper was evaluated.

Safe planning checks from the repository root are:

```bash
uv run --locked --group finetune python scripts/run_finetune_ablation.py --plan
uv run --locked --group finetune python scripts/run_finetune_ablation.py \
  --dry-run --condition matched_d_drp --seed 42 --offline
```

After the development/test protocol is frozen, one GPU run has this form:

```bash
source outputs/finetune/local_env.sh
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/outputs/finetune/triton-cache"
outputs/finetune/venv-gpu/bin/python scripts/run_finetune_ablation.py \
  --train --condition matched_d_drp --seed 42 --offline
```

Run one declared condition/seed at a time. Do not start the remaining 23 runs
until the first run passes the same memory, checkpoint/resume and MLflow checks as
Step 6. A condition is complete only after all three seeds finish; the study is
complete only after all eight conditions finish. A confirmatory claim additionally
requires the predeclared independent evaluation; without it, the completed matrix
remains exploratory and receives descriptive validation analysis only.

For an explicitly exploratory matrix, the restart-safe sequential wrapper
[`run_finetune_ablation_matrix.sh`](../scripts/run_finetune_ablation_matrix.sh)
checks all 24 declared runs, skips outputs that pass endpoint/exposure/checkpoint
and MLflow verification, resumes the latest complete pre-endpoint checkpoint, and
stops on any ambiguous or failed state. It never invokes held-out evaluation.
Inspect without training using `bash scripts/run_finetune_ablation_matrix.sh
--status`; start or continue the matrix using `bash
scripts/run_finetune_ablation_matrix.sh --run`. Per-run logs and the matrix event
log are written under `outputs/finetune/ablation-matrix-logs`.

### Completed exploratory results

All 24 runs (eight conditions × three seeds) completed with exact exposure and
finished MLflow accounting. The reproducible aggregate is generated by
[`analyze_finetune_ablation.py`](../scripts/analyze_finetune_ablation.py) and is
recorded in [`finetune_experiment_report.md`](finetune_experiment_report.md).
Across 48 fixed-endpoint validation outputs, all were valid JSON, 41 retained both
required reference fields, 18 were fully schema valid, and none failed generation
or reached its output limit. The trained endpoint was selected in 14/24 runs.

The strongest reference behavior came from D24+R8 and D24+P8 (both mean reference
F1 0.972 and 6/6 structurally complete outputs). P was selected in all three
seeds but did not improve reference F1 over R. The original fixed-budget
D8+E8+R8 condition reproduced the weakest behavior: reference F1 0.333 and only
2/6 reference-complete outputs. D12+R12 remained 6/6 complete with reference F1
0.969, whereas E-containing conditions without adequate reference supervision
were variable and omitted fields. This supports insufficient D exposure plus E
interference—not an intrinsically invalid R target—as the leading exploratory
explanation. The finding is limited to two validation papers and was not tested
on a new independent corpus.

### Exploratory 30-paper development evaluation

The next exploratory check is frozen to the fixed endpoints of D24, D24+R8,
D24+E8+R8 and D24+P8 for all three seeds (12 adapters, 360 generations). The
thirty Step 7 papers are explicitly reclassified as development data because
their earlier results influenced this design. This evaluation cannot support a
new held-out-test claim and cannot change any training checkpoint.

[`evaluate_finetune_ablation.py`](../scripts/evaluate_finetune_ablation.py) reuses
the audited Step 7 prompts, greedy decoding, schema checks and evaluator while
writing only to `outputs/finetune/ablation-development-evaluation`. It evaluates
the declared training endpoint even when two-paper validation selected the base
model, resumes generation at paper boundaries, reports all three seeds, and does
not attach these reused-corpus metrics to MLflow. Configuration is frozen in
[`finetune_ablation_evaluation.yaml`](../configs/finetune_ablation_evaluation.yaml).
The evaluation completed all 360 generations and produced
`outputs/finetune/ablation-development-evaluation/comparison.{json,md}`. Every
output was valid JSON and retained both reference fields; 337/360 were fully
schema valid, with all remaining failures caused by null funding grant IDs.
D-only led samples, datasets and sample-reference F1. D24+R8 provided the best
balanced structure-aware result, trading some sample-side quality for higher and
more stable facility/facility-reference scores. D24+E8+R8 strengthened facilities
and generic metadata further but reduced samples and sample references. D24+P8
did not show a clear advantage over the original R task. Full analysis is in
[`finetune_experiment_report.md`](finetune_experiment_report.md).

## Step 9 — Targeted follow-up: auxiliary transfer and task order

**Status: Phases A and B complete. Gate B was inconclusive and did not authorize
Phase C under the frozen rule. A separately authorized post-hoc Phase C extension
will follow the no-update diagnostic on the current data (2026-11-09).**

### Purpose and scope

Step 8 showed that the tested configurations with restored complete-JSON exposure
repaired M2's systematic reference omission, but it did not establish whether R
contributes useful relationship knowledge beyond extra optimization. It also
showed a repeatable trade-off: D24+R8 improved facilities and facility
references while reducing
samples and sample references relative to D24. Step 9 will explain that trade-off
with a small, gated study rather than another task-ratio grid.

The primary questions are:

1. Does R add useful information beyond the two extra optimizer updates in
   D24+R8?
2. Is R effective only when it receives gold entities, or does its relationship
   skill survive realistic entity errors?
3. Can task order retain facility-related gains without sacrificing sample-side
   performance?
4. If R has a distinct effect, which of its sample- and facility-reference parts
   causes positive or negative transfer?

P is not included because D24+P8 showed no clear advantage over D24+R8. E is not
included in the first follow-up because it shifted performance further away from
samples without producing a general improvement. Field-weighted losses and
gradient-balancing algorithms are deferred until the simpler controls below show
that a genuine R-specific benefit remains.

### Evidence and data governance

- Keep the eight training papers and their D/E/R targets unchanged for the
  mechanistic controls. Do not add, repair or relabel examples between conditions.
- Seeds remain **42, 123 and 456**. Report every seed; never select a seed.
- The two existing validation papers and thirty Step 7 papers are development
  data for this study. They may be used for exploratory diagnostics and method
  selection but not for a new held-out claim.
- Before choosing a confirmatory model, freeze a new independent test manifest,
  schema version, evaluator version and analysis rule. Do not inspect its gold
  labels or predictions until the shortlist is final.
- If no new independent corpus is available, complete only the exploratory
  analysis and label all conclusions accordingly.

Every new run must record the model revision, adapter configuration, seed,
ordered task schedule and checksum, examples by task and paper, optimizer
updates, input tokens, supervised tokens, wall time, peak VRAM, checkpoints and
MLflow run ID. Equal example counts do not imply equal training signal: D32 and
D24+R8 match example slots and updates, but D and R have different target lengths.
Before any GPU run, the frozen follow-up configuration must replace terms such as
“consistent,” “material” and “retains” in Gates B and C with numeric decision
thresholds derived from measurement resolution and the already available Step 8
seed/paper variability. Those thresholds cannot be revised after D32 is evaluated.

### Phase A — Diagnostics using existing checkpoints

No training is required for this phase.

#### A1. Paired error analysis

Use the saved D24 and D24+R8 predictions for all three seeds to report:

- per-paper precision, recall and F1 for facilities, samples, datasets and both
  reference fields;
- paired D24+R8 minus D24 differences, summarized by seed and paper;
- gold and predicted entity/reference counts and errors involving missing,
  duplicated, hallucinated or mislinked entities;
- whether the facility gain and sample loss are concentrated in a few papers or
  are directionally consistent across papers and seeds.

This analysis must precede new task design. A field-average change alone is not
enough evidence that the corresponding R subtask caused it.

Implementation is frozen in
[`finetune_followup.yaml`](../configs/finetune_followup.yaml) and
[`analyze_finetune_followup.py`](../scripts/analyze_finetune_followup.py). It
validates the six saved fixed-endpoint manifests, evaluator metadata, paper order
and pooled report counts before producing restart-safe JSON and Markdown under
`outputs/finetune/followup/phase-a1`. It performs no model loading, generation or
training. Phase A1 completed over all six runs and 180 run-paper reports; its
paired analysis is stored under `outputs/finetune/followup/phase-a1`.

#### A2. Oracle-versus-predicted R diagnostic

Evaluate each existing D24+R8 fixed endpoint with the R prompt under three input
conditions on development papers:

| Diagnostic | Entity inventory | Question |
|---|---|---|
| Oracle R | Gold entities | Can the adapter recover relationships when entity discovery is perfect? |
| Self-predicted R | Entities from that adapter's frozen one-call D output | How much relationship quality survives realistic entity errors? |
| D24-predicted R | Entities from the corresponding frozen D24 seed | Is the result sensitive to which extractor supplies the inventory? |

Use one frozen and documented rule to map predicted entities and IDs into the R
input. Report both conditional relationship F1 among correctly aligned entities
and end-to-end relationship F1 that penalizes missing or incorrect entities.
Never silently discard unmatched, duplicate or hallucinated entities.

Interpretation is predeclared:

- strong Oracle R but weak predicted-R results indicate entity-error propagation;
- strong Oracle and predicted R but weaker one-call D references indicate an
  integration/output-generation problem;
- weak Oracle R indicates that the current R task itself has not learned a
  reliable relationship function.

Phase A2 is implemented in
[`evaluate_finetune_followup.py`](../scripts/evaluate_finetune_followup.py) with
its frozen contract in
[`finetune_followup_a2.yaml`](../configs/finetune_followup_a2.yaml). It reuses the three
fixed D24+R8 adapters, loads each adapter once and evaluates oracle,
self-predicted and paired-seed D24-predicted inventories sequentially. Generation
is restart-safe at paper boundaries. The 270 R generations are written only under
`outputs/finetune/followup/phase-a2`; invalid R outputs remain explicit and use
empty-reference fallbacks for scoring. Reports separate evaluator-conditional
reference F1 from end-to-end relation-triple F1 that penalizes unaligned datasets
and target entities. No Phase B training should start before these results are
reviewed.

Phase A2 completed all 270 planned generations. All outputs were valid JSON and
258/270 satisfied the stricter R contract. With gold entities, mean sample- and
facility-reference F1 were 0.864 and 0.956. With self-predicted inventories,
end-to-end scores fell to 0.538 and 0.624; with D24-predicted inventories they
were 0.574 and 0.603. The result supports entity/dataset error propagation as the
main practical bottleneck: R links a clean inventory well, but a separate R pass
does not recover entities absent or incorrect upstream and did not outperform
the existing one-call links when given the same self-predicted inventory. Phase
A2 therefore supports proceeding to the predeclared D32 control, but does not by
itself establish R-specific transfer or justify a two-stage deployment.

### Phase B — D32 additional-training control

Train exactly one new condition for all three seeds:

| Condition | D | R | Total examples | Updates | Status |
|---|---:|---:|---:|---:|---|
| D24 | 24 | 0 | 24 | 6 | Existing control |
| D24+R8 | 24 | 8 | 32 | 8 | Existing mixed condition |
| **D32** | **32** | **0** | **32** | **8** | **Three new runs** |

D32 repeats each of the eight frozen D records exactly four times. Within each
seed, use a deterministic order that covers every paper once before repeating it.
Reuse the Step 8 base revision, QLoRA/LoRA settings, learning rate, microbatch,
four-example gradient accumulation, full-context policy and decoding settings.
The fixed step-8 endpoint is the scientific endpoint; intermediate checkpoints
exist for recovery and diagnostics, not post-hoc endpoint selection.

The primary comparison is D32 versus the existing D24+R8 endpoints. Report
facilities, samples, datasets, both reference metrics, schema validity and generic
metadata with seed means/spreads and paired per-paper differences. Also report D24
as the shorter-training reference. Because D32 contains longer supervised targets,
describe it as update/example matched—not token- or compute-matched.

Phase B is implemented in
[`finetune_followup_b.yaml`](../configs/finetune_followup_b.yaml),
[`run_finetune_followup_training.py`](../scripts/run_finetune_followup_training.py)
and
[`run_finetune_followup_phase_b.sh`](../scripts/run_finetune_followup_phase_b.sh).
The launcher runs seeds 42, 123 and 456 sequentially, verifies every completed
endpoint and its MLflow record, resumes only the latest complete checkpoint, and
fails closed on ambiguous nonempty output. The offline dry run verified D32,
four appearances per training paper, four complete eight-paper cycles, eight
optimizer updates and a longest sequence of 32,660 tokens. All three seeds then
completed endpoint 8 with verified D32 exposure and finished MLflow runs.

The development evaluator is frozen in
[`finetune_followup_b_evaluation.yaml`](../configs/finetune_followup_b_evaluation.yaml),
[`evaluate_finetune_followup_phase_b.py`](../scripts/evaluate_finetune_followup_phase_b.py)
and
[`run_finetune_followup_phase_b_evaluation.sh`](../scripts/run_finetune_followup_phase_b_evaluation.sh).
It generates only the 90 new D32 paper outputs, reuses the 180 existing D24 and
D24+R8 outputs, validates their checksums, evaluator provenance and pooled report
counts, and applies Gate B without changing its thresholds. Generation resumes
at paper boundaries; comparison outputs are written under
`outputs/finetune/followup/phase-b/evaluation` and are not attached to MLflow.

The numerical Gate B rules are frozen in the Phase B configuration before any
D32 result is observed. An effect within 0.03 F1 is treated as practically
equivalent. D32 must recover at least 0.03 on both sample-side metrics to support
the simpler D-only decision. Proceeding to Phase C requires D24+R8 advantages of
at least 0.04 facilities F1 and 0.03 facility-reference F1, positive differences
in at least two seeds, more improved than worsened papers for both fields, and no
single paper contributing more than 25% of absolute paired change. These values
are anchored to Phase A1: the observed D24+R8 changes were +0.073 facilities,
+0.045 facility references, -0.049 samples and -0.059 sample references; the
largest-paper shares were 16.7%, 15.5%, 49.7% and 42.3%, respectively.

#### Gate B

- If D32 reproduces the D24+R8 facility and facility-reference gains while
  retaining stronger sample-side performance, there is no current evidence that
  R provides a distinct benefit. Stop the R-order branch and retain D-only as the
  simpler direction.
- If D24+R8 retains a consistent facility-related advantage over D32 across seeds
  and paired papers, treat that as exploratory evidence of R-specific transfer
  and proceed to Phase C.
- If results are inconsistent or dominated by a few papers, do not add more
  objectives. Expand development data before deciding.

Phase B completed all 90 D32 generations and applied the frozen gate without
revision. D32 averaged 0.778 facilities, 0.759 samples, 0.763 datasets, 0.785
sample-reference and 0.782 facility-reference F1. Relative to D32, D24+R8 was
-0.014, -0.021, +0.009, -0.030 and -0.012 on those five fields, respectively.
Its facility and facility-reference advantages were positive in only one of
three seeds, so none of the Phase C checks passed. D32 stayed within the frozen
facility margins and recovered the sample-reference margin, but missed the
required 0.03 sample-F1 recovery. Gate B is therefore **inconclusive**: do not
reinterpret this result as a passed gate or as support for either schedule branch.
The later user-authorized order experiment is a separate post-hoc extension, not
a revision of this frozen decision.

All forty currently annotated papers are already assigned to the frozen eight
training, two validation and thirty reused-development papers. There is no unused
annotated corpus from which to expand development data. The next executable step
is the no-update diagnostic, followed by the user-authorized current-data order
controls. New annotation and a frozen expanded-development manifest are still
required before confirmation, and a separate independent test set must remain
untouched.

### Phase C — Sequential versus mixed D/R training

**Status: user-authorized post-hoc exploratory extension; no-update prerequisite
complete, implementation and evaluation contracts now being frozen.**

Gate B did not support an R-specific effect under its frozen progression rule,
but the user has authorized the two order controls on the current data after the
no-update diagnostic. Reuse the exact same 24 D and 8 R records as the existing
mixed condition and change only their order. Label every result post-hoc and
exploratory: it cannot replace M1, support an independent generalization claim or
be presented as a passed Gate B continuation.

| Schedule | Updates 1–8 | New runs |
|---|---|---:|
| D32 control | 32 D examples | 0 (Phase B) |
| Mixed D24+R8 | Existing deterministic interleaving | 0 (Step 8) |
| **R8 → D24** | 8 R examples, then 24 D examples | **3** |
| **D24 → R8** | 24 D examples, then 8 R examples | **3** |

Keep the optimizer and scheduler state across the stage boundary; do not reset
either. Align the task transition with a gradient-accumulation boundary so no
optimizer update mixes stages. Within each stage, order papers deterministically
from the seed. Save the stage-boundary checkpoint for diagnostics and the fixed
step-8 endpoint for comparison; do not choose between them using the two-paper
validation set.

The main hypothesis is that R8→D24 will preserve useful relationship learning
while the final D stage restores complete-JSON and sample-side behavior. D24→R8
tests whether late auxiliary training causes interference. Compare both schedules
with mixed D24+R8 and D32 using the same three seeds and paired development
evaluation.

#### Gate C

- Shortlist R8→D24 only if, relative to mixed D24+R8, it improves both samples F1
  and sample-reference F1 by at least **0.02**, remains within **0.03** on
  facilities, datasets and facility-reference F1, and is also within **0.03** of
  D32 on all five fields. Both sample-side improvements must be positive in at
  least two of three seeds, improve more paired papers than they worsen and have
  no single paper contribute more than 25% of absolute paired change.
- Treat degradation after D24→R8 as evidence consistent with late-task
  interference only if D24→R8 is at least **0.03** below R8→D24 on samples or
  sample-reference F1, the difference has the same sign in at least two seeds,
  worsens more paired papers than it improves and no paper contributes more than
  25% of absolute paired change. This remains descriptive, not proof of a general
  continual-learning mechanism.
- If neither sequence improves the trade-off, stop schedule experiments rather
  than trying additional ratios or optimizer resets on the reused corpus.

These metrics and numeric thresholds are frozen before either sequential run.
They use the same 0.03 practical-equivalence and 25% concentration rules as Gate
B; the 0.02 paired sample-side recovery threshold targets a material partial
recovery of the completed mixed-versus-D32 deficits without requiring a post-hoc
full reversal. The no-update values did not tune the schedule, exposure, learning
rate, endpoints or thresholds. Evaluate the two sequences against the existing
mixed D24+R8 and D32 fixed endpoints on all three seeds and thirty reused
development papers. Any apparent winner remains only a candidate for later
confirmation on new data.

Phase B added **three unconditional training runs**. The separately authorized
post-hoc Phase C extension adds six sequential runs, for nine Phase B+C runs in
total.

### Phase D — Conditional relationship-task decomposition

Run this phase only if R still provides a distinct benefit after Phases A–C and
the paired error analysis shows that sample and facility relationships behave
differently.

Create two deterministic targets from the frozen R records without changing the
gold entities or relationships:

- `R_sample`: only dataset-to-sample references;
- `R_facility`: only dataset-to-facility references.

Compare D24+R_sample8 and D24+R_facility8 with existing D24+R8, using the same
three seeds, eight updates and exact D exposure. This adds six conditional runs.
The purpose is diagnosis, not choosing a ratio: determine whether facility gains
come specifically from facility-reference supervision and whether either subtask
causes the sample-side loss.

Do not start field-weighted D training unless this decomposition identifies a
specific transferable field. If it does, predeclare a separate span-weighted D32
condition that weights semantic entity/reference value spans, not merely JSON
keys. Compare it with D32 and D24+R8 using free-generation metrics; teacher-forced
loss alone is not an adequate structural safeguard.

### Phase E — Data scaling and independent confirmation

The existing eight training and two validation papers are too small for a strong
generalization claim. After the mechanism study:

1. annotate additional papers and freeze expanded training/development manifests;
2. keep a new representative independent test set untouched;
3. freeze a shortlist containing the appropriate D-only control and at most one
   R-based candidate, plus the analysis script and all checkpoints;
4. evaluate the independent test exactly once with all seeds;
5. report schema validity, precision/recall/F1, paired per-paper differences,
   seed spread, bootstrap intervals, latency and failure categories;
6. do not revise training, prompts, thresholds or checkpoint choice after seeing
   independent-test results.

If enough additional training data exist, compare the shortlisted D-only and
R-based methods at predeclared data scales with matched paper coverage. This is a
separate scaling study and must not recycle independent-test papers into training.

### Proposed implementation boundary

Implementation should reuse the Step 8 data contracts, trainer, evaluator,
resume validation and MLflow accounting. New code should be limited to:

- a frozen follow-up configuration describing D32 and the two sequential
  schedules;
- deterministic schedule construction with task-boundary checks;
- an R-diagnostic evaluator with explicit entity-inventory mapping;
- paired error/aggregate reporting; and
- a restart-safe wrapper that runs only conditions enabled by a recorded gate or
  an explicit post-hoc authorization recorded in this plan.

Proposed outputs belong under `outputs/finetune/followup/`; they must not modify
Step 6–8 artifacts. Planning and dry-run modes must validate schedules, counts,
checksums, context limits and endpoint rules without loading model weights.

### Done when

- Phase A produces reproducible per-paper and oracle/predicted-R reports from
  unchanged existing checkpoints.
- All three D32 runs have exact D32 exposure, eight updates, complete checkpoints
  and MLflow accounting; Gate B is recorded before any sequential run starts.
- If Phase C executes under a passed gate or the recorded post-hoc authorization,
  all six sequential runs preserve the exact D24+R8 multiset, cross task
  boundaries only between optimizer updates and record Gate C.
- Any Phase D run is explicitly justified by the earlier gate and uses frozen
  split-R targets with tests proving equivalence to the relevant part of R.
- All development results are labeled exploratory and update
  [`finetune_experiment_report.md`](finetune_experiment_report.md).
- A confirmatory conclusion is made only after the frozen shortlist is evaluated
  once on a new independent test set.

## Knowledge sharing experiment

**Status: complete and reviewed; no parameter updates or model selection
(2026-11-09). The authorized training-order controls are next.**

### Purpose

Determine whether the D, E and R objectives suggest compatible or conflicting
updates to the shared LoRA parameters. This is a **task-interaction diagnostic**,
not a direct measure of transferred knowledge: gradient agreement is local to a
checkpoint and does not guarantee better free-generation F1.

The lightweight mechanistic diagnostics in
[`finetune_mechanistic_diagnostics.md`](finetune_mechanistic_diagnostics.md) are
complete. They found that 93.8–100% of one-call reference false negatives were
inventory-limited, while D24+R8 exceeded the equal-update D32 control under a
gold inventory by 0.036 sample-reference F1 and 0.039 facility-reference F1. The
gradient analysis may help explain that conditional benefit. It does not make
Gate B pass or change the original M1 decision; Phase C proceeds only through the
separate post-hoc authorization recorded above.

### Fixed scope and inputs

- Use the three frozen **D24 fixed endpoints**, seeds **42, 123 and 456**, as
  separate common starting models. Do not use a single seed or a checkpoint
  selected by this analysis.
- Use the existing D, E and R records for the same eight training papers. Do not
  alter prompts, targets, schema, paper text, IDs or task projections.
- Compute gradients only for trainable LoRA parameters. Keep the quantized base,
  embeddings, output head and vision modules frozen.
- Perform one independent backward pass per paper and task: **24 backward passes
  per seed, 72 total**. Clear gradients between passes and never call an optimizer
  or scheduler.
- Use the production QLoRA precision and full-context policy. Normalize every loss
  by its supervised target tokens, including EOS, using the same label masking as
  training. Record target-token and total-token counts for every example.
- Disable LoRA dropout for the primary deterministic measurement and record this
  diagnostic-mode choice. Fix all random seeds and fail on nonfinite loss,
  gradient or norm.

An optional sensitivity check may repeat a small, predeclared subset with fixed
dropout seeds. It must not replace the deterministic primary result.

### Comparisons

For paper `i`, let `gD_i`, `gE_i` and `gR_i` be the target-token-normalized
gradients at one frozen checkpoint. Compute:

| Comparison | Purpose |
|---|---|
| `gD_i` versus `gE_i` | Same-paper D/E compatibility |
| `gD_i` versus `gR_i` | Same-paper D/R compatibility |
| `gD_i` versus `gD_j`, `i != j` | Within-D baseline for ordinary paper variation |
| `gD_j` versus `gE_i` and `gR_i`, `i != j` | Cross-paper evidence beyond shared content |
| `gE_i` versus `gE_j`; `gR_i` versus `gR_j` | Within-task coherence controls |

Same-paper alignment alone may reflect shared paper content. Cross-paper results
are therefore required before describing an auxiliary task as carrying reusable
information for D. Raw similarity values must be interpreted relative to the D/D
baseline, not relative only to zero.

### Measurements and artifacts

For every task example and pair, record:

- global cosine similarity, raw gradient dot product and each gradient norm;
- the fraction of negative pairings and the distribution across papers and seeds;
- the same quantities by transformer block and LoRA module family, so opposing
  layer-level effects are not hidden by one global value;
- the first-order predicted D-loss change from an auxiliary step,
  `-learning_rate * dot(gD, gAux)`, reported only as a local approximation; and
- runtime, peak VRAM, model/checkpoint hashes, example hashes, precision policy
  and diagnostic-code version.

Accumulate dot products and norms in stable precision. Do not keep full raw
gradient vectors as permanent artifacts; retain only the aggregate statistics and
the minimum temporary data needed for cross-paper comparisons. Write JSON,
per-paper CSV and a concise Markdown report under
`outputs/finetune/knowledge-sharing/gradient-interaction/`. Do not attach this
post-hoc exploratory analysis to the original confirmatory MLflow runs.

Summaries must include each seed separately and, across seeds, the median,
interquartile range, range and negative fraction. With only eight papers and three
seeds, treat intervals and sign frequencies as descriptive rather than population
significance claims.

### Interpretation rules

- Evidence consistent with **E/D conflict** requires D/E similarity to be more
  negative than the within-D baseline across multiple papers and all or most
  seeds, rather than one negative global mean.
- Evidence consistent with **R-to-D compatibility** requires D/R alignment to be
  stable across seeds and present in cross-paper comparisons. Positive same-paper
  similarity alone is insufficient.
- Large auxiliary gradient norms with weak or negative cosine indicate that an
  auxiliary task may dominate shared updates despite equal example counts.
- Opposing layer-level signs indicate localized interaction and may motivate a
  later selective-sharing study; they do not by themselves justify a new adapter
  architecture.
- Near-zero, unstable or baseline-like results mean there is no useful evidence
  of task-specific interference. Prioritize more data and entity/reference
  consistency instead of adding optimization machinery.

Positive gradient similarity does not establish that E or R improves extraction,
and negative similarity does not prove harmful transfer over a complete training
trajectory. Interpret the result together with D32, equal-inventory oracle scores
and free-generation metrics.

### Results and decision

All **72/72** backward passes completed at the three frozen D24 endpoints. The
runner created no optimizer or scheduler, took zero optimizer steps and verified
identical trainable-state and checkpoint hashes before and after every seed. It
produced 212 global comparisons and 7,420 block/module comparisons per seed.

| Comparison | Seed 42 | Seed 123 | Seed 456 | Median | Negative pairs |
|---|---:|---:|---:|---:|---:|
| Cross-paper D/D | 0.081 | 0.056 | 0.089 | **0.081** | 4.8% |
| Cross-paper D/E | 0.016 | 0.015 | 0.029 | 0.016 | 22.6% |
| Cross-paper D/R | 0.020 | 0.013 | 0.012 | 0.013 | 25.6% |
| Cross-paper E/E | 0.204 | 0.193 | 0.161 | 0.193 | 0.0% |
| Cross-paper R/R | 0.466 | 0.261 | 0.167 | 0.261 | 1.2% |
| Same-paper D/E | 0.354 | 0.399 | 0.287 | 0.354 | 0.0% |
| Same-paper D/R | 0.028 | 0.021 | 0.024 | 0.024 | 4.2% |

Values are cosine medians; the final column describes all relevant pairs, while
the cross-seed center uses the three seed medians. D/E and D/R were positive
cross-paper in every seed but substantially below the ordinary D/D baseline.
Same-paper D/E was much larger, consistent with shared source content. E/E and
R/R were internally coherent. Median gradient norms were 0.164 for D, 0.727 for
E and 0.288 for R, so equal auxiliary example counts can dominate update
magnitude even without opposing direction.

There is **no systematic E/D conflict**. R/D compatibility is weak but
reproducibly positive cross-paper; it is not stronger than E/D and is baseline-
like relative to D/D. All module-family medians were positive, with only 2/32
D/E and 1/32 D/R block medians near or below zero. Do not add gradient balancing,
selective sharing or separate adapters. The result does not predict an order
winner, but the no-update prerequisite is satisfied and the separately authorized
R8→D24/D24→R8 behavioral controls may proceed under the frozen rules below.

### Training-order experiments

The immediate current-data order experiment is the narrowly scoped Phase C study
above: existing mixed D24+R8 and D32 controls plus R8→D24 and D24→R8 for all
three seeds. The no-update diagnostic is complete and analyzed, so this study may
now run; keep it post-hoc and exploratory.

After new training and development papers are frozen, repeat the comparison as a
separate predeclared study proportional to the number of training papers `N`:

| Condition | Exposure | Question |
|---|---:|---|
| Standard D-only | `3N D` | Shorter complete-task control |
| Mixed D+R | `3N D + 1N R` | Does simultaneous auxiliary supervision help? |
| Longer D-only | `4N D` | Does extra D explain the mixed result? |
| Auxiliary first | `1N R → 3N D` | Can final D integrate earlier R learning? |
| Auxiliary last | `3N D → 1N R` | Does late R disrupt complete extraction? |

Keep optimizer state through sequential boundaries and align every boundary with
an optimizer update. Use a larger frozen development set for method selection and
evaluate the final shortlist once on an untouched independent test set. Separate
E/R adapters, routing, adapter composition, PCGrad and other gradient-balancing
methods remain deferred unless the diagnostic shows reproducible conflict and the
expanded-data schedules show that task order matters.

### Done when

- All 72 planned backward passes complete without an optimizer step and reproduce
  the frozen D24 checkpoints before and after analysis.
- Same-paper, cross-paper and within-task baselines are complete for every seed,
  with finite losses, norms, dot products and cosine similarities.
- Global and layer/module-level reports include full paper/seed accounting,
  provenance and the declared descriptive summaries.
- The conclusion uses the interpretation rules above, is labeled exploratory and
  is added to [`finetune_experiment_report.md`](finetune_experiment_report.md).
- The diagnostic is complete and reviewed. The user-authorized Phase C order
  controls may now run on the current corpus, but no specialized sharing method
  or additional schedule may be introduced.

## Interpretation and later work

The original M0/M1/M2 comparison remains preliminary because it used one training
seed and unequal task/token exposure. The follow-up used three seeds, controlled
exposure/update families and a thirty-paper development evaluation, but those
papers had already influenced method design. Disclose the benchmark's prior use
and keep the ablation conclusions exploratory until a new independent test exists.

Step 8, the gated Step 9 Phase A/B study and the no-update knowledge-sharing
diagnostic are complete. Gate B was inconclusive. The user-authorized R8→D24 and
D24→R8 controls may now run on the current corpus as a post-hoc exploratory
extension. Broader ratio searches, separate
attribute tasks, optimizer resets, gradient-balancing methods and second-domain
transfer remain stopped. When additional annotations arrive, freeze larger
development and independent test sets before confirmatory method selection.

## Requesting implementation

> Read `docs/finetune_plan.md` and implement Step N only. Reuse completed artifacts,
> run the checks in “Done when,” and report changed files, results, and exact next
> commands. Finish this step before proceeding. GPU execution belongs to the server
> steps and requires an available institute allocation.
