# Step 6 hardware feasibility — 2026-10-28

**Status: complete. BF16 with activation offload and MLP chunking passed the
shortest example but failed the longest training document. The common QLoRA
configuration passed all feasibility/recovery checks and both 18-update
comparative runs completed with selected, reload-verified checkpoints.**

## Environment and unchanged experiment

- GPU: NVIDIA GeForce RTX 5070 Ti; nominal 16 GB, PyTorch-reported capacity
  **15.54 GiB**. One GPU was visible. Native BF16 support was verified.
- NVIDIA driver: **615.71.09**. PyTorch: **2.10.0+cu128**, CUDA runtime **12.8**.
- Python: **3.13.12**. The committed `uv.lock` and `finetune` group were used.
- Model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Full weights were already cached.
- LoRA rank 16, alpha 32, dropout 0.05; BF16 base; SDPA; gradient checkpointing;
  microbatch 1, accumulation 4; no packing or truncation. Production settings
  were preserved apart from the probe's one-update cap and checkpoint interval.
- Frozen split remains 8/2/30. The shortest complete D example is
  `1812.03956:D`, with 15,823 sequence tokens and 1,544 supervised tokens.
- User-authorized storage is the repository's `outputs/finetune/` directory.
  SQLite metadata is in `mlflow.db`, artifacts in `mlflow-artifacts/`.
  `local_env.sh` stores convenience environment settings; source it from the
  repository root. These machine-specific files and all run outputs are ignored
  by Git and need separate preservation when moving machines.

## Executed probes

The initial five probes recorded CUDA out-of-memory failures and terminated. GPU probes
ran sequentially. No optimizer update completed and no training checkpoint or
selected adapter was produced.

| Directory under `outputs/finetune/` | Probe/settings | Peak allocated GiB | Peak reserved GiB | Result |
|---|---|---:|---:|---|
| `smoke_5070ti_short` | Shortest D; locked environment, default allocator | 13.943 | 14.615 | OOM before first update |
| `smoke_5070ti_short_expandable` | Shortest D; expandable allocator | 14.464 | 14.613 | OOM during backward recomputation |
| `smoke_5070ti_validation_loss` | Both validation papers; expandable allocator | 14.454 | 14.527 | OOM during forward-only loss |
| `smoke_5070ti_short_fla` | Shortest D; optimized linear attention + expandable allocator | 14.451 | 14.613 | OOM during backward recomputation |
| `smoke_5070ti_validation_loss_fla` | Validation loss; optimized linear attention + expandable allocator | 14.402 | 14.607 | OOM on `2503.18475` (51,268 tokens) |

Peaks are PyTorch process allocations/reservations, not total device usage or a
measurement of the capacity that would suffice. Failures can occur before the
true peak of a successful step. A small existing desktop GPU process was left
untouched. Available memory and the full error are recorded in each report.

The two expandable-allocator training retries failed on a 280 MiB allocation in
the LoRA MLP path while recomputing activations for backward. The optimized-kernel
validation run advanced to the longer of the two papers, but a further 402 MiB
allocation failed. These results do not establish a minimum sufficient GPU size.

Each directory contains `run_manifest.json`, `mlflow_run.json`, and
`smoke_report.json`. All runs are marked `FAILED` in the shared experiment with
`finetune.phase=smoke`; reports were uploaded. All but the first attempt also
contain/upload `smoke_traceback.txt`. Console logs sit alongside the directories
as `<directory-name>.log`.

## Bounded memory experiments

The allocator retry set `PYTORCH_ALLOC_CONF=expandable_segments:True`, which
changes allocation behavior without changing training inputs or model precision.
See [PyTorch 2.10 allocator documentation](https://docs.pytorch.org/docs/2.10/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf).

The optimized linear-attention experiment installed `flash-linear-attention==0.6.0`,
`fla-core==0.6.0`, and `einops==0.8.2` into the isolated
`outputs/finetune/kernel_probe_deps/` directory. Only those two probe processes
received that directory through `PYTHONPATH`. The standard `.venv`, `pyproject.toml`
and `uv.lock` were not changed by this experiment. Transformers selected the
installed FLA implementation; causal convolution retained its PyTorch fallback.
See the [Transformers Qwen3.5 usage notes](https://huggingface.co/docs/transformers/model_doc/qwen3_5#usage-tips-and-notes)
and [FLA project](https://github.com/fla-org/flash-linear-attention).

For reproducibility of that failed experiment, prefix a smoke command with:

```bash
PYTHONPATH="$PWD/outputs/finetune/kernel_probe_deps" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
TRITON_CACHE_DIR="$PWD/outputs/finetune/triton-cache" \
uv run --locked --group finetune python scripts/smoke_finetune.py \
  --config configs/finetune_json_only.yaml --case short --offline \
  --output-dir outputs/finetune/smoke_new_fla
```

This is diagnostic evidence, not a validated optional dependency configuration
for the comparison. Do not promote it to main runs without the full checks.

## Initial verification and remaining work (before memory changes)

Both offline trainer dry runs passed. The CPU suite passed **80 tests** (the
previous 76 plus four smoke tests). New tests cover deterministic shortest/longest
selection, split isolation, actual optimizer updates with unchanged frozen weights,
forward-only validation, and rejection of nonfinite/missing gradients.

Before production training can proceed, provide more GPU memory or implement and
validate a common memory-saving configuration that preserves the experiment's
requirements. Then rerun shortest and longest D/E/R training, both validation
loss/generation paths, interrupted checkpoint resume and adapter reload, and the
integrated M1/M2 smoke runs. Freeze the common update budget and intervals only
after those checks succeed. Launch both methods freshly from the pinned base.

Longest-example backward, validation generation, real-model resume/reload, and
comparative training were not attempted after the shortest-example training and
long-validation-loss failures. `max_steps` remains unset in both method configs.
Prepared artifacts, test data, schemas, scientific settings and lockfile were not
modified. The 30-paper test evaluation remains Step 7.

## Memory follow-up and common precision policy

The earlier successful training in the sibling repository used NF4 QLoRA,
CPU activation offload, and 2,048-token MLP chunks. Its rank 8, learning rate
5e-5, accumulation 2 and per-example loss reduction differed from this pilot.
We retained this pilot's rank 16 / alpha 32, learning rate 1e-4, accumulation 4,
answer-token-weighted loss, frozen 8/2/30 split, and full context.

Both method configs now enable CPU activation offload, 2,048-token checkpointed
MLP chunks, and NF4 double quantization with BF16 computation. MLP chunking does
not split attention context or remove labels. It changes dropout RNG ordering
relative to an unchunked run; chunk size belongs to the resume contract.
K-bit preparation retains FP32 stability parameters and restores the frozen
input/output vocabulary matrices to BF16, as in the older working setup.

Training, step-zero baseline, adapter validation, and subsequent M0/M1/M2 test
inference must use the same quantized base policy. This is a new NF4 baseline;
old BF16 scores cannot substitute for it. Checkpoints and saved adapters contain
`inference_precision.json`; selection and run manifests record the policy too.
The shared reload helper rejects a precision mismatch. Step 7 must use this
policy rather than merging/reloading the adapter onto a BF16 base silently.

The reproducible Linux GPU dependencies are in the locked `finetune-cuda`
group: bitsandbytes 0.50.2, flash-linear-attention/fla-core 0.6.0, einops 0.8.2.
The CPU test environment remains separate because installed FLA dispatches to
CUDA kernels. Causal convolution retains the PyTorch fallback.

```bash
UV_PROJECT_ENVIRONMENT=outputs/finetune/venv-gpu uv sync --locked --group finetune-cuda
source outputs/finetune/local_env.sh
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/outputs/finetune/triton-cache"
outputs/finetune/venv-gpu/bin/python scripts/smoke_finetune.py \
  --config configs/finetune_structure_aware.yaml --case longest_D --offline \
  --output-dir outputs/finetune/smoke_new_qlora_longest_D
```

The user approved **18 optimizer updates per method, validation/checkpoints every
6 updates** after feasibility and restart checks. That means 72 example exposures
per method: nine JSON-only passes versus three structure-aware passes. It matches
update count, not token exposure; actual D/E/R and token exposure remain logged.

CPU regression coverage now includes loss/gradient equivalence without dropout,
exact interrupted/resumed parameters and validation history with chunked dropout
for both methods, adapter reload, and precision mismatch rejection. The 85-test
suite passed. The controlled `--stop-after-step` flag stops only at a checkpoint
boundary and retains the original maximum step count for a subsequent resume.

Follow-up probe directories use `smoke_bf16_memory_*` and `smoke_qlora_*` under
`outputs/finetune/`; each has its MLflow identity, manifest and memory report.
No test-set inference is part of these checks.

Integrated restart check (two updates, checkpoints every update; uses both held-out
papers at every measurement) uses saved configs under `outputs/finetune/`:

```bash
outputs/finetune/venv-gpu/bin/python scripts/run_finetune.py \
  --config outputs/finetune/smoke_integrated_json_only.yaml --train --smoke --offline --stop-after-step 1
outputs/finetune/venv-gpu/bin/python scripts/run_finetune.py \
  --config outputs/finetune/smoke_integrated_json_only.yaml --train --smoke --offline \
  --resume-from-checkpoint outputs/finetune/smoke_integrated_json_only/checkpoint-1
outputs/finetune/venv-gpu/bin/python scripts/check_finetune_adapter.py \
  --config outputs/finetune/smoke_integrated_json_only.yaml --offline
```

The reload check loads the selected model (fresh quantized base plus selected
checkpoint adapter, when applicable) and reproduces both per-paper validation
losses (relative tolerance 1e-4, absolute 1e-5). The same sequence applies to
`structure_aware`. Smoke runs are explicitly tagged in MLflow and are never
treated as comparative results.

## Completed memory feasibility probes

| Probe directory | Result | Peak allocated GiB |
|---|---|---:|
| `smoke_bf16_memory_short` | out_of_memory | 13.767 |
| `smoke_bf16_memory_short_fla` | passed | 12.862 |
| `smoke_bf16_memory_longest_D_fla` | out_of_memory | 13.775 |
| `smoke_qlora_short` | passed | 7.151 |
| `smoke_qlora_longest_D` | passed | 11.336 |
| `smoke_qlora_longest_E` | passed | 10.919 |
| `smoke_qlora_longest_R` | passed | 10.976 |
| `smoke_qlora_validation_loss` | passed | 9.766 |
| `smoke_qlora_validation_generation` | passed | 10.417 |

All QLoRA training cases completed one optimizer update (four full examples),
with finite nonzero adapter gradients and identical frozen-weight hashes before
and after. Both validation papers completed loss and greedy generation with the
full 16,384-token allowance; both generations ended with EOS and valid JSON.
These are feasibility results, not a claim of extraction accuracy. Both
integrated restart/reload checks passed. JSON-only validation loss moved from
0.075139 at step 0 to 0.069295 at step 1 and 0.063485 at step 2; structure-aware
moved from the same baseline to 0.068675 and 0.063817. Both selected checkpoint
2, and fresh base-plus-adapter reloads reproduced the per-paper losses within the
recorded tolerance.

## Completed comparative runs

Both methods started freshly from the pinned NF4 base and ran 18 optimizer
updates with validation/checkpoints at steps 6, 12 and 18. Selection included the
unadapted step-zero baseline, used the equal mean of the two per-paper target-token
losses, and broke ties toward the earliest step.

| Method | Loss at 0 | Loss at 6 | Loss at 12 | Loss at 18 | Selected |
|---|---:|---:|---:|---:|---:|
| JSON-only | 0.075139 | **0.057769** | 0.074680 | 0.085207 | 6 |
| Structure-aware | 0.075139 | **0.060714** | 0.061765 | 0.068738 | 6 |

The JSON-only run exposed 72 D examples (nine passes through its eight rows). The
structure-aware run exposed 24 examples for each of D, E and R (three passes
through its 24 rows). At the selected step 6, those counts were 24 D examples
versus 8 D + 8 E + 8 R. Actual supervised and total token counts are recorded in
each `selection.json` and `training_summary.json`.

| Method | Output | MLflow run | Wall time | Peak allocated / reserved |
|---|---|---|---:|---:|
| JSON-only | `outputs/finetune/json_only_seed42` | `a3e1f6bf2a494bb7bacb875a5e9edac1` | 47.47 min | 11.456 / 12.379 GiB |
| Structure-aware | `outputs/finetune/structure_aware_seed42` | `6c0cfa0975d546b086dd5b21c60e2cb4` | 48.71 min | 11.456 / 12.379 GiB |

Both MLflow runs are `FINISHED`. Each selected checkpoint contains adapter,
optimizer, scheduler, RNG, exposure, validation and precision files. MLflow also
contains the selected adapter plus `inference_precision.json`. Independent fresh
NF4-base reloads of both selected checkpoint-6 adapters reproduced the selected
per-paper losses exactly.

An external turn interruption terminated an earlier structure-aware attempt after
step-zero validation but before its first checkpoint. Its MLflow run
`93158d5e01a04e07a640edb939b6dd27` is marked `KILLED` with
`ExternalInterruptionBeforeCheckpoint`; local files are preserved under
`outputs/finetune/structure_aware_seed42_interrupted_before_checkpoint`. The
reported structure-aware result is the subsequent fresh run with the same
contract hash.

Step 6 did not use any test papers. Test-set M0/M1/M2 inference remains Step 7 and
must load the selected checkpoint-6 adapters with the recorded NF4 precision
policy.
