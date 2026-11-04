# Lightweight mechanistic fine-tuning diagnostics

**Status (2026-11-04): implementation complete and locally verified; scientific
execution pending on the machine containing the completed fine-tuning artifacts.**
This checkout has neither `outputs/finetune/` nor an accessible NVIDIA runtime,
so it contains no diagnostic results yet.

## Purpose

These exploratory diagnostics answer two narrower questions raised by the completed
D32 and oracle-inventory experiments:

1. How many reference errors require a missing dataset or target entity, and how
   many remain when both entities are available?
2. Under the same gold entity inventory, does explicit R supervision improve
   relationship linking beyond equal-update additional D training?

They use the thirty reused development papers and cannot select a new model or
replace the original held-out pilot result.

## Frozen comparisons

The existing one-call predictions for D24, D24+R8 and D32 are decomposed without
new generation. Every gold reference false negative is assigned to exactly one
category: dataset and target missing, dataset only missing, target only missing,
or link missing/incorrect although both entities align. Predicted false positives
are separated into unaligned dataset, unaligned target and wrong-link categories.

The oracle comparison supplies the same gold entity inventory and R prompt to:

| Condition | Exposure | Role |
|---|---:|---|
| D24+R8 | 24 D + 8 R | Existing oracle outputs; reused |
| D32 | 32 D | Primary equal-update comparator |
| D24 | 24 D | Secondary shorter-training comparator |

The primary contrast is D24+R8 minus D32. It matches examples and optimizer
updates, but not supervised tokens or compute. The secondary D24 comparison helps
describe exposure effects. Paired paper-cluster bootstrap intervals retain all
three seeds for each resampled paper and are descriptive for this reused corpus.

## Execution

The checked-in configuration is
[`finetune_mechanistic_diagnostics.yaml`](../configs/finetune_mechanistic_diagnostics.yaml).
It freezes the completed experiment configurations and prevents these diagnostics
from attaching to the confirmatory MLflow runs.

Inspect the plan locally:

```powershell
uv run --locked --group finetune python scripts/run_finetune_mechanistic_diagnostics.py --mode plan
```

On the machine containing the completed output tree, run the CPU-only decomposition:

```bash
uv run --locked --group finetune python scripts/run_finetune_mechanistic_diagnostics.py --mode decompose
```

Verify the two oracle jobs without loading weights, then generate on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --group finetune python scripts/run_finetune_mechanistic_diagnostics.py --mode oracle --condition all --dry-run --offline
CUDA_VISIBLE_DEVICES=0 uv run --locked --group finetune python scripts/run_finetune_mechanistic_diagnostics.py --mode oracle --condition all --offline
uv run --locked --group finetune python scripts/run_finetune_mechanistic_diagnostics.py --mode compare
```

Generation resumes from an ordered paper prefix. `--score-only` rescans completed
generations without loading a model. Outputs are written under
`outputs/finetune/followup/mechanistic-diagnostics/`.

## Interpretation limits

An oracle advantage for D24+R8 would support an R-supervision effect under the R
prompt, but could also include familiarity with that task format. Gold inventories
differ from predicted inventories in correctness, IDs, ordering and attribute
completeness. Neither diagnostic provides an independent generalization result.
