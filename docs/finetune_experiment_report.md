# Qwen3.5-4B fine-tuning pilot: experiments and results

## Executive summary

We compared two LoRA fine-tuning strategies for one-call metadata extraction from
scientific papers:

- **M0 — Base:** unmodified `Qwen/Qwen3.5-4B`.
- **M1 — JSON-only:** trained only to produce the complete target JSON.
- **M2 — Structure-aware:** trained on complete JSON plus separate entity and
  reference tasks.

Both adapters improved sample and dataset recovery on the thirty-paper held-out
test set. **M1 was the strongest usable model overall:** dataset F1 increased from
0.541 to 0.800 and sample F1 from 0.640 to 0.765, while both reference metrics
also improved. M2 achieved similarly strong entity extraction and the highest
generic-metadata score, but it omitted both dataset reference fields on every
test paper. It therefore failed the required full-schema output contract.

## 1. Experimental design

### Data split

The split was frozen before training and remained unchanged:

| Split | Papers | Purpose |
|---|---:|---|
| Training | 8 | Adapter optimization |
| Validation | 2 | Checkpoint selection only |
| Test | 30 | Final comparison only |

The same parsed paper text, schema, tokenizer, chat template and full-context
policy were used for all methods. Test gold was not used during preparation,
training or checkpoint selection.

### Training tasks

| Task | Input | Target | Used by |
|---|---|---|---|
| D | Paper and full schema | Complete JSON | M1 and M2 |
| E | Paper and entity instructions | Entities and attributes without references | M2 |
| R | Paper, instructions and gold entity inventory | Dataset reference fields | M2 |

M1 contained eight D records. M2 contained 24 records—eight each of D, E and R—
sampled in a 1:1:1 ratio. Validation and test inference always used the D prompt;
gold entities were never supplied at inference time.

### Shared model and optimization settings

- Model: `Qwen/Qwen3.5-4B`
- Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank/alpha/dropout: 16 / 32 / 0.05
- Learning rate: `1e-4`, constant schedule
- Microbatch / gradient accumulation: 1 / 4
- Maximum context: 65,536 tokens, without truncation or packing
- Generation budget: 16,384 tokens, greedy decoding, thinking disabled
- Common training budget: 18 optimizer updates
- Validation/checkpoint interval: 6 updates
- Seed: 42

Both adapters started independently from the same base checkpoint. Checkpoint
selection minimized the equal mean of the two per-paper validation losses, with
ties resolved in favor of the earlier checkpoint.

## 2. Hardware and memory experiments

Experiments ran on one NVIDIA GeForce RTX 5070 Ti with 15.54 GiB of usable VRAM.

Initial BF16 LoRA experiments used CPU activation offload, 2,048-token MLP
chunking and optimized linear-attention kernels. The shortest training example
fit, but the longest complete-JSON example still ran out of memory. Full BF16
training was therefore not feasible on this machine.

We adopted one shared QLoRA configuration for M0, M1 and M2:

- NF4 double-quantized base weights
- BF16 computation
- BF16 frozen input/output embeddings
- CPU activation offload during training
- 2,048-token MLP chunks
- Optimized linear-attention kernels where available

QLoRA passed all required feasibility checks:

- Shortest and longest D training examples
- Longest E and R training examples
- Both validation-paper loss calculations
- Full-budget validation generation
- Interrupted-run resume
- Fresh base-plus-adapter reload

Peak allocation during production training was approximately 11.46 GiB for both
methods. The optional `causal_conv1d` extension was unavailable, so the correct
but slower PyTorch fallback was used consistently.

## 3. Training and checkpoint selection

Both methods completed 18 optimizer updates. Validation losses were:

| Method | Step 0 | Step 6 | Step 12 | Step 18 | Selected |
|---|---:|---:|---:|---:|---:|
| M1 JSON-only | 0.075139 | **0.057769** | 0.074680 | 0.085207 | 6 |
| M2 structure-aware | 0.075139 | **0.060714** | 0.061765 | 0.068738 | 6 |

Both selected checkpoint 6. Independent reloads onto a fresh NF4 base reproduced
the recorded per-paper validation losses exactly.

### Exposure and cost

| Method | Full-run examples | Selected-checkpoint examples | Full-run tokens | Training time |
|---|---|---|---:|---:|
| M1 | 72 D | 24 D | 1,862,089 | 2,846.6 s |
| M2 | 24 D + 24 E + 24 R | 8 D + 8 E + 8 R | 1,783,731 | 2,920.9 s |

The common update budget did not give equal complete-JSON exposure. At the
selected checkpoint, M1 had seen each D record three times; M2 had seen each D
record once, alongside one E and one R record per paper.

## 4. Held-out test results

All three conditions completed generation and scoring for all thirty test papers:

- 90 total generations
- 90/90 valid JSON responses
- No generation exceptions
- No output-limit failures
- No scoring fallbacks

Primary scores include every test paper and use the existing shared alignment and
evaluation rules.

| Metric | M0 base | M1 JSON-only | M2 structure-aware |
|---|---:|---:|---:|
| Full-schema valid | 28/30 | 28/30 | 0/30 |
| Generic metadata macro | 0.897 | 0.909 | **0.923** |
| Facilities F1 | **0.753** | 0.718 | 0.642 |
| Samples F1 | 0.640 | **0.765** | 0.752 |
| Datasets F1 | 0.541 | **0.800** | 0.790 |
| Dataset-to-sample reference F1 | 0.743 | **0.795** | 0.000 |
| Dataset-to-facility reference F1 | 0.700 | **0.734** | 0.000 |
| Mean generation latency per paper | **77.5 s** | 143.1 s | 143.2 s |
| Peak allocated GPU memory | 8.03 GiB | 8.16 GiB | 8.16 GiB |

### M1 interpretation

Relative to M0, M1 improved:

- Samples F1 by 0.125
- Datasets F1 by 0.259
- Sample-reference F1 by 0.052
- Facility-reference F1 by 0.034
- Generic metadata macro by 0.012

Facilities F1 declined by 0.035. At paper level, M1 improved or tied M0 on 26/30
papers for samples and 29/30 for datasets. Its main operational cost was roughly
1.85× slower generation.

### Post-hoc M1 audit (no retraining)

We audited the saved predictions and ran two forward-only validation passes
through the selected M1 checkpoint. No parameters, checkpoints or predictions
were changed.

#### Precision and recall

| Field | M0 precision | M0 recall | M1 precision | M1 recall |
|---|---:|---:|---:|---:|
| Facilities | 0.696 | **0.821** | **0.718** | 0.718 |
| Samples | 0.828 | 0.522 | **0.833** | **0.707** |
| Datasets | 0.635 | 0.471 | **0.853** | **0.753** |
| Sample references | 0.781 | 0.708 | **0.797** | **0.792** |
| Facility references | 0.656 | **0.750** | **0.763** | 0.707 |

M1's sample improvement is primarily a recall gain. Its dataset improvement is
strong in both precision and recall. Facility extraction traded recall for a
small precision increase, explaining its lower F1. Facility-reference F1 improved
through precision despite slightly lower recall.

Paired paper bootstrap intervals used 10,000 resamples with seed 42 and recomputed
pooled F1 counts within each resample:

| M1 − M0 | F1 difference | Paired 95% bootstrap interval |
|---|---:|---:|
| Facilities | -0.035 | [-0.154, 0.061] |
| Samples | +0.125 | **[0.007, 0.288]** |
| Datasets | +0.259 | **[0.135, 0.401]** |
| Sample references | +0.052 | [-0.138, 0.207] |
| Facility references | +0.034 | [-0.093, 0.181] |

The sample and dataset intervals exclude zero; the facilities and reference
intervals do not. These are descriptive uncertainty estimates over this fixed
thirty-paper corpus, not evidence from a new independent population.

#### Validation-loss audit

The unusually low validation loss is not caused by division over the long paper
prompt. Paper text, instructions, empty-thinking prefix, template newline and
padding are masked. Loss is the summed shifted NLL of target JSON tokens plus EOS,
divided by the supervised-token count; validation then takes the equal mean of
the two per-paper losses. The forward-only audit reproduced M1's selected losses:
0.064800 and 0.050739.

Tokens overlapping `sample_refs`/`facility_refs` keys and list values accounted
for 485 of 7,244 supervised validation tokens (**6.70%**) but only **1.07%** of
total validation NLL. Their mean NLL was 0.0091 versus 0.0606 for other target
tokens. Under teacher forcing, reference content is therefore highly predictable
once the correct JSON prefix and gold history are supplied. This explains why
overall NLL is a weak safeguard against free-generation structural omissions:
one early decision to close a dataset object can remove all subsequent reference
tokens even when their conditional token probabilities are excellent.

#### Schema failures and throughput

M1's two invalid papers contained three funding entries with a funder but
`grant_id: null`. In `1804.07163`, two unsupported funder-only entries were added
alongside valid funding pairs. In `2001.00707`, one funder-only entry was emitted
while gold funding was empty. These are semantic over-inclusions, not JSON parsing
failures; a production validator should flag or remove such entries explicitly.

M1 generated 2,156 tokens per paper on average versus 2,003 for M0 (+7.6%), but
throughput fell from 25.85 to 15.07 tokens/s (-41.7%). Output length therefore
explains only part of the 1.85× latency. The slowdown is implementation-specific
and should not be presented as an inherent LoRA cost without additional profiling.

### M2 interpretation

M2 improved generic metadata, sample recovery and dataset recovery over M0, but
every M2 prediction omitted `sample_refs` and `facility_refs` from every predicted
dataset. Its JSON was syntactically valid, yet none of the thirty predictions
satisfied the complete schema.

The training mixture offers a plausible explanation:

- E explicitly teaches dataset objects without reference fields.
- R predicts references in a separate task representation.
- Only D teaches references in their required final location.
- At the selected checkpoint, M2 had one-third of M1's D exposure.

Checkpoint selection used teacher-forced complete-JSON loss. Reference keys form
a small fraction of all supervised tokens, so overall loss could improve without
detecting systematic omission during autoregressive generation. This is a
plausible mechanism, not a causal conclusion from one seed.

M0 and M1 each had two schema-invalid papers caused only by null funding grant
IDs; neither showed M2's structural reference failure.

## 5. Conclusions

1. **M1 is the preferred model for this pilot.** It provides the best balanced
   improvement while preserving the complete extraction contract on 28/30 papers.
2. **M2 is not usable for one-call full-schema extraction in its current form.**
   Its strong entity scores do not compensate for systematic loss of references.
3. **QLoRA is necessary on the available 16 GB GPU** and was reproducible across
   training, resume, validation and adapter reload.
4. **Equal optimizer updates are not equal supervision.** M2 traded complete-JSON
   exposure for auxiliary tasks, which materially affected the output structure.

## 6. Limitations and future work

- Only one training seed was used.
- Checkpoint selection used two validation papers.
- M1 and M2 had unequal task and supervised-token exposure.
- The broader project had previously used this benchmark, so it is not equivalent
  to a newly collected external benchmark.
- Test results must not be used to revise the reported checkpoints.

A future structure-aware experiment could preserve M1's D exposure, increase D
weighting, or make the R representation match the final dataset structure. Such
changes require a new development protocol and an independent test set; they
should not be tuned against these thirty evaluated papers.

## 7. Reproducibility and artifacts

- Workflow and status: [`finetune_plan.md`](finetune_plan.md)
- Hardware and training details: [`finetune_step6_report.md`](finetune_step6_report.md)
- Held-out evaluation audit: [`finetune_step7_report.md`](finetune_step7_report.md)
- Reproducible post-hoc audit: `scripts/analyze_finetune_results.py`
- Post-hoc audit data: `outputs/finetune/evaluation/m1_audit.json`
- Frozen evaluation configuration: `configs/finetune_evaluation.yaml`
- Selected M1 checkpoint: `outputs/finetune/json_only_seed42/checkpoint-6`
- Selected M2 checkpoint: `outputs/finetune/structure_aware_seed42/checkpoint-6`
- Test artifacts: `outputs/finetune/evaluation/`

MLflow runs:

| Condition | Run ID | Status |
|---|---|---|
| M0 | `0f9789feb41e4207ab803ed1cb520a53` | FINISHED |
| M1 | `a3e1f6bf2a494bb7bacb875a5e9edac1` | FINISHED |
| M2 | `6c0cfa0975d546b086dd5b21c60e2cb4` | FINISHED |

