# Qwen3.5-4B fine-tuning pilot: experiments and results

## Executive summary

This pilot tested whether a 4-billion-parameter model could be adapted on one
16 GB GPU for one-call metadata extraction from scientific papers. The model was
`Qwen/Qwen3.5-4B`; training used QLoRA because full-BF16 LoRA did not fit the
longest examples. The frozen data split contained eight training papers, two
validation papers and thirty held-out test papers.

### Experiments at a glance

| Stage | What was compared | Scale | Main purpose and outcome |
|---|---|---:|---|
| Original pilot | M0 base model, M1 complete-JSON training, and M2 complete JSON plus entity/reference tasks | Two adapters, seed 42, 18 updates each | M1 was the strongest usable model; M2 learned entities but systematically omitted both dataset-reference fields. |
| Controlled ablation | Eight D/E/R/P task mixtures with matched-D and fixed-update controls | 24 runs: 8 conditions × 3 seeds | Adequate complete-JSON exposure and explicit reference supervision repaired the omission; the original D8/E8/R8 allocation remained weakest. |
| Development evaluation | Fixed endpoints for D24, D24+R8, D24+E8+R8 and D24+P8 | 12 adapters × 30 papers = 360 generations | Every output was valid JSON and retained both reference fields; no revised method dominated all metrics. |
| Oracle-inventory diagnostic | R inference with gold, self-predicted and D24-predicted entity inventories | 3 D24+R8 adapters × 3 inventories × 30 papers = 270 generations | R linked gold entities well, but predicted entity and dataset errors caused large downstream losses; a second R pass did not repair them. |
| D32 additional-training control | D32 versus the existing D24 and D24+R8 fixed endpoints | 3 new adapters × 30 papers = 90 new generations | D32 led D24+R8 on four of five structural fields, but missed the frozen sample-recovery margin; Gate B was inconclusive and Phase C was stopped. |

Here, D is complete-JSON supervision, E is entity-only supervision, R is the
original reference task, and P is a reference-only target shaped like the final
JSON. All production and ablation training used the same QLoRA, LoRA rank 16,
learning rate `1e-4`, microbatch 1 and four-step gradient accumulation unless the
declared task schedule required a different number of optimizer updates.

### Main results

- **M1 is the preferred model from the original held-out pilot.** Relative to M0,
  it increased sample F1 from 0.640 to 0.765, dataset F1 from 0.541 to 0.800,
  sample-reference F1 from 0.743 to 0.795 and facility-reference F1 from 0.700
  to 0.734. It preserved the full schema on 28/30 papers, although facilities F1
  decreased from 0.753 to 0.718 and inference was about 1.85 times slower.
- **The original M2 is not usable for full-schema extraction.** It achieved
  sample F1 0.752, dataset F1 0.790 and the best generic-metadata score (0.923),
  but omitted `sample_refs` and `facility_refs` on all thirty test papers, giving
  0/30 full-schema outputs and zero F1 for both reference fields.
- **The ablations indicate that M2's problem was the training allocation, not an
  inherently invalid reference task.** At M2's selected checkpoint, it had seen
  only eight complete-JSON examples versus M1's 24. D24+R8 and D24+P8 preserved
  both reference fields in all six two-paper/three-seed outputs, whereas the
  original D8+E8+R8 allocation preserved them in only 2/6 outputs.
- **The thirty-paper development evaluation confirmed that the revised mixtures
  repair the structural failure.** All 360 generations were valid JSON and kept
  both reference fields; 337/360 were fully schema valid, with the remaining
  failures caused only by null funding grant IDs.
- **The revised methods make different trade-offs.** D24 was best for samples,
  datasets and sample references. D24+R8 was the best-balanced structure-aware
  candidate, improving facility and facility-reference performance and reducing
  seed variability at the cost of some sample-side quality. D24+E8+R8 emphasized
  facilities and generic metadata. D24+P8 showed no clear advantage over D24+R8.
- **The oracle-inventory diagnostic locates the main practical bottleneck
  upstream of relationship generation.** With gold entities, R reached sample-
  and facility-reference F1 of 0.864 and 0.956. Self-predicted inventories reduced
  end-to-end scores to 0.538 and 0.624; D24-predicted inventories gave 0.574 and
  0.603. The R task has learned useful linking behavior, but it cannot recover
  entities or datasets missing from its supplied inventory.
- **The D32 control did not establish an R-specific benefit or a decisive D-only
  result.** D32 exceeded D24+R8 on facilities, samples, sample references and
  facility references, while D24+R8 was 0.009 higher on datasets. The R-related
  advantages required for Phase C failed every frozen check, but D32's 0.021
  sample-F1 recovery missed the required 0.03 margin. Gate B was therefore
  inconclusive.

### Decision

Keep **M1** as the preferred pilot model because it won the original predeclared
held-out comparison. The completed Gate B result does not justify promoting
D24+R8 or D32 to a new winner and does not authorize the Phase C schedule runs.
The ablation and follow-up evaluations are exploratory because those thirty
papers had already influenced the experiment design. Expand development data
with newly annotated papers, preserve a separate independent test set, and make
no new model choice until those data are frozen.

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
5. **Rebalanced structure-aware training repairs the missing-reference failure,**
   but the completed development evaluation does not show a general improvement
   over D-only or M1.
6. **Gate B is inconclusive and stops the schedule branch.** D32 did not satisfy
   the frozen simpler-model rule, while D24+R8 failed every requirement for an
   R-specific effect. Additional conclusions require newly annotated development
   data rather than more schedules on the reused corpus.

## 6. Exploratory controlled ablation matrix

We ran the frozen eight-condition matrix with seeds 42, 123 and 456: **24/24
runs completed**, with no generation exception or output-limit failure. Analysis
uses each run's predeclared fixed-exposure endpoint rather than substituting the
selected base checkpoint. This matters because the lexicographic selection rule
chose the trained endpoint in only 14/24 runs and was often influenced by
unrelated null funding IDs.

The scope is deliberately limited to the two validation papers. The previously
evaluated thirty-paper test set was not used.

| Condition | Exposure | Endpoint selected | Full schema | Reference fields present | Reference F1 | Datasets F1 | Validation NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| D only | D24 | 1/3 | 1/6 | 6/6 | 0.917 ± 0.000 | 0.726 ± 0.049 | 0.0575 ± 0.0002 |
| Matched-D DE | D24+E8 | 1/3 | 2/6 | 5/6 | 0.694 ± 0.255 | 0.715 ± 0.068 | 0.0567 ± 0.0003 |
| Matched-D DR | D24+R8 | 2/3 | 2/6 | 6/6 | 0.972 ± 0.048 | 0.666 ± 0.095 | 0.0562 ± 0.0009 |
| Matched-D DER | D24+E8+R8 | 2/3 | 2/6 | 6/6 | 0.944 ± 0.048 | 0.693 ± 0.054 | 0.0563 ± 0.0005 |
| Fixed-update DE | D12+E12 | 2/3 | 4/6 | 4/6 | 0.583 ± 0.382 | 0.677 ± 0.000 | 0.0581 ± 0.0012 |
| Fixed-update DR | D12+R12 | 2/3 | 3/6 | 6/6 | 0.969 ± 0.053 | 0.695 ± 0.053 | 0.0580 ± 0.0005 |
| Fixed-update DER | D8+E8+R8 | 1/3 | 1/6 | 2/6 | 0.333 ± 0.289 | 0.694 ± 0.029 | 0.0601 ± 0.0015 |
| Matched-D DRP | D24+P8 | 3/3 | 3/6 | 6/6 | 0.972 ± 0.048 | 0.703 ± 0.045 | 0.0561 ± 0.0007 |

Values are mean ± sample standard deviation over three seeds. Full-schema counts
cover six endpoint outputs per condition. “Reference fields present” separates
the original M2 failure from unrelated schema errors. Reference F1 is the
selection-compatible macro across both fields and papers; undefined scores count
as zero.

### Interpretation

- **The original R task works when D exposure is adequate.** D24+R8 preserved
  both reference fields in 6/6 outputs and reached reference F1 0.972. Adding E
  while retaining D24+R8 remained structurally complete with reference F1 0.944.
- **P was stable but not clearly superior to R.** D24+P8 selected its endpoint in
  all three seeds and reached 6/6 reference completeness and reference F1 0.972,
  exactly matching D24+R8 on the latter two measures. Its paired dataset-F1
  advantage over D24+R8 was 0.037, based on only two papers.
- **E without sufficient reference counter-supervision was unstable.** D24+E8
  fell to reference F1 0.694, while D12+E12 fell to 0.583; both had missing
  reference fields in at least one endpoint output.
- **The original fixed-budget mixture reproduced the failure mechanism.**
  D8+E8+R8 preserved reference fields in only 2/6 outputs and had the lowest
  reference F1 (0.333) and highest validation NLL (0.0601). In contrast,
  D12+R12 remained complete in 6/6 outputs with reference F1 0.969.
- Facilities F1 (0.929) and samples F1 (1.000) were identical across every
  endpoint condition on these two papers, so they provide no useful separation.

Across all 48 endpoint outputs, 48 were valid JSON, 41 retained all required
reference fields, and 18 were fully schema valid. The schema failures included
54 null `funding[].grant_id` entries and 70 missing reference-field errors (35
for each field). Missing-reference errors occurred only in DE and fixed-budget
DER conditions; R-, P- and D-only endpoints retained the reference structure.

Mean runtime ranged from 18.1 to 31.3 minutes per run. Every condition used about
11.46 GiB peak allocated VRAM. Fixed-update conditions consumed similar total
tokens but substantially different supervised-token counts, so they are matched
controls for update/example budget rather than literal training signal.

The exploratory evidence supports **insufficient D exposure plus interference
from E** as the best current explanation for M2's reference omission. R can
counter that pressure, and changing R to P is not necessary to recover reference
fields in this tiny validation set. These findings should guide a future study;
they do not replace evaluation on a newly collected independent test set.

### Thirty-paper exploratory development evaluation

We then evaluated the fixed endpoints of D24, D24+R8, D24+E8+R8 and D24+P8 for
all three seeds on the thirty Step 7 papers: **12 adapters and 360 generations**.
The checkpoints, shortlist and metrics were frozen before this evaluation. These
papers are nevertheless development data—not an independent test—because their
earlier M0/M1/M2 results influenced the follow-up design.

All 360 generations completed successfully:

- 360/360 valid JSON;
- 360/360 retained both `sample_refs` and `facility_refs`;
- no generation exception, output-limit failure or scoring fallback;
- 337/360 full-schema valid outputs; every remaining schema error was a null
  `funding[].grant_id`, not a missing reference field.

| Fixed endpoint | Full schema | Facilities F1 | Samples F1 | Datasets F1 | Sample refs F1 | Facility refs F1 | Generic metadata | Latency/paper |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D24 | 29.0 ± 0.0 / 30 | 0.691 ± 0.077 | **0.786 ± 0.007** | **0.789 ± 0.019** | **0.814 ± 0.023** | 0.725 ± 0.088 | 0.914 ± 0.014 | 145.3 ± 1.2 s |
| D24+R8 | 27.7 ± 0.6 / 30 | 0.764 ± 0.015 | 0.738 ± 0.007 | 0.771 ± 0.021 | 0.755 ± 0.011 | 0.770 ± 0.026 | 0.921 ± 0.013 | 141.9 ± 2.9 s |
| D24+E8+R8 | 27.7 ± 0.6 / 30 | **0.785 ± 0.019** | 0.702 ± 0.030 | 0.769 ± 0.023 | 0.732 ± 0.021 | **0.778 ± 0.034** | **0.924 ± 0.024** | **137.4 ± 2.0 s** |
| D24+P8 | 28.0 ± 1.0 / 30 | 0.752 ± 0.016 | 0.732 ± 0.029 | 0.745 ± 0.039 | 0.760 ± 0.023 | 0.775 ± 0.031 | 0.922 ± 0.015 | 143.6 ± 6.3 s |

Values are mean ± sample standard deviation across seeds 42, 123 and 456.
Entity and reference values are pooled evaluator F1 within each run, then
summarized across seeds. Full-schema counts are per run.

The development evaluation changes and sharpens the two-paper interpretation:

- **The structural problem is repaired.** Every revised fixed endpoint retained
  reference fields on all thirty papers and all three seeds, unlike original M2's
  0/30 full-schema result caused by systematic reference omission.
- **D-only remains the strongest sample-side configuration.** It led samples,
  datasets and sample-reference F1 and had the best full-schema rate. Its weakness
  was facilities and facility references, especially seed 456.
- **D+R is the best-balanced structure-aware variant.** Relative to D-only it
  raised facilities by 0.073 and facility-reference F1 by 0.045, while lowering
  samples by 0.049, datasets by 0.018 and sample-reference F1 by 0.059. Its simple
  five-field macro was 0.760 versus 0.761 for D-only, effectively the same at this
  descriptive level.
- **E shifts performance toward facilities, not a uniform improvement.** D+E+R
  had the best facilities, facility-reference and generic-metadata means, but the
  lowest samples and sample-reference scores among the three structure-aware
  variants. Thus E no longer destroys structure when D and R are preserved, but
  it changes what the model prioritizes.
- **P does not outperform R.** D+P was slightly higher than D+R on both reference
  means, but lower on facilities, samples and datasets and more variable for
  datasets and latency. The final-shaped target is viable but not necessary.
- **Structure-aware training improved seed stability for facility behavior.**
  D-only facilities ranged from 0.608 to 0.759 and facility-reference F1 from
  0.623 to 0.778. The R/E/P variants had substantially smaller standard
  deviations for those measures.

For historical context, one-seed M1 scored facilities 0.718, samples 0.765,
datasets 0.800, sample references 0.795 and facility references 0.734. The
five-field balance of M1, three-seed D-only and D+R is very similar, but they make
different trade-offs. M1 remains the preferred pilot choice because it was the
predeclared winner of the original held-out comparison; this reused-corpus
analysis should not be used to retroactively replace it.

The practical research candidate is **D24+R8** when facility and relationship
robustness matter, while D-only remains preferable when sample/dataset recovery
is the priority. D24+E8+R8 is worth considering only when facilities and generic
metadata are explicitly more important. A newly collected independent test set
is required before selecting among them for a confirmatory claim.

## 7. Step 9 Phase A2: oracle-versus-predicted R diagnostic

We reused the three fixed D24+R8 adapters and ran the R prompt with three frozen
entity inventories: gold entities, each adapter's own one-call prediction, and
the corresponding D24 prediction. Each condition covered all thirty development
papers and seeds 42, 123 and 456, for **270 R generations**. This is an
exploratory diagnostic on reused development papers, not a new held-out test.

| Entity inventory | R contract valid | Conditional sample refs | Conditional facility refs | End-to-end sample refs | End-to-end facility refs |
|---|---:|---:|---:|---:|---:|
| Gold (`oracle`) | 28.7 ± 0.6 / 30 | **0.864 ± 0.013** | **0.956 ± 0.021** | **0.864 ± 0.013** | **0.956 ± 0.021** |
| D24+R8 one-call output (`self_predicted`) | 29.0 ± 1.0 / 30 | 0.724 ± 0.025 | 0.759 ± 0.040 | 0.538 ± 0.022 | **0.624 ± 0.052** |
| D24 one-call output (`d24_predicted`) | 28.3 ± 0.6 / 30 | **0.773 ± 0.066** | 0.720 ± 0.093 | **0.574 ± 0.081** | 0.603 ± 0.106 |

Values are mean ± sample standard deviation over the three seeds. Conditional
F1 scores links on evaluator-aligned datasets; end-to-end F1 additionally
penalizes missing or incorrect datasets and target entities. The oracle entity
F1 values are 1.0 by construction because gold inventories were supplied; they
do not mean that the model extracted entities perfectly.

### Findings

- **R has learned a useful relationship function.** Oracle sample-reference F1
  was 0.864 and facility-reference F1 was 0.956, with little seed variation.
  This rules against a generally ineffective R task as the main explanation.
- **Predicted inventories are the dominant practical bottleneck.** Relative to
  oracle, self-predicted inventories lowered conditional sample/facility scores
  by 0.140/0.198 and end-to-end scores by 0.326/0.333. D24 inventories lowered
  the same scores by 0.091/0.236 and 0.290/0.353. The larger end-to-end gaps show
  the additional cost of missing, incorrect or unaligned entities and datasets.
- **A separate R pass does not repair the one-call pipeline.** With the
  D24+R8 adapter's own inventory, conditional R scores (0.724 sample, 0.759
  facility) were no better than its existing one-call development scores (0.755
  and 0.770). The current evidence therefore does not support output integration
  as the primary failure mode.
- **Neither predicted inventory dominates.** Compared with D24 inventories,
  self-predicted inventories were better for facilities (+0.073 entity F1 and
  +0.020 end-to-end facility-reference F1) but worse for samples (-0.049 entity
  F1 and -0.037 end-to-end sample-reference F1). Reference differences varied
  substantially by seed, especially for D24-predicted inventories. This mirrors
  the facility-versus-sample trade-off already observed in the one-call models.
- **Generation was reliable but the R contract was not perfect.** All 270
  outputs were valid JSON, with no generation or output-limit failures. Twelve
  outputs failed the stricter requirement to cover every supplied dataset/field
  exactly once and in order, leaving 258/270 contract-valid outputs. These cases
  used the predeclared empty-reference fallback and remained in the scores.

The diagnostic supports continuing to the predeclared D32 additional-training
control: R can learn links under a clean inventory, but Phase A2 alone cannot
show whether D24+R8's facility gains come from R-specific transfer or simply two
additional optimizer updates. It does not justify deploying a two-stage R pass.

## 8. Step 9 Phase B: D32 additional-training control

We trained D32 for seeds 42, 123 and 456 with the same eight complete-JSON
records repeated over eight optimizer updates. The fixed endpoint was evaluated
on the same thirty reused development papers, producing **90/90 new generations**.
The evaluator reused the 180 existing D24 and D24+R8 outputs and applied the
numeric Gate B thresholds frozen before D32 evaluation.

| Condition | Full schema | Facilities F1 | Samples F1 | Datasets F1 | Sample refs F1 | Facility refs F1 |
|---|---:|---:|---:|---:|---:|---:|
| D24 | 29.0 ± 0.0 / 30 | 0.691 ± 0.077 | **0.786 ± 0.007** | **0.789 ± 0.019** | **0.814 ± 0.023** | 0.725 ± 0.088 |
| D24+R8 | 27.7 ± 0.6 / 30 | 0.764 ± 0.015 | 0.738 ± 0.007 | 0.771 ± 0.021 | 0.755 ± 0.011 | 0.770 ± 0.026 |
| D32 | 28.3 ± 0.6 / 30 | **0.778 ± 0.031** | 0.759 ± 0.036 | 0.763 ± 0.015 | 0.785 ± 0.042 | **0.782 ± 0.024** |

Values are mean ± sample standard deviation over the three seeds. All conditions
retained both reference fields on all thirty papers per seed. D32 and D24+R8 are
matched on examples and optimizer updates, but not supervised tokens or compute.

For the primary D24+R8-minus-D32 contrast, the mean deltas were -0.014 facilities,
-0.021 samples, +0.009 datasets, -0.030 sample references and -0.012 facility
references. Facility and facility-reference deltas were positive in only one
seed, improved fewer papers than they worsened, and exceeded the permitted
single-paper concentration. Consequently, every frozen Phase C check failed.
D32 satisfied the facility-equivalence checks and the sample-reference recovery
check, but its 0.021 sample-F1 advantage did not reach the required 0.03.

The frozen Gate B outcome is **inconclusive**. Phase C must not run. All forty
currently annotated papers are already used by the frozen training, validation
and reused-development splits, so the next step requires newly annotated papers.
Those additions should enlarge development data while a separate representative
independent test set remains untouched.

## 9. Limitations and future work

- The original M0/M1/M2 comparison used one seed; the exploratory ablation used
  three seeds.
- Checkpoint selection and the exploratory ablation used only two validation papers.
- The thirty-paper follow-up is a reused development evaluation: the corpus had
  already influenced method design, so its uncertainty is not independent.
- M1 and M2 had unequal task and supervised-token exposure.
- The broader project had previously used this benchmark, so it is not equivalent
  to a newly collected external benchmark.
- Test results must not be used to revise the reported checkpoints.

The controlled matrix suggests preserving D exposure and avoiding an E-heavy
mixture unless explicit reference supervision is retained. A confirmatory study
requires a larger development set and a newly collected independent test set; it
must not be tuned against the thirty previously evaluated papers.

## 10. Reproducibility and artifacts

- Workflow and status: [`finetune_plan.md`](finetune_plan.md)
- Hardware and training details: [`finetune_step6_report.md`](finetune_step6_report.md)
- Held-out evaluation audit: [`finetune_step7_report.md`](finetune_step7_report.md)
- Reproducible post-hoc audit: `scripts/analyze_finetune_results.py`
- Post-hoc audit data: `outputs/finetune/evaluation/m1_audit.json`
- Reproducible ablation aggregate: `scripts/analyze_finetune_ablation.py`
- Ablation aggregate data: `outputs/finetune/ablations/analysis.json`
- Ablation aggregate tables: `outputs/finetune/ablations/analysis.md`
- Ablation development-evaluation runner: `scripts/evaluate_finetune_ablation.py`
- Frozen development-evaluation configuration:
  `configs/finetune_ablation_evaluation.yaml`
- Twelve-run comparison: `outputs/finetune/ablation-development-evaluation/comparison.json`
- Human-readable comparison:
  `outputs/finetune/ablation-development-evaluation/comparison.md`
- Step 9 Phase A2 evaluator: `scripts/evaluate_finetune_followup.py`
- Frozen Phase A2 configuration: `configs/finetune_followup_a2.yaml`
- Phase A2 machine-readable comparison:
  `outputs/finetune/followup/phase-a2/comparison.json`
- Phase A2 human-readable comparison:
  `outputs/finetune/followup/phase-a2/comparison.md`
- Phase B training configuration: `configs/finetune_followup_b.yaml`
- Phase B evaluation configuration: `configs/finetune_followup_b_evaluation.yaml`
- Phase B machine-readable comparison:
  `outputs/finetune/followup/phase-b/evaluation/comparison.json`
- Phase B human-readable comparison:
  `outputs/finetune/followup/phase-b/evaluation/comparison.md`
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
| D32 seed 42 | `2732f835809248379bcda44693559353` | FINISHED |
| D32 seed 123 | `8ce7f83431a74396a7da03d625777a6c` | FINISHED |
| D32 seed 456 | `de6c74f266944774a775e5bd5766802d` | FINISHED |

