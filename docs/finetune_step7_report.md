# Step 7 held-out evaluation — 2026-10-29

## Status

Complete. M0, M1 and M2 each generated and were scored on all thirty frozen test
papers. The process exited successfully, all comparison artifacts are present,
and all three MLflow runs are `FINISHED` with evaluation metrics and artifacts
attached. No checkpoint or training setting was changed after seeing test results.

## Primary results

Primary scores include all thirty papers. Invalid or unscorable generations would
have been evaluated as empty predictions; no such fallback was needed. Entity and
reference F1 pool counts over the corpus. The generic score is the evaluator's
macro summary across bibliographic and document-overview fields.

| Metric | M0 base | M1 JSON-only | M2 structure-aware |
|---|---:|---:|---:|
| Valid JSON | 30/30 | 30/30 | 30/30 |
| Full-schema valid | 28/30 | 28/30 | 0/30 |
| Generic metadata macro | 0.897 | 0.909 | **0.923** |
| Facilities F1 | **0.753** | 0.718 | 0.642 |
| Samples F1 | 0.640 | **0.765** | 0.752 |
| Datasets F1 | 0.541 | **0.800** | 0.790 |
| Dataset-to-sample reference F1 | 0.743 | **0.795** | 0.000 |
| Dataset-to-facility reference F1 | 0.700 | **0.734** | 0.000 |
| Mean generation latency per paper | **77.5 s** | 143.1 s | 143.2 s |
| Peak allocated GPU memory | 8.03 GiB | 8.16 GiB | 8.16 GiB |

Relative to M0, M1 changed pooled F1 by -0.035 for facilities, +0.125 for
samples, +0.259 for datasets, +0.052 for sample references and +0.034 for
facility references. Its generic macro score increased by 0.012. At paper level,
M1 beat/tied/lost to M0 on 9/17/4 papers for samples and 16/13/1 for datasets;
facilities were mixed at 5/19/6.

M2 also improved sample and dataset recovery over M0, but every M2 prediction
omitted `sample_refs` and `facility_refs` from every predicted dataset. All thirty
responses were valid JSON, and the ordinary entity/metadata fields remained
scorable, but none satisfied the complete extraction schema. Consequently both
reference F1 scores are zero. The structure-aware training mixture is a plausible
mechanism: E explicitly presents datasets without reference fields while R
predicts references in a separate task representation, and D supplies only one
third of M2's examples. The single-seed pilot does not prove that explanation.

M0 and M1 each had two schema-invalid papers caused only by null funding grant IDs.
There were no generation exceptions, output-limit failures or scoring fallbacks
for any method.

## Training cost and interpretation

Both adapters ran 18 optimizer updates. M1's full run exposed 72 D examples and
1,862,089 total tokens; its selected checkpoint at step 6 had exposed 24 D
examples and 621,363 total tokens. M2's full run exposed 24 examples of each D/E/R
task and 1,783,731 total tokens; its selected checkpoint had exposed eight of each
task and 594,577 total tokens. Full training took 2,846.6 seconds for M1 and
2,920.9 seconds for M2.

M1 is the strongest usable fine-tuned model in this pilot: it substantially
improves sample and dataset extraction and modestly improves both reference
metrics, at the cost of a small facilities regression and roughly 1.85× inference
latency. M2 is not suitable for one-call full-schema extraction in its current
form despite good generic, sample and dataset scores, because its systematic
reference-field omission violates the required output contract.

These results are preliminary: checkpoint selection used only two validation
papers, there is one training seed, and equal update counts do not equalize task
or supervised-token exposure. Test results must not be used to revise these
checkpoints or tune settings. A future experiment may redesign M2 supervision,
but it must use a new development/test protocol rather than adapting to these
held-out results.

## Reproducibility and tracking

Local artifacts are under `outputs/finetune/evaluation/`, including thirty raw
responses, `ExtractionResult` predictions, per-paper reports and scores for each
condition, plus `comparison.json` and `comparison.md`.

| Condition | MLflow run | Status | Test metrics |
|---|---|---|---:|
| M0 | `0f9789feb41e4207ab803ed1cb520a53` | FINISHED | 62 |
| M1 | `a3e1f6bf2a494bb7bacb875a5e9edac1` | FINISHED | 62 |
| M2 | `6c0cfa0975d546b086dd5b21c60e2cb4` | FINISHED | 62 |

Each run contains the evaluation manifest, evaluator provenance, condition
artifacts and shared comparison artifact. Frozen test-gold fingerprints matched
the split audit during scoring.
