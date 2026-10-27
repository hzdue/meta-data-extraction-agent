# Fine-tuning Step 1 audit

Frozen split: **8 training / 2 validation / 30 test papers**. The original corpus index is unchanged.

## Validation selection

- **2010.02965:** Spectroscopy complements the longest paper's diffraction stratum. Its 51,511 extracted characters are near the calibration median, with 3 samples, 6 datasets, 18 dataset-to-sample links and 2 dataset-to-facility links. The other spectroscopy paper remains in training, preserving all seven technique strata.
- **2503.18475:** Longest calibration paper by complete plain-text extraction (109,810 characters; also most pages at 65). Reserved for validation at the user's request after a prior RTX length-limit issue; diffraction remains represented in training by 2309.13729.

All seven calibration technique strata remain represented in training.

## Calibration inventory

| Paper | Split | Technique stratum | Pages | Text chars | Facilities | Samples | Datasets | Sample links | Facility links |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1806.08116 | train | multi_technique | 27 | 44,379 | 3 | 2 | 3 | 6 | 3 |
| 1812.03956 | train | reflectometry | 11 | 22,097 | 1 | 2 | 2 | 4 | 2 |
| 1912.11816 | train | small_angle_scattering | 8 | 39,093 | 0 | 1 | 1 | 1 | 0 |
| 2009.01093 | train | imaging_tomography | 32 | 60,733 | 2 | 1 | 5 | 5 | 5 |
| 2009.12925 | train | methods_light | 12 | 48,891 | 1 | 2 | 1 | 2 | 1 |
| 2010.02965 | validation | spectroscopy | 28 | 51,511 | 1 | 3 | 6 | 18 | 2 |
| 2104.10884 | train | spectroscopy | 13 | 61,502 | 1 | 9 | 1 | 9 | 1 |
| 2304.08002 | train | small_angle_scattering | 11 | 59,544 | 3 | 8 | 4 | 21 | 3 |
| 2309.13729 | train | diffraction | 12 | 61,904 | 2 | 4 | 4 | 13 | 2 |
| 2503.18475 | validation | diffraction | 65 | 109,810 | 4 | 3 | 6 | 17 | 4 |

## Checks and scope

- All 40 PDFs and gold records exist, open successfully, and have SHA-256 fingerprints in `splits.json`.
- All ten calibration records pass field/type/cardinality/enum/pattern, sequential-ID, and reference checks. This is a structural audit, not a new scientific annotation review.
- Test gold is used only for JSON readability and bibliographic duplicate checks; no test targets, semantic scores, or annotation statistics are used for selection.
- No duplicate candidates were found by PDF hash, normalized text hash, normalized title, DOI, unversioned arXiv identity, or title similarity >= 0.90. This local screening cannot rule out every renamed publication version.
- Schema versions: `{'generic.json': '1.1.0', 'photon_and_neutron.json': '1.1.0'}`; scoring-plan version: `1.1.3`. Schema and scoring declarations agree.
- Source hashes also cover the corpus index, schemas, guidelines, prompt, parser/evaluator code, audit code, and evaluation configuration. Training dependencies and run provenance will be recorded in later steps; adding those dependencies does not change split membership.
- PDF hashes cover raw bytes. Text-source hashes and byte counts normalize only line endings to LF, allowing Windows/Linux checkouts without hiding other content changes.

## Recoverable PDF parser warnings

Text extraction completed, but the following PDFs emitted warnings. Review their final Markdown in Step 2; successful extraction does not guarantee faithful rendering.

- **1812.03956:** bogus font ascent/descent values (40 / -2359)
- **1901.08819:** bogus font ascent/descent values (0 / -207)
- **2006.12074:** syntax error: unknown keyword: 'pagesize'; syntax error: unknown keyword: 'width'; syntax error: unknown keyword: '597.50787pt'; syntax error: unknown keyword: 'height'; syntax error: unknown keyword: '845.04684pt'; bogus font ascent/descent values (0 / -207); encountered syntax errors; page may not be correct
- **2304.08002:** premature end of data in flate filter; ... repeated 3 times...
- **2309.13729:** bogus font ascent/descent values (0 / -252); bogus font ascent/descent values (0 / -209)
- **2407.10282:** syntax error: could not parse color space (252 0 R); Ignoring Page blending colorspace.; syntax error: could not parse color space (1410 0 R); Ignoring Page blending colorspace.
- **2605.14792:** bogus font ascent/descent values (3117 / -2464); ... repeated 2 times...

## Length and hardware follow-up

Lengths use PyMuPDF 1.27.2.2 plain-text extraction of the complete PDFs, including appended material. They are not Qwen token counts or the final Markdown lengths.

Longest paper is excluded from training at the user's request after an RTX length-limit issue. It must still fit full-JSON validation loss and generation; verify exact token length and hardware in later steps.

Step 2 must count complete prompts and targets with the pinned 4B tokenizer. Step 6 must test backward passes only on training examples and forward-only validation on both validation papers. Do not truncate, omit, or train on the long validation paper to work around a resource limit.

The two-paper validation set intentionally includes a length outlier and cannot represent the entire domain. Use mean per-paper validation loss as planned. The existing test benchmark has been used previously and is not a new independent test set.

## Verify without rewriting

```powershell
uv run --no-sync python scripts/audit_finetune_data.py --verify
```

Verification recomputes the audit, checks the fixed 8/2/30 membership and all fingerprints, and fails on drift. The audit command without flags is read-only; `--freeze` refuses to replace existing artifacts.
