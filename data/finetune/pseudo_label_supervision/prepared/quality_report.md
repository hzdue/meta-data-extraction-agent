# Pseudo-label preparation quality report

## Outcome

- Formal validation: 561/565 supervised predictions passed.
- Schema-invalid teacher labels excluded without repair: 4.
- Prepared D records: 504 train and 57 validation.
- Future test: 30 papers; teacher predictions were not read.
- Missing teacher predictions: 5 papers excluded from supervision.
- Parser warnings: 128 papers.
- Low word-presence flags: 15 papers.
- Context exclusions at 65,536 tokens: 0.

Formal validation covers complete nested shape, types, allowed values, cardinalities, unique IDs and reference integrity. It does not establish scientific correctness.

## Label flags

- `datasets_without_facilities`: 1
- `duplicate_names:datasets`: 6
- `duplicate_names:samples`: 3
- `high_count:samples`: 1
- `low_word_presence_pages`: 13
- `pdf_parser_warnings`: 116

## Entity-count distributions

- affiliations: {'minimum': 1, 'p50': 5, 'p95': 10, 'maximum': 19}
- authors: {'minimum': 0, 'p50': 9, 'p95': 18, 'maximum': 45}
- funding: {'minimum': 0, 'p50': 2, 'p95': 9, 'maximum': 23}
- facilities: {'minimum': 0, 'p50': 1, 'p95': 3, 'maximum': 6}
- samples: {'minimum': 1, 'p50': 2, 'p95': 8, 'maximum': 27}
- datasets: {'minimum': 1, 'p50': 3, 'p95': 7, 'maximum': 12}

## Stratified semantic-review sample

- Selected papers: 28 (four per subject group).
- Validation papers in sample: 8.
- Exact normalized evidence strings found: 289/326.
- Exact-match failures require inspection; they are not automatic label errors.
- Detailed sample: `review_sample.jsonl`.
- Codex-assisted review outcome: no additional exclusion was warranted.

A domain expert should independently review the sample before using the data for a paper-quality claim. The Codex-assisted review is sufficient for a provisional pilot once the GPU memory probe passes.
