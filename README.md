# Meta Data Extraction

Code and data for metadata extraction from photon and neutron science papers.
The benchmark contains 40 papers: 10 for calibration and 30 for evaluation,
with 22 experimental conditions evaluated on the same 30 papers.
Released predictions and results allow evaluation without PDFs or model access.

## Setup

Use Python 3.13 or later and run commands from the repository root:

```bash
uv sync --frozen
```

## Released contents

| Path | Contents |
|---|---|
| `configs/` | 22 experiment configs, scoring plan, and alignment helper |
| `data/corpus.json` | Paper IDs, strata, splits, and anonymous annotation pairs |
| `data/gold/` | 40 reconciled gold records |
| `data/annotations/` | 60 independent annotations of the 30 evaluation papers |
| `data/experiment/` | 660 predictions: 22 conditions × 30 papers |
| `data/few_shot/` | 3 fixed calibration examples with parsed text and expected answers |
| `docs/` | Annotation guidelines |
| `schemas/` | Schemas and prompt templates |
| `src/` | Extraction, normalization, alignment, and evaluation code |
| `scripts/` | Experiment runners and paper-result builders |
| `tests/` | Regression and paper-result checks |
| `outputs/` | Evaluation reports, inter-annotator agreement (IAA), and paper results |
| `paper/` | Manuscript sources, tables, and figures |


## Reproduce the paper results

```bash
uv run python scripts/run_evaluation.py
uv run python scripts/compute_iaa.py
uv run python scripts/compute_paper_results.py
uv run python scripts/build_paper_tables.py
uv run python scripts/build_paper_figures.py
```

The first three commands recompute and replace the released results:

- `outputs/evaluation/`: 660 per-paper evaluation reports.
- `outputs/iaa/`: 30 annotation comparisons and pooled `summary.json`.
- `outputs/paper/results.json`: benchmark counts, scores, and comparisons.

The last two commands read `results.json` and regenerate six numerical tables
in `paper/tables/` and `paper/figures/few_shot.pdf`. They can run directly from
the released results.

Optional verification: `uv run python -m pytest`.

## Optional: run new extraction

Obtain PDFs using the IDs in `data/corpus.json` and place them at
`data/raw/<arxiv_id>/paper.pdf`. Copy an experiment config to `tmp/my_run.yaml`,
set `llm.base_url` and `llm.model` for your endpoint, and set
`tracker.output_dir` to `data/processed` to keep new runs separate from the release.
Hosted configs contain a placeholder endpoint and gateway-specific model IDs.

Copy `.env.example` to `.env` and set the credential named by `llm.api_key_env`;
the extraction pipeline loads this file automatically. Then try one paper:

```bash
uv run python scripts/run_experiment.py --config tmp/my_run.yaml --limit 1
```

Remove `--limit 1` to run all 30 evaluation papers. Results go under
`data/processed/my_run/`; existing results are skipped unless `--force` is used.
See `--help` for other options. The evaluation and paper scripts target the
22 published conditions; custom runs can use `evaluate()` in
[evaluation.py](src/meta_data_extraction/evaluation.py).


## Method

See the [manuscript](paper/main.tex),
[annotation guidelines](docs/annotation_guidelines.md),
[scoring plan](configs/evaluation_metrics.yaml), and
[alignment helper](configs/alignment_helper.yaml) for methodological details.
