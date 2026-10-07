"""Compute the paper's numerical results from gold, evaluation, and IAA records.

Run after run_evaluation.py and compute_iaa.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __package__:
    from .corpus import load_corpus
    from .corpus import paper_ids as corpus_paper_ids
else:
    from corpus import load_corpus
    from corpus import paper_ids as corpus_paper_ids

from meta_data_extraction.evaluation import (  # noqa: E402
    DEFAULT_PLAN_PATH,
    aggregate_reports,
    build_evaluator_metadata,
    load_plan,
    validate_evaluator_provenance,
)

EVALUATION_DIR = ROOT / "outputs" / "evaluation"
IAA_SUMMARY_PATH = ROOT / "outputs" / "iaa" / "summary.json"
OUTPUT_PATH = ROOT / "outputs" / "paper" / "results.json"

PRIMARY_FIELDS = (
    ("Affiliations", "bibliographic.affiliations"),
    ("Authors", "bibliographic.authors"),
    ("Funding", "bibliographic.funding"),
    ("Facilities", "facilities"),
    ("Samples", "samples"),
    ("Datasets", "datasets"),
)

ZERO_SHOT_RUNS = {
    model: {"full": f"{prefix}_full_0shot", "no_inst": f"{prefix}_no_inst_0shot"}
    for model, prefix in (
        ("GPT-5.5", "pan_gpt5_5"),
        ("GPT-4o", "pan_gpt4o"),
        ("Mistral Large 3", "pan_mistral_large3"),
        ("Gemma 4 31B", "pan_gemma4_31b_local"),
        ("Qwen3.5 27B", "pan_qwen3_5_27b"),
        ("Qwen3.5 9B", "pan_qwen3_5_9b"),
        ("Qwen3.5 4B", "pan_qwen3_5_4b"),
    )
}
FEW_SHOT_LEVELS = (0, 1, 3)
FEW_SHOT_RUNS = {
    model: {
        shots: ZERO_SHOT_RUNS[model]["full"].replace("_0shot", f"_{shots}shot")
        for shots in FEW_SHOT_LEVELS
    }
    for model in ("GPT-5.5", "Qwen3.5 27B", "Qwen3.5 9B", "Qwen3.5 4B")
}


def load_metrics() -> dict[str, str]:
    """Read the scoring metric for each paper-level field and entity attribute."""
    metrics = {}
    for path, spec in load_plan(DEFAULT_PLAN_PATH).items():
        metrics[path] = spec[0]
        if spec[0] == "entity":
            metrics.update(
                (f"{path}.attrs.{attribute}", scoring[0])
                for attribute, scoring in spec[1].attr_scorers.items()
            )
    return metrics


def load_model_results(run_name: str) -> dict:
    """Validate and aggregate exactly the 30 evaluation reports for one condition."""
    papers = corpus_paper_ids("evaluation", corpus_path=ROOT / "data" / "corpus.json")
    if len(papers) != 30:
        raise ValueError(f"Expected 30 evaluation papers, found {len(papers)}.")
    report_dir = EVALUATION_DIR / run_name
    if not report_dir.is_dir():
        raise FileNotFoundError(f"Missing evaluation directory: {report_dir}")
    paths = {
        path.name.removesuffix("_evaluation.json"): path
        for path in report_dir.glob("*_evaluation.json")
    }
    if set(paths) != set(papers):
        raise ValueError(
            f"Run {run_name!r} does not exactly match the 30 evaluation papers; "
            f"missing={sorted(set(papers) - set(paths))}, "
            f"extra={sorted(set(paths) - set(papers))}."
        )

    expected = build_evaluator_metadata(DEFAULT_PLAN_PATH)
    reports = []
    for paper in papers:
        path = paths[paper]
        report = json.loads(path.read_text(encoding="utf-8"))
        validate_evaluator_provenance(report.get("evaluator", {}), expected, source=str(path))
        reports.append(report)
    return aggregate_reports(reports)


def load_evaluation_results() -> dict[str, dict]:
    """Load and pool all 22 conditions over the same 30 evaluation papers."""
    run_names = {
        name for runs in ZERO_SHOT_RUNS.values() for name in runs.values()
    } | {
        name for runs in FEW_SHOT_RUNS.values() for name in runs.values()
    }
    return {name: load_model_results(name) for name in sorted(run_names)}


def load_iaa_results(
    summary_path: Path | None = None,
    plan_path: Path | None = None,
) -> dict:
    """Read the complete pooled IAA result and validate its evaluator settings."""
    summary_path = summary_path if summary_path is not None else IAA_SUMMARY_PATH
    plan_path = plan_path if plan_path is not None else DEFAULT_PLAN_PATH
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("n_papers_total") != 30
        or summary.get("n_papers_scored") != 30
        or summary.get("n_papers_missing") != 0
        or summary.get("n_papers_failed") != 0
    ):
        raise ValueError(f"{summary_path}: expected complete formal N=30 IAA")
    validate_evaluator_provenance(
        summary.get("evaluator", {}), build_evaluator_metadata(plan_path),
        source=str(summary_path),
    )
    return summary["aggregated"]


def _metric_value(aggregate: dict, path: str, *, metric_key: str | None = None) -> float:
    """Select a field's score; heading order requires explicit 'mean_order'."""
    result = aggregate[path]
    if metric_key is not None:
        if metric_key not in {"f1", "mean_score", "mean_order"}:
            raise ValueError(f"Unsupported metric key {metric_key!r} for {path!r}.")
        value = result.get(metric_key)
        if value is None:
            raise ValueError(f"Aggregate result for {path!r} has no {metric_key!r}.")
        return float(value)
    for key in ("f1", "mean_score", "mean_order"):
        if result.get(key) is not None:
            return float(result[key])
    raise ValueError(f"Aggregate result for {path!r} has no paper-facing score.")


def compute_benchmark_statistics(root: Path = ROOT) -> dict[str, dict[str, int]]:
    """Count the 40 selected gold records and the annotation workload.

    All three annotators labelled calibration papers; two labelled each evaluation
    paper.
    """
    papers = load_corpus(root / "data" / "corpus.json")["papers"]
    calibration = sum(paper["split"] == "calibration" for paper in papers)
    evaluation = sum(paper["split"] == "evaluation" for paper in papers)
    if (len(papers), calibration, evaluation) != (40, 10, 30):
        raise ValueError("Expected 40 papers: 10 calibration and 30 evaluation.")

    entity_counts = dict.fromkeys(
        ("Affiliations", "Authors", "Funding items", "Facilities", "Samples", "Datasets"), 0
    )
    reference_counts = dict.fromkeys(
        ("Author--affiliation", "Dataset--sample", "Dataset--facility"), 0
    )
    for paper in papers:
        path = root / "data" / "gold" / f"{paper['arxiv_id']}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        bibliographic = record["bibliographic"]
        entities = {
            "Affiliations": bibliographic["affiliations"],
            "Authors": bibliographic["authors"],
            "Funding items": bibliographic["funding"],
            "Facilities": record["facilities"],
            "Samples": record["samples"],
            "Datasets": record["datasets"],
        }
        for label, items in entities.items():
            if not isinstance(items, list):
                raise ValueError(f"{path}: {label} must be a list")
            entity_counts[label] += len(items)
        for label, items, key in (
            ("Author--affiliation", entities["Authors"], "affiliation_refs"),
            ("Dataset--sample", entities["Datasets"], "sample_refs"),
            ("Dataset--facility", entities["Datasets"], "facility_refs"),
        ):
            for item in items:
                references = item[key]
                if not isinstance(references, list):
                    raise ValueError(f"{path}: {key} must be a list")
                reference_counts[label] += len(references)

    return {
        "Corpus and annotation": {
            "Papers": len(papers),
            "Calibration papers": calibration,
            "Evaluation papers": evaluation,
            "Working annotations": 3 * calibration + 2 * evaluation,
        },
        "Repeated entities": entity_counts,
        "Paper-local reference edges": reference_counts,
    }


def compute_iaa_results(iaa: dict, metrics: dict[str, str]) -> dict:
    """Prepare headline IAA counts and complete appendix scores from pooled IAA."""
    fields = {}
    for path, metric in metrics.items():
        agreement = iaa[path]
        fields[path] = {"score": _metric_value(iaa, path)}
        if metric == "ordered_set_f1":
            fields[path]["order_score"] = _metric_value(iaa, path, metric_key="mean_order")
        if "f1" not in agreement:
            non_null, total = (
                ("n_pairs_non_null", "n_pairs") if ".attrs." in path
                else ("n_non_null", "n_papers")
            )
            fields[path]["support"] = [agreement[non_null], agreement[total]]

    headline = []
    for outcome, path in PRIMARY_FIELDS:
        agreement = iaa[path]
        headline.append({
            "outcome": outcome, "path": path, "agreement": agreement["f1"],
            "matched": agreement["tp"], "unmatched": agreement["fp"] + agreement["fn"],
        })
    return {"headline": headline, "fields": fields}


def compute_zero_shot_results(evaluations: dict, metrics: dict[str, str]) -> dict:
    """Prepare seven-model headline and appendix results from full-instruction runs."""
    models = [evaluations[runs["full"]] for runs in ZERO_SHOT_RUNS.values()]
    fields = {}
    for path, metric in metrics.items():
        scores = [model[path] for model in models]
        fields[path] = {"scores": [_metric_value(model, path) for model in models]}
        if metric == "ordered_set_f1":
            fields[path]["order_scores"] = [
                _metric_value(model, path, metric_key="mean_order") for model in models
            ]
        if "f1" not in scores[0]:
            if ".attrs." in path:
                # Attribute support differs with each model's parent alignments.
                fields[path]["support"] = [
                    max(score["n_pairs_non_null"] for score in scores),
                    max(score["n_pairs"] for score in scores),
                ]
            else:
                fields[path]["support"] = [scores[0]["n_non_null"], scores[0]["n_papers"]]

    headline = [
        {"outcome": outcome, "path": path, "scores": fields[path]["scores"]}
        for outcome, path in PRIMARY_FIELDS
    ]
    return {"models": list(ZERO_SHOT_RUNS), "headline": headline, "fields": fields}


def compute_instruction_effects(evaluations: dict) -> list[dict]:
    """Subtract each model's no-instruction recovery from its full-instruction recovery."""
    rows = []
    for outcome, path in PRIMARY_FIELDS:
        deltas = []
        for runs in ZERO_SHOT_RUNS.values():
            full = evaluations[runs["full"]][path]["f1"]
            no_instruction = evaluations[runs["no_inst"]][path]["f1"]
            deltas.append(full - no_instruction)
        rows.append({"outcome": outcome, "path": path, "deltas": deltas})
    return rows


def compute_few_shot_results(evaluations: dict) -> dict:
    """Compare zero, one, and three shots with each model's own zero-shot baseline."""
    outcomes = []
    for outcome, path in PRIMARY_FIELDS:
        deltas = []
        for runs in FEW_SHOT_RUNS.values():
            baseline = evaluations[runs[0]][path]["f1"]
            deltas.append([
                evaluations[runs[shots]][path]["f1"] - baseline
                for shots in FEW_SHOT_LEVELS
            ])
        outcomes.append({"outcome": outcome, "path": path, "deltas": deltas})
    return {
        "models": list(FEW_SHOT_RUNS), "shots": list(FEW_SHOT_LEVELS),
        "outcomes": outcomes,
    }


def compute_paper_results() -> dict:
    """Load and validate inputs, then prepare the paper's numerical result sections."""
    metrics = load_metrics()
    evaluations = load_evaluation_results()
    iaa = load_iaa_results()
    for name, aggregate in [*evaluations.items(), ("IAA", iaa)]:
        if set(aggregate) != set(metrics):
            raise ValueError(f"{name}: aggregate fields do not match the scoring plan")
        incomplete = [path for path, value in aggregate.items() if value.get("n_papers") != 30]
        if incomplete:
            raise ValueError(f"{name}: expected all 30 papers for fields {incomplete}")

    return {
        "n_papers": 30,
        "metrics": metrics,
        "benchmark_statistics": compute_benchmark_statistics(ROOT),
        "iaa": compute_iaa_results(iaa, metrics),
        "zero_shot": compute_zero_shot_results(evaluations, metrics),
        "instruction_effects": compute_instruction_effects(evaluations),
        "few_shot": compute_few_shot_results(evaluations),
    }


def main(argv: list[str] | None = None) -> int:
    """Recompute and save results; an unsuccessful rerun removes the previous file."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    try:
        OUTPUT_PATH.unlink(missing_ok=True)
        results = compute_paper_results()
        content = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(content, encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        print(f"Cannot compute paper results: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
