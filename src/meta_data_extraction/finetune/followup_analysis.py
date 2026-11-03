"""Frozen Step 9 follow-up analyses over completed development predictions."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Literal
import unicodedata

from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
import yaml


SEEDS = (42, 123, 456)
CONTROL = "d_only"
TREATMENT = "matched_d_dr"
EXPECTED_EXPOSURE = {CONTROL: {"D": 24}, TREATMENT: {"D": 24, "R": 8}}
EXPECTED_ENDPOINT = {CONTROL: 6, TREATMENT: 8}
FIELD_SPECS = {
    "facilities": {"report_key": "facilities", "kind": "entity", "label": "Facilities"},
    "samples": {"report_key": "samples", "kind": "entity", "label": "Samples"},
    "datasets": {"report_key": "datasets", "kind": "entity", "label": "Datasets"},
    "sample_refs": {
        "report_key": "datasets.attrs.sample_refs", "kind": "reference",
        "label": "Sample references",
    },
    "facility_refs": {
        "report_key": "datasets.attrs.facility_refs", "kind": "reference",
        "label": "Facility references",
    },
}
ENTITY_FIELDS = tuple(field for field, spec in FIELD_SPECS.items() if spec["kind"] == "entity")
REFERENCE_FIELDS = tuple(field for field, spec in FIELD_SPECS.items() if spec["kind"] == "reference")


class PhaseA1Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_dir: str
    output_json: str
    output_markdown: str
    corpus_role: Literal["exploratory_reused_step7_test_as_development"]
    endpoint_policy: Literal["fixed_exposure"]
    control: str
    treatment: str
    seeds: list[int]
    expected_papers: int
    bootstrap_samples: int
    bootstrap_seed: int

    @model_validator(mode="after")
    def frozen_contract(self):
        if (self.control, self.treatment) != (CONTROL, TREATMENT):
            raise ValueError("Phase A1 is frozen to D24 versus D24+R8")
        if self.seeds != list(SEEDS) or self.expected_papers != 30:
            raise ValueError("Phase A1 seed or paper contract changed")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if Path(self.output_json).parent != Path(self.output_markdown).parent:
            raise ValueError("Phase A1 outputs must share one directory")
        return self


class PhaseA2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evaluation_config: str
    ablation_config: str
    source_evaluation_dir: str
    output_dir: str
    corpus_role: Literal["exploratory_reused_step7_test_as_development"]
    endpoint_policy: Literal["fixed_exposure"]
    adapter_condition: str
    inventories: list[str]
    seeds: list[int]
    expected_papers: int
    max_seq_length: PositiveInt
    max_new_tokens: PositiveInt
    decoding: Literal["greedy"]
    enable_thinking: Literal[False]
    attach_mlflow: Literal[False]

    @model_validator(mode="after")
    def frozen_contract(self):
        if self.adapter_condition != "matched_d_dr":
            raise ValueError("Phase A2 must use the fixed D24+R8 endpoints")
        if self.inventories != ["oracle", "self_predicted", "d24_predicted"]:
            raise ValueError("Phase A2 inventory order changed")
        if self.seeds != list(SEEDS) or self.expected_papers != 30:
            raise ValueError("Phase A2 seed or paper contract changed")
        if self.max_seq_length != 65536 or self.max_new_tokens != 4096:
            raise ValueError("Phase A2 context or generation budget changed")
        return self


class FollowupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    phase_a1: PhaseA1Config


class PhaseA2FileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    phase_a2: PhaseA2Config


def read_config(path: Path) -> FollowupConfig:
    return FollowupConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def read_phase_a2_config(path: Path) -> PhaseA2FileConfig:
    return PhaseA2FileConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_sha256(paths: list[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda candidate: str(candidate)):
        try:
            label = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            label = str(path.resolve())
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def prf(counts: dict) -> dict:
    tp, fp, fn = (int(counts[key]) for key in ("tp", "fp", "fn"))
    if min(tp, fp, fn) < 0:
        raise ValueError("Metric counts cannot be negative")
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 0.0 if precision == 0 or recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def add_counts(rows: list[dict]) -> dict:
    return {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn")}


def report_field_counts(report: dict, field: str) -> dict:
    key = FIELD_SPECS[field]["report_key"]
    value = report.get("fields", {}).get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Report is missing evaluator counts for {key}")
    if all(name in value for name in ("tp", "fp", "fn")):
        return {name: int(value[name]) for name in ("tp", "fp", "fn")}
    if FIELD_SPECS[field]["kind"] == "reference" and value == {"n_pairs": 0}:
        return {"tp": 0, "fp": 0, "fn": 0}
    raise ValueError(f"Report is missing evaluator counts for {key}")


def report_field_f1(report: dict, field: str) -> float | None:
    key = FIELD_SPECS[field]["report_key"]
    value = report["fields"][key]
    if FIELD_SPECS[field]["kind"] == "reference" and value == {"n_pairs": 0}:
        return None
    return prf(report_field_counts(report, field))["f1"]


def _normalized(value) -> str:
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def _duplicate_excess(values: list) -> int:
    counts = Counter(value for value in values if value not in (None, "", ()))
    return sum(count - 1 for count in counts.values() if count > 1)


def prediction_error_profile(prediction: dict, report: dict) -> dict:
    output = prediction.get("output")
    if not isinstance(output, dict):
        raise ValueError("Prediction does not contain a parsed output object")
    collections = {field: output.get(field) for field in ENTITY_FIELDS}
    if any(not isinstance(value, list) for value in collections.values()):
        raise ValueError("Prediction entity collections are missing or not lists")

    identities = {
        "facilities": [_normalized(row.get("name")) for row in collections["facilities"]],
        "samples": [_normalized(row.get("sample_name")) for row in collections["samples"]],
        "datasets": [(_normalized(row.get("dataset_name")),
                      _normalized(row.get("measurement_technique")))
                     for row in collections["datasets"]],
    }
    duplicate_ids = sum(_duplicate_excess([row.get("id") for row in rows])
                        for rows in collections.values())
    duplicate_identities = sum(_duplicate_excess(identities[field]) for field in ENTITY_FIELDS)

    sample_ids = {row.get("id") for row in collections["samples"] if row.get("id") is not None}
    facility_ids = {row.get("id") for row in collections["facilities"] if row.get("id") is not None}
    dangling = repeated = 0
    for dataset in collections["datasets"]:
        for key, valid in (("sample_refs", sample_ids), ("facility_refs", facility_ids)):
            refs = dataset.get(key)
            if not isinstance(refs, list):
                raise ValueError(f"Prediction dataset is missing list field {key}")
            dangling += sum(ref not in valid for ref in refs)
            repeated += _duplicate_excess(refs)

    entity_counts = {field: report_field_counts(report, field) for field in ENTITY_FIELDS}
    reference_counts = {field: report_field_counts(report, field) for field in REFERENCE_FIELDS}
    return {
        "missing_entities": sum(value["fn"] for value in entity_counts.values()),
        "unmatched_predictions": sum(value["fp"] for value in entity_counts.values()),
        "duplicate_id_excess": duplicate_ids,
        "duplicate_identity_excess": duplicate_identities,
        "missing_references": sum(value["fn"] for value in reference_counts.values()),
        "extra_or_mislinked_references": sum(value["fp"] for value in reference_counts.values()),
        "dangling_references": dangling,
        "repeated_reference_excess": repeated,
    }


def gold_counts(gold: dict) -> dict:
    result = {field: len(gold.get(field, [])) for field in ENTITY_FIELDS}
    datasets = gold.get("datasets", [])
    result["sample_refs"] = sum(len(row.get("sample_refs", [])) for row in datasets)
    result["facility_refs"] = sum(len(row.get("facility_refs", [])) for row in datasets)
    return result


def _validate_manifest(manifest: dict, config: PhaseA1Config, condition: str, seed: int) -> list[str]:
    training = manifest.get("training") or {}
    expected_label = f"{condition}_seed{seed}"
    actual = (
        manifest.get("study_type"), manifest.get("corpus_role"), manifest.get("endpoint_policy"),
        manifest.get("condition"), training.get("condition"), training.get("seed"),
        training.get("label"), training.get("endpoint_step"), training.get("planned_exposure"),
    )
    expected = (
        "exploratory", config.corpus_role, config.endpoint_policy, expected_label,
        condition, seed, expected_label, EXPECTED_ENDPOINT[condition], EXPECTED_EXPOSURE[condition],
    )
    if actual != expected:
        raise ValueError(f"Saved evaluation contract differs for {expected_label}")
    evaluator = manifest.get("evaluator") or {}
    if evaluator.get("schema_version_mismatch") is not False:
        raise ValueError(f"Evaluator schema mismatch in {expected_label}")
    paper_ids = (manifest.get("data") or {}).get("paper_ids")
    if (not isinstance(paper_ids, list) or len(paper_ids) != config.expected_papers
            or len(set(paper_ids)) != config.expected_papers):
        raise ValueError(f"Paper contract differs in {expected_label}")
    return paper_ids


def _load_run(root: Path, config: PhaseA1Config, condition: str, seed: int,
              expected_papers: list[str] | None) -> tuple[dict, list[Path]]:
    directory = root / config.source_dir / f"{condition}_seed{seed}"
    manifest_path, metrics_path = directory / "manifest.json", directory / "metrics.json"
    manifest, metrics = read_json(manifest_path), read_json(metrics_path)
    paper_ids = _validate_manifest(manifest, config, condition, seed)
    if expected_papers is not None and paper_ids != expected_papers:
        raise ValueError("Paper order differs across Phase A1 runs")

    reports, predictions, gold_by_paper, source_paths = {}, {}, {}, [manifest_path, metrics_path]
    for paper_id in paper_ids:
        report_path = directory / "reports" / f"{paper_id}.json"
        prediction_path = directory / "predictions" / f"{paper_id}.json"
        report, prediction = read_json(report_path), read_json(prediction_path)
        if Path(report["pred"]).resolve() != prediction_path.resolve():
            raise ValueError(f"Report/prediction path mismatch: {condition} seed {seed} {paper_id}")
        gold_path = Path(report["gold"])
        if not gold_path.is_absolute():
            gold_path = root / gold_path
        gold = read_json(gold_path)
        for field in FIELD_SPECS:
            report_field_counts(report, field)
        reports[paper_id], predictions[paper_id], gold_by_paper[paper_id] = report, prediction, gold
        source_paths.extend((report_path, prediction_path, gold_path))

    aggregate = metrics.get("all_papers", {}).get("fields", {})
    for field, spec in FIELD_SPECS.items():
        stored = aggregate.get(spec["report_key"])
        pooled = add_counts([report_field_counts(reports[paper_id], field) for paper_id in paper_ids])
        if not isinstance(stored, dict) or any(int(stored[key]) != pooled[key]
                                               for key in ("tp", "fp", "fn")):
            raise ValueError(f"Stored aggregate counts differ for {condition} seed {seed} {field}")

    return {
        "condition": condition, "seed": seed, "paper_ids": paper_ids,
        "manifest": manifest, "reports": reports, "predictions": predictions,
        "gold": gold_by_paper,
    }, source_paths


def load_phase_a1_inputs(root: Path, config: PhaseA1Config) -> tuple[dict, list[Path]]:
    runs, source_paths, paper_ids = {}, [], None
    evaluator = None
    for condition in (config.control, config.treatment):
        runs[condition] = {}
        for seed in config.seeds:
            run, paths = _load_run(root, config, condition, seed, paper_ids)
            paper_ids = run["paper_ids"] if paper_ids is None else paper_ids
            current_evaluator = run["manifest"]["evaluator"]
            evaluator = current_evaluator if evaluator is None else evaluator
            if current_evaluator != evaluator:
                raise ValueError("Evaluator metadata differs across Phase A1 runs")
            runs[condition][seed] = run
            source_paths.extend(paths)
    return {"runs": runs, "paper_ids": paper_ids, "evaluator": evaluator}, source_paths


def percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_cluster_bootstrap(runs: dict, paper_ids: list[str], field: str, *,
                             samples: int, seed: int) -> dict:
    rng = random.Random(seed)

    def statistic(indices: list[int]) -> float:
        deltas = []
        selected = [paper_ids[index] for index in indices]
        for run_seed in SEEDS:
            metrics = []
            for condition in (CONTROL, TREATMENT):
                counts = add_counts([
                    report_field_counts(runs[condition][run_seed]["reports"][paper_id], field)
                    for paper_id in selected
                ])
                metrics.append(prf(counts)["f1"])
            deltas.append(metrics[1] - metrics[0])
        return math.fsum(deltas) / len(deltas)

    observed = statistic(list(range(len(paper_ids))))
    estimates = [statistic([rng.randrange(len(paper_ids)) for _ in paper_ids])
                 for _ in range(samples)]
    return {
        "definition": "paired paper-cluster bootstrap; each sampled paper retains all three seeds",
        "mean_seed_delta_f1": observed,
        "paired_bootstrap_95_ci": [percentile(estimates, .025), percentile(estimates, .975)],
        "probability_delta_above_zero": sum(value > 0 for value in estimates) / samples,
        "samples": samples, "seed": seed,
    }


def _seed_metrics(runs: dict, paper_ids: list[str], field: str) -> dict:
    result = {}
    for seed in SEEDS:
        by_condition = {}
        paper_f1 = {}
        for condition in (CONTROL, TREATMENT):
            reports = runs[condition][seed]["reports"]
            by_condition[condition] = prf(add_counts(
                [report_field_counts(reports[paper_id], field) for paper_id in paper_ids]))
            paper_f1[condition] = {
                paper_id: report_field_f1(reports[paper_id], field)
                for paper_id in paper_ids
            }
        deltas = {
            paper_id: (paper_f1[TREATMENT][paper_id] - paper_f1[CONTROL][paper_id]
                       if paper_f1[TREATMENT][paper_id] is not None
                       and paper_f1[CONTROL][paper_id] is not None else None)
            for paper_id in paper_ids
        }
        result[str(seed)] = {
            "control": by_condition[CONTROL], "treatment": by_condition[TREATMENT],
            "delta": {
                metric: by_condition[TREATMENT][metric] - by_condition[CONTROL][metric]
                for metric in ("precision", "recall", "f1")
            },
            "paper_f1_direction": {
                "improved": sum(value is not None and value > 1e-12 for value in deltas.values()),
                "tied": sum(value is not None and abs(value) <= 1e-12 for value in deltas.values()),
                "worsened": sum(value is not None and value < -1e-12 for value in deltas.values()),
                "both_undefined": sum(paper_f1[CONTROL][paper] is None
                                      and paper_f1[TREATMENT][paper] is None for paper in paper_ids),
                "control_only_undefined": sum(paper_f1[CONTROL][paper] is None
                                              and paper_f1[TREATMENT][paper] is not None
                                              for paper in paper_ids),
                "treatment_only_undefined": sum(paper_f1[CONTROL][paper] is not None
                                                and paper_f1[TREATMENT][paper] is None
                                                for paper in paper_ids),
            },
        }
    f1_deltas = [result[str(seed)]["delta"]["f1"] for seed in SEEDS]
    return {
        "by_seed": result,
        "delta_f1_across_seeds": {
            "mean": math.fsum(f1_deltas) / len(f1_deltas),
            "sample_sd": statistics.stdev(f1_deltas),
            "min": min(f1_deltas), "max": max(f1_deltas),
        },
    }


def _paper_comparisons(runs: dict, paper_ids: list[str]) -> list[dict]:
    rows = []
    for paper_id in paper_ids:
        fields = {}
        for field in FIELD_SPECS:
            by_seed = {}
            for seed in SEEDS:
                values = {}
                for condition in (CONTROL, TREATMENT):
                    report = runs[condition][seed]["reports"][paper_id]
                    values[condition] = report_field_f1(report, field)
                by_seed[str(seed)] = (
                    values[TREATMENT] - values[CONTROL]
                    if values[TREATMENT] is not None and values[CONTROL] is not None else None)
            deltas = [value for value in by_seed.values() if value is not None]
            fields[field] = {
                "delta_f1_by_seed": by_seed,
                "mean_delta_f1": math.fsum(deltas) / len(deltas) if deltas else None,
                "comparable_seeds": len(deltas),
                "undefined_seed_pairs": len(SEEDS) - len(deltas),
                "improved_seeds": sum(value > 1e-12 for value in deltas),
                "tied_seeds": sum(abs(value) <= 1e-12 for value in deltas),
                "worsened_seeds": sum(value < -1e-12 for value in deltas),
            }
        gold = runs[CONTROL][SEEDS[0]]["gold"][paper_id]
        expected_gold = gold_counts(gold)
        for condition in (CONTROL, TREATMENT):
            for seed in SEEDS:
                if gold_counts(runs[condition][seed]["gold"][paper_id]) != expected_gold:
                    raise ValueError(f"Gold counts differ across runs for {paper_id}")
        rows.append({"paper_id": paper_id, "gold_counts": expected_gold, "fields": fields})
    return rows


def _concentration(papers: list[dict], field: str) -> dict:
    all_values = [(row["paper_id"], row["fields"][field]["mean_delta_f1"]) for row in papers]
    values = [(paper, value) for paper, value in all_values if value is not None]
    absolute_total = math.fsum(abs(value) for _, value in values)
    ranked_gain = sorted(values, key=lambda row: (-row[1], row[0]))
    ranked_loss = sorted(values, key=lambda row: (row[1], row[0]))
    return {
        "papers_improved": sum(value > 1e-12 for _, value in values),
        "papers_tied": sum(abs(value) <= 1e-12 for _, value in values),
        "papers_worsened": sum(value < -1e-12 for _, value in values),
        "papers_not_comparable": len(all_values) - len(values),
        "largest_gains": [{"paper_id": paper, "mean_delta_f1": value}
                          for paper, value in ranked_gain[:5]],
        "largest_losses": [{"paper_id": paper, "mean_delta_f1": value}
                           for paper, value in ranked_loss[:5]],
        "largest_absolute_paper_share": (max((abs(value) for _, value in values), default=0.0)
                                         / absolute_total if absolute_total else 0.0),
    }


def _error_totals(runs: dict, paper_ids: list[str]) -> dict:
    result = {}
    for condition in (CONTROL, TREATMENT):
        by_seed = {}
        for seed in SEEDS:
            profiles = [prediction_error_profile(
                runs[condition][seed]["predictions"][paper_id],
                runs[condition][seed]["reports"][paper_id]) for paper_id in paper_ids]
            by_seed[str(seed)] = {key: sum(row[key] for row in profiles)
                                  for key in profiles[0]}
        result[condition] = {
            "by_seed": by_seed,
            "all_seed_paper_outputs": {
                key: sum(by_seed[str(seed)][key] for seed in SEEDS)
                for key in next(iter(by_seed.values()))
            },
        }
    return result


def build_phase_a1(root: Path, config_path: Path, config: FollowupConfig) -> dict:
    loaded, source_paths = load_phase_a1_inputs(root, config.phase_a1)
    runs, paper_ids = loaded["runs"], loaded["paper_ids"]
    papers = _paper_comparisons(runs, paper_ids)
    fields = {}
    for index, (field, spec) in enumerate(FIELD_SPECS.items()):
        seed_metrics = _seed_metrics(runs, paper_ids, field)
        fields[field] = {
            "label": spec["label"], **seed_metrics,
            "paired_paper_cluster_bootstrap": paired_cluster_bootstrap(
                runs, paper_ids, field, samples=config.phase_a1.bootstrap_samples,
                seed=config.phase_a1.bootstrap_seed + index),
            "paper_concentration": _concentration(papers, field),
        }
    unique_sources = sorted(set(source_paths), key=lambda path: str(path))
    return {
        "version": config.version, "phase": "A1", "study_type": "exploratory",
        "corpus_role": config.phase_a1.corpus_role,
        "comparison": {"control": CONTROL, "treatment": TREATMENT,
                       "definition": "fixed endpoints paired by seed and paper"},
        "seeds": list(SEEDS), "paper_ids": paper_ids,
        "field_definitions": {
            "unmatched_predictions": "evaluator entity false positives; may include hallucinations or semantic duplicates",
            "duplicate_id_excess": "repeated entity IDs beyond the first occurrence",
            "duplicate_identity_excess": "repeated normalized entity identity keys beyond the first occurrence",
            "extra_or_mislinked_references": "evaluator reference false positives; includes extra or wrong-target links",
            "dangling_references": "predicted references to IDs absent from the corresponding predicted entity list",
            "repeated_reference_excess": "duplicate IDs within one predicted dataset reference list",
        },
        "evaluator": loaded["evaluator"],
        "provenance": {
            "config": str(config_path.relative_to(root)), "config_sha256": sha256(config_path),
            "source_files": len(unique_sources),
            "source_bundle_sha256": combined_sha256(unique_sources, root=root),
        },
        "accounting": {
            "runs": 6, "run_paper_outputs": 6 * len(paper_ids),
            "paired_seed_paper_comparisons": len(SEEDS) * len(paper_ids),
            "independent_test_papers": 0,
        },
        "fields": fields,
        "error_totals": _error_totals(runs, paper_ids),
        "papers": papers,
        "caveats": [
            "The thirty papers are reused development data, not an independent test.",
            "Bootstrap intervals resample papers and are descriptive for this fixed corpus.",
            "Entity false positives do not by themselves distinguish hallucinations from semantic duplicates.",
            "Reference false positives do not by themselves distinguish extra links from wrong-target links.",
            "Per-paper conditional reference F1 is undefined when no dataset pair aligns; such pairs are marked NA rather than scored as perfect or zero.",
        ],
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def markdown_report(analysis: dict) -> str:
    control, treatment = analysis["comparison"]["control"], analysis["comparison"]["treatment"]
    lines = [
        "# Step 9 Phase A1: paired D24 versus D24+R8 error analysis", "",
        "This is an exploratory analysis of the thirty reused development papers. It reads the "
        "saved fixed-endpoint reports and performs no generation or training.", "",
        "## Pooled field metrics by seed", "",
        "| Field | Seed | D24 P/R/F1 | D24+R8 P/R/F1 | ΔP | ΔR | ΔF1 | Papers +/=/-/NA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for field in FIELD_SPECS:
        result = analysis["fields"][field]
        for seed in SEEDS:
            row = result["by_seed"][str(seed)]
            base, model, delta = row["control"], row["treatment"], row["delta"]
            direction = row["paper_f1_direction"]
            undefined = (direction["both_undefined"] + direction["control_only_undefined"]
                         + direction["treatment_only_undefined"])
            lines.append(
                f"| {result['label']} | {seed} | {_fmt(base['precision'])}/{_fmt(base['recall'])}/{_fmt(base['f1'])} | "
                f"{_fmt(model['precision'])}/{_fmt(model['recall'])}/{_fmt(model['f1'])} | "
                f"{delta['precision']:+.3f} | {delta['recall']:+.3f} | {delta['f1']:+.3f} | "
                f"{direction['improved']}/{direction['tied']}/{direction['worsened']}/{undefined} |"
            )
    lines += ["", "## Paired paper-cluster summary", "",
              "| Field | Mean ΔF1 across seeds | Seed SD | Paired 95% interval | Papers +/=/-/NA | Largest-paper share of |Δ| |",
              "|---|---:|---:|---:|---:|---:|"]
    for field in FIELD_SPECS:
        result = analysis["fields"][field]
        delta = result["delta_f1_across_seeds"]
        bootstrap = result["paired_paper_cluster_bootstrap"]
        concentration = result["paper_concentration"]
        interval = bootstrap["paired_bootstrap_95_ci"]
        lines.append(
            f"| {result['label']} | {delta['mean']:+.3f} | {delta['sample_sd']:.3f} | "
            f"[{interval[0]:+.3f}, {interval[1]:+.3f}] | "
            f"{concentration['papers_improved']}/{concentration['papers_tied']}/{concentration['papers_worsened']}/"
            f"{concentration['papers_not_comparable']} | "
            f"{concentration['largest_absolute_paper_share']:.1%} |"
        )
    lines += ["", "The interval resamples papers as clusters and retains all three seeds for each sampled paper.",
              "It is descriptive for this reused corpus, not an independent-population confidence claim.", "",
              "## Error accounting across all 90 outputs per condition", "",
              "| Condition | Missing entities | Unmatched predictions | Duplicate IDs | Duplicate identities | Missing refs | Extra/mislinked refs | Dangling refs | Repeated refs |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in (control, treatment):
        row = analysis["error_totals"][condition]["all_seed_paper_outputs"]
        lines.append(
            f"| `{condition}` | {row['missing_entities']} | {row['unmatched_predictions']} | "
            f"{row['duplicate_id_excess']} | {row['duplicate_identity_excess']} | "
            f"{row['missing_references']} | {row['extra_or_mislinked_references']} | "
            f"{row['dangling_references']} | {row['repeated_reference_excess']} |"
        )
    lines += ["", "Evaluator entity false positives are reported as unmatched predictions; they may include "
              "hallucinations or semantic duplicates. Evaluator reference false positives are reported as "
              "extra/mislinked because those causes are not separable from aggregate counts.", "",
              "## Per-paper mean F1 differences", "",
              "Values are D24+R8 minus D24, averaged across the three paired seeds.", "",
              "| Paper | Gold F/S/D/SR/FR | Facilities | Samples | Datasets | Sample refs | Facility refs |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for paper in analysis["papers"]:
        gold = paper["gold_counts"]
        values = [paper["fields"][field]["mean_delta_f1"] for field in FIELD_SPECS]
        rendered = ["NA" if value is None else f"{value:+.3f}" for value in values]
        lines.append(
            f"| `{paper['paper_id']}` | {gold['facilities']}/{gold['samples']}/{gold['datasets']}/"
            f"{gold['sample_refs']}/{gold['facility_refs']} | " + " | ".join(rendered) + " |"
        )
    lines += ["", "## Accounting and caveats", "",
              f"- Inputs: {analysis['accounting']['runs']} runs and {analysis['accounting']['run_paper_outputs']} run-paper reports.",
              f"- Paired comparisons: {analysis['accounting']['paired_seed_paper_comparisons']} seed-paper pairs.",
              f"- Source bundle SHA-256: `{analysis['provenance']['source_bundle_sha256']}`.",
              "- Independent test papers: 0.", ""]
    lines.extend(f"- {caveat}" for caveat in analysis["caveats"])
    return "\n".join(lines) + "\n"
