"""Step 9 Phase A2 oracle-versus-predicted inventory evaluation."""
from __future__ import annotations

from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import statistics
import time
from typing import Literal

import torch
from transformers import set_seed

from meta_data_extraction.data_models import ExtractionResult, Provenance
from meta_data_extraction.evaluation import (
    aggregate_reports, build_evaluator_metadata, evaluate, validate_evaluator_provenance,
)
from meta_data_extraction.utils import get_prompt_version, get_schema_refs
from .ablation import read_ablation_config
from .ablation_evaluation import endpoint_adapter
from .data import COLLECTIONS, REFERENCE_FIELDS, project_entities, reconstruct, task_prompt
from .evaluation import EvaluationConfig, read_evaluation_config, runtime_versions
from .followup_analysis import PhaseA2Config, PhaseA2FileConfig
from .parsing import text_hash
from .tokenization import render_prompt
from .tracking import checksum
from .training import load_base, reload_adapter
from .training_data import TrainingConfig, file_hash, verify_prepared
from .validation import generate_answer, validation_context, write_json


INVENTORIES = ("oracle", "self_predicted", "d24_predicted")
SEEDS = (42, 123, 456)
SOURCE_CONDITIONS = {"self_predicted": "matched_d_dr", "d24_predicted": "d_only"}
ENTITY_REPORT_FIELDS = ("facilities", "samples", "datasets")
REFERENCE_REPORT_FIELDS = {
    "sample_refs": "datasets.attrs.sample_refs",
    "facility_refs": "datasets.attrs.facility_refs",
}


def validate_followup_config(config: PhaseA2FileConfig) -> PhaseA2Config:
    return config.phase_a2


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                 for row in rows), encoding="utf-8")
    temporary.replace(path)


def label(inventory: str, seed: int) -> str:
    if inventory not in INVENTORIES or seed not in SEEDS:
        raise ValueError("Inventory/seed is outside the frozen Phase A2 matrix")
    return f"{inventory}_seed{seed}"


def strict_json_object(text: str) -> dict:
    def reject_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value


def validate_inventory(inventory: dict) -> dict:
    if set(inventory) != set(COLLECTIONS):
        raise ValueError("R inventory must contain exactly facilities, samples and datasets")
    result = deepcopy(inventory)
    for collection in COLLECTIONS:
        rows = result[collection]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"R inventory {collection} must be a list of objects")
        ids = [row.get("id") for row in rows]
        if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"R inventory {collection} IDs must be nonempty and unique")
    if any(field in dataset for dataset in result["datasets"] for field in REFERENCE_FIELDS):
        raise ValueError("R inventory leaked reference answers")
    return result


def inventory_from_output(output: dict) -> dict:
    if not isinstance(output, dict):
        raise ValueError("Saved D prediction does not contain an output object")
    return validate_inventory(project_entities(output))


def empty_references(inventory: dict) -> dict:
    return {"references": [
        {"dataset_id": dataset["id"], "field": field, "target_ids": []}
        for dataset in inventory["datasets"] for field in REFERENCE_FIELDS
    ]}


def validate_reference_output(value: dict, inventory: dict) -> dict:
    if set(value) != {"references"} or not isinstance(value["references"], list):
        raise ValueError("R output must contain exactly one references list")
    for row in value["references"]:
        if not isinstance(row, dict) or set(row) != {"dataset_id", "field", "target_ids"}:
            raise ValueError("Every R record requires exactly dataset_id, field and target_ids")
        if (not isinstance(row["dataset_id"], str) or not isinstance(row["field"], str)
                or not isinstance(row["target_ids"], list)
                or any(not isinstance(item, str) for item in row["target_ids"])):
            raise ValueError("R record values have incorrect types")
    reconstruct(inventory, value)
    return value


def _source_prediction(root: Path, config: PhaseA2Config, condition: str,
                       seed: int, paper_id: str) -> tuple[dict, dict]:
    directory = root / config.source_evaluation_dir / f"{condition}_seed{seed}"
    manifest = read_json(directory / "manifest.json")
    training = manifest.get("training") or {}
    if (manifest.get("corpus_role"), manifest.get("endpoint_policy"),
            training.get("condition"), training.get("seed")) != (
                config.corpus_role, config.endpoint_policy, condition, seed):
        raise ValueError(f"Source D prediction contract changed: {condition} seed {seed}")
    progress = {row["paper_id"]: row for row in read_jsonl(directory / "generation.jsonl")}
    if len(progress) != config.expected_papers or paper_id not in progress:
        raise ValueError(f"Source D generation is incomplete: {condition} seed {seed}")
    record = progress[paper_id]
    prediction_path = directory / record["prediction_file"]
    if checksum(prediction_path) != record["prediction_sha256"]:
        raise ValueError(f"Source D prediction changed: {condition} seed {seed} {paper_id}")
    prediction = read_json(prediction_path)
    return inventory_from_output(prediction.get("output")), {
        "condition": condition, "path": str(prediction_path.relative_to(root)),
        "sha256": record["prediction_sha256"],
    }


def _frozen_gold(root: Path, split: dict, paper_id: str) -> tuple[dict, Path, str]:
    path = root / "data/gold" / f"{paper_id}.json"
    actual = file_hash(path)
    expected = split["audit"]["source_files"][f"data/gold/{paper_id}.json"]["sha256"]
    if actual != expected:
        raise ValueError(f"Frozen development gold changed: {paper_id}")
    return read_json(path), path, actual


def prepare_context(root: Path, config: PhaseA2Config, training: TrainingConfig,
                    tokenizer_identity: dict) -> dict:
    _, manifest, drift = verify_prepared(root, training)
    if tokenizer_identity != manifest["tokenizer"]:
        raise ValueError("Tokenizer identity differs from the preparation manifest")
    prepare = manifest["config"]
    split = read_json(root / prepare["split_manifest"])
    paper_ids = split["test"]
    if len(paper_ids) != config.expected_papers or len(set(paper_ids)) != config.expected_papers:
        raise ValueError("Phase A2 requires the thirty frozen development papers")
    return {
        "manifest": manifest, "prepare": prepare, "split": split,
        "paper_ids": paper_ids, "schemas": prepare["schemas"],
        "preparation_manifest_sha256": file_hash(
            root / training.prepared_dir / "preparation_manifest.json"),
        "preparation_environment_changes": drift,
    }


def prepare_rows(root: Path, config: PhaseA2Config, context: dict, tokenizer,
                 inventory_type: str, seed: int) -> list[dict]:
    rows = []
    for paper_id in context["paper_ids"]:
        gold, gold_path, gold_sha = _frozen_gold(root, context["split"], paper_id)
        if inventory_type == "oracle":
            inventory = validate_inventory(project_entities(gold))
            source = {"condition": "gold", "path": str(gold_path.relative_to(root)),
                      "sha256": gold_sha}
        else:
            inventory, source = _source_prediction(
                root, config, SOURCE_CONDITIONS[inventory_type], seed, paper_id)
        text_path = root / context["prepare"]["output_dir"] / "text" / f"{paper_id}.md"
        text = text_path.read_text(encoding="utf-8")
        prompt = task_prompt("R", text, context["schemas"], inventory)
        rendered = render_prompt(tokenizer, [{"role": "user", "content": prompt}])
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False, truncation=False)
        if len(prompt_ids) + config.max_new_tokens > config.max_seq_length:
            raise ValueError(f"Phase A2 prompt exceeds context window: {inventory_type} {seed} {paper_id}")
        rows.append({
            "paper_id": paper_id, "prompt_ids": prompt_ids, "prompt_tokens": len(prompt_ids),
            "prompt_sha256": text_hash(rendered), "text_sha256": file_hash(text_path),
            "inventory": inventory,
            "inventory_sha256": text_hash(json.dumps(inventory, ensure_ascii=False, sort_keys=True)),
            "inventory_source": source,
        })
    return rows


def evaluation_contract(root: Path, config_path: Path, config: PhaseA2Config,
                        base_config: EvaluationConfig, context: dict, tokenizer_identity: dict,
                        inventory_type: str, seed: int, rows: list[dict],
                        training: TrainingConfig, checkpoint: Path, identity: dict) -> dict:
    source_paths = [
        config_path, root / config.evaluation_config, root / config.ablation_config,
        root / base_config.metrics_config, root / base_config.alignment_config,
        root / "scripts/evaluate_finetune_followup.py",
        root / "src/meta_data_extraction/finetune/followup_evaluation.py",
        root / "src/meta_data_extraction/finetune/data.py",
        root / "src/meta_data_extraction/finetune/evaluation.py",
        root / "src/meta_data_extraction/evaluation.py",
        root / "src/meta_data_extraction/alignment.py",
    ]
    evaluator = build_evaluator_metadata(root / base_config.metrics_config)
    if evaluator["schema_version_mismatch"]:
        raise ValueError("Evaluator schema versions do not match")
    return {
        "version": "1.0.0", "phase": "A2", "study_type": "exploratory",
        "corpus_role": config.corpus_role, "condition": label(inventory_type, seed),
        "inventory_type": inventory_type, "endpoint_policy": config.endpoint_policy,
        "training": identity,
        "model": {
            "model_id": context["prepare"]["model_id"],
            "revision": context["prepare"]["revision"],
            "precision": training.precision_policy(), "adapter": str(checkpoint.resolve()),
            "adapter_sha256": identity["adapter_sha256"],
        },
        "decoding": {
            "method": config.decoding, "enable_thinking": config.enable_thinking,
            "max_new_tokens": config.max_new_tokens, "max_seq_length": config.max_seq_length,
            "seed": seed, "num_beams": 1, "repetition_penalty": 1.0,
        },
        "data": {
            "source_split": "step7_test", "analysis_role": "exploratory_development",
            "paper_ids": context["paper_ids"],
            "gold_entity_inventory_in_prompt": inventory_type == "oracle",
            "gold_read_for_scoring": True,
            "prompts": [{
                "paper_id": row["paper_id"], "prompt_tokens": row["prompt_tokens"],
                "prompt_sha256": row["prompt_sha256"], "text_sha256": row["text_sha256"],
                "inventory_sha256": row["inventory_sha256"],
                "inventory_source": row["inventory_source"],
            } for row in rows],
            "preparation_manifest_sha256": context["preparation_manifest_sha256"],
            "prior_results_influenced_design": True,
        },
        "tokenizer": tokenizer_identity, "evaluator": evaluator,
        "mlflow_attachment_allowed": False,
        "sources": {str(path.relative_to(root)): file_hash(path) for path in source_paths},
    }


def prepare_output(output: Path, contract: dict) -> None:
    manifest = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not manifest.is_file() or read_json(manifest) != contract:
            raise ValueError(f"Phase A2 output belongs to a different contract: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest, contract)


def _reconstructed_result(base_config: EvaluationConfig, condition: str, row: dict,
                          generated: dict, reconstructed: dict, warnings: list[str],
                          latency_ms: float, schemas: list[str]) -> dict:
    return ExtractionResult(
        document_id=row["paper_id"], domain_name=base_config.domain_name,
        output=reconstructed, raw_output=generated.get("text"),
        provenance=Provenance(
            config_name=condition, prompt_version=get_prompt_version(),
            schemas=get_schema_refs(schemas), input_tokens=row["prompt_tokens"],
            output_tokens=generated.get("generated_tokens"), latency_ms=round(latency_ms, 2),
        ), warnings=warnings,
    ).model_dump(mode="json")


def generate_condition(model, tokenizer, rows: list[dict], config: PhaseA2Config,
                       base_config: EvaluationConfig, inventory_type: str, seed: int,
                       output: Path, schemas: list[str]) -> list[dict]:
    progress_path = output / "generation.jsonl"
    progress = read_jsonl(progress_path)
    expected_ids = [row["paper_id"] for row in rows]
    if [row["paper_id"] for row in progress] != expected_ids[:len(progress)]:
        raise ValueError("Phase A2 progress is not an ordered paper prefix")
    for old in progress:
        for file_key, hash_key in (
            ("raw_file", "raw_sha256"), ("inventory_file", "inventory_sha256"),
            ("reference_file", "reference_sha256"),
            ("reconstructed_file", "reconstructed_sha256"),
        ):
            if checksum(output / old[file_key]) != old[hash_key]:
                raise ValueError(f"Saved Phase A2 artifact changed: {old['paper_id']} {file_key}")

    condition = label(inventory_type, seed)
    for row in rows[len(progress):]:
        paper_id = row["paper_id"]
        raw_path = output / "raw" / f"{paper_id}.txt"
        inventory_path = output / "inventories" / f"{paper_id}.json"
        reference_path = output / "reference_predictions" / f"{paper_id}.json"
        reconstructed_path = output / "reconstructed_predictions" / f"{paper_id}.json"
        write_json(inventory_path, row["inventory"])
        started = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        warnings: list[str] = []
        try:
            with validation_context(model):
                generated = generate_answer(model, tokenizer, row, config.max_new_tokens)
            status = "generated"
        except torch.OutOfMemoryError as exc:
            generated = {"text": "", "generated_tokens": 0, "ended_with_eos": False,
                         "hit_output_limit": False}
            status = "generation_failed"
            warnings.append(f"OutOfMemoryError: {exc}")
            torch.cuda.empty_cache()
        except Exception as exc:
            generated = {"text": "", "generated_tokens": 0, "ended_with_eos": False,
                         "hit_output_limit": False}
            status = "generation_failed"
            warnings.append(f"{type(exc).__name__}: {exc}")
        latency_ms = (time.monotonic() - started) * 1000
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(generated["text"], encoding="utf-8")

        parsed, json_valid, contract_valid = {}, False, False
        if status == "generated":
            try:
                parsed = strict_json_object(generated["text"])
                json_valid = True
            except (ValueError, json.JSONDecodeError) as exc:
                status = "invalid_json"
                warnings.append(f"Strict JSON parse failed: {exc}")
            if json_valid:
                try:
                    validate_reference_output(parsed, row["inventory"])
                    contract_valid = True
                except (ValueError, KeyError, TypeError) as exc:
                    status = "invalid_reference_contract"
                    warnings.append(f"Reference contract failed: {exc}")
        if generated["hit_output_limit"]:
            warnings.append("Generation reached max_new_tokens without EOS.")
        scored_references = parsed if contract_valid else empty_references(row["inventory"])
        reconstructed = reconstruct(row["inventory"], scored_references)
        write_json(reference_path, parsed)
        write_json(reconstructed_path, _reconstructed_result(
            base_config, condition, row, generated, reconstructed, warnings, latency_ms, schemas))
        record = {
            "paper_id": paper_id, "status": status, "json_valid": json_valid,
            "reference_contract_valid": contract_valid,
            "used_empty_reference_fallback": not contract_valid,
            "prompt_tokens": row["prompt_tokens"],
            "generated_tokens": generated["generated_tokens"],
            "ended_with_eos": generated["ended_with_eos"],
            "hit_output_limit": generated["hit_output_limit"],
            "latency_ms": round(latency_ms, 2),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
            "raw_file": raw_path.relative_to(output).as_posix(),
            "inventory_file": inventory_path.relative_to(output).as_posix(),
            "reference_file": reference_path.relative_to(output).as_posix(),
            "reconstructed_file": reconstructed_path.relative_to(output).as_posix(),
            "raw_sha256": checksum(raw_path), "inventory_sha256": checksum(inventory_path),
            "reference_sha256": checksum(reference_path),
            "reconstructed_sha256": checksum(reconstructed_path), "warnings": warnings,
        }
        progress.append(record)
        write_jsonl(progress_path, progress)
    return progress


def _match_map(report: dict, field: str) -> dict[str, str]:
    rows = report.get("fields", {}).get(field, {}).get("match_stages", [])
    result = {}
    for row in rows:
        pred, gold = row.get("pred"), row.get("gold")
        if isinstance(pred, str) and isinstance(gold, str):
            if pred in result:
                raise ValueError(f"Duplicate evaluator alignment for {field}: {pred}")
            result[pred] = gold
    return result


def end_to_end_reference_counts(gold: dict, reconstructed: dict, report: dict,
                                field: str) -> dict:
    if field not in REFERENCE_FIELDS:
        raise ValueError(f"Unknown reference field: {field}")
    dataset_map = _match_map(report, "datasets")
    target_collection = "samples" if field == "sample_refs" else "facilities"
    target_map = _match_map(report, target_collection)
    gold_links = {(dataset["id"], target) for dataset in gold["datasets"]
                  for target in dataset[field]}
    mapped, invalid = set(), 0
    for dataset in reconstructed["datasets"]:
        for target in dataset[field]:
            if dataset["id"] not in dataset_map or target not in target_map:
                invalid += 1
            else:
                mapped.add((dataset_map[dataset["id"]], target_map[target]))
    tp = len(mapped & gold_links)
    return {"tp": tp, "fp": invalid + len(mapped - gold_links), "fn": len(gold_links) - tp}


def prf(counts: dict) -> dict:
    tp, fp, fn = (int(counts[key]) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 0.0 if precision == 0 or recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def _sum_counts(rows: list[dict]) -> dict:
    return {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn")}


def score_condition(root: Path, config: PhaseA2Config, base_config: EvaluationConfig,
                    inventory_type: str, seed: int, output: Path) -> dict:
    manifest = read_json(output / "manifest.json")
    expected_evaluator = build_evaluator_metadata(root / base_config.metrics_config)
    validate_evaluator_provenance(manifest["evaluator"], expected_evaluator,
                                  source=str(output / "manifest.json"))
    generation = read_jsonl(output / "generation.jsonl")
    expected_ids = manifest["data"]["paper_ids"]
    if [row["paper_id"] for row in generation] != expected_ids or len(generation) != 30:
        raise ValueError(f"Phase A2 generation is incomplete: {label(inventory_type, seed)}")
    reports, paper_scores = [], []
    end_to_end = {field: [] for field in REFERENCE_FIELDS}
    for row in generation:
        paper_id = row["paper_id"]
        for file_key, hash_key in (("inventory_file", "inventory_sha256"),
                                   ("reconstructed_file", "reconstructed_sha256")):
            if checksum(output / row[file_key]) != row[hash_key]:
                raise ValueError(f"Phase A2 scored artifact changed: {paper_id}")
        gold_path = root / "data/gold" / f"{paper_id}.json"
        reconstructed_path = output / row["reconstructed_file"]
        report = evaluate(gold_path, reconstructed_path, root / base_config.metrics_config)
        report_path = output / "reports" / f"{paper_id}.json"
        write_json(report_path, report)
        reports.append(report)
        gold = read_json(gold_path)
        reconstructed = read_json(reconstructed_path)["output"]
        end_scores = {}
        for field in REFERENCE_FIELDS:
            counts = end_to_end_reference_counts(gold, reconstructed, report, field)
            end_to_end[field].append(counts)
            end_scores[field] = prf(counts)
        paper_scores.append({
            "paper_id": paper_id, "status": row["status"],
            "reference_contract_valid": row["reference_contract_valid"],
            "entity_f1": {field: report["fields"][field]["f1"]
                          for field in ENTITY_REPORT_FIELDS},
            "conditional_reference_f1": {
                field: report["fields"][path].get("f1")
                for field, path in REFERENCE_REPORT_FIELDS.items()
            },
            "end_to_end_reference": end_scores,
            "report_file": report_path.relative_to(output).as_posix(),
        })
    fields = aggregate_reports(reports)
    end_summary = {field: prf(_sum_counts(rows)) for field, rows in end_to_end.items()}
    summary = {
        "papers_total": 30,
        "papers_json_valid": sum(row["json_valid"] for row in generation),
        "papers_reference_contract_valid": sum(row["reference_contract_valid"] for row in generation),
        "generation_failures": sum(row["status"] == "generation_failed" for row in generation),
        "output_limit_failures": sum(row["hit_output_limit"] for row in generation),
        "facilities_f1": fields["facilities"]["f1"],
        "samples_f1": fields["samples"]["f1"],
        "datasets_f1": fields["datasets"]["f1"],
        "conditional_sample_refs_f1": fields["datasets.attrs.sample_refs"].get("f1"),
        "conditional_facility_refs_f1": fields["datasets.attrs.facility_refs"].get("f1"),
        "end_to_end_sample_refs_f1": end_summary["sample_refs"]["f1"],
        "end_to_end_facility_refs_f1": end_summary["facility_refs"]["f1"],
        "mean_latency_ms": math.fsum(row["latency_ms"] for row in generation) / 30,
        "mean_generated_tokens": math.fsum(row["generated_tokens"] for row in generation) / 30,
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in generation),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in generation),
    }
    metrics = {
        "version": "1.0.0", "phase": "A2", "study_type": "exploratory",
        "condition": label(inventory_type, seed), "inventory_type": inventory_type,
        "seed": seed, "summary": summary,
        "aggregation_policy": {
            "conditional_reference": "shared evaluator pooled TP/FP/FN on aligned dataset pairs",
            "end_to_end_reference": "pooled relation triples; unaligned datasets or target entities count as false positives and all unmatched gold links as false negatives",
            "invalid_R_output": "empty reference lists over the supplied inventory",
        },
        "all_papers": {"fields": fields, "end_to_end_reference": end_summary},
        "papers": paper_scores, "evaluator": expected_evaluator,
    }
    write_json(output / "metrics.json", metrics)
    write_jsonl(output / "paper_scores.jsonl", paper_scores)
    return metrics


SUMMARY_METRICS = (
    "papers_reference_contract_valid", "facilities_f1", "samples_f1", "datasets_f1",
    "conditional_sample_refs_f1", "conditional_facility_refs_f1",
    "end_to_end_sample_refs_f1", "end_to_end_facility_refs_f1",
    "mean_latency_ms", "mean_generated_tokens",
)


def _summary(values: dict[int, float]) -> dict:
    if set(values) != set(SEEDS):
        raise ValueError("Phase A2 comparison requires all three seeds")
    numbers = list(values.values())
    return {"mean": math.fsum(numbers) / len(numbers), "sample_sd": statistics.stdev(numbers),
            "min": min(numbers), "max": max(numbers),
            "by_seed": {str(seed): values[seed] for seed in SEEDS}}


def compare(root: Path, config: PhaseA2Config) -> dict:
    conditions = {}
    for inventory in INVENTORIES:
        runs = {seed: read_json(root / config.output_dir / label(inventory, seed) / "metrics.json")
                for seed in SEEDS}
        conditions[inventory] = {
            "runs": {str(seed): runs[seed]["summary"] for seed in SEEDS},
            "aggregate": {
                metric: _summary({seed: float(runs[seed]["summary"][metric] or 0.0)
                                  for seed in SEEDS}) for metric in SUMMARY_METRICS
            },
        }
    contrasts = {}
    for name, treatment, control in (
        ("self_minus_oracle", "self_predicted", "oracle"),
        ("d24_minus_oracle", "d24_predicted", "oracle"),
        ("self_minus_d24", "self_predicted", "d24_predicted"),
    ):
        metrics = {}
        for metric in SUMMARY_METRICS:
            deltas = {seed: conditions[treatment]["runs"][str(seed)][metric]
                      - conditions[control]["runs"][str(seed)][metric] for seed in SEEDS}
            metrics[metric] = _summary(deltas)
        contrasts[name] = {"treatment": treatment, "control": control, "metrics": metrics}
    comparison = {
        "version": "1.0.0", "phase": "A2", "study_type": "exploratory",
        "corpus_role": config.corpus_role, "conditions": conditions, "contrasts": contrasts,
        "accounting": {"adapters": 3, "inventory_conditions": 3, "papers": 30,
                       "generations": 270, "independent_test_papers": 0},
        "caveats": [
            "These thirty papers are reused development data, not an independent test.",
            "Oracle inventories intentionally expose gold entities but not gold references.",
            "Conditional reference F1 excludes unaligned dataset pairs; end-to-end reference F1 penalizes entity and dataset alignment failures.",
        ],
    }
    write_json(root / config.output_dir / "comparison.json", comparison)
    lines = [
        "# Step 9 Phase A2: oracle versus predicted entity inventories", "",
        "Exploratory results on the thirty reused development papers; all values are mean ± sample standard deviation over seeds 42, 123 and 456.", "",
        "| Inventory | R contract valid | Facilities F1 | Samples F1 | Datasets F1 | Conditional sample refs | Conditional facility refs | End-to-end sample refs | End-to-end facility refs | Latency/paper |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for inventory in INVENTORIES:
        agg = conditions[inventory]["aggregate"]
        show = lambda metric, scale=1.0, digits=3: (  # noqa: E731
            f"{agg[metric]['mean'] / scale:.{digits}f} ± {agg[metric]['sample_sd'] / scale:.{digits}f}")
        lines.append(
            f"| `{inventory}` | {show('papers_reference_contract_valid', digits=1)}/30 | "
            f"{show('facilities_f1')} | {show('samples_f1')} | {show('datasets_f1')} | "
            f"{show('conditional_sample_refs_f1')} | {show('conditional_facility_refs_f1')} | "
            f"{show('end_to_end_sample_refs_f1')} | {show('end_to_end_facility_refs_f1')} | "
            f"{show('mean_latency_ms', 1000, 1)} s |"
        )
    lines += ["", "## Predicted-inventory changes from oracle", "",
              "| Contrast | Δ conditional sample refs | Δ conditional facility refs | Δ end-to-end sample refs | Δ end-to-end facility refs |",
              "|---|---:|---:|---:|---:|"]
    for key in ("self_minus_oracle", "d24_minus_oracle", "self_minus_d24"):
        row = contrasts[key]["metrics"]
        lines.append(
            f"| `{key}` | {row['conditional_sample_refs_f1']['mean']:+.3f} | "
            f"{row['conditional_facility_refs_f1']['mean']:+.3f} | "
            f"{row['end_to_end_sample_refs_f1']['mean']:+.3f} | "
            f"{row['end_to_end_facility_refs_f1']['mean']:+.3f} |"
        )
    lines += ["", "Conditional scores measure linking among evaluator-aligned datasets. End-to-end scores also penalize missing or incorrect datasets and target entities.",
              "", "## Accounting", "", "- Adapters: 3 fixed D24+R8 endpoints.",
              "- Inventory conditions: 3.", "- Papers per condition: 30.",
              "- Total generations: 270.", "- Independent test papers: 0.", ""]
    markdown = "\n".join(lines) + "\n"
    path = root / config.output_dir / "comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(markdown, encoding="utf-8")
    temporary.replace(path)
    return comparison


def generation_complete(output: Path, rows: list[dict]) -> bool:
    progress = read_jsonl(output / "generation.jsonl")
    return [row["paper_id"] for row in progress] == [row["paper_id"] for row in rows]


def unload(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_endpoint_model(root: Path, prepare: dict, training: TrainingConfig,
                        checkpoint: Path, *, offline: bool):
    return reload_adapter(load_base(prepare, training, offline=offline), checkpoint,
                          prepare, config=training)
