"""Held-out M0/M1/M2 generation, scoring, comparison, and MLflow attachment."""
from __future__ import annotations

import gc
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import sys
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
import torch
from transformers import set_seed
import yaml

from meta_data_extraction.data_models import ExtractionResult, Provenance
from meta_data_extraction.evaluation import (
    aggregate_reports,
    build_evaluator_metadata,
    evaluate,
    validate_evaluator_provenance,
)
from meta_data_extraction.utils import (
    get_prompt_version,
    get_schema_refs,
)
from .data import task_prompt
from .memory import chunk_text_mlps
from .parsing import text_hash
from .schema import schema_errors
from .tokenization import load_tokenizer, render_prompt
from .tracking import RunTracker, attach_evaluation, checksum, read_tracking_config
from .training import load_base, reload_adapter
from .training_data import TrainingConfig, file_hash, read_config, verify_prepared
from .validation import generate_answer, validation_context, write_json


CONDITIONS = ("m0", "m1", "m2")
METHODS = {"m0": "base", "m1": "json_only", "m2": "structure_aware"}
DIRECT_ENTITY_FIELDS = ("facilities", "samples", "datasets")
REFERENCE_FIELDS = ("datasets.attrs.sample_refs", "datasets.attrs.facility_refs")


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["1.0.0"]
    prepare_config: str
    json_only_config: str
    structure_aware_config: str
    metrics_config: str
    alignment_config: str
    output_dir: str
    base_run_dir: str
    domain_name: str = Field(min_length=1)
    seed: int = 42
    max_seq_length: PositiveInt = 65536
    max_new_tokens: PositiveInt = 16384
    enable_thinking: Literal[False] = False
    decoding: Literal["greedy"] = "greedy"


def read_evaluation_config(path: Path) -> EvaluationConfig:
    return EvaluationConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _strict_json_object(text: str) -> dict:
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
        raise ValueError("Expected a JSON object")
    return value


def _shared_training_contract(root: Path, config: EvaluationConfig) -> tuple[TrainingConfig, TrainingConfig, dict]:
    if ((root / config.metrics_config).resolve() != (root / "configs/evaluation_metrics.yaml").resolve()
            or (root / config.alignment_config).resolve() != (root / "configs/alignment_helper.yaml").resolve()):
        raise ValueError("Step 7 must use the shared evaluation and alignment configurations")
    json_config = read_config(root / config.json_only_config)
    structure_config = read_config(root / config.structure_aware_config)
    shared = ("prepare_config", "prepared_dir", "seed", "validation_max_new_tokens",
              "max_seq_length", "dtype", "load_in_4bit", "attn_implementation",
              "mlp_chunk_size")
    differences = [name for name in shared if getattr(json_config, name) != getattr(structure_config, name)]
    if differences:
        raise ValueError(f"M1/M2 inference contracts differ: {differences}")
    prepare = yaml.safe_load((root / config.prepare_config).read_text(encoding="utf-8"))
    if (json_config.prepare_config != config.prepare_config
            or prepare["max_seq_length"] != config.max_seq_length
            or prepare["max_new_tokens"] != config.max_new_tokens
            or config.enable_thinking != prepare["enable_thinking"]
            or config.seed != json_config.seed):
        raise ValueError("Evaluation, preparation, and training contracts disagree")
    return json_config, structure_config, prepare


def prepare_test_rows(root: Path, config: EvaluationConfig, tokenizer,
                      tokenizer_identity: dict | None = None) -> tuple[list[dict], dict]:
    """Build audited D prompts from cached text; this function never opens test gold."""
    json_config, _, prepare = _shared_training_contract(root, config)
    _, manifest, drift = verify_prepared(root, json_config)
    if tokenizer_identity is not None and tokenizer_identity != manifest["tokenizer"]:
        raise ValueError("Tokenizer/runtime identity differs from preparation")
    split = json.loads((root / prepare["split_manifest"]).read_text(encoding="utf-8"))
    test_ids = split["test"]
    if len(test_ids) != 30 or len(set(test_ids)) != 30:
        raise ValueError("Expected exactly 30 distinct held-out test papers")
    report = json.loads((root / json_config.prepared_dir / "length_report.json").read_text(encoding="utf-8"))
    expected = {row["paper_id"]: row for row in report["inference"] if row["split"] == "test"}
    if set(expected) != set(test_ids):
        raise ValueError("Length report does not cover the frozen test split")
    schemas = prepare["schemas"]
    rows = []
    for paper_id in test_ids:
        text_path = root / json_config.prepared_dir / "text" / f"{paper_id}.md"
        paper_text = text_path.read_text(encoding="utf-8")
        prompt = task_prompt("D", paper_text, schemas)
        rendered = render_prompt(tokenizer, [{"role": "user", "content": prompt}])
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False, truncation=False)
        audited = expected[paper_id]
        if (len(prompt_ids) != audited["prompt_tokens"]
                or len(prompt_ids) + config.max_new_tokens != audited["required_tokens"]
                or not audited["fits"] or len(prompt_ids) + config.max_new_tokens > config.max_seq_length):
            raise ValueError(f"Test prompt/context audit changed: {paper_id}")
        rows.append({"paper_id": paper_id, "split": "test", "task": "D",
                     "prompt_ids": prompt_ids, "prompt_tokens": len(prompt_ids),
                     "prompt_sha256": text_hash(rendered), "text_sha256": file_hash(text_path)})
    summary = {"test_ids": test_ids, "schemas": schemas, "length_audit": expected,
               "preparation_manifest_sha256": file_hash(root / json_config.prepared_dir / "preparation_manifest.json"),
               "preparation_environment_changes": drift, "manifest": manifest}
    return rows, summary


def _selected_adapter(root: Path, training_config: TrainingConfig) -> tuple[Path, dict]:
    output = root / training_config.output_dir
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    if selection.get("status") != "complete" or selection.get("kind") != "adapter":
        raise ValueError(f"Training selection is not a completed adapter: {output}")
    checkpoint = output / selection["checkpoint"]
    if checksum(checkpoint / "adapter_model.safetensors") != selection["adapter_sha256"]:
        raise ValueError(f"Selected adapter changed: {checkpoint}")
    if json.loads((checkpoint / "inference_precision.json").read_text(encoding="utf-8")) != training_config.precision_policy():
        raise ValueError(f"Selected adapter precision changed: {checkpoint}")
    return checkpoint, selection


def condition_contract(root: Path, config_path: Path, config: EvaluationConfig, condition: str,
                       rows: list[dict], data: dict, tokenizer_identity: dict) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    json_config, structure_config, prepare = _shared_training_contract(root, config)
    training_config = {"m0": json_config, "m1": json_config, "m2": structure_config}[condition]
    adapter, selection = (None, None) if condition == "m0" else _selected_adapter(root, training_config)
    source_paths = [config_path, root / config.metrics_config, root / config.alignment_config,
                    root / "src/meta_data_extraction/finetune/evaluation.py",
                    root / "src/meta_data_extraction/finetune/schema.py",
                    root / "scripts/evaluate_finetune.py",
                    root / "src/meta_data_extraction/evaluation.py",
                    root / "src/meta_data_extraction/alignment.py"]
    contract = {
        "version": config.version, "condition": condition, "method": METHODS[condition],
        "model": {"model_id": prepare["model_id"], "revision": prepare["revision"],
                  "precision": training_config.precision_policy(),
                  "adapter": str(adapter.resolve()) if adapter else None,
                  "adapter_sha256": selection["adapter_sha256"] if selection else None,
                  "selected_step": selection["step"] if selection else 0},
        "decoding": {"method": config.decoding, "enable_thinking": config.enable_thinking,
                     "max_new_tokens": config.max_new_tokens, "max_seq_length": config.max_seq_length,
                     "seed": config.seed, "num_beams": 1, "repetition_penalty": 1.0},
        "data": {"split": "test", "paper_ids": [row["paper_id"] for row in rows],
                 "prompts": [{k: row[k] for k in ("paper_id", "prompt_tokens", "prompt_sha256", "text_sha256")}
                             for row in rows],
                 "preparation_manifest_sha256": data["preparation_manifest_sha256"],
                 "test_gold_in_prompt": False},
        "tokenizer": tokenizer_identity,
        "evaluator": build_evaluator_metadata(root / config.metrics_config),
        "sources": {str(path.relative_to(root)): file_hash(path) for path in source_paths},
    }
    if contract["evaluator"]["schema_version_mismatch"]:
        raise ValueError("Evaluator schema versions do not match the current schemas")
    return contract


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
                         encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_condition_output(output: Path, contract: dict):
    manifest = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")) != contract:
            raise ValueError(f"Evaluation output is nonempty or belongs to a different contract: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest, contract)


def _prediction_result(config: EvaluationConfig, condition: str, row: dict, generated: dict,
                       parsed: dict, warnings: list[str], latency_ms: float,
                       schemas: list[str]) -> ExtractionResult:
    return ExtractionResult(
        document_id=row["paper_id"], domain_name=config.domain_name, output=parsed,
        raw_output=generated.get("text"),
        provenance=Provenance(config_name=condition, prompt_version=get_prompt_version(),
                              schemas=get_schema_refs(schemas), input_tokens=row["prompt_tokens"],
                              output_tokens=generated.get("generated_tokens"), latency_ms=round(latency_ms, 2)),
        warnings=warnings,
    )


def generate_condition(model, tokenizer, rows: list[dict], config: EvaluationConfig,
                       condition: str, output: Path, schemas: list[str]) -> list[dict]:
    """Generate missing papers and atomically commit progress after each one."""
    progress_path = output / "generation.jsonl"
    progress = _read_jsonl(progress_path)
    expected_ids = [row["paper_id"] for row in rows]
    if [row["paper_id"] for row in progress] != expected_ids[:len(progress)]:
        raise ValueError("Generation progress is not an ordered prefix of the test split")
    for old in progress:
        prediction = output / old["prediction_file"]
        raw = output / old["raw_file"]
        if checksum(prediction) != old["prediction_sha256"] or checksum(raw) != old["raw_sha256"]:
            raise ValueError(f"Saved generation changed: {old['paper_id']}")

    for row in rows[len(progress):]:
        paper_id = row["paper_id"]
        raw_path = output / "raw" / f"{paper_id}.txt"
        prediction_path = output / "predictions" / f"{paper_id}.json"
        started = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        warnings: list[str] = []
        try:
            with validation_context(model):
                generated = generate_answer(model, tokenizer, row, config.max_new_tokens)
            generated["schemas"] = schemas
            status = "generated"
            try:
                parsed = _strict_json_object(generated["text"])
            except (ValueError, json.JSONDecodeError) as exc:
                parsed, status = {}, "invalid_json"
                warnings.append(f"Strict JSON parse failed: {exc}")
            validation_errors = schema_errors(parsed, schemas) if status == "generated" else []
            if validation_errors:
                warnings.append(f"Schema validation found {len(validation_errors)} error(s).")
            if generated["hit_output_limit"]:
                warnings.append("Generation reached max_new_tokens without an EOS marker.")
        except torch.OutOfMemoryError as exc:
            generated = {"text": "", "generated_tokens": 0, "ended_with_eos": False,
                         "hit_output_limit": False, "schemas": schemas}
            parsed, validation_errors, status = {}, [], "generation_failed"
            warnings.append(f"OutOfMemoryError: {exc}")
            torch.cuda.empty_cache()
        except Exception as exc:
            generated = {"text": "", "generated_tokens": 0, "ended_with_eos": False,
                         "hit_output_limit": False, "schemas": schemas}
            parsed, validation_errors, status = {}, [], "generation_failed"
            warnings.append(f"{type(exc).__name__}: {exc}")
        latency_ms = (time.monotonic() - started) * 1000
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(generated["text"], encoding="utf-8")
        result = _prediction_result(config, condition, row, generated, parsed, warnings, latency_ms, schemas)
        write_json(prediction_path, result.model_dump(mode="json"))
        record = {
            "paper_id": paper_id, "status": status, "json_valid": status == "generated",
            "schema_valid": status == "generated" and not validation_errors,
            "schema_errors": validation_errors, "prompt_tokens": row["prompt_tokens"],
            "generated_tokens": generated["generated_tokens"],
            "ended_with_eos": generated["ended_with_eos"],
            "hit_output_limit": generated["hit_output_limit"], "latency_ms": round(latency_ms, 2),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
            "raw_file": raw_path.relative_to(output).as_posix(),
            "prediction_file": prediction_path.relative_to(output).as_posix(),
            "raw_sha256": checksum(raw_path), "prediction_sha256": checksum(prediction_path),
            "warnings": warnings,
        }
        progress.append(record)
        _write_jsonl(progress_path, progress)
    return progress


def _empty_prediction(path: Path, paper_id: str, config: EvaluationConfig, condition: str, warning: str):
    result = ExtractionResult(document_id=paper_id, domain_name=config.domain_name, output={},
                              provenance=Provenance(config_name=condition), warnings=[warning])
    write_json(path, result.model_dump(mode="json"))


def _mean_field_score(fields: dict, prefixes: tuple[str, ...]) -> float:
    values = []
    for path, result in fields.items():
        if path.startswith(prefixes):
            value = result.get("f1", result.get("mean_score"))
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
    return math.fsum(values) / len(values) if values else 0.0


def _summary(generation: list[dict], fields: dict, fallback_count: int) -> dict[str, float | int | None]:
    total = len(generation)
    if total != 30:
        raise ValueError(f"Expected 30 generated papers, found {total}")
    result: dict[str, float | int | None] = {
        "papers_total": total,
        "papers_json_valid": sum(row["json_valid"] for row in generation),
        "papers_schema_valid": sum(row["schema_valid"] for row in generation),
        "generation_failures": sum(row["status"] == "generation_failed" for row in generation),
        "output_limit_failures": sum(row["hit_output_limit"] for row in generation),
        "scoring_fallbacks": fallback_count,
        "json_valid_rate": sum(row["json_valid"] for row in generation) / total,
        "schema_valid_rate": sum(row["schema_valid"] for row in generation) / total,
        "mean_latency_ms": math.fsum(row["latency_ms"] for row in generation) / total,
        "mean_generated_tokens": math.fsum(row["generated_tokens"] for row in generation) / total,
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in generation),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in generation),
        "generic_metadata_macro_score": _mean_field_score(fields, ("bibliographic.", "document_overview.")),
    }
    for path in DIRECT_ENTITY_FIELDS:
        result[f"{path}_f1"] = fields[path]["f1"]
    # Reference quality is conditional on aligned datasets. If there are no
    # aligned pairs, preserve the evaluator's undefined result instead of
    # inventing either a zero or a perfect score.
    result["sample_refs_f1"] = fields[REFERENCE_FIELDS[0]].get("f1")
    result["facility_refs_f1"] = fields[REFERENCE_FIELDS[1]].get("f1")
    return result


def score_condition(root: Path, config: EvaluationConfig, condition: str, output: Path) -> dict:
    """Score cached predictions; broken outputs receive an explicit empty fallback."""
    contract = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected_evaluator = build_evaluator_metadata(root / config.metrics_config)
    validate_evaluator_provenance(contract["evaluator"], expected_evaluator, source=str(output / "manifest.json"))
    generation = _read_jsonl(output / "generation.jsonl")
    expected_ids = contract["data"]["paper_ids"]
    if [row["paper_id"] for row in generation] != expected_ids or len(generation) != 30:
        raise ValueError("Generation is incomplete or does not match the frozen test split")

    split = json.loads((root / "data/finetune/splits.json").read_text(encoding="utf-8"))
    frozen_sources = split["audit"]["source_files"]
    reports, valid_reports, paper_scores = [], [], []
    gold_hashes, fallback_count = {}, 0
    for generated in generation:
        paper_id = generated["paper_id"]
        prediction_path = output / generated["prediction_file"]
        if checksum(prediction_path) != generated["prediction_sha256"]:
            raise ValueError(f"Prediction changed after generation: {paper_id}")
        gold_path = root / "data" / "gold" / f"{paper_id}.json"
        gold_hashes[paper_id] = file_hash(gold_path)
        expected_gold_hash = frozen_sources[f"data/gold/{paper_id}.json"]["sha256"]
        if gold_hashes[paper_id] != expected_gold_hash:
            raise ValueError(f"Frozen test gold changed: {paper_id}")
        used_prediction = generated["json_valid"]
        if not used_prediction:
            fallback_count += 1
        try:
            report = evaluate(gold_path, prediction_path, root / config.metrics_config)
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            if used_prediction:
                fallback_count += 1
            used_prediction = False
            fallback_path = output / "fallback_predictions" / f"{paper_id}.json"
            _empty_prediction(fallback_path, paper_id, config, condition,
                              f"Evaluator fallback after {type(exc).__name__}: {exc}")
            report = evaluate(gold_path, fallback_path, root / config.metrics_config)
        report_path = output / "reports" / f"{paper_id}.json"
        write_json(report_path, report)
        reports.append(report)
        if generated["json_valid"] and used_prediction:
            valid_reports.append(report)
        paper_scores.append({
            "paper_id": paper_id, "generation_status": generated["status"],
            "json_valid": generated["json_valid"], "schema_valid": generated["schema_valid"],
            "used_generated_prediction": used_prediction,
            "report_file": report_path.relative_to(output).as_posix(),
            "entity_f1": {field: report["fields"][field]["f1"] for field in DIRECT_ENTITY_FIELDS},
            "sample_refs_f1": report["fields"][REFERENCE_FIELDS[0]].get("f1"),
            "facility_refs_f1": report["fields"][REFERENCE_FIELDS[1]].get("f1"),
        })

    all_fields = aggregate_reports(reports)
    valid_fields = aggregate_reports(valid_reports) if valid_reports else {}
    metrics = {
        "version": "1.0.0", "condition": condition,
        "method": METHODS.get(condition, "ablation_fixed_endpoint"),
        "aggregation_policy": {
            "primary": "all 30 papers; invalid/unscorable generations are evaluated as empty predictions",
            "secondary": "JSON-valid predictions successfully accepted by the evaluator",
            "entity_and_reference_metrics": "pooled TP/FP/FN",
            "scalar_metrics": "paper mean; entity attributes are matched-pair weighted",
        },
        "summary": _summary(generation, all_fields, fallback_count),
        "all_papers": {"n_papers": len(reports), "fields": all_fields},
        "valid_predictions_only": {"n_papers": len(valid_reports), "fields": valid_fields},
        "papers": paper_scores, "gold_sha256": gold_hashes,
        "evaluator": expected_evaluator,
    }
    write_json(output / "metrics.json", metrics)
    _write_jsonl(output / "paper_scores.jsonl", paper_scores)
    return metrics


def _training_cost(root: Path, training_config: TrainingConfig | None) -> dict:
    if training_config is None:
        return {"optimizer_updates": 0, "training_runtime_seconds": 0.0,
                "full_run_exposure": None, "selected_exposure": None}
    summary = json.loads((root / training_config.output_dir / "training_summary.json").read_text(encoding="utf-8"))
    return {"optimizer_updates": summary["global_step"],
            "training_runtime_seconds": summary["metrics"]["train_runtime"],
            "full_run_exposure": summary["exposure"],
            "selected_step": summary["selection"]["step"],
            "selected_exposure": summary["selection"]["exposure"]}


def compare_conditions(root: Path, config: EvaluationConfig) -> dict:
    output_root = root / config.output_dir
    json_config, structure_config, _ = _shared_training_contract(root, config)
    metrics = {condition: json.loads((output_root / condition / "metrics.json").read_text(encoding="utf-8"))
               for condition in CONDITIONS}
    paper_orders = [[row["paper_id"] for row in metrics[c]["papers"]] for c in CONDITIONS]
    if (len(paper_orders[0]) != 30 or len(set(paper_orders[0])) != 30
            or any(order != paper_orders[0] for order in paper_orders[1:])):
        raise ValueError("M0/M1/M2 metrics do not cover identical ordered papers")
    if any(metrics[c].get("condition") not in {None, c} for c in CONDITIONS):
        raise ValueError("A metrics file is labeled for the wrong condition")
    evaluator = metrics["m0"]["evaluator"]
    if any(metrics[c]["evaluator"] != evaluator for c in CONDITIONS[1:]):
        raise ValueError("M0/M1/M2 evaluator provenance differs")
    costs = {"m0": _training_cost(root, None), "m1": _training_cost(root, json_config),
             "m2": _training_cost(root, structure_config)}
    comparison = {
        "version": "1.0.0", "paper_ids": paper_orders[0], "evaluator": evaluator,
        "fallback_policy": metrics["m0"]["aggregation_policy"]["primary"],
        "conditions": {condition: {"method": METHODS[condition],
                                    "quality": metrics[condition]["summary"],
                                    "training_cost": costs[condition]}
                       for condition in CONDITIONS},
        "caveats": ["Two validation papers", "One training seed",
                    "M1 and M2 have equal optimizer updates but unequal task/token exposure",
                    "Test results must not be used to change checkpoints or training settings"],
    }
    write_json(output_root / "comparison.json", comparison)
    def display(value):
        return "n/a" if value is None else f"{value:.3f}"

    headers = ["condition", "JSON valid", "schema valid", "facilities F1", "samples F1",
               "datasets F1", "sample refs F1", "facility refs F1", "generic macro"]
    lines = ["# Fine-tuning test comparison", "", "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    for condition in CONDITIONS:
        quality = comparison["conditions"][condition]["quality"]
        values = [condition.upper(), display(quality["json_valid_rate"]), display(quality["schema_valid_rate"]),
                  display(quality["facilities_f1"]), display(quality["samples_f1"]),
                  display(quality["datasets_f1"]), display(quality["sample_refs_f1"]),
                  display(quality["facility_refs_f1"]), display(quality["generic_metadata_macro_score"])]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Primary aggregation includes all 30 papers; malformed or unscorable outputs are treated as empty predictions.",
                  "", "These held-out results must not be used to revise checkpoint selection or training settings."])
    (output_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def _mlflow_metrics(metrics: dict) -> dict[str, float]:
    result = {key: float(value) for key, value in metrics["summary"].items()
              if isinstance(value, (int, float)) and math.isfinite(value)}
    for path, values in metrics["all_papers"]["fields"].items():
        for key in ("f1", "mean_score"):
            value = values.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                result[f"fields/{path}/{key}"] = float(value)
    return result


def attach_all_to_mlflow(root: Path, config: EvaluationConfig, comparison: Path) -> dict[str, str]:
    """Attach a complete three-way result set; create M0's evaluation-only run once."""
    json_config, structure_config, _ = _shared_training_contract(root, config)
    outputs = {"m0": root / config.base_run_dir,
               "m1": root / json_config.output_dir, "m2": root / structure_config.output_dir}
    base_output = outputs["m0"]
    if not (base_output / "mlflow_run.json").exists():
        settings = read_tracking_config(root / json_config.tracking_config, require_ready=True)
        eval_manifest = json.loads((root / config.output_dir / "m0" / "manifest.json").read_text(encoding="utf-8"))
        contract = {"config": {"method": "base", "seed": config.seed,
                               "max_seq_length": config.max_seq_length,
                               "max_new_tokens": config.max_new_tokens},
                    "data": {"tokenizer": eval_manifest["tokenizer"], "split": eval_manifest["data"]},
                    "runtime": {"python": sys.version.split()[0], "torch": torch.__version__},
                    "sources": eval_manifest["sources"], "evaluation_only": True}
        tracker = RunTracker.start(settings, base_output, contract, method="base")
        tracker.finish("FINISHED")
    run_ids = {}
    for condition in CONDITIONS:
        condition_output = root / config.output_dir / condition
        metrics = json.loads((condition_output / "metrics.json").read_text(encoding="utf-8"))
        run_ids[condition] = attach_evaluation(
            outputs[condition], _mlflow_metrics(metrics),
            {"condition": condition_output, "comparison": comparison}, metrics["evaluator"])
    return run_ids


def runtime_versions() -> dict:
    packages = {}
    for package in ("torch", "transformers", "tokenizers", "peft", "bitsandbytes",
                    "flash-linear-attention", "fla-core"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"python": sys.version.split()[0], **packages}


def run_condition(root: Path, config_path: Path, config: EvaluationConfig, condition: str,
                  tokenizer, tokenizer_identity: dict, rows: list[dict], data: dict, *,
                  offline: bool, score_only: bool) -> dict:
    output = root / config.output_dir / condition
    contract = condition_contract(root, config_path, config, condition, rows, data, tokenizer_identity)
    contract["runtime"] = runtime_versions()
    prepare_condition_output(output, contract)
    if not score_only:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("Generation requires one CUDA GPU with native BF16 support")
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU for evaluation")
        json_config, structure_config, prepare = _shared_training_contract(root, config)
        training_config = {"m0": json_config, "m1": json_config, "m2": structure_config}[condition]
        set_seed(config.seed)
        model = load_base(prepare, training_config, offline=offline)
        if condition != "m0":
            checkpoint, _ = _selected_adapter(root, training_config)
            model = reload_adapter(model, checkpoint, prepare, config=training_config)
        else:
            # Adapters are reloaded with the training-time chunk contract. Apply
            # it explicitly to M0 so all three inference paths are identical.
            chunk_text_mlps(model, training_config.mlp_chunk_size)
        try:
            generate_condition(model, tokenizer, rows, config, condition, output, data["schemas"])
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    return score_condition(root, config, condition, output)

