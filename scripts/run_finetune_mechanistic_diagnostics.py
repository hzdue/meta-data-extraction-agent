#!/usr/bin/env python
"""Run the frozen lightweight mechanistic diagnostics over completed adapters."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import set_seed

from meta_data_extraction.evaluation import build_evaluator_metadata
from meta_data_extraction.finetune.evaluation import read_evaluation_config, runtime_versions
from meta_data_extraction.finetune.followup_analysis import read_phase_a2_config
from meta_data_extraction.finetune.followup_evaluation import (
    generate_condition, generation_complete, load_endpoint_model, prepare_context,
    prepare_output, prepare_rows, score_condition, unload,
)
from meta_data_extraction.finetune.mechanistic_diagnostics import (
    CONDITIONS, SEEDS, build_error_decomposition, compare_oracle, endpoint,
    error_markdown, oracle_contract, oracle_markdown, oracle_output, read_config,
    relabel_oracle_predictions, verify_frozen_configs, write_result,
)
from meta_data_extraction.finetune.tokenization import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def _paths(config):
    output = ROOT / config.output_dir
    return {
        "error_json": output / "reference_error_decomposition.json",
        "error_markdown": output / "reference_error_decomposition.md",
        "oracle_json": output / "oracle_comparison.json",
        "oracle_markdown": output / "oracle_comparison.md",
    }


def _run_error_decomposition(config_path, config):
    result = build_error_decomposition(ROOT, config_path, config)
    paths = _paths(config)
    write_result(paths["error_json"], result)
    write_result(paths["error_markdown"], error_markdown(result))
    print(error_markdown(result))
    return result


def _oracle_context(config, offline):
    phase_a2 = read_phase_a2_config(ROOT / config.phase_a2_config).phase_a2
    base_config = read_evaluation_config(ROOT / phase_a2.evaluation_config)
    training, _, _ = endpoint(ROOT, config, "d_only", SEEDS[0])
    preparation = json.loads((ROOT / training.prepared_dir /
                              "preparation_manifest.json").read_text(encoding="utf-8"))["config"]
    tokenizer, tokenizer_identity = load_tokenizer(preparation, local_files_only=offline)
    context = prepare_context(ROOT, phase_a2, training, tokenizer_identity)
    evaluator = build_evaluator_metadata(ROOT / base_config.metrics_config)
    if evaluator["schema_version_mismatch"]:
        raise ValueError("Evaluator schema versions do not match")
    return phase_a2, base_config, tokenizer, tokenizer_identity, context, evaluator


def _oracle_runs(config_path, config, condition, *, offline, dry_run, score_only):
    selected = list(config.oracle_comparison.generated_conditions) if condition == "all" else [condition]
    if any(value not in config.oracle_comparison.generated_conditions for value in selected):
        raise ValueError("Only D32 and D24 require new oracle generations")
    phase_a2, base_config, tokenizer, tokenizer_identity, context, evaluator = \
        _oracle_context(config, offline)
    jobs = []
    for model_condition in selected:
        for seed in SEEDS:
            training, checkpoint, identity = endpoint(ROOT, config, model_condition, seed)
            rows = prepare_rows(ROOT, phase_a2, context, tokenizer, "oracle", seed)
            contract = oracle_contract(
                ROOT, config_path, config, phase_a2, context, tokenizer_identity,
                model_condition, seed, rows, training, checkpoint, identity, evaluator)
            contract["runtime"] = runtime_versions()
            jobs.append((model_condition, seed, training, checkpoint, rows, contract))
    if dry_run:
        print(json.dumps({"status": "dry_run_passed", "runs": [{
            "condition": condition, "seed": seed,
            "adapter": str(checkpoint.relative_to(ROOT)),
            "adapter_sha256": contract["model"]["adapter_sha256"],
            "papers": len(rows), "max_prompt_tokens": max(row["prompt_tokens"] for row in rows),
            "max_required_tokens": max(row["prompt_tokens"] + phase_a2.max_new_tokens for row in rows),
        } for condition, seed, _, checkpoint, rows, contract in jobs]}, indent=2))
        return
    if not score_only:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("Oracle generation requires one CUDA GPU with native BF16 support")
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU for oracle generation")
    for model_condition, seed, training, checkpoint, rows, contract in jobs:
        output = oracle_output(ROOT, config, model_condition, seed)
        prepare_output(output, contract)
        model = None
        if not score_only and not generation_complete(output, rows):
            set_seed(seed)
            model = load_endpoint_model(
                ROOT, context["prepare"], training, checkpoint, offline=offline)
        try:
            if not score_only:
                generate_condition(model, tokenizer, rows, phase_a2, base_config,
                                   "oracle", seed, output, context["schemas"])
                relabel_oracle_predictions(output, model_condition, seed)
            metrics = score_condition(ROOT, phase_a2, base_config, "oracle", seed, output)
            metrics["phase"] = "mechanistic_oracle"
            metrics["condition"] = f"oracle_{model_condition}_seed{seed}"
            metrics["model_condition"] = model_condition
            write_result(output / "metrics.json", metrics)
            print(json.dumps({"condition": model_condition, "seed": seed,
                              "summary": metrics["summary"]}, indent=2), flush=True)
        finally:
            if model is not None:
                unload(model)
            else:
                gc.collect()


def _compare_oracle(config_path, config):
    result = compare_oracle(ROOT, config_path, config)
    paths = _paths(config)
    write_result(paths["oracle_json"], result)
    write_result(paths["oracle_markdown"], oracle_markdown(result))
    print(oracle_markdown(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/finetune_mechanistic_diagnostics.yaml"))
    parser.add_argument("--mode", required=True,
                        choices=("plan", "decompose", "oracle", "compare", "all"))
    parser.add_argument("--condition", choices=("d_only", "d32", "all"), default="all")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)
    config_path = (ROOT / args.config).resolve()
    config = read_config(config_path)
    verify_frozen_configs(ROOT, config)
    if args.mode == "plan":
        print(json.dumps({
            "status": "planned_not_executed",
            "error_decomposition": {"conditions": list(CONDITIONS), "new_generations": 0},
            "oracle": {"reused": {"condition": "matched_d_dr", "generations": 90},
                       "new": {"conditions": ["d32", "d_only"], "generations": 180},
                       "primary_contrast": config.oracle_comparison.primary_contrast,
                       "model_selection_allowed": False},
            "outputs": {key: str(path.relative_to(ROOT)) for key, path in _paths(config).items()},
        }, indent=2))
        return 0
    if args.mode in {"decompose", "all"}:
        _run_error_decomposition(config_path, config)
    if args.mode in {"oracle", "all"}:
        _oracle_runs(config_path, config, args.condition, offline=args.offline,
                     dry_run=args.dry_run, score_only=args.score_only)
    if args.mode in {"compare", "all"} and not args.dry_run:
        _compare_oracle(config_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
