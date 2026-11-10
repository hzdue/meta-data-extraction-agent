#!/usr/bin/env python
"""Plan or run the frozen no-update LoRA gradient interaction diagnostic."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import torch
from transformers import set_seed
import yaml

from meta_data_extraction.finetune.ablation_evaluation import endpoint_adapter
from meta_data_extraction.finetune.gradient_interaction import (
    combine_results, diagnostic_train_mode, install_fused_answer_loss, prepare_examples,
    read_gradient_config, seed_result, verify_contract,
)
from meta_data_extraction.finetune.training import load_base, reload_adapter
from meta_data_extraction.finetune.validation import write_json


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/finetune_gradient_interaction.yaml"))
    parser.add_argument("--mode", choices=("plan", "run", "summarize"), required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, choices=(42, 123, 456),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config_path = (ROOT / args.config).resolve()
    config = read_gradient_config(config_path)
    matrix, base_training, endpoints = verify_contract(ROOT, config_path, config)
    if args.mode == "run" and args.seed is None:
        # Activation offload uses a pinned-memory allocator that is intentionally
        # process-scoped. Fresh children guarantee it is released between seeds.
        for seed in config.seeds:
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--config", str(config_path), "--mode", "run", "--seed", str(seed)]
            if args.offline:
                command.append("--offline")
            subprocess.run(command, cwd=ROOT, check=True)
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--config", str(config_path), "--mode", "summarize"]
        if args.offline:
            command.append("--offline")
        subprocess.run(command, cwd=ROOT, check=True)
        return 0
    tokenizer, examples, papers, data_summary = prepare_examples(
        ROOT, base_training, config, offline=args.offline)
    output = ROOT / config.output_dir
    plan = {"status": "planned_not_executed", "seeds": config.seeds, "tasks": config.tasks,
            "papers": papers, "backward_passes": len(examples) * len(config.seeds),
            "endpoints": endpoints, "output_dir": config.output_dir,
            "optimizer_steps": 0, "mlflow_attachment": False}
    if args.mode == "plan":
        print(json.dumps(plan, indent=2))
        return 0
    if args.mode == "summarize":
        paths = [output / f"seed_{seed}.json" for seed in config.seeds]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"Missing completed seed artifacts: {missing}")
        results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        final = combine_results(ROOT, config_path, config, endpoints, results, data_summary)
        print(json.dumps(final["summary"], indent=2))
        return 0
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("Gradient diagnostic requires one CUDA GPU with native BF16 support")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU for the diagnostic")
    preparation = yaml.safe_load((ROOT / base_training.prepare_config).read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plan.json", plan)
    seed = args.seed
    identity = endpoints[config.seeds.index(seed)]
    seed_path = output / f"seed_{seed}.json"
    if seed_path.is_file():
        saved = json.loads(seed_path.read_text(encoding="utf-8"))
        if saved["endpoint"]["adapter_sha256"] != identity["adapter_sha256"]:
            raise ValueError("Existing seed artifact belongs to a different endpoint")
        print(json.dumps({"seed": seed, "status": "reused_complete_seed_artifact"}))
        return 0
    training, checkpoint, checked = endpoint_adapter(ROOT, matrix, config.condition, seed)
    if checked != {key: identity[key] for key in checked}:
        raise ValueError("Endpoint identity changed after planning")
    set_seed(seed, deterministic=True)
    model = reload_adapter(load_base(preparation, training, offline=args.offline), checkpoint,
                           preparation, trainable=True, config=training)
    install_fused_answer_loss(model)
    diagnostic_train_mode(model)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    try:
        result = seed_result(ROOT, config, seed, identity, training, model, tokenizer, examples)
        write_json(seed_path, result)
        print(json.dumps({"seed": seed, "status": "complete",
                          "runtime_seconds": result["runtime_seconds"],
                          "checkpoint_unchanged": result["checkpoint_sha256_before"]
                          == result["checkpoint_sha256_after"]}))
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
