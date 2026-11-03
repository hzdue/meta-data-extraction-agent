#!/usr/bin/env python
"""Run one frozen Step 9 follow-up evaluation phase."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import set_seed

from meta_data_extraction.finetune.ablation import read_ablation_config
from meta_data_extraction.finetune.ablation_evaluation import endpoint_adapter
from meta_data_extraction.finetune.evaluation import read_evaluation_config, runtime_versions
from meta_data_extraction.finetune.followup_analysis import read_phase_a2_config
from meta_data_extraction.finetune.followup_evaluation import (
    INVENTORIES, compare, evaluation_contract, generate_condition, generation_complete,
    label, load_endpoint_model, prepare_context, prepare_output, prepare_rows,
    score_condition, unload, validate_followup_config,
)
from meta_data_extraction.finetune.tokenization import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("a2",))
    parser.add_argument("--config", type=Path, default=Path("configs/finetune_followup_a2.yaml"))
    parser.add_argument("--inventory", choices=(*INVENTORIES, "all"), default="all")
    parser.add_argument("--offline", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    mode.add_argument("--compare-only", action="store_true")
    args = parser.parse_args(argv)

    config_path = (ROOT / args.config).resolve()
    config = validate_followup_config(read_phase_a2_config(config_path))
    base_config = read_evaluation_config(ROOT / config.evaluation_config)
    matrix = read_ablation_config(ROOT / config.ablation_config)
    selected = list(INVENTORIES) if args.inventory == "all" else [args.inventory]

    endpoints = {}
    for seed in config.seeds:
        training, checkpoint, identity = endpoint_adapter(
            ROOT, matrix, config.adapter_condition, seed)
        endpoints[seed] = (training, checkpoint, identity)
    if args.plan:
        rows = [{
            "seed": seed, "inventory": inventory,
            "adapter": str(endpoints[seed][1].relative_to(ROOT)),
            "adapter_sha256": endpoints[seed][2]["adapter_sha256"],
            "output": f"{config.output_dir}/{label(inventory, seed)}",
            "papers": config.expected_papers,
        } for seed in config.seeds for inventory in selected]
        print(json.dumps({"status": "planned_not_executed", "phase": "A2",
                          "generation_count": len(rows) * config.expected_papers,
                          "runs": rows}, indent=2))
        return 0
    if args.compare_only:
        if args.inventory != "all":
            raise ValueError("--compare-only requires --inventory all")
        compare(ROOT, config)
        return 0

    preparation = endpoints[config.seeds[0]][0]
    prepared_config = json.loads((ROOT / preparation.prepared_dir /
                                  "preparation_manifest.json").read_text(encoding="utf-8"))["config"]
    tokenizer, tokenizer_identity = load_tokenizer(prepared_config, local_files_only=args.offline)
    context = prepare_context(ROOT, config, preparation, tokenizer_identity)

    rowsets, contracts = {}, {}
    for seed in config.seeds:
        training, checkpoint, identity = endpoints[seed]
        for inventory in selected:
            key = (seed, inventory)
            rows = prepare_rows(ROOT, config, context, tokenizer, inventory, seed)
            contract = evaluation_contract(
                ROOT, config_path, config, base_config, context, tokenizer_identity,
                inventory, seed, rows, training, checkpoint, identity)
            contract["runtime"] = runtime_versions()
            rowsets[key], contracts[key] = rows, contract
    if args.dry_run:
        result = [{
            "condition": contract["condition"], "papers": len(rowsets[key]),
            "max_prompt_tokens": max(row["prompt_tokens"] for row in rowsets[key]),
            "max_required_tokens": max(row["prompt_tokens"] + config.max_new_tokens
                                       for row in rowsets[key]),
            "gold_entity_inventory_in_prompt": contract["data"]["gold_entity_inventory_in_prompt"],
            "adapter_sha256": contract["model"]["adapter_sha256"],
        } for key, contract in contracts.items()]
        print(json.dumps({"status": "dry_run_passed", "phase": "A2", "runs": result}, indent=2))
        return 0

    if not args.score_only:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("Phase A2 generation requires one CUDA GPU with native BF16 support")
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU for Phase A2")

    for seed in config.seeds:
        training, checkpoint, _ = endpoints[seed]
        pending = []
        for inventory in selected:
            output = ROOT / config.output_dir / label(inventory, seed)
            prepare_output(output, contracts[(seed, inventory)])
            if not generation_complete(output, rowsets[(seed, inventory)]):
                pending.append(inventory)
        model = None
        if pending and not args.score_only:
            print(f"model_load_start seed={seed} pending={','.join(pending)}", flush=True)
            set_seed(seed)
            model = load_endpoint_model(
                ROOT, context["prepare"], training, checkpoint, offline=args.offline)
        try:
            for inventory in selected:
                output = ROOT / config.output_dir / label(inventory, seed)
                print(f"phase_a2_start inventory={inventory} seed={seed}", flush=True)
                if not args.score_only:
                    generate_condition(
                        model, tokenizer, rowsets[(seed, inventory)], config, base_config,
                        inventory, seed, output, context["schemas"])
                metrics = score_condition(ROOT, config, base_config, inventory, seed, output)
                print(json.dumps({"inventory": inventory, "seed": seed,
                                  "summary": metrics["summary"]}, indent=2), flush=True)
        finally:
            if model is not None:
                unload(model)
            else:
                gc.collect()
    if args.inventory == "all":
        compare(ROOT, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
