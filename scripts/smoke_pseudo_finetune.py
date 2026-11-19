#!/usr/bin/env python
"""Plan or run an isolated Stage-A pseudo-label GPU feasibility probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from meta_data_extraction.finetune.pseudo_experiment import (
    build_stage_a_contract,
    prepare_pseudo_stage_a,
)
from meta_data_extraction.finetune.smoke import CASES, hardware_info, probe_plan, run_probe
from meta_data_extraction.finetune.training import inspect_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/finetune_pseudo_only.yaml")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case", choices=("plan", *CASES), default="plan")
    parser.add_argument("--output-dir", type=Path,
                        help="Required for GPU cases; must be new or empty")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.case != "plan" and args.output_dir is None:
        parser.error("GPU cases require --output-dir on persistent storage")

    try:
        config_path = (ROOT / args.config).resolve()
        config, tokenizer, records, validation, summary, _ = prepare_pseudo_stage_a(
            ROOT, config_path, offline=args.offline
        )
        preparation = yaml.safe_load(
            (ROOT / config.prepare_config).read_text(encoding="utf-8")
        )
        architecture = inspect_model(preparation, offline=args.offline)
        if args.case == "plan":
            result = {
                "status": "plan_only",
                "gpu_feasibility": "not_tested",
                "hardware": hardware_info(),
                "production_policy": {
                    key: config.model_dump()[key]
                    for key in (
                        "max_seq_length", "microbatch_size",
                        "gradient_accumulation_steps", "gradient_checkpointing",
                        "offload_activations", "mlp_chunk_size", "dtype",
                        "load_in_4bit", "attn_implementation",
                    )
                },
                "data": summary,
                "cases": probe_plan(records, validation, config),
            }
        else:
            output = (ROOT / args.output_dir).resolve()
            result = run_probe(
                ROOT, config, tokenizer, records, validation, summary, architecture,
                preparation, args.case, output, offline=args.offline,
                contract_builder=lambda probe_config: build_stage_a_contract(
                    ROOT, config_path, probe_config, summary, architecture
                ),
                entrypoint="scripts/smoke_pseudo_finetune.py",
            )
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"Pseudo smoke check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
