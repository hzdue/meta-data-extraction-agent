"""Reload a completed run's selected model and reproduce its validation loss."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from meta_data_extraction.finetune.training import build_trainer, load_base, reload_adapter
from meta_data_extraction.finetune.training_data import prepare_training, read_config
from meta_data_extraction.finetune.validation import validation_context, validation_losses, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = read_config(ROOT / args.config)
    output = (ROOT / config.output_dir).resolve()
    manifest = json.loads((output / "run_manifest.json").read_text())
    if manifest["config"] != config.model_dump():
        raise ValueError("Reload configuration differs from the saved run")
    history = json.loads((output / "validation_history.json").read_text())["history"]
    selection = json.loads((output / "selection.json").read_text())
    if selection["status"] != "complete":
        raise ValueError("Reload check requires completed checkpoint selection")
    if history[-1]["step"] != config.max_steps:
        raise ValueError("Reload check requires a completed run")
    tokenizer, records, validation, _ = prepare_training(ROOT, config, offline=args.offline)
    preparation = yaml.safe_load((ROOT / config.prepare_config).read_text())
    base = load_base(preparation, config, offline=args.offline)
    if selection["kind"] == "adapter":
        model = reload_adapter(base, output / selection["checkpoint"], preparation, config=config)
    else:
        model = base
    trainer = build_trainer(model, tokenizer, records, config, output / "reload_check")
    with validation_context(model):
        actual = validation_losses(trainer, validation)
    expected = next(row["papers"] for row in history if row["step"] == selection["step"])
    passed = all(a["paper_id"] == e["paper_id"] and a["supervised_tokens"] == e["supervised_tokens"]
                 and math.isclose(a["loss"], e["loss"], rel_tol=1e-4, abs_tol=1e-5)
                 for a, e in zip(actual, expected, strict=True))
    report = {"status": "passed" if passed else "failed", "selected_step": selection["step"],
              "selected_kind": selection["kind"], "expected": expected, "actual": actual,
              "inference_precision": config.precision_policy(), "relative_tolerance": 1e-4,
              "absolute_tolerance": 1e-5}
    write_json(output / "adapter_reload_report.json", report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise ValueError("Reloaded adapter validation losses differ")


if __name__ == "__main__":
    main()
