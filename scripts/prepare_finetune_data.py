"""Prepare Step 2 artifacts locally; never load model weights or train."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import yaml

if __package__:
    from .audit_finetune_data import fingerprint, read_json, validate_fields, validate_references
else:
    from audit_finetune_data import fingerprint, read_json, validate_fields, validate_references

from meta_data_extraction.finetune.data import make_record, task_prompt
from meta_data_extraction.finetune.parsing import load_cached, pretty_json, save_unchanged_or_new, text_hash
from meta_data_extraction.finetune.tokenization import inference_lengths, load_tokenizer, record_lengths
from meta_data_extraction.utils import load_merged_schema

ROOT = Path(__file__).resolve().parents[1]


def source_fingerprint(path: Path) -> dict:
    """Preparation source files are text, including TOML and the uv lockfile."""
    text = path.read_text(encoding="utf-8")
    return {"sha256": text_hash(text), "bytes": len(text.encode("utf-8")),
            "normalization": "LF newlines"}


def frozen_inputs(root: Path, config: dict) -> dict:
    manifest = read_json(root / config["split_manifest"])
    sets = [set(manifest[key]) for key in ("train", "validation", "test")]
    if [len(s) for s in sets] != [8, 2, 30] or len(set.union(*sets)) != 40:
        raise ValueError("Expected a disjoint frozen 8/2/30 split")
    if manifest["status"] != "frozen" or manifest["audit"]["errors"]:
        raise ValueError("Split must be frozen with no audit errors")
    for path, expected in manifest["audit"]["source_files"].items():
        # Checksums only, including test gold; never load test gold for targets.
        if fingerprint(root / path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    return manifest


def check_config(config: dict) -> None:
    for key in ("max_seq_length", "max_new_tokens", "parse_workers", "parse_timeout_seconds"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_new_tokens"] >= config["max_seq_length"]:
        raise ValueError("Reserve less than the full sequence cap for generation")
    if config["schemas"] != ["generic.json", "photon_and_neutron.json"]:
        raise ValueError("Use the frozen full extraction schema")
    if config["enable_thinking"] is not False:
        raise ValueError("Thinking must be disabled")


def parse_papers(root: Path, output: Path, manifest: dict, config: dict) -> None:
    ids = manifest["train"] + manifest["validation"] + manifest["test"]

    def run(paper_id):
        if load_cached(root, output, paper_id):
            return f"{paper_id}: verified cached Markdown"
        print(f"Parsing {paper_id} ...", flush=True)
        completed = subprocess.run(
            [sys.executable, "-m", "meta_data_extraction.finetune.parsing",
             "--root", str(root), "--output", str(output), "--paper", paper_id],
            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=config["parse_timeout_seconds"],
        )
        if completed.returncode:
            raise ValueError(f"Parsing failed for {paper_id}: {completed.stderr[-3000:]}")
        if not load_cached(root, output, paper_id):
            raise ValueError(f"Missing parse output: {paper_id}")
        return completed.stdout.strip().splitlines()[-1]

    failures = []
    with ThreadPoolExecutor(max_workers=config["parse_workers"]) as pool:
        futures = {pool.submit(run, pid): pid for pid in ids}
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append(f"{futures[future]}: {exc}")
                print(failures[-1], file=sys.stderr, flush=True)
    if failures:
        raise ValueError("Some PDFs did not parse; completed caches retained for resume.\n" + "\n".join(failures))


def prepare(root: Path, output: Path, manifest: dict, config: dict, *,
            offline: bool = False, verify_only: bool = False) -> dict:
    tokenizer, tokenizer_identity = load_tokenizer(config, local_files_only=offline)
    cap = config["max_seq_length"]
    if cap > tokenizer.model_max_length:
        raise ValueError("Configured cap exceeds the pinned tokenizer's declared context")
    schema = load_merged_schema(config["schemas"])
    shared = {
        "split_manifest_sha256": fingerprint(root / config["split_manifest"])["sha256"],
        "config_sha256": text_hash(pretty_json(config)),
        "schema_versions": manifest["audit"]["schema_versions"],
        "model_id": config["model_id"], "revision": config["revision"],
        "chat_template_sha256": tokenizer_identity["chat_template_sha256"],
        "task_prompt_version": "1.0.0",
    }
    datasets = {"train_json_only.jsonl": [], "train_structure_aware.jsonl": [], "validation.jsonl": []}
    lengths, inference, papers = [], [], []
    for split in ("train", "validation", "test"):
        for paper_id in manifest[split]:
            cached = load_cached(root, output, paper_id)
            if cached is None:
                raise ValueError(f"Missing cached Markdown for {paper_id}; run --stage parse first")
            text, metadata = cached
            papers.append({"paper_id": paper_id, "split": split, **metadata})
            inference.append(inference_lengths(tokenizer, paper_id, split,
                             task_prompt("D", text, config["schemas"]), cap, config["max_new_tokens"]))
            if split == "test":
                continue  # Never open test gold, even to estimate generation lengths.
            gold_path = root / "data/gold" / f"{paper_id}.json"
            gold = read_json(gold_path)
            errors = validate_fields(gold, schema) or validate_references(gold)
            if errors:
                raise ValueError(f"Invalid calibration gold {paper_id}: {errors}")
            provenance = {**shared, "text_sha256": metadata["text_sha256"],
                          "gold_sha256": fingerprint(gold_path)["sha256"]}
            for task in (("D", "E", "R") if split == "train" else ("D",)):
                record = make_record(paper_id, split, task, text, gold, config["schemas"], provenance)
                stats = record_lengths(tokenizer, record, cap)
                record["token_counts"] = stats
                lengths.append(stats)
                if split == "validation":
                    datasets["validation.jsonl"].append(record)
                else:
                    datasets["train_structure_aware.jsonl"].append(record)
                    if task == "D":
                        datasets["train_json_only.jsonl"].append(record)
            print(f"Counted {paper_id} ({split})", flush=True)
    counts = {filename: len(records) for filename, records in datasets.items()}
    if list(counts.values()) != [8, 24, 2]:
        raise ValueError(f"Unexpected example counts: {counts}")
    overflows = ([{"kind": "supervised", **row} for row in lengths if not row["fits"]]
                 + [{"kind": "inference", **row} for row in inference if not row["fits"]]
                 + [{"kind": "known_D_target_exceeds_output_reservation", **row}
                    for row in lengths if row["task"] == "D" and row["supervised_tokens"] > config["max_new_tokens"]])
    report = {
        "version": "1.0.0", "status": "blocked_overflow" if overflows else "length_checks_passed",
        "config": config, "tokenizer": tokenizer_identity, "counts": counts,
        "supervised_examples": lengths, "inference": inference, "overflows": overflows,
        "longest_training": max((r for r in lengths if r["split"] == "train"), key=lambda r: r["sequence_tokens"]),
        "validation": [r for r in lengths if r["split"] == "validation"],
        "longest_inference": max(inference, key=lambda r: r["required_tokens"]),
        "paper_quality_flags": [{"paper_id": p["paper_id"], "warnings": p["warnings"],
                                 "low_word_presence_pages": p["low_word_presence_pages"]}
                                for p in papers if p["warnings"] or p["low_word_presence_pages"]],
        "policy": {
            "truncation": False, "packing": False, "test_gold_used": False,
            "supervision": "JSON target plus one EOS; mask prompt, empty thinking prefix, trailing template newline and padding",
            "output_reservation": f"Fixed {config['max_new_tokens']:,}-token reservation; not fitted to test gold; not a guarantee against generation cutoff",
            "gpu_feasibility": "Not tested; forward/backward and validation memory checks remain Step 6",
            "pdf_quality": "Word-presence checks are heuristic, not proof of reading order; inspect flagged PDFs",
        },
    }
    sources = ["pyproject.toml", "uv.lock", "scripts/prepare_finetune_data.py",
               "src/meta_data_extraction/finetune/data.py", "src/meta_data_extraction/finetune/parsing.py",
               "src/meta_data_extraction/finetune/tokenization.py", "src/meta_data_extraction/utils.py",
               "schemas/prompt_template.md", *[f"schemas/{s}" for s in config["schemas"]]]
    payloads = {filename: "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
                for filename, rows in datasets.items()}
    provenance = {
        "version": "1.0.0", "inputs": shared, "config": config,
        "tokenizer": tokenizer_identity, "sources": {s: source_fingerprint(root / s) for s in sources},
        "runtime": {"python": sys.version.split()[0], "pyyaml": version("pyyaml")},
        "papers": papers,
        "datasets": {name: {"records": counts[name], "sha256": text_hash(value)} for name, value in payloads.items()},
        "serialization": "UTF-8; indent=2 targets; preserve source object/list order, nulls and values; LF newlines",
    }
    outputs = {"length_report.json": pretty_json(report), "preparation_manifest.json": pretty_json(provenance)}
    if not overflows:
        outputs.update(payloads)
    # Preflight every existing output before writing anything, so drift never
    # produces a mixed old/new dataset. New outputs are exclusive, resumable writes.
    for name, value in outputs.items():
        path = output / name
        if path.exists() and path.read_text(encoding="utf-8") != value:
            raise ValueError(f"Prepared output differs: {path}; use a new output directory")
        if verify_only and not path.exists():
            raise ValueError(f"Missing prepared output: {path}")
    if not verify_only:
        for name, value in outputs.items():
            save_unchanged_or_new(output / name, value)
    if overflows:
        raise ValueError(f"{len(overflows)} length checks failed; see length_report.json. No JSONL exported.")
    print(f"{'Verified' if verify_only else 'Prepared'} 8/24/2 examples; all 40 inference inputs fit the configured cap.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/finetune_prepare.yaml")
    parser.add_argument("--stage", choices=["parse", "build", "all", "verify"], default="all")
    parser.add_argument("--offline", action="store_true", help="Use only cached tokenizer files")
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        check_config(config)
        manifest = frozen_inputs(ROOT, config)
        output = ROOT / config["output_dir"]
        if args.stage in {"parse", "all"}:
            parse_papers(ROOT, output, manifest, config)
        if args.stage in {"build", "all", "verify"}:
            prepare(ROOT, output, manifest, config, offline=args.offline, verify_only=args.stage == "verify")
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
