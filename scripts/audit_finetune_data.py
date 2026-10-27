"""Audit Step 1 inputs, freeze an explicit 8/2/30 split, or verify it unchanged.

Uses fast PDF plain-text extraction for approximate lengths, not the training
Markdown parser or a tokenizer. Evaluation annotations are read for document
identity/duplicate checks only; annotation statistics use calibration papers only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import pymupdf
import yaml

if __package__:
    from .corpus import load_corpus
else:
    from corpus import load_corpus


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("data/finetune/splits.json")
REPORT = Path("data/finetune/audit.md")
SCHEMAS = ("generic.json", "photon_and_neutron.json")
COLLECTIONS = {
    "bibliographic.affiliations": "aff",
    "bibliographic.authors": "author",
    "facilities": "facility",
    "samples": "sample",
    "datasets": "dataset",
}
REFERENCES = (
    ("bibliographic.authors", "affiliation_refs", "bibliographic.affiliations"),
    ("datasets", "sample_refs", "samples"),
    ("datasets", "facility_refs", "facilities"),
)
PROVENANCE_PATHS = (
    "data/corpus.json", "docs/annotation_guidelines.md",
    "configs/evaluation_metrics.yaml", "configs/alignment_helper.yaml",
    "schemas/prompt_template.md",
    "src/meta_data_extraction/components/parser.py",
    "src/meta_data_extraction/utils.py", "src/meta_data_extraction/evaluation.py",
    "src/meta_data_extraction/alignment.py", "src/meta_data_extraction/normalization.py",
    "scripts/corpus.py", "scripts/audit_finetune_data.py",
)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


def fingerprint(path: Path) -> dict:
    """Normalize text newlines so Windows and Linux checkouts verify equally."""
    if path.suffix in {".json", ".yaml", ".md", ".py"}:
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return {"sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content), "normalization": "LF newlines"}
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size, "normalization": "none"}


def field_value(record: dict, path: str):
    value = record
    for key in path.split("."):
        value = value[key]
    return value


def validate_fields(value, definition: dict, path: str = "gold") -> list[str]:
    """Validate this repository's field-definition format, not JSON Schema."""
    errors = []
    kind = definition.get("type")
    if kind is None:
        if not isinstance(value, dict):
            return [f"{path}: expected object"]
        fields = {key: item for key, item in definition.items() if key != "_meta"}
        for key in sorted(set(fields) - set(value)):
            errors.append(f"{path}.{key}: missing field")
        for key in sorted(set(value) - set(fields)):
            errors.append(f"{path}.{key}: unexpected field")
        for key in fields.keys() & value.keys():
            errors.extend(validate_fields(value[key], fields[key], f"{path}.{key}"))
        return sorted(errors)
    if value is None and "null" in kind.split("|"):
        return []
    base_kind = kind.split("|")[0]
    if base_kind.startswith("list["):
        if not isinstance(value, list):
            return [f"{path}: expected {kind}"]
        cardinality = definition.get("cardinality", {})
        if not cardinality.get("min", 0) <= len(value) <= cardinality.get("max", float("inf")):
            errors.append(f"{path}: cardinality {len(value)} outside {cardinality}")
        item_definition = definition.get("item_schema") or {
            "type": base_kind[5:-1], "constraints": definition.get("constraints", {}),
        }
        for index, item in enumerate(value):
            errors.extend(validate_fields(item, item_definition, f"{path}[{index}]"))
    elif base_kind == "string":
        if not isinstance(value, str):
            return [f"{path}: expected {kind}"]
        constraints = definition.get("constraints", {})
        if "pattern" in constraints and re.fullmatch(constraints["pattern"], value) is None:
            errors.append(f"{path}: does not match declared pattern")
        if "allowed_values" in constraints and value not in constraints["allowed_values"]:
            errors.append(f"{path}: disallowed value {value!r}")
    elif base_kind == "boolean":
        if type(value) is not bool:
            errors.append(f"{path}: expected boolean")
    else:
        errors.append(f"{path}: unsupported schema type {kind!r}")
    return errors


def validate_references(gold: dict) -> list[str]:
    errors = []
    identifiers = {}
    for collection, prefix in COLLECTIONS.items():
        ids = [item["id"] for item in field_value(gold, collection)]
        identifiers[collection] = set(ids)
        if len(ids) != len(set(ids)):
            errors.append(f"{collection}: duplicate IDs")
        if ids != [f"{prefix}_{index}" for index in range(1, len(ids) + 1)]:
            errors.append(f"{collection}: IDs do not follow the declared sequential convention")
    for source, field, target in REFERENCES:
        for item in field_value(gold, source):
            refs = item[field]
            if len(refs) != len(set(refs)):
                errors.append(f"{source}.{item['id']}.{field}: repeated references")
            for ref in refs:
                if ref not in identifiers[target]:
                    errors.append(f"{source}.{item['id']}.{field}: dangling reference {ref}")
    return errors


def normalized_title(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def duplicate_groups(values: dict[str, str]) -> list[list[str]]:
    groups = defaultdict(list)
    for paper, value in values.items():
        if value:
            groups[value].append(paper)
    return sorted(sorted(ids) for ids in groups.values() if len(ids) > 1)


def audit(root: Path) -> dict:
    corpus = load_corpus(root / "data/corpus.json")
    calibration = sorted(p["arxiv_id"] for p in corpus["papers"] if p["split"] == "calibration")
    evaluation = sorted(p["arxiv_id"] for p in corpus["papers"] if p["split"] == "evaluation")
    if len(calibration) != 10 or len(evaluation) != 30:
        raise ValueError("Expected the existing 10 calibration / 30 evaluation corpus")
    merged_schema, schema_versions, files = {}, {}, {}
    for name in SCHEMAS:
        path = f"schemas/{name}"
        schema = read_json(root / path)
        schema_versions[name] = schema.pop("_meta")["schema_version"]
        merged_schema.update(schema)
        files[path] = fingerprint(root / path)
    for path in PROVENANCE_PATHS:
        files[path] = fingerprint(root / path)
    scoring = yaml.safe_load((root / "configs/evaluation_metrics.yaml").read_text(encoding="utf-8"))
    errors = []
    if scoring["schema_versions"] != schema_versions:
        errors.append("Scoring plan and extraction schema versions disagree")
    identities, inventory, pdf_warnings = {}, [], {}
    for paper in sorted(corpus["papers"], key=lambda item: item["arxiv_id"]):
        paper_id = paper["arxiv_id"]
        pdf_path, gold_path = f"data/raw/{paper_id}/paper.pdf", f"data/gold/{paper_id}.json"
        files[pdf_path], files[gold_path] = fingerprint(root / pdf_path), fingerprint(root / gold_path)
        gold = read_json(root / gold_path)
        bibliographic = gold["bibliographic"]
        pymupdf.TOOLS.mupdf_warnings(reset=True)
        with pymupdf.open(root / pdf_path) as pdf:
            pages = len(pdf)
            text = "\n\n".join(page.get_text() for page in pdf)
        warnings = pymupdf.TOOLS.mupdf_warnings(reset=True).strip()
        if warnings:
            pdf_warnings[paper_id] = warnings.splitlines()
        if not text.strip():
            errors.append(f"{paper_id}: PDF has no extractable text")
        title = bibliographic.get("title") or ""
        if not title:
            errors.append(f"{paper_id}: missing title for duplicate screening")
        doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", bibliographic.get("doi") or "", flags=re.I).strip().casefold()
        arxiv_id = re.sub(r"v\d+$", "", bibliographic.get("arxiv_id") or paper_id)
        if arxiv_id != paper_id:
            errors.append(f"{paper_id}: explicit gold arXiv identity disagrees with corpus ID ({arxiv_id})")
        identities[paper_id] = {
            "pdf_sha256": files[pdf_path]["sha256"],
            "normalized_text_sha256": hashlib.sha256(" ".join(unicodedata.normalize("NFKC", text).casefold().split()).encode("utf-8")).hexdigest(),
            "title": normalized_title(title), "doi": doi, "arxiv_id": arxiv_id,
        }
        if paper_id not in calibration:
            continue
        paper_errors = validate_fields(gold, merged_schema)
        if not paper_errors:
            paper_errors = validate_references(gold)
        errors.extend(f"{paper_id}: {error}" for error in paper_errors)
        if paper_errors:
            continue
        inventory.append({
            "paper_id": paper_id, "stratum": paper["stratum"], "title": title,
            "pages": pages, "plain_text_chars": len(text), "approx_words": len(text.split()),
            "facilities": len(gold["facilities"]), "samples": len(gold["samples"]),
            "datasets": len(gold["datasets"]),
            "sample_links": sum(len(d["sample_refs"]) for d in gold["datasets"]),
            "facility_links": sum(len(d["facility_refs"]) for d in gold["datasets"]),
            "affiliation_links": sum(len(a["affiliation_refs"]) for a in bibliographic["authors"]),
        })
    duplicates = {
        field: duplicate_groups({pid: identity[field] for pid, identity in identities.items()})
        for field in ("pdf_sha256", "normalized_text_sha256", "title", "doi", "arxiv_id")
    }
    similar_titles = []
    ids = sorted(identities)
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            similarity = SequenceMatcher(None, identities[left]["title"], identities[right]["title"]).ratio()
            if similarity >= 0.90:
                similar_titles.append({"papers": [left, right], "similarity": round(similarity, 4)})
    if any(duplicates.values()) or similar_titles:
        errors.append("Potential duplicate publications require review before freezing")
    return {
        "calibration": calibration, "test": evaluation, "schema_versions": schema_versions,
        "scoring_plan_version": scoring["version"], "source_files": files,
        "length_method": {"library": "PyMuPDF", "version": pymupdf.VersionBind,
                          "method": "page.get_text(); pages joined with two newlines; no truncation",
                          "limitation": "Approximate PDF text length, not training Markdown or model tokens"},
        "calibration_inventory": inventory,
        "pdf_warnings": pdf_warnings,
        "duplicate_checks": {"exact_matches": duplicates, "similar_titles_threshold": 0.90,
                             "similar_title_candidates": similar_titles,
                             "scope": "All 40 papers; local identity/content screening, not an exhaustive external version search"},
        "errors": sorted(errors),
    }


def build_manifest(result: dict, validation: list[str], reasons: list[str]) -> dict:
    if result["errors"]:
        raise ValueError("Audit errors must be resolved before freezing")
    calibration = set(result["calibration"])
    if len(validation) != 2 or len(set(validation)) != 2 or not set(validation) <= calibration:
        raise ValueError("Validation must contain two distinct calibration IDs")
    if len(reasons) != 2 or not all(reason.strip() for reason in reasons):
        raise ValueError("Provide one selection rationale for each validation paper")
    inventory = result["calibration_inventory"]
    longest = max(inventory, key=lambda row: row["plain_text_chars"])["paper_id"]
    if longest not in validation:
        raise ValueError(f"User requires the longest calibration paper ({longest}) in validation")
    train = sorted(calibration - set(validation))
    if {row["stratum"] for row in inventory if row["paper_id"] in train} != {row["stratum"] for row in inventory}:
        raise ValueError("The chosen split removes a technique stratum from training")
    return {
        "version": "1.0.0", "status": "frozen", "split_unit": "paper",
        "train": train, "validation": sorted(validation), "test": result["test"],
        "selection": {
            "method": "Fixed, intentional selection by length and calibration annotation inventory; no model scores",
            "longest_paper": longest, "longest_metric": "plain_text_chars",
            "validation_rationale": dict(zip(validation, reasons, strict=True)),
            "refit_on_validation": False,
            "resource_note": "Longest paper is excluded from training at the user's request after an RTX length-limit issue. It must still fit full-JSON validation loss and generation; verify exact token length and hardware in later steps.",
        },
        "audit": result,
    }


def render_report(manifest: dict) -> str:
    result = manifest["audit"]
    lines = [
        "# Fine-tuning Step 1 audit", "",
        "Frozen split: **8 training / 2 validation / 30 test papers**. The original corpus index is unchanged.", "",
        "## Validation selection", "",
    ]
    for pid in manifest["validation"]:
        lines.append(f"- **{pid}:** {manifest['selection']['validation_rationale'][pid]}")
    lines += ["", "All seven calibration technique strata remain represented in training.", "",
              "## Calibration inventory", "",
              "| Paper | Split | Technique stratum | Pages | Text chars | Facilities | Samples | Datasets | Sample links | Facility links |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["calibration_inventory"]:
        split = "validation" if row["paper_id"] in manifest["validation"] else "train"
        lines.append(f"| {row['paper_id']} | {split} | {row['stratum']} | {row['pages']} | {row['plain_text_chars']:,} | {row['facilities']} | {row['samples']} | {row['datasets']} | {row['sample_links']} | {row['facility_links']} |")
    lines += [
        "", "## Checks and scope", "",
        "- All 40 PDFs and gold records exist, open successfully, and have SHA-256 fingerprints in `splits.json`.",
        "- All ten calibration records pass field/type/cardinality/enum/pattern, sequential-ID, and reference checks. This is a structural audit, not a new scientific annotation review.",
        "- Test gold is used only for JSON readability and bibliographic duplicate checks; no test targets, semantic scores, or annotation statistics are used for selection.",
        "- No duplicate candidates were found by PDF hash, normalized text hash, normalized title, DOI, unversioned arXiv identity, or title similarity >= 0.90. This local screening cannot rule out every renamed publication version.",
        f"- Schema versions: `{result['schema_versions']}`; scoring-plan version: `{result['scoring_plan_version']}`. Schema and scoring declarations agree.",
        "- Source hashes also cover the corpus index, schemas, guidelines, prompt, parser/evaluator code, audit code, and evaluation configuration. Training dependencies and run provenance will be recorded in later steps; adding those dependencies does not change split membership.",
        "- PDF hashes cover raw bytes. Text-source hashes and byte counts normalize only line endings to LF, allowing Windows/Linux checkouts without hiding other content changes.",
        "", "## Recoverable PDF parser warnings", "",
    ]
    if result["pdf_warnings"]:
        lines.append("Text extraction completed, but the following PDFs emitted warnings. Review their final Markdown in Step 2; successful extraction does not guarantee faithful rendering.")
        lines.append("")
        for pid, warnings in result["pdf_warnings"].items():
            lines.append(f"- **{pid}:** " + "; ".join(warnings))
    else:
        lines.append("None recorded.")
    lines += [
        "", "## Length and hardware follow-up", "",
        f"Lengths use PyMuPDF {result['length_method']['version']} plain-text extraction of the complete PDFs, including appended material. They are not Qwen token counts or the final Markdown lengths.",
        "", manifest["selection"]["resource_note"], "",
        "Step 2 must count complete prompts and targets with the pinned 4B tokenizer. Step 6 must test backward passes only on training examples and forward-only validation on both validation papers. Do not truncate, omit, or train on the long validation paper to work around a resource limit.",
        "", "The two-paper validation set intentionally includes a length outlier and cannot represent the entire domain. Use mean per-paper validation loss as planned. The existing test benchmark has been used previously and is not a new independent test set.",
        "", "## Verify without rewriting", "",
        "```powershell", "uv run --no-sync python scripts/audit_finetune_data.py --verify", "```", "",
        "Verification recomputes the audit, checks the fixed 8/2/30 membership and all fingerprints, and fails on drift. The audit command without flags is read-only; `--freeze` refuses to replace existing artifacts.", "",
    ]
    return "\n".join(lines)


def freeze(root: Path, manifest: dict) -> None:
    paths = [root / MANIFEST, root / REPORT]
    if any(path.exists() for path in paths):
        raise ValueError("Frozen artifacts already exist; use --verify, do not overwrite the split")
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for path, content in zip(paths, [json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", render_report(manifest)], strict=True):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)


def verify(root: Path, current: dict) -> None:
    saved = read_json(root / MANIFEST)
    rationale = saved["selection"]["validation_rationale"]
    expected = build_manifest(current, saved["validation"], [rationale[pid] for pid in saved["validation"]])
    if saved != expected:
        raise ValueError("Frozen split, source fingerprints, or audit metadata changed; review drift explicitly")
    if (root / REPORT).read_text(encoding="utf-8") != render_report(saved):
        raise ValueError("Audit report differs from the frozen manifest")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--validation", nargs=2)
    parser.add_argument("--reason", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        result = audit(ROOT)
        print("paper         stratum                  pages   chars   fac  samples  datasets  sample_links  facility_links")
        for row in result["calibration_inventory"]:
            print(f"{row['paper_id']:13} {row['stratum']:24} {row['pages']:5} {row['plain_text_chars']:7} {row['facilities']:5} {row['samples']:8} {row['datasets']:9} {row['sample_links']:13} {row['facility_links']:15}")
        if result["pdf_warnings"]:
            print("Recoverable PDF warnings recorded for: " + ", ".join(result["pdf_warnings"]))
        if result["errors"]:
            print(json.dumps({"errors": result["errors"], "duplicates": result["duplicate_checks"]}, indent=2))
            return 1
        if args.freeze:
            manifest = build_manifest(result, args.validation or [], args.reason)
            freeze(ROOT, manifest)
            print(f"Frozen {MANIFEST.as_posix()} and {REPORT.as_posix()}")
        elif args.verify:
            verify(ROOT, result)
            print("Verified: frozen 8/2/30 split, all source hashes, and audit report unchanged.")
        else:
            print("Audit passed; no files written. No duplicate candidates found across 40 papers.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
