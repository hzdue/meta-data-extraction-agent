"""Deterministic pseudo-label splits and full-PDF parsing for the 600-paper corpus."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys

import pymupdf
import pymupdf4llm

from meta_data_extraction.components.parser import PdfParser


EXPECTED_SUBJECT_GROUPS = {
    "small_angle_scattering",
    "diffraction",
    "spectroscopy",
    "multi_technique",
    "reflectometry",
    "methods_light",
    "imaging_tomography",
}
SPLIT_NAMES = ("pseudo_train", "pseudo_validation", "future_test")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pretty_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def read_text_exact(path: Path) -> str:
    """Decode UTF-8 without universal-newline normalization."""
    return path.read_bytes().decode("utf-8")


def save_unchanged_or_new(path: Path, content: str) -> None:
    if path.exists():
        if read_text_exact(path) != content:
            raise ValueError(f"Existing artifact differs: {path}. Use a new output directory.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def parser_identity(root: Path) -> dict:
    """Stable identity of the content-producing parser, not orchestration code."""
    return {
        "name": "PdfParser",
        "options": "repository defaults; complete PDF; no truncation",
        "pymupdf": version("pymupdf"),
        "pymupdf4llm": version("pymupdf4llm"),
        "layout_enabled": bool(pymupdf4llm._use_layout),
        "source_sha256": text_hash(
            (root / "src/meta_data_extraction/components/parser.py").read_text(
                encoding="utf-8")),
    }


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def check_config(config: dict) -> None:
    expected = {
        "version", "source_dir", "source_manifest", "existing_split_manifest",
        "split_manifest", "output_dir", "split_seed", "split_counts",
        "parse_workers", "parse_timeout_seconds",
    }
    if set(config) != expected or config["version"] != "1.0.0":
        raise ValueError("Unexpected pseudo-label preparation configuration")
    if type(config["split_seed"]) is not int:
        raise ValueError("split_seed must be an integer")
    if (set(config["split_counts"]) != set(SPLIT_NAMES)
            or any(type(value) is not int or value <= 0
                   for value in config["split_counts"].values())
            or sum(config["split_counts"].values()) != 595):
        raise ValueError("Split counts must define 508/57/30 over 595 predictions")
    if config["split_counts"] != {
            "pseudo_train": 508, "pseudo_validation": 57, "future_test": 30}:
        raise ValueError("The reviewed pseudo-label split is 508/57/30")
    for key in ("parse_workers", "parse_timeout_seconds"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")


def page_bin(record: dict) -> str:
    pages = record["page_count"]
    if type(pages) is not int or pages <= 0:
        raise ValueError(f"Invalid page count for {record.get('paper_id')}")
    if pages <= 10:
        return "short_1_10"
    if pages <= 20:
        return "medium_11_20"
    if pages <= 40:
        return "long_21_40"
    return "very_long_41_plus"


def publication_period(record: dict) -> str:
    published = record["published"]
    if not isinstance(published, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", published) is None:
        raise ValueError(f"Invalid publication date for {record.get('paper_id')}")
    year = int(published[:4])
    if year <= 2015:
        return "2010_2015"
    if year <= 2020:
        return "2016_2020"
    if year <= 2023:
        return "2021_2023"
    return "2024_2026"


def _stable_key(seed: int, role: str, paper_id: str) -> str:
    return hashlib.sha256(f"{seed}:{role}:{paper_id}".encode()).hexdigest()


def proportional_quotas(counts: Counter, total: int) -> dict[str, int]:
    """Largest-remainder allocation with stable lexical tie-breaking."""
    population = sum(counts.values())
    if total < 0 or total > population or not counts:
        raise ValueError("Invalid proportional allocation")
    ideal = {key: total * value / population for key, value in counts.items()}
    result = {key: int(value) for key, value in ideal.items()}
    remaining = total - sum(result.values())
    order = sorted(counts, key=lambda key: (-(ideal[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    if sum(result.values()) != total or any(result[key] > counts[key] for key in result):
        raise ValueError("Proportional allocation failed")
    return result


def balanced_select(records: list[dict], count: int, seed: int, role: str) -> list[dict]:
    """Match subject quotas exactly and jointly balance page/date margins."""
    subject_targets = proportional_quotas(
        Counter(row["subject_group"] for row in records), count)
    page_targets = proportional_quotas(Counter(page_bin(row) for row in records), count)
    period_targets = proportional_quotas(
        Counter(publication_period(row) for row in records), count)
    subject_selected: Counter = Counter()
    page_selected: Counter = Counter()
    period_selected: Counter = Counter()
    remaining = list(records)
    selected = []
    while len(selected) < count:
        eligible = [row for row in remaining
                    if subject_selected[row["subject_group"]]
                    < subject_targets[row["subject_group"]]]
        if not eligible:
            raise ValueError(f"Subject quotas became infeasible for {role}")

        def score(row):
            subject = row["subject_group"]
            page = page_bin(row)
            period = publication_period(row)
            subject_need = subject_targets[subject] - subject_selected[subject]
            page_need = max(page_targets[page] - page_selected[page], 0)
            period_need = max(period_targets[period] - period_selected[period], 0)
            return (
                subject_need / max(subject_targets[subject], 1)
                + page_need / max(page_targets[page], 1)
                + period_need / max(period_targets[period], 1)
            )

        best = min(eligible, key=lambda row: (
            -score(row), _stable_key(seed, role, row["paper_id"])))
        remaining.remove(best)
        selected.append(best)
        subject_selected[best["subject_group"]] += 1
        page_selected[page_bin(best)] += 1
        period_selected[publication_period(best)] += 1
    if len(selected) != count or len({row["paper_id"] for row in selected}) != count:
        raise ValueError(f"Could not select {count} unique records for {role}")
    if subject_selected != Counter(subject_targets):
        raise ValueError(f"Subject quotas differ for {role}")
    return sorted(selected, key=lambda row: row["paper_id"])


def split_statistics(ids: list[str], by_id: dict[str, dict]) -> dict:
    rows = [by_id[paper_id] for paper_id in ids]
    return {
        "papers": len(rows),
        "subject_groups": dict(sorted(Counter(row["subject_group"] for row in rows).items())),
        "page_bins": dict(sorted(Counter(page_bin(row) for row in rows).items())),
        "publication_periods": dict(sorted(
            Counter(publication_period(row) for row in rows).items())),
        "page_count": {
            "minimum": min(row["page_count"] for row in rows),
            "maximum": max(row["page_count"] for row in rows),
        },
    }


def create_split_manifest(root: Path, config: dict) -> dict:
    source_path = root / config["source_manifest"]
    rows = read_jsonl(source_path)
    if len(rows) != 600 or len({row["paper_id"] for row in rows}) != 600:
        raise ValueError("Expected 600 unique source-manifest entries")
    if set(row["subject_group"] for row in rows) != EXPECTED_SUBJECT_GROUPS:
        raise ValueError("Source subject groups changed")
    available = [row for row in rows if row["global_seed_prediction"]["path"] is not None]
    missing = [row for row in rows if row["global_seed_prediction"]["path"] is None]
    if (len(available), len(missing)) != (595, 5):
        raise ValueError("Expected 595 available and 5 missing predictions")
    if any(row["global_seed_prediction"]["status"] == "missing" for row in available):
        raise ValueError("Available prediction has missing status")
    if any(row["global_seed_prediction"]["status"] != "missing" for row in missing):
        raise ValueError("Missing prediction has non-missing status")

    benchmark = read_json(root / config["existing_split_manifest"])
    benchmark_ids = set(benchmark["train"] + benchmark["validation"] + benchmark["test"])
    overlap = sorted(benchmark_ids & {row["paper_id"] for row in rows})
    if overlap:
        raise ValueError(f"Pseudo corpus overlaps the existing benchmark: {overlap}")

    seed = config["split_seed"]
    future = balanced_select(available, config["split_counts"]["future_test"], seed,
                             "future_test")
    future_ids = {row["paper_id"] for row in future}
    remainder = [row for row in available if row["paper_id"] not in future_ids]
    validation = balanced_select(
        remainder, config["split_counts"]["pseudo_validation"], seed,
        "pseudo_validation")
    validation_ids = {row["paper_id"] for row in validation}
    train = sorted((row for row in remainder if row["paper_id"] not in validation_ids),
                   key=lambda row: row["paper_id"])
    roles = {
        "pseudo_train": [row["paper_id"] for row in train],
        "pseudo_validation": [row["paper_id"] for row in validation],
        "future_test": [row["paper_id"] for row in future],
    }
    if {key: len(value) for key, value in roles.items()} != config["split_counts"]:
        raise ValueError("Generated split counts differ from configuration")
    sets = [set(roles[key]) for key in SPLIT_NAMES]
    if len(set.union(*sets)) != 595 or sum(map(len, sets)) != 595:
        raise ValueError("Pseudo-label splits are not disjoint and complete")

    by_id = {row["paper_id"]: row for row in rows}
    manifest = {
        "version": "1.0.0",
        "status": "frozen",
        "seed": seed,
        "source_manifest": config["source_manifest"],
        "source_manifest_sha256": file_hash(source_path),
        "selection_inputs": ["paper_id", "subject_group", "page_count", "published"],
        "selection_policy": (
            "Exact proportional subject-group quotas; deterministic joint greedy balance "
            "of page-count bins and publication periods"
        ),
        "strata": {
            "page_bins": ["short_1_10", "medium_11_20", "long_21_40",
                          "very_long_41_plus"],
            "publication_periods": ["2010_2015", "2016_2020", "2021_2023",
                                    "2024_2026"],
        },
        **roles,
        "excluded_missing_prediction": sorted(row["paper_id"] for row in missing),
        "statistics": {key: split_statistics(ids, by_id) for key, ids in roles.items()},
        "leakage_checks": {
            "existing_benchmark_manifest": config["existing_split_manifest"],
            "existing_benchmark_overlap": overlap,
            "future_test_predictions_permitted_in_training": False,
        },
    }
    save_unchanged_or_new(root / config["split_manifest"], pretty_json(manifest))
    return manifest


def external_parser_identity(root: Path) -> dict:
    return parser_identity(root)


def _resolved_source_pdf(root: Path, source_dir: Path, relative_pdf: str) -> Path:
    source = (root / source_dir).resolve()
    pdf = (root / source_dir / relative_pdf).resolve()
    if not pdf.is_relative_to(source):
        raise ValueError(f"PDF escaped the declared source directory: {relative_pdf}")
    return pdf


def external_cache_paths(output: Path, paper_id: str) -> tuple[Path, Path]:
    """Return the two exact cache paths for one corpus paper."""
    if re.fullmatch(r"\d{4}\.\d{4,5}", paper_id) is None:
        raise ValueError("Expected a bare corpus arXiv ID")
    text_dir = output / "text"
    return text_dir / f"{paper_id}.md", text_dir / f"{paper_id}.json"


def clear_external_cache(output: Path, paper_id: str) -> list[Path]:
    """Remove only one paper's Markdown/metadata pair for targeted repair."""
    removed = []
    for path in external_cache_paths(output, paper_id):
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def load_external_cached(root: Path, output: Path, source_dir: Path, paper_id: str,
                         relative_pdf: str, role: str) -> tuple[str, dict] | None:
    text_path, meta_path = external_cache_paths(output, paper_id)
    if not text_path.exists() and not meta_path.exists():
        return None
    if not text_path.exists() or not meta_path.exists():
        raise ValueError(f"Incomplete pseudo-label Markdown cache: {paper_id}")
    metadata = read_json(meta_path)
    text = read_text_exact(text_path)
    pdf_path = _resolved_source_pdf(root, source_dir, relative_pdf)
    expected_parser = external_parser_identity(root)
    saved_parser = metadata["parser"]
    if (metadata["paper_id"] != paper_id
            or metadata["split_role"] != role
            or metadata["source_pdf"] != relative_pdf.replace("\\", "/")
            or any(saved_parser.get(key) != value for key, value in expected_parser.items())
            or metadata["pdf_sha256"] != file_hash(pdf_path)
            or metadata["text_sha256"] != text_hash(text)):
        raise ValueError(f"Stale or modified pseudo-label Markdown cache: {paper_id}")
    return text, metadata


def parse_external_one(root: Path, output: Path, source_dir: Path, paper_id: str,
                       relative_pdf: str, role: str) -> dict:
    if re.fullmatch(r"\d{4}\.\d{4,5}", paper_id) is None:
        raise ValueError("Expected a bare corpus arXiv ID")
    if role not in {*SPLIT_NAMES, "excluded_missing_prediction"}:
        raise ValueError(f"Unexpected split role: {role}")
    cached = load_external_cached(root, output, source_dir, paper_id, relative_pdf, role)
    if cached:
        return cached[1]
    pdf_path = _resolved_source_pdf(root, source_dir, relative_pdf)
    pymupdf.TOOLS.mupdf_warnings(reset=True)
    document = PdfParser().parse(pdf_path)
    text = document.text
    if not text.strip():
        raise ValueError(f"No Markdown extracted for {paper_id}")
    warnings = pymupdf.TOOLS.mupdf_warnings(reset=True).strip().splitlines()
    words = lambda value: set(re.findall(r"[^\W_]{3,}", value.casefold()))
    markdown_words = words(text)
    coverage = []
    with pymupdf.open(pdf_path) as pdf:
        page_count = len(pdf)
        plain_pages = [page.get_text() for page in pdf]
    for index, plain in enumerate(plain_pages, 1):
        page_words = words(plain)
        coverage.append({
            "page": index,
            "plain_text_chars": len(plain),
            "word_presence": (round(len(page_words & markdown_words) / len(page_words), 4)
                              if page_words else None),
        })
    metadata = {
        "paper_id": paper_id,
        "split_role": role,
        "source_pdf": relative_pdf.replace("\\", "/"),
        "pdf_sha256": file_hash(pdf_path),
        "parser": external_parser_identity(root),
        "preparation_source_sha256": text_hash(Path(__file__).read_text(encoding="utf-8")),
        "text_sha256": text_hash(text),
        "text_chars": len(text),
        "pages": page_count,
        "plain_text_chars": len("\n\n".join(plain_pages)),
        "warnings": warnings,
        "page_word_presence": coverage,
        "low_word_presence_pages": [
            row["page"] for row in coverage
            if row["word_presence"] is not None and row["word_presence"] < 0.75
        ],
    }
    save_unchanged_or_new(output / "text" / f"{paper_id}.md", text)
    save_unchanged_or_new(output / "text" / f"{paper_id}.json", pretty_json(metadata))
    return metadata


def _roles(split: dict) -> dict[str, str]:
    result = {}
    for role in SPLIT_NAMES:
        for paper_id in split[role]:
            if paper_id in result:
                raise ValueError(f"Paper appears in multiple splits: {paper_id}")
            result[paper_id] = role
    for paper_id in split["excluded_missing_prediction"]:
        if paper_id in result:
            raise ValueError(f"Missing prediction entered a supervised split: {paper_id}")
        result[paper_id] = "excluded_missing_prediction"
    return result


def parse_all(root: Path, config: dict, split: dict,
              repair_stale_cache: bool = False) -> dict:
    source_manifest_path = root / config["source_manifest"]
    if file_hash(source_manifest_path) != split["source_manifest_sha256"]:
        raise ValueError("Source manifest changed after split creation")
    rows = read_jsonl(source_manifest_path)
    by_id = {row["paper_id"]: row for row in rows}
    roles = _roles(split)
    if set(by_id) != set(roles) or len(roles) != 600:
        raise ValueError("Split roles do not cover all 600 source papers")
    output = root / config["output_dir"]
    source_dir = Path(config["source_dir"])
    parser_at_start = external_parser_identity(root)
    repaired = []

    def run(paper_id: str):
        relative_pdf = by_id[paper_id]["pdf"]["path"]
        role = roles[paper_id]
        if external_parser_identity(root) != parser_at_start:
            raise ValueError("Parser implementation changed during preparation")
        try:
            cached = load_external_cached(
                root, output, source_dir, paper_id, relative_pdf, role)
        except (OSError, ValueError, KeyError) as exc:
            if not repair_stale_cache:
                raise
            clear_external_cache(output, paper_id)
            repaired.append(paper_id)
            cached = None
            print(f"{paper_id}: repairing stale cache ({exc})", flush=True)
        if cached:
            return f"{paper_id}: verified cached Markdown"
        completed = subprocess.run(
            [sys.executable, "-m",
             "meta_data_extraction.finetune.pseudo_label_preparation",
             "--root", str(root), "--output", str(output),
             "--source-dir", str(source_dir), "--paper", paper_id,
             "--pdf", relative_pdf, "--role", role],
            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=config["parse_timeout_seconds"],
        )
        if completed.returncode:
            raise ValueError(f"Parsing failed for {paper_id}: {completed.stderr[-3000:]}")
        if not load_external_cached(root, output, source_dir, paper_id, relative_pdf, role):
            raise ValueError(f"Missing parse output: {paper_id}")
        return completed.stdout.strip().splitlines()[-1]

    failures = []
    ids = sorted(by_id)
    with ThreadPoolExecutor(max_workers=config["parse_workers"]) as pool:
        futures = {pool.submit(run, paper_id): paper_id for paper_id in ids}
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                message = f"{futures[future]}: {exc}"
                failures.append(message)
                print(message, file=sys.stderr, flush=True)
    if failures:
        raise ValueError(
            "Some PDFs did not parse; completed caches were retained for resume.\n"
            + "\n".join(failures))
    if external_parser_identity(root) != parser_at_start:
        raise ValueError(
            "Parser implementation changed during preparation; rerun with "
            "--repair-stale-cache")

    metadata = [load_external_cached(
        root, output, source_dir, paper_id, by_id[paper_id]["pdf"]["path"],
        roles[paper_id])[1] for paper_id in ids]
    summary = {
        "version": "1.0.0",
        "status": "complete",
        "papers": len(metadata),
        "split_manifest": config["split_manifest"],
        "split_manifest_sha256": file_hash(root / config["split_manifest"]),
        "source_manifest": config["source_manifest"],
        "source_manifest_sha256": split["source_manifest_sha256"],
        "parser": parser_at_start,
        "summary_source_sha256": text_hash(Path(__file__).read_text(encoding="utf-8")),
        "paper_preparation_source_counts": dict(sorted(Counter(
            row["preparation_source_sha256"] for row in metadata).items())),
        "role_counts": dict(sorted(Counter(row["split_role"] for row in metadata).items())),
        "page_count": {
            "minimum": min(row["pages"] for row in metadata),
            "maximum": max(row["pages"] for row in metadata),
        },
        "text_chars": {
            "minimum": min(row["text_chars"] for row in metadata),
            "maximum": max(row["text_chars"] for row in metadata),
        },
        "papers_with_parser_warnings": [row["paper_id"] for row in metadata
                                         if row["warnings"]],
        "papers_with_low_word_presence_pages": [
            {"paper_id": row["paper_id"],
             "pages": row["low_word_presence_pages"]}
            for row in metadata if row["low_word_presence_pages"]
        ],
        "policy": {
            "parser": "Existing PdfParser with repository defaults",
            "complete_pdf": True,
            "truncation": False,
            "prediction_files_read": False,
            "future_test_predictions_read": False,
            "preparation_provenance_note": (
                "The resumable run spans preparation-wrapper revisions used for "
                "cache validation and retry handling; the content-producing parser "
                "identity is uniform across all papers"
            ),
            "quality_note": (
                "Word-presence checks are heuristic; flagged Markdown requires review"
            ),
        },
    }
    save_unchanged_or_new(output / "parse_summary.json", pretty_json(summary))
    if repaired:
        print(f"Repaired {len(set(repaired))} stale or incomplete cache pairs", flush=True)
    return summary


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Parse one pseudo-label corpus PDF")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--role", required=True)
    args = parser.parse_args(argv)
    try:
        result = parse_external_one(
            args.root.resolve(), args.output.resolve(), args.source_dir,
            args.paper, args.pdf, args.role)
        print(f"{args.paper}: {result['pages']} pages, "
              f"{result['text_chars']} Markdown chars", flush=True)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"Parsing failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

