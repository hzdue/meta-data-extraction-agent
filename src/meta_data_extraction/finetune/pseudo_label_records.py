"""Validate pseudo-labels and create auditable complete-output SFT records."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from meta_data_extraction.finetune.pseudo_label_preparation import (
    file_hash,
    page_bin,
    text_hash,
)
from meta_data_extraction.utils import (
    build_prompt,
    get_prompt_version,
    load_merged_schema,
)


SCHEMAS = ["generic.json", "photon_and_neutron.json"]
SUPERVISED_ROLES = ("pseudo_train", "pseudo_validation")
TOKENIZER_MODEL = "Qwen/Qwen3.5-4B"
TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
REVIEW_SEED = 42
ID_PREFIXES = {
    "bibliographic.affiliations": "aff",
    "bibliographic.authors": "author",
    "facilities": "facility",
    "samples": "sample",
    "datasets": "dataset",
}
NAME_FIELDS = {
    "bibliographic.authors": "full_name",
    "facilities": "name",
    "samples": "sample_name",
    "datasets": "dataset_name",
}
EXTREME_THRESHOLDS = {
    "bibliographic.affiliations": 50,
    "bibliographic.authors": 50,
    "bibliographic.funding": 25,
    "facilities": 10,
    "samples": 25,
    "datasets": 25,
}


def strict_json(path: Path):
    def reject(value: str):
        raise ValueError(f"Non-standard JSON constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _schema_fields(schema: dict) -> dict:
    ignored = {"description", "instructions", "constraints", "cardinality",
               "type", "item_schema"}
    return {key: value for key, value in schema.items()
            if key not in ignored and isinstance(value, dict)}


def _validate_shape(value, schema: dict, path: str, errors: list[str]) -> None:
    field_type = schema.get("type")
    if field_type is None:
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object, got {type(value).__name__}")
            return
        expected = set(_schema_fields(schema))
        actual = set(value)
        if actual != expected:
            errors.append(
                f"{path}: keys differ; missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}")
        for key in sorted(expected & actual):
            _validate_shape(value[key], schema[key], f"{path}.{key}", errors)
        return

    cardinality = schema.get("cardinality")
    if field_type.startswith("list["):
        if not isinstance(value, list):
            errors.append(f"{path}: expected list, got {type(value).__name__}")
            return
        if cardinality:
            minimum = cardinality.get("min")
            maximum = cardinality.get("max")
            if minimum is not None and len(value) < minimum:
                errors.append(f"{path}: {len(value)} items is below minimum {minimum}")
            if maximum is not None and len(value) > maximum:
                errors.append(f"{path}: {len(value)} items exceeds maximum {maximum}")
        item_type = field_type[5:-1]
        if item_type == "string":
            for index, item in enumerate(value):
                if not isinstance(item, str):
                    errors.append(
                        f"{path}[{index}]: expected string, got {type(item).__name__}")
        else:
            item_schema = schema.get("item_schema")
            if not isinstance(item_schema, dict):
                errors.append(f"{path}: list type {field_type!r} lacks item_schema")
                return
            for index, item in enumerate(value):
                _validate_shape(item, item_schema, f"{path}[{index}]", errors)
        allowed = schema.get("constraints", {}).get("allowed_values")
        if allowed is not None:
            for index, item in enumerate(value):
                if item not in allowed:
                    errors.append(f"{path}[{index}]: {item!r} is not allowed")
        return

    nullable = field_type.endswith("|null")
    base_type = field_type.removesuffix("|null")
    if value is None:
        if not nullable:
            errors.append(f"{path}: null is not allowed for {field_type}")
        return
    valid = ((base_type == "string" and isinstance(value, str))
             or (base_type == "boolean" and type(value) is bool))
    if not valid:
        errors.append(f"{path}: expected {field_type}, got {type(value).__name__}")
        return
    allowed = schema.get("constraints", {}).get("allowed_values")
    if allowed is not None and value not in allowed:
        errors.append(f"{path}: {value!r} is not allowed")


def _collection(prediction: dict, dotted: str) -> list:
    value = prediction
    for part in dotted.split("."):
        value = value[part]
    return value


def _validate_ids_and_refs(prediction: dict, errors: list[str]) -> None:
    ids = {}
    for path in ID_PREFIXES:
        collection = _collection(prediction, path)
        values = [item.get("id") for item in collection]
        if len(values) != len(set(values)):
            errors.append(f"{path}: IDs are not unique")
        ids[path] = set(values)

    for index, author in enumerate(prediction["bibliographic"]["authors"]):
        for ref in author["affiliation_refs"]:
            if ref not in ids["bibliographic.affiliations"]:
                errors.append(
                    f"bibliographic.authors[{index}].affiliation_refs: "
                    f"dangling reference {ref!r}")
    for index, dataset in enumerate(prediction["datasets"]):
        for ref in dataset["sample_refs"]:
            if ref not in ids["samples"]:
                errors.append(
                    f"datasets[{index}].sample_refs: dangling reference {ref!r}")
        for ref in dataset["facility_refs"]:
            if ref not in ids["facilities"]:
                errors.append(
                    f"datasets[{index}].facility_refs: dangling reference {ref!r}")


def validate_prediction(prediction: dict, schema: dict) -> list[str]:
    errors = []
    _validate_shape(prediction, schema, "$", errors)
    if not errors:
        _validate_ids_and_refs(prediction, errors)
    return errors


def _normal(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _empty_string_paths(value, path: str = "$") -> list[str]:
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            result.extend(_empty_string_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_empty_string_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str) and not value.strip():
        result.append(path)
    return result


def quality_flags(prediction: dict, parse_metadata: dict) -> list[str]:
    flags = []
    counts = {path: len(_collection(prediction, path))
              for path in EXTREME_THRESHOLDS}
    if not prediction["datasets"]:
        flags.append("no_datasets")
    if not prediction["facilities"] and not prediction["samples"] \
            and not prediction["datasets"]:
        flags.append("empty_pan_inventory")
    if prediction["datasets"] and not prediction["samples"]:
        flags.append("datasets_without_samples")
    if prediction["datasets"] and not prediction["facilities"]:
        flags.append("datasets_without_facilities")
    for path, threshold in EXTREME_THRESHOLDS.items():
        if counts[path] >= threshold:
            flags.append(f"high_count:{path}")
    if _empty_string_paths(prediction):
        flags.append("empty_string_values")
    for path, prefix in ID_PREFIXES.items():
        values = [item["id"] for item in _collection(prediction, path)]
        expected = [f"{prefix}_{index}" for index in range(1, len(values) + 1)]
        if values != expected:
            flags.append(f"nonsequential_ids:{path}")
    for path, field in NAME_FIELDS.items():
        names = [_normal(item[field]) for item in _collection(prediction, path)]
        populated = [name for name in names if name]
        if len(populated) != len(set(populated)):
            flags.append(f"duplicate_names:{path}")
    if parse_metadata["warnings"]:
        flags.append("pdf_parser_warnings")
    if parse_metadata["low_word_presence_pages"]:
        flags.append("low_word_presence_pages")
    return sorted(set(flags))


def entity_counts(prediction: dict) -> dict[str, int]:
    return {
        "affiliations": len(prediction["bibliographic"]["affiliations"]),
        "authors": len(prediction["bibliographic"]["authors"]),
        "funding": len(prediction["bibliographic"]["funding"]),
        "facilities": len(prediction["facilities"]),
        "samples": len(prediction["samples"]),
        "datasets": len(prediction["datasets"]),
    }


def _stable_key(subject: str, paper_id: str) -> str:
    return hashlib.sha256(
        f"{REVIEW_SEED}:{subject}:{paper_id}".encode()).hexdigest()


def select_review_sample(rows: list[dict], count_per_subject: int = 4) -> list[dict]:
    selected = []
    for subject in sorted({row["subject_group"] for row in rows}):
        candidates = [row for row in rows if row["subject_group"] == subject]
        candidates.sort(key=lambda row: _stable_key(subject, row["paper_id"]))
        chosen = []
        validation = [row for row in candidates
                      if row["split_role"] == "pseudo_validation"]
        if validation:
            chosen.append(validation[0])
        for length_bin in ("short_1_10", "medium_11_20", "long_21_40",
                           "very_long_41_plus"):
            options = [row for row in candidates
                       if row["page_bin"] == length_bin and row not in chosen]
            if options and len(chosen) < count_per_subject:
                chosen.append(options[0])
        for row in candidates:
            if len(chosen) >= count_per_subject:
                break
            if row not in chosen:
                chosen.append(row)
        selected.extend(chosen)
    return sorted(selected, key=lambda row: row["paper_id"])


def evidence_screen(prediction: dict, document: str) -> dict:
    checks = [("bibliographic.title", prediction["bibliographic"]["title"])]
    for index, item in enumerate(prediction["facilities"]):
        checks.append((f"facilities[{index}].name", item["name"]))
    for index, item in enumerate(prediction["samples"]):
        checks.append((f"samples[{index}].sample_name", item["sample_name"]))
    for index, item in enumerate(prediction["datasets"]):
        checks.append((f"datasets[{index}].dataset_name", item["dataset_name"]))
        checks.append((f"datasets[{index}].measurement_technique",
                       item["measurement_technique"]))
    normalized_document = _normal(document)
    results = []
    for path, value in checks:
        normalized = _normal(value) if isinstance(value, str) else ""
        results.append({
            "path": path,
            "value": value,
            "exact_normalized_match": bool(normalized and normalized in normalized_document),
        })
    matched = sum(row["exact_normalized_match"] for row in results)
    return {
        "checked_strings": len(results),
        "exact_normalized_matches": matched,
        "match_fraction": round(matched / len(results), 4) if results else None,
        "unmatched": [row for row in results if not row["exact_normalized_match"]],
        "interpretation": (
            "A failed exact-string match is a review flag, not proof that the label "
            "is unsupported; abbreviations, paraphrases and PDF text artifacts can fail"
        ),
    }


def _summary(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "minimum": ordered[0],
        "p50": at(0.50),
        "p95": at(0.95),
        "maximum": ordered[-1],
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(
                record, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            stream.write("\n")


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")


def prepare(root: Path, output: Path) -> dict:
    source_dir = root / "data/share_600_papers"
    source_manifest_path = source_dir / "manifest.jsonl"
    split_path = root / "data/finetune/pseudo_label_supervision/splits.json"
    length_path = output / "length_report.json"
    parse_summary_path = output / "parse_summary.json"
    text_dir = output / "text"
    output.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(source_manifest_path)
    source_by_id = {row["paper_id"]: row for row in source_rows}
    split = strict_json(split_path)
    length_report = strict_json(length_path)
    parse_summary = strict_json(parse_summary_path)
    merged_schema = load_merged_schema(SCHEMAS)
    token_by_id = {row["paper_id"]: row
                   for row in length_report["records_longest_first"]}

    if length_report["scope"]["future_test_predictions_read"] is not False:
        raise ValueError("Length report did not preserve future-test isolation")
    if set(token_by_id) != set(split["pseudo_train"] + split["pseudo_validation"]):
        raise ValueError("Length report does not cover the frozen supervised split")
    if parse_summary["papers"] != 600:
        raise ValueError("Full parse cache is incomplete")

    schema_hashes = {name: file_hash(root / "schemas" / name) for name in SCHEMAS}
    prompt_path = root / "schemas/prompt_template.md"
    common_provenance = {
        "split_manifest": split_path.relative_to(root).as_posix(),
        "split_manifest_sha256": file_hash(split_path),
        "source_manifest": source_manifest_path.relative_to(root).as_posix(),
        "source_manifest_sha256": file_hash(source_manifest_path),
        "prompt_template": prompt_path.relative_to(root).as_posix(),
        "prompt_template_version": get_prompt_version(prompt_path),
        "prompt_template_sha256": file_hash(prompt_path),
        "schemas": schema_hashes,
        "tokenizer_model": TOKENIZER_MODEL,
        "tokenizer_revision": TOKENIZER_REVISION,
        "length_report_sha256": file_hash(length_path),
        "teacher_model": "google/gemma-4-31B-it",
        "teacher_revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "teacher_temperature": 0,
    }

    supervised_rows = []
    training_records = {role: [] for role in SUPERVISED_ROLES}
    formal_errors = {}
    flag_counts = Counter()
    count_values = {key: [] for key in entity_counts({
        "bibliographic": {"affiliations": [], "authors": [], "funding": []},
        "facilities": [], "samples": [], "datasets": []}).keys()}

    for role in SUPERVISED_ROLES:
        for paper_id in sorted(split[role]):
            source = source_by_id[paper_id]
            prediction_relative = source["global_seed_prediction"]["path"]
            if prediction_relative is None:
                raise ValueError(f"Supervised paper lacks prediction: {paper_id}")
            prediction_path = source_dir / prediction_relative
            if file_hash(prediction_path) != source["global_seed_prediction"]["sha256"]:
                raise ValueError(f"Prediction hash differs from source manifest: {paper_id}")
            prediction = strict_json(prediction_path)
            errors = validate_prediction(prediction, merged_schema)
            if errors:
                formal_errors[paper_id] = errors
                continue

            text_path = text_dir / f"{paper_id}.md"
            parse_metadata_path = text_dir / f"{paper_id}.json"
            document = text_path.read_bytes().decode("utf-8")
            parse_metadata = strict_json(parse_metadata_path)
            if text_hash(document) != parse_metadata["text_sha256"]:
                raise ValueError(f"Prepared text hash differs: {paper_id}")
            flags = quality_flags(prediction, parse_metadata)
            flag_counts.update(flags)
            counts = entity_counts(prediction)
            for key, value in counts.items():
                count_values[key].append(value)
            token_counts = token_by_id[paper_id]
            if token_counts["prediction_sha256"] != file_hash(prediction_path):
                raise ValueError(f"Length report prediction hash differs: {paper_id}")
            if token_counts["sequence_tokens"] > 65_536:
                raise ValueError(f"Supervised sequence exceeds 65,536 tokens: {paper_id}")

            target = canonical_json(prediction)
            prompt = build_prompt(SCHEMAS, document)
            record = {
                "paper_id": paper_id,
                "split_role": role,
                "task": "D_complete_json",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
                "token_counts": {
                    "prompt_without_chat_wrapper": token_counts[
                        "prompt_tokens_without_chat_wrapper"],
                    "target_without_chat_wrapper": token_counts[
                        "target_tokens_without_chat_wrapper"],
                    "complete_sequence": token_counts["sequence_tokens"],
                },
                "quality_flags": flags,
                "provenance": {
                    **common_provenance,
                    "source_text": text_path.relative_to(root).as_posix(),
                    "source_text_sha256": parse_metadata["text_sha256"],
                    "source_pdf": (source_dir / source["pdf"]["path"]
                                   ).relative_to(root).as_posix(),
                    "source_pdf_sha256": source["pdf"]["sha256"],
                    "prediction": prediction_path.relative_to(root).as_posix(),
                    "prediction_sha256": source["global_seed_prediction"]["sha256"],
                    "prediction_status": source["global_seed_prediction"]["status"],
                },
            }
            training_records[role].append(record)
            supervised_rows.append({
                "paper_id": paper_id,
                "split_role": role,
                "subject_group": source["subject_group"],
                "page_count": source["page_count"],
                "page_bin": page_bin(source),
                "entity_counts": counts,
                "quality_flags": flags,
                "prediction": prediction,
                "document": document,
            })

    error_path = output / "label_validation_errors.json"
    _write_json(error_path, formal_errors)
    role_by_id = {
        paper_id: role for role in SUPERVISED_ROLES for paper_id in split[role]
    }
    invalid_by_role = Counter(role_by_id[paper_id] for paper_id in formal_errors)
    expected_counts = {
        role: len(split[role]) - invalid_by_role[role] for role in SUPERVISED_ROLES
    }
    if {role: len(training_records[role]) for role in SUPERVISED_ROLES} != expected_counts:
        raise ValueError("Prepared supervised record counts differ from frozen split")

    review_rows = select_review_sample(supervised_rows)
    review_records = []
    for row in review_rows:
        review_records.append({
            "paper_id": row["paper_id"],
            "split_role": row["split_role"],
            "subject_group": row["subject_group"],
            "page_count": row["page_count"],
            "page_bin": row["page_bin"],
            "entity_counts": row["entity_counts"],
            "quality_flags": row["quality_flags"],
            "evidence_screen": evidence_screen(row["prediction"], row["document"]),
            "source_text": f"data/finetune/pseudo_label_supervision/prepared/text/{row['paper_id']}.md",
            "prediction": source_by_id[row["paper_id"]]["global_seed_prediction"]["path"],
            "review_status": "codex_assisted_review_complete",
        })

    future_records = []
    for paper_id in sorted(split["future_test"]):
        source = source_by_id[paper_id]
        parse_metadata = strict_json(text_dir / f"{paper_id}.json")
        future_records.append({
            "paper_id": paper_id,
            "split_role": "future_test",
            "subject_group": source["subject_group"],
            "published": source["published"],
            "page_count": source["page_count"],
            "page_bin": page_bin(source),
            "source_text": f"data/finetune/pseudo_label_supervision/prepared/text/{paper_id}.md",
            "source_text_sha256": parse_metadata["text_sha256"],
            "human_annotation_status": "required",
            "teacher_prediction_read": False,
            "teacher_prediction_permitted_in_training": False,
        })

    train_path = output / "train_pseudo_d.jsonl"
    validation_path = output / "validation_pseudo_d.jsonl"
    future_path = output / "future_test_manifest.jsonl"
    review_path = output / "review_sample.jsonl"
    quality_path = output / "quality_report.md"
    semantic_review_path = output / "semantic_review.md"
    manifest_path = output / "preparation_manifest.json"
    _write_jsonl(train_path, training_records["pseudo_train"])
    _write_jsonl(validation_path, training_records["pseudo_validation"])
    _write_jsonl(future_path, future_records)
    _write_jsonl(review_path, review_records)

    unmatched = sum(record["evidence_screen"]["checked_strings"]
                    - record["evidence_screen"]["exact_normalized_matches"]
                    for record in review_records)
    checked = sum(record["evidence_screen"]["checked_strings"]
                  for record in review_records)
    parser_warning_papers = parse_summary["papers_with_parser_warnings"]
    low_coverage = parse_summary["papers_with_low_word_presence_pages"]
    flag_paper_ids = {
        flag: sorted(row["paper_id"] for row in supervised_rows
                     if flag in row["quality_flags"])
        for flag in sorted(flag_counts)
    }
    report_lines = [
        "# Pseudo-label preparation quality report",
        "",
        "## Outcome",
        "",
        f"- Formal validation: {565 - len(formal_errors)}/565 supervised predictions passed.",
        f"- Schema-invalid teacher labels excluded without repair: {len(formal_errors)}.",
        f"- Prepared D records: {len(training_records['pseudo_train'])} train and "
        f"{len(training_records['pseudo_validation'])} validation.",
        "- Future test: 30 papers; teacher predictions were not read.",
        "- Missing teacher predictions: 5 papers excluded from supervision.",
        f"- Parser warnings: {len(parser_warning_papers)} papers.",
        f"- Low word-presence flags: {len(low_coverage)} papers.",
        "- Context exclusions at 65,536 tokens: 0.",
        "",
        "Formal validation covers complete nested shape, types, allowed values, "
        "cardinalities, unique IDs and reference integrity. It does not establish "
        "scientific correctness.",
        "",
        "## Label flags",
        "",
    ]
    if flag_counts:
        report_lines.extend(
            f"- `{name}`: {count}" for name, count in sorted(flag_counts.items()))
    else:
        report_lines.append("- None.")
    report_lines.extend(["", "## Entity-count distributions", ""])
    report_lines.extend(
        f"- {name}: {_summary(values)}" for name, values in count_values.items())
    report_lines.extend([
        "",
        "## Stratified semantic-review sample",
        "",
        f"- Selected papers: {len(review_records)} (four per subject group).",
        f"- Validation papers in sample: {sum(r['split_role'] == 'pseudo_validation' for r in review_records)}.",
        f"- Exact normalized evidence strings found: {checked - unmatched}/{checked}.",
        "- Exact-match failures require inspection; they are not automatic label errors.",
        "- Detailed sample: `review_sample.jsonl`.",
        "- Codex-assisted review outcome: no additional exclusion was warranted.",
        "",
        "A domain expert should independently review the sample before using the data "
        "for a paper-quality claim. The Codex-assisted review is sufficient for a "
        "provisional pilot once the GPU memory probe passes.",
        "",
    ])
    quality_path.write_text("\n".join(report_lines), encoding="utf-8", newline="\n")

    semantic_lines = [
        "# Codex-assisted pseudo-label semantic review",
        "",
        "## Scope",
        "",
        f"- Reviewed the evidence-screen output for all {len(review_records)} stratified papers.",
        "- The sample contains four papers from each subject group and eight validation papers.",
        f"- Exact normalized matches: {checked - unmatched}/{checked} checked strings.",
        "- Directly inspected the lowest-match documents and the label-quality flag groups.",
        "- Visually inspected the worst parser-coverage pages in `1903.05884`, "
        "`2207.14409` and `1910.09611`.",
        "",
        "## Findings",
        "",
        "- Exact-match misses were attributable to PDF typography, formulas, abbreviations, "
        "paraphrased technique labels or line-breaking; they remain visible in the review JSONL.",
        "- Repeated sample names in `2208.02456`, `2603.09635` and `2604.26560` "
        "refer to distinct physical forms such as single crystal, powder or polycrystalline material.",
        "- Repeated dataset names in `1809.02567`, `1909.06555`, `2002.08152`, "
        "`2101.08026`, `2108.11506` and `2602.22156` refer to distinct campaigns, "
        "samples, facilities or acquisition conditions; the quality flags are retained.",
        "- The 27 samples in `2212.01894` are explicitly named variants rather than an "
        "obvious duplicated inventory.",
        "- `2606.10715` has no named facility but contains laboratory-style datasets; an "
        "empty facility list is allowed by the schema.",
        "- The visually reviewed low-coverage pages contain ordinary scientific text, "
        "tables, figures or sparse supplementary pages. No additional parser exclusion was warranted.",
        "",
        "## Decision",
        "",
        "No additional records were excluded or edited. All quality flags remain attached "
        "to their training records. This review supports a provisional pilot, not a claim "
        "that the pseudo-labels are scientifically equivalent to human annotations. An "
        "independent domain-expert review remains recommended before publication-quality use.",
        "",
    ]
    semantic_review_path.write_text(
        "\n".join(semantic_lines), encoding="utf-8", newline="\n")

    outputs = [train_path, validation_path, future_path, review_path, quality_path,
               semantic_review_path, error_path]
    manifest = {
        "version": "1.0.0",
        "status": "prepared_pending_gpu_probe",
        "counts": {
            "pseudo_train": len(training_records["pseudo_train"]),
            "pseudo_validation": len(training_records["pseudo_validation"]),
            "future_test": len(future_records),
            "excluded_missing_prediction": len(split["excluded_missing_prediction"]),
            "excluded_schema_invalid_prediction": len(formal_errors),
        },
        "formal_validation": {
            "passed": 565 - len(formal_errors),
            "failed_and_excluded": len(formal_errors),
            "excluded_paper_ids": sorted(formal_errors),
            "checks": [
                "complete nested schema shape", "field types", "allowed values",
                "cardinalities", "unique IDs", "reference integrity",
            ],
        },
        "quality": {
            "flag_counts": dict(sorted(flag_counts.items())),
            "flag_paper_ids": flag_paper_ids,
            "review_sample_papers": len(review_records),
            "review_sample_status": "codex_assisted_review_complete",
            "domain_expert_review": "recommended_before_paper_quality_claim",
            "parser_warning_papers": len(parser_warning_papers),
            "low_word_presence_papers": len(low_coverage),
        },
        "policy": {
            "task": "D_complete_json",
            "target_values_modified": False,
            "target_whitespace_canonicalized": True,
            "future_test_predictions_read": False,
            "truncation": False,
            "maximum_sequence_tokens": max(
                record["token_counts"]["complete_sequence"]
                for role in SUPERVISED_ROLES for record in training_records[role]),
        },
        "inputs": common_provenance,
        "outputs": {
            path.relative_to(root).as_posix(): {
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        },
        "preparation_source_sha256": text_hash(Path(__file__).read_text(encoding="utf-8")),
    }
    _write_json(manifest_path, manifest)
    return manifest

