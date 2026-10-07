"""Build the six numerical LaTeX tables from outputs/paper/results.json.

Run from the repository root with:

    uv run python scripts/build_paper_tables.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "paper" / "tables"
RESULTS_PATH = ROOT / "outputs" / "paper" / "results.json"


def _write_text(path: Path, content: str) -> Path:
    """Write a generated UTF-8 artifact and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    return path


def _load_paper_results() -> dict[str, Any]:
    """Read the saved numerical results produced by compute_paper_results.py."""

    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{RESULTS_PATH} is missing. Run "
            "'uv run python scripts/compute_paper_results.py' first."
        ) from exc


def _model_labels(models: Sequence[str]) -> tuple[str, ...]:
    """Shorten saved model names for the table column headings."""

    labels = {
        "Mistral Large 3": "Mistral",
        "Gemma 4 31B": "Gemma",
        "Qwen3.5 27B": "Qwen 27B",
        "Qwen3.5 9B": "Qwen 9B",
        "Qwen3.5 4B": "Qwen 4B",
    }
    return tuple(labels.get(model, model) for model in models)


def build_benchmark_statistics(output_dir: Path = TABLE_DIR) -> Path:
    """Build ``tab:benchmark-statistics`` from ``outputs/paper/results.json``.

    Write ``paper/tables/benchmark_statistics.tex``."""

    groups = _load_paper_results()["benchmark_statistics"]
    rows = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\begin{tabular}{@{}lr@{}}",
        "    \\toprule",
        "    Item & Count \\\\",
        "    \\midrule",
    ]
    for group_index, (label, counts) in enumerate(groups.items()):
        if group_index:
            rows.append("    \\addlinespace")
        rows.append(
            f"    \\multicolumn{{2}}{{@{{}}l}}{{\\textbf{{{label}}}}} \\\\"
        )
        rows.extend(f"    {label} & {count:,} \\\\" for label, count in counts.items())
    rows.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Scale of the PaN metadata benchmark. Entity and",
            "  reference counts are derived from the final gold records.}",
            "  \\label{tab:benchmark-statistics}",
            "\\end{table}",
            "",
        ]
    )
    return _write_text(output_dir / "benchmark_statistics.tex", "\n".join(rows))


def build_iaa_recovery(output_dir: Path = TABLE_DIR) -> Path:
    """Build ``tab:iaa-recovery`` from ``outputs/paper/results.json``.

    Write the six entity F1 scores and matched/unmatched counts to
    ``paper/tables/iaa_recovery.tex``."""

    rows = _load_paper_results()["iaa"]["headline"]
    latex = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\begin{tabular}{@{}lrrr@{}}",
        "    \\toprule",
        r"    Entity collection & F1 & Matched & Unmatched \\",
        "    \\midrule",
    ]
    for row in rows:
        latex.append(
            rf"    {row['outcome']} & {row['agreement']:.3f} & {row['matched']} & {row['unmatched']} \\"
        )
    latex.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Pairwise agreement on entity recovery outcomes for",
            "  the 30 evaluation papers.}",
            "  \\label{tab:iaa-recovery}",
            "\\end{table}",
            "",
        ]
    )
    return _write_text(output_dir / "iaa_recovery.tex", "\n".join(latex))


def build_zero_shot_recovery(output_dir: Path = TABLE_DIR) -> Path:
    """Build ``tab:zero-shot-recovery`` from ``outputs/paper/results.json``.

    Write six entity F1 scores per model to ``paper/tables/zero_shot_recovery.tex``."""

    data = _load_paper_results()["zero_shot"]
    header = "    Entity collection & " + " & ".join(_model_labels(data["models"])) + r" \\"

    latex = [
        "\\begin{table*}[!t]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "  \\begin{tabular}{@{}p{0.19\\textwidth}rrrrrrr@{}}",
        "    \\toprule",
        header,
        "    \\midrule",
    ]
    for row in data["headline"]:
        values = " & ".join(f"{score:.3f}" for score in row['scores'])
        latex.append(f"    {row['outcome']} & {values} \\\\")
    latex.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Full-instruction zero-shot entity recovery for the six",
            "  entity collections on the 30 evaluation papers. All values",
            "  are entity F1.}",
            "  \\label{tab:zero-shot-recovery}",
            "\\end{table*}",
            "",
        ]
    )
    return _write_text(output_dir / "zero_shot_recovery.tex", "\n".join(latex))


def build_instruction_effects(output_dir: Path = TABLE_DIR) -> Path:
    """Build ``tab:instruction-effects`` from ``outputs/paper/results.json``.

    Write full-minus-no-instruction entity F1 differences to
    ``paper/tables/instruction_effects.tex``."""

    data = _load_paper_results()
    models = data["zero_shot"]["models"]
    latex = [
        "\\begin{table*}[!t]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\begin{tabular}{@{}p{0.18\\textwidth}rrrrrrr@{}}",
        "    \\toprule",
        "    Entity collection & " + " & ".join(_model_labels(models)) + " \\\\",
        "    \\midrule",
    ]
    for row in data["instruction_effects"]:
        values = " & ".join(f"{delta:+.3f}" for delta in row['deltas'])
        latex.append(f"    {row['outcome']} & {values} \\\\")
    latex.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Effect of detailed field instructions on entity recovery",
            "  for the six entity collections on the 30 evaluation papers,",
            "  reported as full-instruction minus no-instruction entity F1.",
            "  Positive values favor the full instructions.}",
            "  \\label{tab:instruction-effects}",
            "\\end{table*}",
            "",
        ]
    )
    return _write_text(output_dir / "instruction_effects.tex", "\n".join(latex))


# The PDF parser discards the rotated arXiv stamp on some papers. These fields
# are flagged only for model results; human annotation is not parser-limited.
PARSER_AFFECTED_PATHS = frozenset(
    {
        "bibliographic.arxiv_id",
        "bibliographic.publication_date",
        "bibliographic.venue",
    }
)


@dataclass(frozen=True)
class AppendixField:
    """One appendix row, with an optional secondary metric."""

    label: str
    path: str
    metric_key: str | None = None


APPENDIX_BLOCKS = (
    ("Paper-level fields", (
        AppendixField("Title", "bibliographic.title"),
        AppendixField("Subtitle", "bibliographic.subtitle"),
        AppendixField("Abstract", "bibliographic.abstract"),
        AppendixField("DOI", "bibliographic.doi"),
        AppendixField("arXiv ID", "bibliographic.arxiv_id"),
        AppendixField("URL", "bibliographic.url"),
        AppendixField("Publication date", "bibliographic.publication_date"),
        AppendixField("Venue", "bibliographic.venue"),
        AppendixField("Publisher", "bibliographic.publisher"),
        AppendixField("Language", "bibliographic.language"),
        AppendixField("Keywords", "bibliographic.keywords"),
        AppendixField("Section-heading content", "document_overview.section_headings", "f1"),
        AppendixField("Section-heading order", "document_overview.section_headings", "mean_order"),
        AppendixField("Has references", "document_overview.has_references"),
        AppendixField("Has figures", "document_overview.has_figures"),
        AppendixField("Has tables", "document_overview.has_tables"),
    )),
    ("Affiliations", (
        AppendixField("Affiliation entities", "bibliographic.affiliations"),
        AppendixField("Affiliation label", "bibliographic.affiliations.attrs.label"),
        AppendixField("Affiliation text", "bibliographic.affiliations.attrs.text"),
    )),
    ("Authors", (
        AppendixField("Author entities", "bibliographic.authors"),
        AppendixField("Author full name", "bibliographic.authors.attrs.full_name"),
        AppendixField("Author ORCID", "bibliographic.authors.attrs.orcid"),
        AppendixField("Author email", "bibliographic.authors.attrs.email"),
        AppendixField("Corresponding-author flag", "bibliographic.authors.attrs.is_corresponding"),
        AppendixField("Author--affiliation refs", "bibliographic.authors.attrs.affiliation_refs"),
    )),
    ("Funding", (
        AppendixField("Funding entities", "bibliographic.funding"),
        AppendixField("Funder", "bibliographic.funding.attrs.funder"),
        AppendixField("Grant ID", "bibliographic.funding.attrs.grant_id"),
    )),
    ("Facilities", (
        AppendixField("Facility entities", "facilities"),
        AppendixField("Facility name", "facilities.attrs.name"),
    )),
    ("Samples", (
        AppendixField("Sample entities", "samples"),
        AppendixField("Sample name", "samples.attrs.sample_name"),
        AppendixField("Chemical composition", "samples.attrs.chemical_composition"),
        AppendixField("Sample state", "samples.attrs.sample_state"),
        AppendixField("Sample mass", "samples.attrs.sample_mass"),
        AppendixField("Sample description", "samples.attrs.sample_description"),
        AppendixField("Supplier / creator", "samples.attrs.supplier_or_creator"),
    )),
    ("Datasets", (
        AppendixField("Dataset entities", "datasets"),
        AppendixField("Dataset name", "datasets.attrs.dataset_name"),
        AppendixField("Measurement technique", "datasets.attrs.measurement_technique"),
        AppendixField("Dataset description", "datasets.attrs.dataset_description"),
        AppendixField("Measurement mode", "datasets.attrs.measurement_mode"),
        AppendixField("Dataset purpose", "datasets.attrs.purpose"),
        AppendixField("Dataset--sample refs", "datasets.attrs.sample_refs"),
        AppendixField("Dataset--facility refs", "datasets.attrs.facility_refs"),
    )),
)


APPENDIX_METRIC_LABELS = {
    "entity": "Entity F1",
    "ordered_set_f1": "Set F1",
    "set_f1": "Set F1",
    "reference": "Reference F1",
    "rouge_l": "ROUGE-L",
    "normalized_exact": "Normalized exact",
    "exact": "Exact",
}


def validate_appendix_registry(metrics: dict[str, str]) -> None:
    """Require complete, non-duplicative coverage of all 44 frozen paths."""

    fields = [field for _, group in APPENDIX_BLOCKS for field in group]
    paths = [field.path for field in fields]
    expected = set(metrics)
    if len(expected) != 44:
        raise ValueError(f"expected 44 frozen paths, found {len(expected)}")
    if set(paths) != expected:
        raise ValueError(
            "appendix registry does not cover the frozen plan: "
            f"missing={sorted(expected - set(paths))}, "
            f"extra={sorted(set(paths) - expected)}"
        )
    duplicates = {path for path in paths if paths.count(path) > 1}
    if duplicates != {"document_overview.section_headings"}:
        raise ValueError(f"unexpected appendix path duplication: {sorted(duplicates)}")
    heading_fields = [
        field for field in fields if field.path == "document_overview.section_headings"
    ]
    if {field.metric_key for field in heading_fields} != {"f1", "mean_order"}:
        raise ValueError("headings must expose content F1 and mean_order exactly once")
    if len(fields) != 45:
        raise ValueError(f"expected 45 appendix outcomes, found {len(fields)}")


def _appendix_metric_label(field: AppendixField, metrics: dict[str, str]) -> str:
    """Return a display label for the saved native metric."""

    if field.metric_key == "mean_order":
        return r"Kendall $\tau$"
    metric_kind = metrics[field.path]
    try:
        return APPENDIX_METRIC_LABELS[metric_kind]
    except KeyError as exc:
        raise ValueError(f"unsupported appendix metric {metric_kind!r}") from exc


def _support_marker(result: dict[str, Any]) -> str:
    """Mark null-only or sparse scalar support; F1 and order scores have none."""

    numerator, denominator = result.get("support", (0, 0))
    if denominator <= 0:
        return ""
    if numerator == 0:
        return r"$^{\dagger}$"
    if numerator / denominator <= 0.1:
        return r"$^{\ddagger}$"
    return ""


def build_appendix_iaa(output_dir: Path = TABLE_DIR) -> Path:
    """Build ``tab:appendix-iaa`` from ``outputs/paper/results.json``.

    Write all 45 outcomes to ``paper/tables/appendix_iaa.tex``."""

    data = _load_paper_results()
    fields = data["iaa"]["fields"]
    metrics = data["metrics"]
    validate_appendix_registry(metrics)
    latex = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{6pt}",
        "  \\begin{tabular}{@{}p{0.34\\textwidth}p{0.18\\textwidth}r@{}}",
        "    \\toprule",
        r"     & Metric & Agreement \\",
        "    \\midrule",
    ]
    for index, (block_label, block_fields) in enumerate(APPENDIX_BLOCKS):
        if index:
            latex.extend(("    \\addlinespace", "    \\midrule"))
        latex.append(rf"    \multicolumn{{3}}{{@{{}}l}}{{\textit{{{block_label}}}}} \\")
        for field in block_fields:
            result = fields[field.path]
            score = result["order_score" if field.metric_key == "mean_order" else "score"]
            latex.append(
                f"    {field.label}{_support_marker(result)} & "
                f"{_appendix_metric_label(field, metrics)} & "
                rf"{score:.3f} \\"
            )
    latex.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Complete pairwise annotation agreement on the 30",
            "  evaluation papers. Attribute and reference rows are",
            "  conditional on aligned parents. A $^{\\dagger}$ marks a field with",
            "  no non-null support and a $^{\\ddagger}$ one with at most 10\\%",
            "  non-null support.}",
            "  \\label{tab:appendix-iaa}",
            "\\end{table*}",
            "",
        ]
    )
    return _write_text(output_dir / "appendix_iaa.tex", "\n".join(latex))


def build_appendix_zero_shot(output_dir: Path = TABLE_DIR) -> Path:
    """Build ``tab:appendix-zero-shot`` from ``outputs/paper/results.json``.

    Write all 45 outcomes per model to ``paper/tables/appendix_zero_shot.tex``."""

    data = _load_paper_results()
    fields = data["zero_shot"]["fields"]
    models = data["zero_shot"]["models"]
    metrics = data["metrics"]
    validate_appendix_registry(metrics)
    latex = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\fontsize{8.5pt}{9.5pt}\\selectfont",
        "  \\setlength{\\tabcolsep}{2pt}",
        "  \\begin{tabular}{@{}p{0.24\\textwidth}p{0.13\\textwidth}rrrrrrr@{}}",
        "    \\toprule",
        "     & Metric & "
        + " & ".join(_model_labels(models))
        + r" \\",
        "    \\midrule",
    ]
    for index, (block_label, block_fields) in enumerate(APPENDIX_BLOCKS):
        if index:
            latex.extend(("    \\addlinespace", "    \\midrule"))
        latex.append(
            rf"    \multicolumn{{9}}{{@{{}}l}}{{\textit{{{block_label}}}}} \\"
        )
        for field in block_fields:
            result = fields[field.path]
            scores = result["order_scores" if field.metric_key == "mean_order" else "scores"]
            values = " & ".join(f"{score:.3f}" for score in scores)
            parser_flag = (
                r"$^{\S}$" if field.path in PARSER_AFFECTED_PATHS else ""
            )
            latex.append(
                f"    {field.label}{_support_marker(result)}"
                f"{parser_flag} & "
                f"{_appendix_metric_label(field, metrics)} & "
                rf"{values} \\"
            )
    latex.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Complete full-instruction zero-shot results on the 30 evaluation",
            "  papers. Paper-level scalar and order scores are averaged",
            "  over papers; scalar entity-attribute scores are averaged over aligned",
            "  pairs; count-based set, entity, and reference F1 scores are",
            "  micro-aggregated across the corpus. Attribute and reference scores are",
            "  conditional",
            "  on aligned parents. A $^{\\dagger}$ marks a field",
            "  with no gold non-null values and a $^{\\ddagger}$ one with at most",
            "  10\\% gold non-null support. A $^{\\S}$ marks fields whose gold evidence is",
            "  discarded during PDF-to-text conversion on some papers, as",
            "  detailed in Appendix~\\ref{app:complete-results}.}",
            "  \\label{tab:appendix-zero-shot}",
            "\\end{table*}",
            "",
        ]
    )
    return _write_text(output_dir / "appendix_zero_shot.tex", "\n".join(latex))


def main(argv: Sequence[str] | None = None) -> int:
    """Build all six numerical tables from the saved paper results."""

    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    build_benchmark_statistics()
    build_iaa_recovery()
    build_zero_shot_recovery()
    build_instruction_effects()
    build_appendix_iaa()
    build_appendix_zero_shot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
