"""Contract tests for the current paper table builders."""

from __future__ import annotations

import sys
import json
import shutil
import subprocess
import pytest
import yaml
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_paper_tables as tables  # noqa: E402


@pytest.fixture(scope="module")
def paper_results_path(tmp_path_factory) -> Path:
    """Prepare synthetic scores for the real scoring-plan fields, without outputs/."""
    plan = yaml.safe_load((ROOT / "configs/evaluation_metrics.yaml").read_text(encoding="utf-8"))
    metrics = {}
    for field, spec in plan["scoring_plan"].items():
        metrics[field] = spec["metric"]
        metrics.update((f"{field}.attrs.{name}", attribute["metric"])
                       for name, attribute in spec.get("attrs", {}).items())
    models = ["GPT-5.5", "GPT-4o", "Mistral Large 3", "Gemma 4 31B",
              "Qwen3.5 27B", "Qwen3.5 9B", "Qwen3.5 4B"]
    entities = (
        ("Affiliations", "bibliographic.affiliations"),
        ("Authors", "bibliographic.authors"), ("Funding", "bibliographic.funding"),
        ("Facilities", "facilities"), ("Samples", "samples"), ("Datasets", "datasets"),
    )
    scores = [0.125, 0.234, 0.345, 0.456, 0.567, 0.678, 0.789]
    iaa_fields = {field: {"score": 0.625} for field in metrics}
    model_fields = {field: {"scores": scores} for field in metrics}
    for field, metric in metrics.items():
        if metric in {"rouge_l", "normalized_exact", "exact"}:
            iaa_fields[field]["support"] = model_fields[field]["support"] = [30, 30]
    for field, support in (
        ("bibliographic.subtitle", [0, 30]),
        ("bibliographic.authors.attrs.orcid", [3, 30]),
        ("samples.attrs.sample_mass", [0, 0]),
    ):
        iaa_fields[field]["support"] = model_fields[field]["support"] = support
    iaa_fields["document_overview.section_headings"]["order_score"] = -0.25
    model_fields["document_overview.section_headings"]["order_scores"] = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    iaa_headline, zero_headline = [], []
    for index, (outcome, field) in enumerate(entities):
        agreement = 0.601 + index * 0.01
        entity_scores = [score + index * 0.01 for score in scores]
        iaa_fields[field]["score"] = agreement
        model_fields[field]["scores"] = entity_scores
        iaa_headline.append({"outcome": outcome, "path": field, "agreement": agreement,
                             "matched": 17 + index, "unmatched": 3 + index})
        zero_headline.append({"outcome": outcome, "path": field, "scores": entity_scores})
    data = {
        "n_papers": 30, "metrics": metrics,
        "benchmark_statistics": {
            "Corpus and annotation": {"Papers": 40},
            "Repeated entities": {"Samples": 17},
            "Paper-local reference edges": {"Dataset--sample": 21},
        },
        "iaa": {"headline": iaa_headline, "fields": iaa_fields},
        "zero_shot": {"models": models, "headline": zero_headline, "fields": model_fields},
        "instruction_effects": [
            {"outcome": outcome, "path": field,
             "deltas": [-0.12345, 0, 0.01234, 0.1, -0.2, 0.03, 0.04]}
            for outcome, field in entities
        ],
    }
    path = tmp_path_factory.mktemp("paper_results") / "results.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def saved_paper_results(monkeypatch, paper_results_path: Path) -> Path:
    """Point numerical table builders at the prepared snapshot."""

    monkeypatch.setattr(tables, "RESULTS_PATH", paper_results_path)
    return paper_results_path


@pytest.mark.usefixtures("saved_paper_results")
def test_benchmark_statistics_latex_has_reference_and_groups(tmp_path: Path) -> None:
    latex = tables.build_benchmark_statistics(output_dir=tmp_path).read_text(encoding="utf-8")
    assert r"\label{tab:benchmark-statistics}" in latex
    assert "Paper-local reference edges" in latex


@pytest.mark.usefixtures("saved_paper_results")
def test_complete_appendix_registry_covers_all_frozen_paths() -> None:
    metrics = tables._load_paper_results()["metrics"]
    tables.validate_appendix_registry(metrics)

    fields = [field for _, group in tables.APPENDIX_BLOCKS for field in group]
    paths = [field.path for field in fields]
    assert len(set(metrics)) == 44
    assert len(fields) == 45
    assert set(paths) == set(metrics)
    assert paths.count("document_overview.section_headings") == 2


@pytest.mark.usefixtures("saved_paper_results")
def test_compact_iaa_recovery_preserves_all_headline_outcomes(tmp_path: Path) -> None:
    latex = tables.build_iaa_recovery(output_dir=tmp_path).read_text(encoding="utf-8")

    assert r"\begin{table}[t]" in latex
    assert r"\label{tab:iaa-recovery}" in latex
    assert latex.count("Entity F1") == 0
    for outcome in (
        "Affiliations",
        "Authors",
        "Funding",
        "Facilities",
        "Samples",
        "Datasets",
    ):
        assert outcome in latex
    assert "Matched & Unmatched" in latex
    assert "evaluation papers" in latex
    assert "Affiliations & 0.601 & 17 & 3" in latex
    assert "Datasets & 0.651 & 22 & 8" in latex


@pytest.mark.usefixtures("saved_paper_results")
def test_complete_iaa_appendix_uses_native_metrics_and_support(tmp_path: Path) -> None:
    combined = tables.build_appendix_iaa(output_dir=tmp_path).read_text(encoding="utf-8")

    # Merged profile spans paper-level and entity blocks in one table, with
    # the paper-level fields grouped ahead of the repeated entities.
    assert r"\textit{Paper-level fields}" in combined
    assert r"\textit{Datasets}" in combined
    # Heading content is scored with the same set F1 as other set-valued
    # fields; its ordering is the separate Kendall-tau row below.
    assert "Section-heading content & Set F1 & 0.625" in combined
    assert r"Section-heading order & Kendall $\tau$ & -0.250" in combined
    assert "Author--affiliation refs & Reference F1" in combined
    assert "Chemical composition & Normalized exact" in combined
    assert "Dataset--facility refs & Reference F1" in combined
    # The conditionality column is dropped; parent is conveyed by grouping.
    assert "Cond. on" not in combined
    # The Support column is dropped; its low-support flag moves to the field
    # name (Subtitle has no gold non-null values), and no support strings remain.
    assert r"Subtitle$^{\dagger}$ & ROUGE-L" in combined
    assert r"Author ORCID$^{\ddagger}$ & Normalized exact" in combined
    assert "Sample mass & Normalized exact & 0.625" in combined
    assert "non-null/aligned" not in combined
    assert "TP/FP/FN" not in combined


@pytest.mark.usefixtures("saved_paper_results")
def test_zero_shot_appendix_values_and_flags_without_support_column(tmp_path: Path) -> None:
    values = tables.build_appendix_zero_shot(output_dir=tmp_path).read_text(encoding="utf-8")

    # Full profile spans paper-level and entity blocks in one table, without a
    # conditionality column (parent is conveyed by grouping).
    assert r"\textit{Paper-level fields}" in values
    assert r"\textit{Datasets}" in values
    assert "Title & ROUGE-L &" in values
    assert "Author entities & Entity F1 &" in values
    assert "Dataset entities & Entity F1 &" in values
    assert "Chemical composition & Normalized exact &" in values
    assert "Dataset--facility refs & Reference F1 &" in values
    assert "Cond. on" not in values
    assert "Section-heading content & Set F1 & 0.125 & 0.234 & 0.345" in values
    assert r"Section-heading order & Kendall $\tau$ & -0.300 & -0.200 & -0.100" in values
    # The Support column is dropped; its low-support flag moves to the field
    # name, and no support strings remain in the body.
    assert r"Subtitle$^{\dagger}$ & ROUGE-L &" in values
    assert r"Author ORCID$^{\ddagger}$ & Normalized exact" in values
    assert "Sample mass & Normalized exact & 0.125" in values
    assert "non-null/aligned" not in values
    assert "gold items" not in values


@pytest.mark.usefixtures("saved_paper_results")
def test_parser_affected_flags_are_exact_and_model_only(tmp_path: Path) -> None:
    zero_shot = tables.build_appendix_zero_shot(output_dir=tmp_path).read_text(
        encoding="utf-8"
    )
    iaa = tables.build_appendix_iaa(output_dir=tmp_path).read_text(encoding="utf-8")

    for label in ("arXiv ID", "Publication date", "Venue"):
        assert f"{label}$^{{\\S}}$ &" in zero_shot
        assert f"{label}$^{{\\S}}$" not in iaa

    # Three flagged rows plus the zero-shot caption's marker definition.
    assert zero_shot.count(r"$^{\S}$") == 4
    assert r"A $^{\S}$ marks fields whose gold evidence is" in zero_shot
    assert "discarded during PDF-to-text conversion" in zero_shot
    assert r"Appendix~\ref{app:complete-results}" in zero_shot
    assert r"$^{\S}$" not in iaa


@pytest.mark.usefixtures("saved_paper_results")
def test_zero_shot_recovery_latex_is_scoped_and_nonconditional(tmp_path: Path) -> None:
    recovery = tables.build_zero_shot_recovery(output_dir=tmp_path).read_text(encoding="utf-8")
    assert r"\label{tab:zero-shot-recovery}" in recovery
    assert "Cond. on" not in recovery
    # The uniform metric column was dropped; all rows are entity F1 (caption).
    assert "Metric" not in recovery
    assert "Entity F1" not in recovery
    assert "GPT-5.5 & GPT-4o & Mistral & Gemma & Qwen 27B & Qwen 9B & Qwen 4B" in recovery
    assert "Affiliations & 0.125 & 0.234 & 0.345 & 0.456 & 0.567 & 0.678 & 0.789" in recovery
    assert "Datasets & 0.175 & 0.284 & 0.395 & 0.506 & 0.617 & 0.728 & 0.839" in recovery


@pytest.mark.usefixtures("saved_paper_results")
def test_instruction_effect_latex_is_scoped_and_unaveraged(tmp_path: Path) -> None:
    latex = tables.build_instruction_effects(output_dir=tmp_path).read_text(encoding="utf-8")

    assert r"\label{tab:instruction-effects}" in latex
    assert "30 evaluation papers" in latex
    assert "full-instruction minus no-instruction" in latex
    assert r"\textbf" not in latex
    assert "Samples & -0.123 & +0.000 & +0.012 & +0.100 & -0.200 & +0.030 & +0.040" in latex


def test_result_tables_render_from_saved_json_without_computation_sources(
    paper_results_path: Path, tmp_path: Path,
) -> None:
    """Only the renderer and saved JSON are present."""

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(ROOT / "scripts/build_paper_tables.py", scripts / "build_paper_tables.py")
    saved = json.loads(paper_results_path.read_text(encoding="utf-8"))
    saved["iaa"]["headline"][0]["agreement"] = 0.125
    saved["zero_shot"]["headline"][0]["scores"][0] = 0.456
    saved["benchmark_statistics"]["Repeated entities"]["Samples"] = 1_234
    path = tmp_path / "outputs" / "paper" / "results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(saved), encoding="utf-8")
    command = [sys.executable, "-B", str(scripts / "build_paper_tables.py")]

    help_result = subprocess.run(
        [*command, "--help"], cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    assert "--artifact" not in help_result.stdout
    obsolete_option = subprocess.run(
        [*command, "--artifact", "iaa-recovery"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert obsolete_option.returncode == 2
    assert "unrecognized arguments" in obsolete_option.stderr
    assert not (tmp_path / "paper").exists()

    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)

    generated = tmp_path / "paper" / "tables"
    assert {path.name for path in generated.glob("*.tex")} == {
        "iaa_recovery.tex", "appendix_iaa.tex", "zero_shot_recovery.tex",
        "appendix_zero_shot.tex", "instruction_effects.tex", "benchmark_statistics.tex",
    }
    assert "Affiliations & 0.125 &" in (generated / "iaa_recovery.tex").read_text()
    assert "Affiliations & 0.456 &" in (generated / "zero_shot_recovery.tex").read_text()
    assert "Samples & 1,234" in (generated / "benchmark_statistics.tex").read_text()


def test_missing_paper_results_explains_required_command(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tables, "RESULTS_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="compute_paper_results.py"):
        tables.build_iaa_recovery(tmp_path)
