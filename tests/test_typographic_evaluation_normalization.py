"""Regression tests for scoring-safe typographic normalization."""

from meta_data_extraction.alignment import (
    AlignmentResult,
    align_samples,
    align_similarity,
)
from meta_data_extraction.evaluation import load_plan, _score_entities
from meta_data_extraction.normalization import (
    normalize_author_name_typography,
    normalize_sample_formula_formatting,
)


def test_author_spacing_acute_is_repaired_without_stripping_accents():
    assert normalize_author_name_typography(
        "Andr´e Heinemann"
    ) == "André Heinemann"
    assert normalize_author_name_typography(
        "M. Garc´ıa-Fern´andez"
    ) == "M. García-Fernández"
    assert normalize_author_name_typography(
        "André Heinemann"
    ) == "André Heinemann"
    assert normalize_author_name_typography(
        "Andre Heinemann"
    ) != "André Heinemann"


def test_author_typography_normalization_is_idempotent():
    value = "M. Garc´ıa-Fern´andez"
    once = normalize_author_name_typography(value)
    assert normalize_author_name_typography(once) == once


def test_sample_formula_formatting_folds_only_presentation_differences():
    assert normalize_sample_formula_formatting(
        "Ni0 . 90V0 . 10"
    ) == "Ni0.90V0.10"
    assert normalize_sample_formula_formatting(
        "Sr1-xGdxTiO3 (x = 0.06)"
    ) == "Sr1-xGdxTiO3 x=0.06"
    assert normalize_sample_formula_formatting(
        "Sr1-xGdxTiO3 x = 0.06"
    ) == "Sr1-xGdxTiO3 x=0.06"
    assert normalize_sample_formula_formatting("Fe(OH)3") == "Fe(OH)3"
    assert normalize_sample_formula_formatting("Co") != (
        normalize_sample_formula_formatting("CO")
    )


def test_sample_formula_formatting_is_idempotent():
    value = "Sr1-xGdxTiO3 (x = 0 . 06)"
    once = normalize_sample_formula_formatting(value)
    assert normalize_sample_formula_formatting(once) == once


def test_active_plan_wires_typography_normalization_to_alignment_and_scoring():
    plan = load_plan("configs/evaluation_metrics.yaml")
    authors = plan["bibliographic.authors"][1]
    samples = plan["samples"][1]

    assert authors.key_fn(
        {"full_name": "Andr´e Heinemann"},
        {"full_name": "André Heinemann"},
    ) == 1.0
    assert authors.attr_scorers["full_name"] == (
        "normalized_exact", normalize_author_name_typography,
    )
    assert samples.attr_scorers["sample_name"] == (
        "rouge_l", normalize_sample_formula_formatting,
    )
    assert samples.attr_scorers["chemical_composition"] == (
        "normalized_exact", normalize_sample_formula_formatting,
    )


def test_author_repair_applies_symmetrically_to_linkage_and_score():
    config = load_plan(
        "configs/evaluation_metrics.yaml"
    )["bibliographic.authors"][1]
    malformed = [{"id": "pred_author", "full_name": "Andr´e Heinemann"}]
    canonical = [{"id": "gold_author", "full_name": "André Heinemann"}]

    affiliations = {"bibliographic.affiliations": AlignmentResult([])}
    forward_alignment = align_similarity(
        malformed, canonical,
        key_fn=config.key_fn, tau=config.tau,
    )
    reverse_alignment = align_similarity(
        canonical, malformed,
        key_fn=config.key_fn, tau=config.tau,
    )
    forward = _score_entities(
        malformed, canonical, config, forward_alignment, affiliations,
    )
    reverse = _score_entities(
        canonical, malformed, config, reverse_alignment, affiliations,
    )

    assert forward["tp"] == reverse["tp"] == 1
    assert forward["attrs"]["full_name"]["mean_score"] == 1.0
    assert reverse["attrs"]["full_name"]["mean_score"] == 1.0


def test_spaced_decimal_sample_aligns_and_scores_as_the_same_formula():
    config = load_plan("configs/evaluation_metrics.yaml")["samples"][1]
    prediction = [{
        "id": "pred_sample",
        "sample_name": "Ni0 . 90V0 . 10",
        "chemical_composition": "Ni0 . 90V0 . 10",
    }]
    gold = [{
        "id": "sample_1",
        "sample_name": "Ni0.90V0.10",
        "chemical_composition": "Ni0.90V0.10",
    }]
    forward_alignment = align_samples(
        prediction, gold, document_id="2001.02706", identity_gold=gold,
    )
    reverse_alignment = align_samples(
        gold, prediction, document_id="2001.02706", identity_gold=gold,
    )
    forward = _score_entities(
        prediction, gold, config, forward_alignment, {},
    )
    reverse = _score_entities(
        gold, prediction, config, reverse_alignment, {},
    )

    assert forward["tp"] == reverse["tp"] == 1
    assert forward["attrs"]["sample_name"]["mean_score"] == 1.0
    assert forward["attrs"]["chemical_composition"]["mean_score"] == 1.0
    assert reverse["attrs"]["chemical_composition"]["mean_score"] == 1.0


def test_parenthesized_sample_variable_aligns_and_composition_scores_exactly():
    config = load_plan("configs/evaluation_metrics.yaml")["samples"][1]
    prediction = [{
        "id": "pred_sample",
        "sample_name": "SGTO x = 0.06",
        "chemical_composition": "Sr1-xGdxTiO3 (x = 0.06)",
    }]
    gold = [{
        "id": "sample_2",
        "sample_name": "SGTO6",
        "chemical_composition": "Sr1−xGdxTiO3 x=0.06",
    }]
    aligned = align_samples(
        prediction, gold, document_id="2010.00094",
    )
    result = _score_entities(prediction, gold, config, aligned, {})

    assert result["tp"] == 1
    assert result["attrs"]["chemical_composition"]["mean_score"] == 1.0
    # Formatting normalization must not erase the extra name qualifier.
    assert result["attrs"]["sample_name"]["mean_score"] < 1.0
