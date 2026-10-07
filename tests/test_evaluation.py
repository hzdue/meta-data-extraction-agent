"""Tests for src/meta_data_extraction/evaluation.py.

Covers the format bridge and the evaluator/provenance block in the report.
The evaluation conventions are summarized in README.md#scoring-conventions.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from meta_data_extraction.evaluation import (
    DEFAULT_PLAN_PATH,
    _unwrap_extraction_result,
    evaluate,
)


# -----------------------------------------------------------------------------
# _unwrap_extraction_result — unit-level
# -----------------------------------------------------------------------------


def test_unwrap_returns_output_when_signature_matches():
    pred = {
        "document_id": "1812.03956",
        "domain_name": "photon_and_neutron",
        "output": {"bibliographic": {"title": "some title"}},
        "provenance": {},
        "warnings": [],
    }
    assert _unwrap_extraction_result(pred) == {"bibliographic": {"title": "some title"}}


def test_unwrap_passes_annotation_shape_through():
    pred = {
        "_annotation": {"annotator": "annotator_1"},
        "bibliographic": {"title": "some title"},
    }
    assert _unwrap_extraction_result(pred) is pred


def test_unwrap_requires_both_signature_keys():
    """`document_id` alone or `output` alone must not trigger the unwrap."""
    only_output = {"output": {"bibliographic": {"title": "x"}}}
    only_doc_id = {"document_id": "x", "bibliographic": {"title": "x"}}
    assert _unwrap_extraction_result(only_output) is only_output
    assert _unwrap_extraction_result(only_doc_id) is only_doc_id


# -----------------------------------------------------------------------------
# evaluate() end-to-end with ExtractionResult-shaped pred
# -----------------------------------------------------------------------------


def _minimal_schema_content() -> dict:
    """A minimal v1.1.0 dict that populates enough fields to exercise a
    handful of dispatch entries. Used as both gold content and (wrapped)
    pred content, so a working bridge yields a perfect score."""
    return {
        "bibliographic": {
            "title": "A shared title",
            "subtitle": None,
            "affiliations": [],
            "authors": [],
            "abstract": "A shared abstract for both sides.",
            "keywords": [],
            "doi": None,
            "arxiv_id": "1234.56789",
            "url": None,
            "publication_date": "2024-01-15",
            "venue": None,
            "publisher": None,
            "language": "en",
            "funding": [],
        },
        "document_overview": {
            "section_headings": [],
            "has_references": True,
            "has_figures": True,
            "has_tables": False,
        },
        "facilities": [],
        "samples": [],
        "datasets": [],
    }


def test_evaluate_scores_llm_result_shape_against_gold(tmp_path):
    """An ExtractionResult-shaped pred must score against its `output`
    payload, not treat every field as a one-null mismatch. Without the
    bridge, every score below would be 0.0 (gold populated, pred `None`)."""
    content = _minimal_schema_content()
    gold = {
        "_annotation": {"arxiv_id": "test", "annotator": "gold"},
        **content,
    }
    pred = {
        "document_id": "test",
        "domain_name": "photon_and_neutron",
        "output": content,
        "provenance": {"extractor": "llm", "model": "test-model"},
        "warnings": [],
    }

    gold_path = tmp_path / "gold.json"
    pred_path = tmp_path / "pred.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    pred_path.write_text(json.dumps(pred), encoding="utf-8")

    report = evaluate(gold_path, pred_path)

    # Scalar scoring exercises the normalized_exact / exact / rouge_l paths.
    assert report["fields"]["bibliographic.title"]["score"] == 1.0
    assert report["fields"]["bibliographic.abstract"]["score"] == 1.0
    assert report["fields"]["bibliographic.arxiv_id"]["score"] == 1.0
    assert report["fields"]["bibliographic.publication_date"]["score"] == 1.0
    assert report["fields"]["bibliographic.language"]["score"] == 1.0
    assert report["fields"]["document_overview.has_references"]["score"] == 1.0
    assert report["fields"]["document_overview.has_figures"]["score"] == 1.0
    assert report["fields"]["document_overview.has_tables"]["score"] == 1.0

    # Both-empty entity lists exercise the null-handling rule.
    assert report["fields"]["facilities"]["f1"] == 1.0
    assert report["fields"]["samples"]["f1"] == 1.0
    assert report["fields"]["datasets"]["f1"] == 1.0


# -----------------------------------------------------------------------------
# Report metadata
# -----------------------------------------------------------------------------


def test_report_has_evaluator_block():
    """The report header carries the plan and alignment-helper versions,
    and both declared and actual schema versions."""
    report = evaluate(
        "data/gold/1804.07163.json",
        "data/experiment/pan_gpt4o_full_0shot/1804.07163.json",
    )
    ev = report["evaluator"]
    assert ev["plan_version"] == str(yaml.safe_load(DEFAULT_PLAN_PATH.read_text(encoding="utf-8"))["version"])
    assert ev["alignment_helper_version"] == str(yaml.safe_load(Path("configs/alignment_helper.yaml").read_text(encoding="utf-8"))["version"])
    funding_source = report["identity_sources"]["bibliographic.funding"]
    assert Path(funding_source["path"]) == Path("data/gold/1804.07163.json")
    assert set(funding_source) == {"path"}
    assert report["identity_sources"]["samples"] == funding_source
    assert report["identity_sources"]["datasets"] == funding_source
    assert ev["schema_versions_declared"] == {
        "generic.json": "1.1.0",
        "photon_and_neutron.json": "1.1.0",
    }
    assert ev["schema_versions_actual"] == {
        "generic.json": "1.1.0",
        "photon_and_neutron.json": "1.1.0",
    }
    assert ev["schema_version_mismatch"] is False
    assert "extraction_provenance" not in report


def test_schema_version_mismatch_flag(tmp_path):
    """A declared version that differs from what's on disk flips the
    schema_version_mismatch flag to True."""
    # Build a minimal plan file that declares an intentionally wrong version.
    fake_plan = {
        "version": "0.1",
        "schema_versions": {"generic.json": "9.9.9"},  # wrong on purpose
        "scoring_plan": {
            "bibliographic.title": {"metric": "normalized_exact"},
        },
    }
    plan_path = tmp_path / "mismatch.yaml"
    plan_path.write_text(yaml.safe_dump(fake_plan), encoding="utf-8")

    gold = {"bibliographic": {"title": "X"}}
    pred = {"bibliographic": {"title": "X"}}
    gold_path = tmp_path / "gold.json"
    pred_path = tmp_path / "pred.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    pred_path.write_text(json.dumps(pred), encoding="utf-8")

    report = evaluate(gold_path, pred_path, plan_path=plan_path)
    assert report["evaluator"]["schema_version_mismatch"] is True
    assert report["evaluator"]["schema_versions_declared"] == {"generic.json": "9.9.9"}
    assert report["evaluator"]["schema_versions_actual"] == {"generic.json": "1.1.0"}


# -----------------------------------------------------------------------------
# Null-support reporting (gold_null flag + n_non_null aggregation)
# -----------------------------------------------------------------------------


def test_scalar_metrics_flag_gold_null():
    from meta_data_extraction.evaluation import _exact, _normalized_exact, _rouge_l

    for metric in (_exact, _normalized_exact, _rouge_l):
        both = metric(None, None)
        assert both["score"] == 1.0 and both["gold_null"] is True
        gold_null = metric("hallucinated", None)
        assert gold_null["score"] == 0.0 and gold_null["gold_null"] is True
        pred_null = metric(None, "value")
        assert pred_null["score"] == 0.0 and "gold_null" not in pred_null
        real = metric("value", "value")
        assert real["score"] == 1.0 and "gold_null" not in real


def test_aggregate_reports_scalar_n_non_null():
    from meta_data_extraction.evaluation import aggregate_reports

    reports = [
        {"fields": {"bibliographic.subtitle": {"score": 1.0, "gold_null": True}}},
        {"fields": {"bibliographic.subtitle": {"score": 1.0, "gold_null": True}}},
        {"fields": {"bibliographic.subtitle": {"score": 0.0}}},
        {"fields": {"bibliographic.subtitle": {"score": 1.0}}},
    ]
    agg = aggregate_reports(reports)["bibliographic.subtitle"]
    # mean over all papers unchanged; support counts only gold-non-null.
    assert agg == {"mean_score": 0.75, "n_papers": 4, "n_non_null": 2}


def test_aggregate_pairs_reports_non_null_support():
    from meta_data_extraction.evaluation import _aggregate_attribute_scores

    pairs = [
        {"score": 1.0, "gold_null": True},
        {"score": 1.0},
        {"score": 0.0},
    ]
    out = _aggregate_attribute_scores(pairs)
    assert out == {"mean_score": 2.0 / 3.0, "n_pairs": 3, "n_pairs_non_null": 2}


# -----------------------------------------------------------------------------
# Kendall-order aggregation for ordered_set_f1 fields
# -----------------------------------------------------------------------------


def test_aggregate_reports_rolls_up_order():
    from meta_data_extraction.evaluation import aggregate_reports

    reports = [
        {
            "fields": {
                "document_overview.section_headings": {
                    "tp": 4,
                    "fp": 0,
                    "fn": 0,
                    "f1": 1.0,
                    "order": 1.0,
                }
            }
        },
        {
            "fields": {
                "document_overview.section_headings": {
                    "tp": 3,
                    "fp": 1,
                    "fn": 1,
                    "f1": 0.75,
                    "order": 0.5,
                }
            }
        },
        {
            "fields": {
                "document_overview.section_headings": {
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                    "f1": 1.0,
                    "order": None,
                }
            }
        },
    ]
    agg = aggregate_reports(reports)["document_overview.section_headings"]
    assert agg["mean_order"] == 0.75  # mean over defined orders only
    assert agg["n_ordered"] == 2
    assert agg["tp"] == 8  # content counts untouched


def test_aggregate_reports_no_order_keys_for_plain_set_fields():
    from meta_data_extraction.evaluation import aggregate_reports

    reports = [
        {"fields": {"bibliographic.keywords": {"tp": 2, "fp": 0, "fn": 1, "f1": 0.8}}},
    ]
    agg = aggregate_reports(reports)["bibliographic.keywords"]
    assert "mean_order" not in agg and "n_ordered" not in agg


# -----------------------------------------------------------------------------
# Similarity assignment and facility containment
# -----------------------------------------------------------------------------


def test_token_containment_short_inside_long():
    from meta_data_extraction.alignment import token_containment, token_overlap_f1

    short, long = "ISIS", "WISH beamline, ISIS, United Kingdom"
    assert token_containment(short, long) == 1.0
    assert token_overlap_f1(short, long) < 0.5  # why F1 alone fails here


def test_unsupported_similarity_options_are_rejected():
    import pytest
    from meta_data_extraction.evaluation import _spec_from_yaml

    for option in (
        {"key_mode": "containment_max"},
        {"assignment": "hungarian"},
        {"key_attr_normalizers": {"name": "samples.sample_name"}},
    ):
        with pytest.raises(ValueError):
            _spec_from_yaml({
                "metric": "entity", "key_attrs": ["name"], "attrs": {}, **option
            })


def test_similarity_assignment_maximizes_cardinality_before_similarity():
    from meta_data_extraction.alignment import align_similarity

    pred = [{"id": "p1"}, {"id": "p2"}]
    gold = [{"id": "g1"}, {"id": "g2"}]
    # The strongest single edge would block the only two-pair solution.
    similarities = {
        ("p1", "g1"): 1.0,
        ("p1", "g2"): 0.6,
        ("p2", "g1"): 0.6,
        ("p2", "g2"): 0.0,
    }
    result = align_similarity(
        pred, gold, key_fn=lambda p, g: similarities[p["id"], g["id"]]
    )
    assert len(result.pairs) == 2
    assert result.id_map == {"p1": "g2", "p2": "g1"}


# -----------------------------------------------------------------------------
# Facility alias normalization
# -----------------------------------------------------------------------------


def test_facility_alias_expands_acronyms():
    from meta_data_extraction.alignment import normalize_facility

    assert (
        normalize_facility("LLB (Saclay, France)")
        == "laboratoire léon brillouin (saclay, france)"
    )
    assert (
        normalize_facility("measured at the ILL")
        == "measured at the institut laue-langevin"
    )
    # non-acronym text passes through (casefolded), no partial-word hits
    assert normalize_facility("Illinois facility") == "illinois facility"
    assert normalize_facility(None) is None


def test_sample_registry_alignment_drives_sample_references(tmp_path):
    gold_path = Path("data/gold/2309.13729.json")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    pred = {
        "samples": [
            {
                **gold["samples"][0],
                "id": "pred_sample_1",
            },
            {
                **gold["samples"][1],
                "id": "pred_sample_2",
                "sample_name": "nanocomposite prepared at 350 °C",
                "chemical_composition": "CoFe2O4/Co-Fe",
            },
            {
                **gold["samples"][2],
                "id": "pred_sample_3",
                "sample_name": "nanocomposite prepared at 400 °C",
                "chemical_composition": "CoFe2O4/Co-Fe",
            },
            {
                **gold["samples"][3],
                "id": "pred_sample_4",
                "sample_name": "nanocomposite prepared at 450 °C",
                "chemical_composition": "CoFe2O4/Co-Fe",
            },
        ],
        "datasets": [
            {
                **gold["datasets"][1],
                "id": "pred_dataset_2",
                "sample_refs": [
                    "pred_sample_1",
                    "pred_sample_2",
                    "pred_sample_3",
                    "pred_sample_4",
                ],
            }
        ],
    }
    pred_path = tmp_path / "prediction.json"
    pred_path.write_text(json.dumps(pred), encoding="utf-8")
    report = evaluate(gold_path, pred_path)
    assert report["fields"]["samples"]["tp"] == 4
    assert {item["stage"] for item in report["fields"]["samples"]["match_stages"]} == {
        "exact_name",
        "shared_composition_discriminator",
    }
    assert report["fields"]["datasets.attrs.sample_refs"] == {
        "tp": 4,
        "fp": 0,
        "fn": 0,
        "f1": 1.0,
        "n_pairs": 1,
    }


def test_active_plan_loads():
    from meta_data_extraction.evaluation import load_plan

    plan = load_plan("configs/evaluation_metrics.yaml")
    matchers = {
        "facilities": "facility_staged",
        "samples": "sample_registry",
        "datasets": "dataset_registry",
        "bibliographic.funding": "funding_registry",
    }
    for path, matcher in matchers.items():
        assert plan[path][0] == "entity"
        assert plan[path][1].matcher == matcher
        assert plan[path][1].key_fn is None
    assert plan["facilities"][1].tau == 1.0
    for path in ("bibliographic.authors", "bibliographic.affiliations"):
        assert plan[path][1].matcher == "similarity"
        assert callable(plan[path][1].key_fn)


def test_all_entity_matchers_record_correspondences_and_stages():
    report = evaluate(
        "data/gold/1804.07163.json",
        "data/experiment/pan_gpt4o_full_0shot/1804.07163.json",
        "configs/evaluation_metrics.yaml",
    )
    for field in (
        "bibliographic.authors", "bibliographic.affiliations", "bibliographic.funding",
        "facilities", "samples", "datasets",
    ):
        result = report["fields"][field]
        assert len(result["match_stages"]) == result["tp"]
        assert all(set(pair) == {"pred", "gold", "stage"} for pair in result["match_stages"])
        if field in ("bibliographic.authors", "bibliographic.affiliations"):
            assert {pair["stage"] for pair in result["match_stages"]} == {"similarity"}


def test_exact_unique_dataset_name_separates_identity_from_technique_scoring():
    report = evaluate(
        "data/gold/2010.12832.json",
        "data/experiment/pan_qwen3_5_4b_full_1shot/2010.12832.json",
        "configs/evaluation_metrics.yaml",
    )
    datasets = report["fields"]["datasets"]
    assert datasets["tp"] == 2
    assert datasets["fp"] == 0
    assert datasets["fn"] == 0
    assert {
        item["stage"]
        for item in datasets["match_stages"]
        if item["pred"] == "dataset_2"
    } == {"exact_unique_name"}

    technique = report["fields"]["datasets.attrs.measurement_technique"]
    assert technique["n_pairs"] == 2
    assert 0.0 < technique["mean_score"] < 1.0


def test_aggregate_reports_sums_alignment_abstentions():
    from meta_data_extraction.evaluation import aggregate_reports

    reports = [
        {
            "fields": {
                "datasets": {
                    "tp": 1,
                    "fp": 2,
                    "fn": 0,
                    "alignment_abstentions": {"pred_dataset_technique_collision": 3},
                }
            }
        },
        {
            "fields": {
                "datasets": {
                    "tp": 2,
                    "fp": 1,
                    "fn": 1,
                    "alignment_abstentions": {
                        "pred_dataset_technique_collision": 2,
                        "gold_dataset_technique_collision": 4,
                    },
                }
            }
        },
    ]
    assert aggregate_reports(reports)["datasets"]["alignment_abstentions"] == {
        "gold_dataset_technique_collision": 4,
        "pred_dataset_technique_collision": 5,
    }
