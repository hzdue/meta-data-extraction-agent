"""Metric-definition checks for the exploratory ablation aggregate."""
from __future__ import annotations

import math

from scripts.analyze_finetune_ablation import endpoint_metrics, summarize


def diagnostic(paper_id, *, schema_valid, errors, sample_f1, facility_f1):
    return {
        "paper_id": paper_id, "status": "scored", "json_valid": True,
        "schema_valid": schema_valid, "schema_errors": errors,
        "hit_output_limit": False, "generated_tokens": 10,
        "entity_f1": {"facilities": .5, "samples": .75, "datasets": 1.0},
        "conditional_reference_f1": {
            "sample_refs": {"f1": sample_f1, "n_pairs": 1 if sample_f1 is not None else 0},
            "facility_refs": {"f1": facility_f1, "n_pairs": 1 if facility_f1 is not None else 0},
        },
    }


def test_endpoint_metrics_separate_reference_structure_from_full_schema():
    result = endpoint_metrics({"loss": .2, "diagnostics": [
        diagnostic("one", schema_valid=False,
                   errors=["$out.bibliographic.funding[0].grant_id: null is not allowed"],
                   sample_f1=1.0, facility_f1=.5),
        diagnostic("two", schema_valid=False,
                   errors=["$out.datasets[0].sample_refs: missing field",
                           "$out.datasets[0].facility_refs: missing field"],
                   sample_f1=None, facility_f1=None),
    ]})
    assert result["schema_valid_papers"] == 0
    assert result["reference_complete_papers"] == 1
    assert result["mean_reference_f1"] == .375
    assert result["sample_refs_f1"] == .5
    assert result["facility_refs_f1"] == .25
    assert result["json_valid_papers"] == 2


def test_three_seed_summary_uses_sample_standard_deviation():
    result = summarize({42: 1.0, 123: 2.0, 456: 3.0})
    assert result["mean"] == 2.0
    assert result["sample_sd"] == 1.0
    assert result["min"] == 1.0 and result["max"] == 3.0
    assert result["by_seed"] == {"42": 1.0, "123": 2.0, "456": 3.0}
    assert math.isfinite(result["sample_sd"])
