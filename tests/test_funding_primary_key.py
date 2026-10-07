import json
import itertools
from pathlib import Path
import pytest
import yaml

from meta_data_extraction.alignment import (
    _match_funding,
    _prepare_funding_identities,
    grant_anchors,
    grant_identity,
)


def _pairs(matches):
    return {(pred, gold, stage) for pred, (gold, stage) in matches.items()}


def _entries(gold):
    return _prepare_funding_identities(gold)


def _match(pred, gold):
    return _match_funding(pred, _entries(gold))


def test_missing_either_required_field_cannot_match():
    gold = [{"funder": "Department of Energy", "grant_id": "DE-SC-0019154"}]
    assert _match([{"funder": "Department of Energy", "grant_id": None}], gold) == {}
    assert _match([{"funder": None, "grant_id": "DE-SC-0019154"}], gold) == {}


def test_same_grant_aligns_despite_different_funder_wording():
    pred = [{"funder": "DOE", "grant_id": "DE-FG02-07ER46382"}]
    gold = [{"funder": "Department of Energy", "grant_id": "DE-FG02-07ER46382"}]
    assert _pairs(_match(pred, gold)) == {(0, 0, "grant")}


def test_same_funder_cannot_link_different_grants():
    pred = [{"funder": "Department of Energy", "grant_id": "DE-AC02-05CH11231"}]
    gold = [{"funder": "Department of Energy", "grant_id": "DE-SC-0019154"}]
    assert _match(pred, gold) == {}


def test_grant_identity_ignores_labels_and_separator_presentation():
    values = ["Grant No. DMR-2105081", "DMR2105081", "No. DMR 2105081"]
    assert {grant_identity(value) for value in values} == {"dmr2105081"}


def test_enriched_grant_phrases_share_identifier_anchors():
    cases = [
        ("834134", "834134 (WATUSO)"),
        ("DP160100545", "Discovery Grant DP160100545"),
        ("759696", "N 759696 (acronym TOP)"),
        ("DNRF-93", "DNRF-93 (Center for Materials Crystallography)"),
        ("283883", "NMI3-II Grant 283883"),
        ("15639149", "AFR Grant No. 15639149"),
    ]
    for left, right in cases:
        assert grant_anchors(left) & grant_anchors(right)


def test_enriched_grant_phrases_align_in_anchor_stage():
    pred = [{"funder": "ERC", "grant_id": "834134"}]
    gold = [{"funder": "European Research Council", "grant_id": "834134 (WATUSO)"}]
    assert _pairs(_match(pred, gold)) == {(0, 0, "grant_anchor")}


@pytest.mark.parametrize("merged", ["15563, 16554, 16566", "15563, 16554, and 16566"])
def test_merged_grants_abstain_even_after_individual_grants_are_consumed(merged):
    gold = [{"funder": "University of Tokyo", "grant_id": grant}
            for grant in ("15563", "16554", "16566")]
    pred = [gold[0].copy(), gold[1].copy(),
            {"funder": "University of Tokyo", "grant_id": merged}]
    for ordered_gold in itertools.permutations(gold):
        for ordered_pred in itertools.permutations(pred):
            diagnostics = {}
            matches = _match_funding(
                list(ordered_pred), _entries(list(ordered_gold)), diagnostics=diagnostics
            )
            assert {ordered_pred[i]["grant_id"] for i in matches} == {"15563", "16554"}
            assert all(ordered_pred[i]["grant_id"] == ordered_gold[j]["grant_id"]
                       for i, (j, _) in matches.items())
            assert diagnostics == {"ambiguous_grant_anchors": 1}


def test_anchor_fallback_preserves_one_grant_with_multiple_cofunders():
    gold = [{"funder": f, "grant_id": "834134 (WATUSO)"}
            for f in ("European Research Council", "Example Foundation")]
    pred = [{"funder": f, "grant_id": "834134"}
            for f in ("Example Foundation", "European Research Council")]
    matches = _match(pred, gold)
    assert _pairs(matches) == {(0, 1, "grant_anchor"), (1, 0, "grant_anchor")}


@pytest.mark.parametrize("run", ["pan_gpt4o_full_0shot", "pan_mistral_large3_no_inst_0shot"])
def test_actual_merged_tokyo_grants_stay_in_unmatched_population(run):
    from meta_data_extraction.evaluation import evaluate

    root = Path(__file__).resolve().parents[1]
    gold_path = root / "data/gold/2105.04167.json"
    pred_path = root / "data/experiment" / run / "2105.04167.json"
    gold = json.loads(gold_path.read_text(encoding="utf8"))["bibliographic"]["funding"]
    pred = json.loads(pred_path.read_text(encoding="utf8"))["output"]["bibliographic"]["funding"]
    merged_index = next(i for i, entity in enumerate(pred)
                        if all(g in entity["grant_id"] for g in ("15563", "16554", "16566")))
    matches = _match(pred, gold)
    assert merged_index not in matches
    result = evaluate(gold_path, pred_path, root / "configs/evaluation_metrics.yaml")["fields"]["bibliographic.funding"]
    # Three individual grants match; the merged record remains unmatched.
    # Keep this expectation independent of regeneratable evaluation reports.
    assert (result["tp"], result["fp"], result["fn"]) == (3, 1, 3)
    assert result["n_pred"] == len(pred) and result["n_gold"] == len(gold)
    assert result["alignment_abstentions"] == {"pred_ambiguous_grant_anchors": 1}


def test_different_doe_grants_share_no_identifier_anchor():
    assert not (grant_anchors("DE-AC02-05CH11231") & grant_anchors("DE-SC-0019154"))


@pytest.mark.parametrize("separator", ["-", "/", "."])
def test_compound_grants_cannot_match_on_a_shared_numeric_suffix(separator):
    pred = [{"funder": "Foundation", "grant_id": f"ABC{separator}12345"}]
    gold = [{"funder": "Foundation", "grant_id": f"XYZ{separator}12345"}]
    assert _match(pred, gold) == {}


def test_compound_grant_anchor_is_not_ambiguous_with_a_different_prefix():
    pred = [{"funder": "Foundation", "grant_id": "ABC-12345 (project)"}]
    gold = [{"funder": "Foundation", "grant_id": grant}
            for grant in ("ABC-12345", "XYZ-12345")]
    assert _pairs(_match(pred, gold)) == {(0, 0, "grant_anchor")}


def test_grant_anchor_preserves_a_separately_printed_project_number():
    pred = [{"funder": "DFG", "grant_id": "403191981"}]
    gold = [{
        "funder": "DFG",
        "grant_id": "SPP 2137 (Skyrmionics) under grant PF393/19 (project-id 403191981)",
    }]
    assert _pairs(_match(pred, gold)) == {(0, 0, "grant_anchor")}


def test_shared_grant_uses_funder_as_secondary_key():
    pred = [
        {
            "funder": "Fondo Europeo de Desarrollo Regional",
            "grant_id": "ProID2024010034",
        },
        {"funder": "Agencia Canaria", "grant_id": "ProID2024010034"},
    ]
    gold = [
        {"funder": "Agencia Canaria", "grant_id": "ProID2024010034"},
        {
            "funder": "Fondo Europeo de Desarrollo Regional",
            "grant_id": "ProID2024010034",
        },
    ]
    assert _pairs(_match(pred, gold)) == {
        (0, 1, "grant_then_funder"),
        (1, 0, "grant_then_funder"),
    }


def test_duplicate_prediction_does_not_hide_false_positive():
    pred = [
        {"funder": "DOE", "grant_id": "DE-SC-0019154"},
        {"funder": "Department of Energy", "grant_id": "DE-SC-0019154"},
    ]
    gold = [{"funder": "Department of Energy", "grant_id": "DE-SC-0019154"}]
    matches = _match(pred, gold)
    assert len(matches) == 1


def test_shared_grant_requires_positive_funder_evidence_and_cannot_fall_back():
    pred = [
        {
            "funder": "National Institute of Standards and Technology",
            "grant_id": "DMR-1508249",
        },
        {"funder": "National Science Foundation", "grant_id": "DMR-1508249"},
        {
            "funder": "Center for High Resolution Neutron Scattering",
            "grant_id": None,
        },
    ]
    gold = [
        {
            "funder": "Center for High Resolution Neutron Scattering",
            "grant_id": "DMR-1508249",
        }
    ]
    assert _match(pred, gold) == {}
    assert _match(list(reversed(pred)), gold) == {}


def test_shared_grant_selects_the_record_with_positive_funder_evidence():
    pred = [
        {"funder": "DOE", "grant_id": "DE-SC-0019154"},
        {"funder": "Unrelated wording", "grant_id": "DE-SC-0019154"},
    ]
    gold = [
        {"funder": "Department of Energy", "grant_id": "DE-SC-0019154"}
    ]
    for ordered in itertools.permutations(pred):
        matches = _match(list(ordered), gold)
        assert {
            (ordered[index]["funder"], target, stage)
            for index, (target, stage) in matches.items()
        } == {("DOE", 0, "grant_then_funder")}


def test_curated_funding_helper_contains_only_grounded_aliases():
    root = Path(__file__).resolve().parents[1]
    helper = (
        yaml.safe_load(
            (root / "configs" / "alignment_helper.yaml").read_text(encoding="utf-8")
        )
        or {}
    )
    funding = helper["entities"]["bibliographic.funding"]
    assert set(funding) == {"aliases"}
    assert len(funding["aliases"]) == 3
    doe = next(item for item in funding["aliases"] if item["alias"] == "DOE")
    assert doe == {"alias": "DOE", "canonical_name": "Department of Energy"}


def test_runtime_identity_builder_covers_40_papers_and_104_gold_entities():
    root = Path(__file__).resolve().parents[1]
    corpus = json.loads(
        (root / "data" / "corpus.json").read_text(encoding="utf-8")
    )
    paper_ids = {
        paper["arxiv_id"]
        for paper in corpus["papers"]
    }
    registry = {}
    for paper_id in paper_ids:
        gold = json.loads(
            (root / "data" / "gold" / f"{paper_id}.json").read_text(encoding="utf-8")
        )
        funding = gold.get("bibliographic", {}).get("funding") or []
        registry[paper_id] = _prepare_funding_identities(funding)

    assert len(registry) == 40
    assert sum(len(entries) for entries in registry.values()) == 104
    assert all(
        entry["anchors"] for entries in registry.values() for entry in entries
    )
