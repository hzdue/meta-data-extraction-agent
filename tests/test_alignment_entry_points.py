"""Shared contracts for the public entity-alignment entry points."""

import copy

import pytest

from meta_data_extraction import alignment


@pytest.mark.parametrize("align, record, options", [
    (alignment.align_funding,
     {"grant_id": "ABC-12345", "funder": "Example Foundation"}, {}),
    (alignment.align_samples,
     {"id": "sample_1", "sample_name": "alpha specimen", "chemical_composition": "TiN"},
     {"document_id": "synthetic-paper"}),
    (alignment.align_datasets,
     {"id": "dataset_1", "dataset_name": "alpha dataset", "measurement_technique": "XRD"},
     {"document_id": "synthetic-paper"}),
])
def test_entity_entry_points_use_reconciled_identities_without_rewriting_records(
    align, record, options,
):
    identity_gold = [record.copy()]
    gold, pred = [record.copy()], [record.copy()]
    if "id" in record:
        gold[0]["id"], pred[0]["id"] = "left_local_id", "right_local_id"
    before = copy.deepcopy((pred, gold, identity_gold))

    result = align(pred, gold, identity_gold=identity_gold, **options)
    assert [(pair.pred_index, pair.gold_index) for pair in result.pairs] == [(0, 0)]
    assert result.id_map == ({"right_local_id": "left_local_id"} if "id" in record else {})
    assert (pred, gold, identity_gold) == before

    # An explicitly empty reconciled source must not fall back to the scored gold.
    assert align(pred, gold, identity_gold=[], **options).pairs == []
    assert len(align(pred, gold, **options).pairs) == 1


@pytest.mark.parametrize("align, options", [
    (alignment.align_similarity, {"key_fn": alignment.make_key_scorer("name")}),
    (alignment.align_funding, {}),
    (alignment.align_samples, {"document_id": "synthetic-paper"}),
    (alignment.align_datasets, {"document_id": "synthetic-paper"}),
    (alignment.align_facilities, {}),
])
def test_entity_entry_points_handle_absent_collections(align, options):
    result = align(None, None, **options)
    assert result.pairs == []
    assert result.id_map == {}
    assert result.diagnostics == {"match_stages": []}
