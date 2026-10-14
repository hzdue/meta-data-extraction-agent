"""An invented missing entity must count even if every emitted entity is correct."""
import json

import pytest

from test_scoring import frozen, pair, ROOT
from meta_data_extraction.offline.scoring import evaluate_pair, summaries


@pytest.mark.parametrize("collection,item", [
    ("samples", {"id": "sample_2", "sample_name": "omega specimen", "chemical_composition": "Au"}),
    ("datasets", {"id": "dataset_2", "dataset_name": "omega spectroscopy",
                  "measurement_technique": "XAS", "sample_refs": ["sample_1"], "facility_refs": []}),
])
def test_omitted_gold_entity_reduces_recall_without_corrupting_emitted_fields(pair, frozen, collection, item):
    gold, prediction = pair
    baseline = evaluate_pair(ROOT, gold, prediction, frozen)
    reference = json.loads(gold.read_bytes())
    reference[collection].append(item)
    gold.write_text(json.dumps(reference), encoding="utf-8")
    before = prediction.read_bytes()
    result = evaluate_pair(ROOT, gold, prediction, frozen)
    entities = result["fields"][collection]
    assert baseline["fields"][collection]["fn"] == 0
    assert entities["n_gold"] == 2
    assert entities["tp"] == 1 and entities["fn"] == 1 and entities["fp"] == 0
    rows = {row.field: row for row in summaries([result], frozen)}
    assert rows[collection].category == "entity_recovery"
    assert rows[collection].support_count == 2
    assert rows[collection].metrics["f1"] == pytest.approx(2 / 3)
    assert prediction.read_bytes() == before
