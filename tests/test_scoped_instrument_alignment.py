"""Source-linked instrument forms must preserve paper and record boundaries."""
import copy
import json
from pathlib import Path

from meta_data_extraction.alignment import _match_datasets, _prepare_dataset_identities

ROOT = Path(__file__).resolve().parents[1]


def gold(paper):
    return json.loads((ROOT / 'data/gold' / f'{paper}.json').read_text(encoding='utf8'))['datasets']


def test_instrument_link_is_paper_scoped_and_does_not_rewrite_values():
    records = gold('2405.14324')
    predictions = [{'dataset_name': 'vibrating sample magnetometer',
                    'measurement_technique': 'vibrating sample magnetometer'}]
    original = copy.deepcopy(predictions)
    entries = _prepare_dataset_identities(records, '2405.14324')
    assert _match_datasets(predictions, entries) == {0: (3, 'unique_technique')}
    assert predictions == original
    unrelated = _prepare_dataset_identities(records, 'no-source-link')
    assert _match_datasets(predictions, unrelated) == {}


def test_instrument_does_not_choose_between_duplicate_predictions():
    entries = _prepare_dataset_identities(gold('2202.11658'), '2202.11658')
    predictions = [{'dataset_name': 'vibrating sample magnetometer',
                    'measurement_technique': 'vibrating sample magnetometer'}] * 2
    assert _match_datasets(predictions, entries) == {}


def test_combined_sem_eds_form_does_not_match_split_gold():
    predictions = [{'dataset_name': 'scanning electron microscope and energy dispersive spectroscopy',
                    'measurement_technique': 'scanning electron microscope and energy dispersive spectroscopy'}]
    combined = _prepare_dataset_identities(gold('2404.06853'), '2404.06853')
    assert {p:g for p,(g,_) in _match_datasets(predictions, combined).items()} == {0: 2}
    split = _prepare_dataset_identities(gold('2212.04169'), '2212.04169')
    assert _match_datasets(predictions, split) == {}


def test_current_voltage_name_does_not_broaden_transport_technique():
    entries = _prepare_dataset_identities(gold('2603.01364'), '2603.01364')
    named = [{'dataset_name': 'Current-voltage (I-V) measurements',
              'measurement_technique': 'electrical transport'}]
    assert {p:g for p,(g,_) in _match_datasets(named, entries).items()} == {0: 4}
    generic = [{'dataset_name': 'electrical transport data',
                'measurement_technique': 'electrical transport'}]
    assert _match_datasets(generic, entries) == {}
