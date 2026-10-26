"""Unified identity, reserved completion budget, duplicate safety and model routing."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import extract
from test_pan_fact_repair import Model, record, frozen, SOURCES, EV
from test_pan_fact_v15 import CandidateModel
from test_pan_repair import SCHEMA, admission


class RepairV17Tests(unittest.TestCase):
    def run_fact(self, model, rec=None):
        rec = deepcopy(rec or record())
        trace = extract.run_pan_evidence_repair(record=rec, anchors={}, detector=frozen(rec), schema=SCHEMA,
            call_model=model, retrieve=lambda *a: {'sources': SOURCES}, max_rounds=2,
            enable_coverage=False, enable_measurement_reconciliation=True)
        result, applied = extract.apply_pan_reviewed_transactions(rec, trace, SCHEMA)
        self.assertFalse(applied['skipped'])
        replayed, _, _ = extract.replay_pan_evidence_repair(record=rec, trace=trace, detector=frozen(rec), schema=SCHEMA)
        self.assertEqual(replayed, result)
        return result, trace

    def test_admission_alone_is_complete_no_old_identity_required(self):
        class Unified(CandidateModel):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review':
                    for key in ('identity', 'preparation_evidence', 'measurement_evidence', 'existing_sample_id'):
                        response.pop(key, None)
                return response
        result, trace = self.run_fact(Unified())
        self.assertEqual(len(result['samples']), 3)
        self.assertEqual(trace['summary']['fact_review_completion_attempts'], 0)
        self.assertNotIn('top-level identity or preparation_evidence', extract.PAN_FACT_REVIEW_V15)
        self.assertNotIn('For add_sample also return', extract.PAN_FACT_REVIEW_V17)

    def test_old_protocol_completion_still_requires_old_fields(self):
        proposal = {'op': 'add_sample'}
        review = {'admission': admission(EV), 'pair_checks': []}
        self.assertEqual(extract._pan_fact_review_missing(proposal, review, 17), [])
        self.assertIn('identity', extract._pan_fact_review_missing(proposal, review, 16))

    def test_small_remaining_budget_defers_discovery_not_half_finished_candidates(self):
        model = Model()
        with patch.object(extract, 'PAN_REPAIR_FACT_CALLS', 4):
            result, trace = self.run_fact(model)
        self.assertEqual(result, record())
        self.assertFalse(model.calls)
        self.assertGreater(trace['summary']['measurement_deferred_count'], 0)

    def test_exact_duplicates_need_independent_context_approval_and_remap(self):
        rec = record()
        rec['samples'][1] = {**rec['samples'][0], 'id': 'sample_2'}
        class Duplicate(Model):
            def __init__(self, confirmed=True):
                super().__init__()
                self.confirmed = confirmed
            def __call__(self, prompt, payload, kind, limit):
                if payload['mode'] == 'duplicate_samples':
                    old, keep = payload['context']['pair']
                    return {'proposals': [{'op': 'merge_sample', 'old_id': old, 'keep_id': keep,
                        'basis': 'duplicate_identity', 'evidence': EV,
                        'links': [{'dataset_id': 'dataset_2', 'evidence': EV}]}]}
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review':
                    response['duplicate_identity'] = {'same_object': self.confirmed, 'contexts_checked': True, 'evidence': EV}
                return response
        result, trace = self.run_fact(Duplicate(), rec)
        self.assertEqual(len(result['samples']), 1)
        self.assertNotIn('sample_2', result['datasets'][1]['sample_refs'])
        self.assertIn('sample_1', result['datasets'][1]['sample_refs'])
        result, _ = self.run_fact(Duplicate(False), rec)
        self.assertEqual(len(result['samples']), 2)
        proposal = {'op': 'merge_sample', 'old_id': 'sample_2', 'keep_id': 'sample_1', 'basis': 'duplicate_identity',
                    'evidence': EV, 'links': [{'dataset_id': 'dataset_2', 'evidence': EV}]}
        rec['samples'][1]['sample_description'] = 'different treatment'
        with self.assertRaisesRegex(ValueError, 'not_exact_duplicates'):
            extract._pan_fact_compile(rec, proposal, SOURCES, SCHEMA)

    def test_repair_model_route_does_not_mutate_upstream_client(self):
        client = Mock()
        routed = Mock()
        routed.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id='repair-35b')])
        client.with_options.return_value = routed
        with patch.dict(extract.os.environ, {'REPAIR_MODEL': 'repair-35b', 'REPAIR_BASE_URL': 'http://localhost:8001/v1'}):
            actual, model = extract._pan_repair_model_client(client, 'seed-9b')
        self.assertIs(actual, routed)
        self.assertEqual(model, 'repair-35b')
        client.with_options.assert_called_once_with(max_retries=0, base_url='http://localhost:8001/v1')
        client.models.list.assert_not_called()

    def test_wrong_served_model_refused_without_fallback(self):
        client = Mock()
        client.with_options.return_value.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id='seed-9b')])
        with patch.dict(extract.os.environ, {'REPAIR_MODEL': 'repair-35b', 'REPAIR_BASE_URL': ''}):
            with self.assertRaisesRegex(ValueError, 'refusing model fallback'):
                extract._pan_repair_model_client(client, 'seed-9b')

    def test_unchanged_model_route_has_no_network_preflight(self):
        client = Mock()
        with patch.dict(extract.os.environ, {'REPAIR_MODEL': '', 'REPAIR_BASE_URL': ''}):
            _, model = extract._pan_repair_model_client(client, 'seed-9b')
        self.assertEqual(model, 'seed-9b')
        client.with_options.return_value.models.list.assert_not_called()


if __name__ == '__main__':
    unittest.main()
