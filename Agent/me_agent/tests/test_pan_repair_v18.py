"""Synthetic protocol/graph regressions; no development-paper IDs or annotations."""
from copy import deepcopy
import unittest

import extract
from test_pan_fact_repair import record, frozen, SOURCES, EV, merge
from test_pan_fact_v15 import CandidateModel
from test_pan_repair import SCHEMA


def run(model, retrieve=None):
    rec = record()
    trace = extract.run_pan_evidence_repair(record=rec, anchors={}, detector=frozen(rec), schema=SCHEMA,
        call_model=model, retrieve=retrieve or (lambda *a: {'sources': SOURCES}), max_rounds=2,
        enable_coverage=False, enable_measurement_reconciliation=True)
    result, applied = extract.apply_pan_reviewed_transactions(rec, trace, SCHEMA)
    assert not applied['skipped'], applied['skipped']
    replayed, _, checked = extract.replay_pan_evidence_repair(record=rec, trace=trace,
        detector=frozen(rec), schema=SCHEMA)
    assert not checked['skipped']
    assert result == replayed
    return result, trace


class Independent(CandidateModel):
    def __call__(self, prompt, payload, kind, limit):
        response = super().__call__(prompt, payload, kind, limit)
        if payload['mode'] == 'missing_samples':
            response['proposals'][0]['links'].append({'dataset_id': 'dataset_2', 'evidence': EV})
        if payload['mode'] == 'review' and payload['context']['proposal']['op'] == 'add_sample':
            response['decision'] = 'reject'  # Summary rejects the full bundle, not the admitted entity.
            response['entity_decision'] = 'approve'
            response['pair_checks'][1].update(scope='contradicted', evidence=[])
        return response


class RepairV18Tests(unittest.TestCase):
    def test_fixed_scope_retains_existing_merge_destination_and_datasets(self):
        rec = record()
        for d in rec['datasets']:
            d['facility_refs'].append('facility_1')
        proposal = merge()
        old = extract._pan_fact_review_request(rec, proposal, SOURCES, SCHEMA, 17)
        new = extract._pan_fact_review_request(rec, proposal, SOURCES, SCHEMA, 18)
        self.assertFalse(old['context']['after']['datasets'])
        self.assertEqual(len(new['context']['after']['datasets']), 2)
        self.assertEqual([f['id'] for f in new['context']['after']['facilities']], ['facility_1'])
        self.assertEqual(len(new['context']['before']['facilities']), 2)
        self.assertNotIn('review_protocol', old['context'])
        self.assertEqual(new, extract._pan_fact_review_request(rec, proposal, SOURCES, SCHEMA, 18))

    def test_partial_entity_chain_is_independent_and_replayable(self):
        result, trace = run(Independent())
        new = next(s['id'] for s in result['samples'] if s['sample_name'] == 'Maple')
        self.assertIn(new, result['datasets'][0]['sample_refs'])
        self.assertNotIn(new, result['datasets'][1]['sample_refs'])
        tx = next(t for t in trace['transactions'] if t.get('proposal', {}).get('op') == 'add_sample')
        self.assertEqual(tx['review']['decision'], 'reject')  # Never silently rewrite the model verdict.
        self.assertEqual(len(tx['proposal']['links']), 1)
        self.assertEqual(trace['summary']['fact_evidence_expansion_attempts'], 0)

    def test_v19_admission_replaces_redundant_v18_entity_verdict(self):
        class Bad(Independent):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review':
                    response.pop('entity_decision', None)
                return response
        result, _ = run(Bad())
        self.assertEqual(len(result['samples']), 3)
        self.assertFalse(extract._pan_fact_review_approved({'op': 'add_sample'}, {'admission': {}}, 18))

    def test_no_explicit_pair_means_no_orphan_entity(self):
        class NoPair(Independent):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review':
                    for check in response['pair_checks']:
                        check.update(scope='contradicted', evidence=[])
                return response
        result, _ = run(NoPair())
        self.assertEqual(result, record())

    def test_rejected_entity_does_not_get_salvaged_by_explicit_pairs(self):
        class Rejected(Independent):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review':
                    response['entity_decision'] = 'reject'
                return response
        result, _ = run(Rejected())
        self.assertEqual(len(result['samples']), 2)

    def test_pair_dictionary_lossless_and_bound_to_original_response(self):
        class Dictionary(Independent):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review':
                    response['pair_checks'] = {p['pair_id']: {k: v for k, v in p.items() if k != 'pair_id'}
                                               for p in response['pair_checks']}
                return response
        result, trace = run(Dictionary())
        self.assertEqual(len(result['samples']), 3)
        tx = next(t for t in trace['transactions'] if t.get('proposal', {}).get('op') == 'add_sample')
        self.assertIsInstance(tx['review']['pair_checks'], dict)
        bad = deepcopy(tx['review'])
        bad['pair_checks']['p999'] = {'scope': 'explicit_pair', 'evidence': EV}
        with self.assertRaisesRegex(ValueError, 'invalid_pair_dictionary'):
            extract._pan_fact_normalize_review(bad, tx['review_request'])
        bad = deepcopy(tx['review'])
        bad['pair_checks']['p1']['pair_id'] = 'p2'
        with self.assertRaises(ValueError):
            extract._pan_fact_normalize_review(bad, tx['review_request'])
        tx['review']['entity_decision'] = 'reject'
        with self.assertRaises(ValueError):
            extract._pan_fact_verify_requests(tx, trace['model_responses'], SCHEMA)

    def test_empty_measurement_evidence_is_not_filled(self):
        class Empty(Independent):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review':
                    response['admission']['measurement_evidence'] = []
                return response
        result, _ = run(Empty())
        self.assertEqual(len(result['samples']), 2)

    def test_state_or_known_object_is_not_admitted_as_new_sample(self):
        for relation, existing in [('measurement_state', 'sample_1'), ('uncertain', None),
                                   ('distinct_object', 'sample_1')]:
            class State(CandidateModel):
                def __call__(self, prompt, payload, kind, limit):
                    response = super().__call__(prompt, payload, kind, limit)
                    if payload['mode'] == 'candidate_identity':
                        response['object_identity'].update(relation=relation, existing_sample_id=existing)
                    return response
            with self.subTest(relation=relation, existing=existing):
                result, _ = run(State())
                self.assertEqual(len(result['samples']), 2)

    def test_unknown_pair_retrieves_once_even_with_rejected_summary(self):
        queried = []
        class Unknown(Independent):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review' and payload['context']['proposal']['op'] == 'add_sample':
                    expanded = any(e['page'] == 2 for e in payload['excerpts'])
                    response['pair_checks'][1].update(scope='explicit_pair' if expanded else 'unknown',
                                                      evidence=['e2'] if expanded else [])
                return response
        def retrieve(task, plan, *args):
            queried.extend(plan['queries'])
            if 'Maple spectroscopy' in plan['queries']:
                return {'sources': {'page:2': {'page_number': 2, 'text': 'Maple was measured by spectroscopy.'}}}
            return {'sources': SOURCES}
        result, trace = run(Unknown(), retrieve)
        new = next(s['id'] for s in result['samples'] if s['sample_name'] == 'Maple')
        self.assertIn(new, result['datasets'][1]['sample_refs'])
        self.assertEqual(queried.count('Maple spectroscopy'), 1)
        self.assertNotIn('Maple diffraction', queried)
        tx = next(t for t in trace['transactions'] if t.get('proposal', {}).get('op') == 'add_sample')
        self.assertEqual(tx['proposal']['links'][1]['evidence'], ['e2'])
        self.assertFalse(any(a.get('held_links') for a in trace['actions']))


if __name__ == '__main__':
    unittest.main()
