"""Protocol and graph regressions using fictional objects, never development gold."""
from copy import deepcopy
import unittest

import extract
from test_pan_fact_repair import Model, record, frozen, SOURCES, EV
from test_pan_fact_v15 import CandidateModel
from test_pan_inventory import InventoryModel, EMPTY, engine
from test_pan_repair import SCHEMA, admission, FakeModel, RECORD, edit, run, detector, MEASURED_EVIDENCE, MEASURED_SOURCES


class RepairV16Tests(unittest.TestCase):
    def run_fact(self, model, retrieve=None):
        rec = record()
        trace = extract.run_pan_evidence_repair(record=rec, anchors={}, detector=frozen(rec), schema=SCHEMA,
            call_model=model, retrieve=retrieve or (lambda *a: {"sources": SOURCES}), max_rounds=2,
            enable_coverage=False, enable_measurement_reconciliation=True)
        result, applied = extract.apply_pan_reviewed_transactions(rec, trace, SCHEMA)
        self.assertFalse(applied['skipped'])
        replayed, _, _ = extract.replay_pan_evidence_repair(record=rec, trace=trace, detector=frozen(rec), schema=SCHEMA)
        self.assertEqual(result, replayed)
        return result, trace

    def test_missing_review_fields_completed_without_reversing_decision(self):
        class Missing(Model):
            def __call__(self, prompt, payload, kind, limit):
                if payload['mode'] == 'batch_completion':
                    return {'completions': [{'candidate_id': row['candidate_id'], 'fields': {
                        'pair_checks': [{'pair_id': p['pair_id'], 'scope': 'explicit_pair', 'evidence': EV}
                        for p in row['review_context']['pairs']]}} for row in payload['context']['items']]}
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review':
                    response.pop('pair_checks')
                return response
        result, trace = self.run_fact(Missing())
        self.assertIn('sample_2', result['datasets'][0]['sample_refs'])
        self.assertTrue(trace['transactions'][0].get('review_completion'))
        broken = deepcopy(trace)
        broken['model_responses'] = [r for r in broken['model_responses'] if r['call_kind'] != 'pan_repair_fact_batch_completion']
        with self.assertRaises(ValueError):
            extract.replay_pan_evidence_repair(record=record(), trace=broken, detector=frozen(record()), schema=SCHEMA)

    def test_completion_cannot_change_existing_judgment(self):
        with self.assertRaises(ValueError):
            extract._pan_fact_complete_review({'decision': 'reject'}, {'decision': 'approve', 'pair_checks': []}, ['pair_checks'])
        self.assertFalse(extract._pan_repair_completion_valid(
            [{'edit_id': 'x', 'preferred': 'uncertain'}],
            [{'edit_id': 'x', 'preferred': 'after', 'admission': admission(EV)}]))

    def test_unresolved_without_reason_code_triggers_one_new_evidence_review(self):
        class Need(Model):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review' and not any(e['page'] == 2 for e in payload['excerpts']):
                    response['decision'] = 'unresolved'
                    response.pop('reason_code', None)
                return response
        def retrieve(task, plan, *args):
            if any('Birch diffraction' in q for q in plan['queries']):
                return {'sources': {'page:2': {'page_number': 2, 'text': 'Independent diffraction profiles were collected for Birch in transmission geometry.'}}}
            return {'sources': SOURCES}
        result, trace = self.run_fact(Need(), retrieve)
        self.assertIn('sample_2', result['datasets'][0]['sample_refs'])
        self.assertEqual(trace['summary']['fact_evidence_expansion_attempts'], 1)

    def test_valid_new_specimen_survives_one_unknown_link(self):
        class Partial(CandidateModel):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'missing_samples':
                    response['proposals'][0]['links'].append({'dataset_id': 'dataset_2', 'evidence': EV})
                if payload['mode'] == 'review' and payload['context']['proposal']['op'] == 'add_sample':
                    response['pair_checks'][1]['scope'] = 'unknown'
                    response['pair_checks'][1]['evidence'] = []
                return response
        result, trace = self.run_fact(Partial())
        new = next(s['id'] for s in result['samples'] if s['sample_name'] == 'Maple')
        self.assertIn(new, result['datasets'][0]['sample_refs'])
        self.assertNotIn(new, result['datasets'][1]['sample_refs'])
        self.assertTrue(any(a.get('held_links') for a in trace['actions']))

    def test_hierarchy_cannot_replace_facility_with_instrument(self):
        class BadHierarchy(Model):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review':
                    response['facility_identity'] = {'relation': 'umbrella_of', 'old_kind': 'facility',
                                                    'keep_kind': 'instrument', 'evidence': EV}
                return response
        result, trace = self.run_fact(BadHierarchy(merge_site=True))
        self.assertEqual(result['facilities'], record()['facilities'])
        self.assertIn('repair_facility_identity_unproven', trace['summary']['fact_validation_failures'])

    def test_inventory_cannot_bypass_uncertain_identity_or_scope(self):
        for field, value in [('identity', 'state'), ('identity', 'existing'), ('scope', 'uncertain')]:
            class Unsafe(InventoryModel):
                def __call__(self, prompt, payload, kind, limit):
                    response = super().__call__(prompt, payload, kind, limit)
                    if kind.startswith('pan_repair_challenge'):
                        for r in response['reviews']:
                            r['admission'][field] = value
                    return response
            trace = engine(Unsafe())
            result, applied = extract.apply_pan_reviewed_transactions(EMPTY, trace, SCHEMA)
            self.assertEqual(result, EMPTY)
            self.assertFalse(applied['skipped'])

    def test_inventory_missing_admission_is_completed_and_replay_bound(self):
        class Missing(InventoryModel):
            def __call__(self, prompt, payload, kind, limit):
                if kind == 'pan_repair_challenge_admission_completion':
                    return {'reviews': [{**r, 'admission': admission(r['evidence'],
                        next(e['target_id'].split(':')[0] for e in payload['edits'] if e['edit_id'] == r['edit_id']))}
                        for r in payload['completion_of']]}
                response = super().__call__(prompt, payload, kind, limit)
                if kind.startswith('pan_repair_challenge'):
                    for r in response['reviews']:
                        r.pop('admission', None)
                return response
        trace = engine(Missing())
        result, applied = extract.apply_pan_reviewed_transactions(EMPTY, trace, SCHEMA)
        self.assertEqual(len(result['samples']), 1)
        replayed, _, _ = extract.replay_pan_evidence_repair(record=EMPTY, trace=trace, detector={'target_judgments': []}, schema=SCHEMA)
        self.assertEqual(result, replayed)

    def test_additive_subset_has_no_orphans_and_removals_stay_atomic(self):
        def e(i, op, target):
            return {'edit_id': i, 'operation': op, 'target_id': target}
        component = [e('s', 'add_entity', 'samples:new'), e('d', 'add_entity', 'datasets:new'),
                     e('good', 'add_relation', 'datasets:new:sample_refs:new'),
                     e('bad', 'add_relation', 'datasets:old:sample_refs:new')]
        self.assertEqual([x['edit_id'] for x in extract._pan_repair_safe_additive_subset(component, {'s', 'd', 'good'})], ['s', 'd', 'good'])
        self.assertEqual(extract._pan_repair_safe_additive_subset(component, {'s', 'd'}), [])
        component.append(e('remove', 'remove_entity', 'samples:old'))
        self.assertEqual(extract._pan_repair_safe_additive_subset(component, {'s', 'd', 'good', 'remove'}), [])

    def test_calibration_is_not_blanket_excluded_but_needs_scope_evidence(self):
        value = admission(EV)
        value['role'] = 'calibration'
        self.assertTrue(extract._pan_repair_admission_valid(value, 'samples', SOURCES, short=True))
        value['scope_evidence'] = []
        self.assertFalse(extract._pan_repair_admission_valid(value, 'samples', SOURCES, short=True))

    def test_normal_repair_partial_addition_replays_without_rejected_edge(self):
        proposals = [edit('add_entity', 'samples:sample_new_measured_1', 'explicit_measurement',
                         entity={'sample_name': 'specimen B', 'sample_state': 'solid'}, evidence=MEASURED_EVIDENCE),
                     edit('add_relation', 'datasets:dataset_1:sample_refs:sample_new_measured_1', 'explicit_measurement', evidence=MEASURED_EVIDENCE),
                     edit('add_relation', 'datasets:dataset_2:sample_refs:sample_new_measured_1', 'explicit_measurement', evidence=MEASURED_EVIDENCE)]
        class Partial(FakeModel):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if kind == 'pan_repair_challenge':
                    for r in response['reviews']:
                        if next(e for e in payload['edits'] if e['edit_id'] == r['edit_id'])['target_id'].startswith('datasets:dataset_2:'):
                            r['decision'] = 'reject'
                return response
        trace = run(Partial(proposals, verdict='supported'), sources=MEASURED_SOURCES)
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 2)
        self.assertNotIn('sample_new_measured_1', result['datasets'][1]['sample_refs'])
        replayed, _, _ = extract.replay_pan_evidence_repair(record=RECORD, trace=trace, detector=detector(), schema=SCHEMA)
        self.assertEqual(result, replayed)


if __name__ == '__main__':
    unittest.main()
