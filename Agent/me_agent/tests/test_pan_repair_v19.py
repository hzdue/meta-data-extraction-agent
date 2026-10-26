"""Fictional candidates test executable contracts, not development-paper accuracy."""
from copy import deepcopy
import unittest

import extract
from test_pan_fact_v15 import CandidateModel
from test_pan_fact_repair import record, frozen, SOURCES, EV
from test_pan_repair import SCHEMA, admission
from test_pan_repair_v18 import run


class BatchModel(CandidateModel):
    def __init__(self, missing_proposal=False, uncertain=False):
        super().__init__()
        self.missing_proposal, self.uncertain = missing_proposal, uncertain
        self.batches = []

    def __call__(self, prompt, payload, kind, limit):
        mode = payload['mode']
        if mode == 'batch_completion':
            self.batches.append(deepcopy(payload))
            rows = []
            for row in payload['context']['items']:
                values = {'evidence': EV, 'admission': admission(EV),
                    'pair_checks': [{'pair_id': p['pair_id'], 'scope': 'explicit_pair', 'evidence': EV}
                                    for p in row.get('review_context', {}).get('pairs', [])]}
                if self.uncertain:
                    values['admission']['identity'] = 'uncertain'
                rows.append({'candidate_id': row['candidate_id'],
                             'fields': {k: values[k] for k in row['missing_fields']}})
            return {'completions': rows}
        response = super().__call__(prompt, payload, kind, limit)
        if mode == 'missing_samples':
            second = deepcopy(response['proposals'][0])
            second['entity']['sample_name'] = 'Elm'
            response['proposals'].append(second)
            if self.missing_proposal:
                for p in response['proposals']:
                    p.pop('evidence')
        if mode == 'review' and payload['context']['proposal']['op'] == 'add_sample':
            response = {'reason': 'Needs independent admission and pair checks'}
        return response


class RepairV19Tests(unittest.TestCase):
    def test_two_candidates_share_one_completion_and_both_replay(self):
        model = BatchModel()
        result, trace = run(model)
        self.assertEqual(len(result['samples']), 4)
        self.assertEqual(len(model.batches), 1)
        self.assertEqual(len(model.batches[0]['context']['items']), 2)
        self.assertEqual(trace['summary']['fact_review_completion_attempts'], 1)
        additions = [t for t in trace['transactions'] if t.get('proposal', {}).get('op') == 'add_sample']
        self.assertEqual(len(additions), 2)
        self.assertNotEqual(additions[1]['before'], additions[1]['review_record'])
        self.assertNotIn('decision', additions[0]['review'])
        self.assertNotIn('entity_decision', additions[0]['review'])

    def test_proposal_evidence_requires_semantic_completion_not_copying_links(self):
        model = BatchModel(missing_proposal=True)
        result, trace = run(model)
        self.assertEqual(len(result['samples']), 4)
        self.assertEqual(len(model.batches), 2)  # One discovery batch, one review batch, not per candidate.
        self.assertEqual(model.batches[0]['context']['items'][0]['missing_fields'], ['evidence'])
        broken = deepcopy(trace)
        broken['model_responses'] = [r for r in broken['model_responses']
            if r.get('fact_request') != model.batches[0]]
        with self.assertRaises(ValueError):
            extract.replay_pan_evidence_repair(record=record(), trace=broken, detector=frozen(record()), schema=SCHEMA)

    def test_uncertain_completion_does_not_gain_approval(self):
        result, _ = run(BatchModel(uncertain=True))
        self.assertEqual(len(result['samples']), 2)

    def test_batch_cannot_change_existing_fields_or_candidate_ids(self):
        request = {'context': {'items': [{'candidate_id': '0', 'missing_fields': ['admission']}]}}
        for rows in [[{'candidate_id': 'foreign', 'fields': {'admission': {}}}],
                     [{'candidate_id': '0', 'fields': {'admission': {}, 'decision': 'approve'}}],
                     [{'candidate_id': '0', 'fields': {'admission': {}}}]*2]:
            with self.assertRaises(ValueError):
                extract._pan_fact_batch_fields(request, {'completions': rows}, '0')

    def test_citation_adapter_only_accepts_unique_known_id(self):
        raw = {'evidence': ['e1: The quoted passage'], 'reason': 'e1: leave prose alone'}
        normalized = extract._pan_fact_normalize_citations(raw, SOURCES)
        self.assertEqual(normalized['evidence'], ['e1'])
        self.assertEqual(normalized['reason'], raw['reason'])
        self.assertEqual(raw['evidence'], ['e1: The quoted passage'])
        for bad in ['e999: absent', 'e1: or e2', 'e1/e2', 'page 1']:
            self.assertEqual(extract._pan_fact_normalize_citations({'evidence': [bad]}, SOURCES)['evidence'], [bad])

    def test_annotated_citation_is_bound_to_original_response(self):
        class Annotated(CandidateModel):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review':
                    response['admission']['identity_evidence'] = ['e1: explicit preparation']
                    response.pop('decision', None)
                    response.pop('entity_decision', None)
                return response
        result, trace = run(Annotated())
        self.assertEqual(len(result['samples']), 3)
        raw = next(r for r in trace['model_responses'] if r['call_kind'] == 'pan_repair_fact_review')
        self.assertEqual(raw['raw_response']['admission']['identity_evidence'], ['e1: explicit preparation'])

    def test_snapshot_rebase_rejects_removal_or_changed_identity(self):
        _, trace = run(BatchModel())
        tx = next(t for t in trace['transactions'] if t.get('proposal', {}).get('op') == 'add_sample')
        for change in ['remove', 'rename']:
            current = deepcopy(tx['before'])
            if change == 'remove':
                current['samples'].pop()
            else:
                current['samples'][0]['sample_name'] = 'Other specimen'
            with self.assertRaises(ValueError):
                extract._pan_fact_review_record(current, tx)

    def test_unknown_pair_does_not_require_redundant_decision(self):
        class Unknown(CandidateModel):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload['mode'] == 'review' and payload['context']['proposal']['op'] == 'add_sample':
                    response.pop('decision', None)
                    response.pop('entity_decision', None)
                    if not any(e['page'] == 2 for e in payload['excerpts']):
                        response['pair_checks'][0].update(scope='unknown', evidence=[])
                return response
        def retrieve(task, plan, *args):
            if 'Maple diffraction' in plan['queries']:
                return {'sources': {'page:2': {'page_number': 2, 'text': 'Maple is measured by diffraction.'}}}
            return {'sources': SOURCES}
        result, trace = run(Unknown(), retrieve)
        self.assertEqual(len(result['samples']), 3)
        self.assertEqual(trace['summary']['fact_evidence_expansion_attempts'], 1)


if __name__ == '__main__':
    unittest.main()
