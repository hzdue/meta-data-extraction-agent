"""Repair protocol/graph regressions. No inference, evaluator or annotation access."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import unittest

import extract
from test_pan_repair import SCHEMA, RECORD, FakeModel, detector, edit
from test_pan_inventory import InventoryModel, SOURCES, EVIDENCE, EMPTY, observation, engine


class InventoryProtocolTests(unittest.TestCase):
    def test_fact_schema_removes_graph_fields_and_instructions_without_changing_schema(self):
        schema = deepcopy(SCHEMA)
        schema['datasets']['instructions'] = ['Report actual measurements.', 'Set sample_refs to sample IDs.']
        original = deepcopy(schema)
        facts = extract._pan_repair_inventory_schema(schema)
        self.assertEqual(schema, original)
        for group in extract.PAN_REPAIR_GROUPS:
            self.assertFalse({'id', 'sample_refs', 'facility_refs'} & facts[group]['fields'].keys())
        self.assertEqual(facts['datasets']['instructions'], ['Report actual measurements.'])

    def test_generated_ids_and_edges_are_discarded_not_imported(self):
        row = observation()
        row['sample']['id'] = 'sample_9'
        row['dataset'].update(id='dataset_3', sample_refs=['sample_9', 'invented'], facility_refs=['unknown'])
        raw = {'observations': [row], 'gaps': []}
        before = deepcopy(raw)
        normalized, notes = extract._pan_repair_inventory_normalize(raw, SCHEMA)
        rows, errors = extract._pan_repair_inventory_rows(raw, SOURCES, SCHEMA)
        self.assertFalse(errors)
        self.assertEqual(len(rows), 1)
        self.assertTrue(notes)
        self.assertEqual(raw, before)
        self.assertNotIn('id', normalized['observations'][0]['sample'])
        self.assertNotIn('sample_refs', normalized['observations'][0]['dataset'])
        model = InventoryModel([row])
        trace = engine(model)
        output, _ = extract.apply_pan_reviewed_transactions(EMPTY, trace, SCHEMA)
        self.assertEqual(len(output['samples']), 1)
        self.assertEqual(output['datasets'][0]['sample_refs'], [output['samples'][0]['id']])
        self.assertNotIn('sample_9', output['datasets'][0]['sample_refs'])
        self.assertGreater(trace['summary']['inventory_normalization_count'], 0)

    def test_shared_nested_citation_and_explicit_scope_are_losslessly_relocated(self):
        row = observation()
        evidence = row.pop('evidence')
        role, scope, campaign = row.pop('object_role'), row.pop('relation_scope'), row.pop('campaign')
        row['sample'].update(evidence=evidence, object_role=role, relation_scope=scope, campaign=campaign)
        row['dataset']['evidence'] = deepcopy(evidence)
        rows, errors = extract._pan_repair_inventory_rows({'observations': [row]}, SOURCES, SCHEMA)
        self.assertFalse(errors)
        self.assertEqual(rows[0]['object_role'], role)
        self.assertTrue(rows[0]['evidence'])
        # Different endpoint excerpts do not become invented relation evidence.
        row['dataset']['evidence'] = [{'source_id': 'page:1', 'quote': 'not the shared source citation'}]
        rows, errors = extract._pan_repair_inventory_rows({'observations': [row]}, SOURCES, SCHEMA)
        self.assertFalse(rows)
        self.assertEqual(errors[0]['reason'], 'ungrounded_inventory_evidence')

    def test_conflicting_roles_and_missing_measurement_evidence_are_not_inferred(self):
        row = observation()
        row['sample']['object_role'] = 'measured_mixture'
        rows, errors = extract._pan_repair_inventory_rows({'observations': [row]}, SOURCES, SCHEMA)
        self.assertFalse(rows)
        self.assertEqual(errors[0]['reason'], 'conflicting_inventory_scope')
        row = observation()
        row.pop('evidence')
        rows, errors = extract._pan_repair_inventory_rows({'observations': [row]}, SOURCES, SCHEMA)
        self.assertFalse(rows)
        self.assertEqual(errors[0]['reason'], 'ungrounded_inventory_evidence')

    def test_correction_is_bounded_to_failed_indexes_and_bound_in_replay(self):
        broken = observation()
        broken.pop('object_role')
        base = InventoryModel([broken])
        def model(prompt, payload, kind, limit):
            if kind == 'pan_repair_inventory_protocol_correction':
                self.assertEqual(payload['failed_indexes'], [0])
                self.assertNotIn('metadata', payload)
                return {'observation_updates': [{'index': 0, 'observation': observation()}]}
            return base(prompt, payload, kind, limit)
        trace = engine(model)
        self.assertEqual(trace['summary']['inventory_represented_count'], 1)
        result, replay, applied = extract.replay_pan_evidence_repair(record=EMPTY, trace=trace,
            detector={'target_judgments': []}, schema=SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 3)
        altered = deepcopy(trace)
        altered['measurement_inventory']['extractions'][0]['protocol_correction']['patch']['observation_updates'][0]['observation']['sample']['sample_name'] = 'another specimen'
        with self.assertRaises(ValueError):
            extract.replay_pan_evidence_repair(record=EMPTY, trace=altered, detector={'target_judgments': []}, schema=SCHEMA)
        raw = {'observations': [observation(), broken]}
        with self.assertRaises(extract.PanRepairResponseError):
            extract._pan_repair_inventory_patch(raw, {'observation_updates': [{'index': 0, 'observation': observation('rewritten')}]}, SOURCES, SCHEMA)

    def test_zero_admitted_facts_is_explicit_degradation_not_success(self):
        row = observation()
        row['relation_scope'] = 'paper_level_cooccurrence'
        base = InventoryModel([row])
        def model(prompt, payload, kind, limit):
            if kind == 'pan_repair_inventory_protocol_correction':
                return {'observation_updates': []}
            return base(prompt, payload, kind, limit)
        trace = engine(model)
        self.assertTrue(trace['summary']['inventory_degraded'])
        self.assertEqual(trace['summary']['inventory_raw_row_count'], 1)
        self.assertEqual(trace['summary']['inventory_admitted_row_count'], 0)
        self.assertEqual(trace['summary']['inventory_binding_request_count'], 0)
        self.assertFalse(trace['transactions'])

    def test_unknown_site_does_not_block_proven_specimen_measurement_chain(self):
        base = InventoryModel([observation(facility=True)], {'sample': 'new', 'dataset': 'new', 'facility': None})
        trace = engine(base)
        output, applied = extract.apply_pan_reviewed_transactions(EMPTY, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 3)
        self.assertEqual(len(output['samples']), 1)
        self.assertFalse(output['facilities'])
        self.assertEqual(output['datasets'][0]['sample_refs'], [output['samples'][0]['id']])
        self.assertEqual(trace['summary']['inventory_represented_count'], 0)
        self.assertEqual(trace['summary']['inventory_pending_count'], 1)
        replayed, _, _ = extract.replay_pan_evidence_repair(record=EMPTY, trace=trace,
            detector={'target_judgments': []}, schema=SCHEMA)
        self.assertEqual(replayed, output)

    def test_grounded_prepared_variant_is_qualified_only_when_creating_new_specimen(self):
        row = observation('specimen Amber')
        row['specimen_variant'] = {'kind': 'prepared_or_recovered', 'label': 'cured', 'evidence': EVIDENCE}
        record = deepcopy(RECORD)
        frozen = detector('samples:sample_1', 'supported')
        model = InventoryModel([row], {'sample': 'new', 'dataset': 'datasets:dataset_1', 'facility': None})
        trace = engine(model, record, frozen)
        result, applied = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 2)
        self.assertEqual(result['samples'][:2], record['samples'])
        self.assertEqual(result['samples'][-1]['sample_name'], 'specimen Amber (cured)')
        extract._pan_repair_assert_protected(result, extract._pan_repair_protected_targets(record, frozen))
        replayed, _, _ = extract.replay_pan_evidence_repair(record=record, trace=trace, detector=frozen, schema=SCHEMA)
        self.assertEqual(replayed, result)

    def test_scan_parameter_or_unprinted_variant_is_not_a_prepared_specimen(self):
        for kind, label in [('measurement_parameter', 'cured'), ('prepared_or_recovered', 'invented treatment')]:
            row = observation()
            row['specimen_variant'] = {'kind': kind, 'label': label, 'evidence': EVIDENCE}
            rows, errors = extract._pan_repair_inventory_rows({'observations': [row]}, SOURCES, SCHEMA)
            self.assertFalse(rows)
            self.assertTrue(errors)

    def test_source_context_bridges_methods_and_results_without_seed_input(self):
        row = observation()
        parsed, errors = extract._pan_repair_inventory_rows({'observations': [row]}, SOURCES, SCHEMA)
        self.assertFalse(errors)
        inventory = {'observations': {'fact': {'row': parsed[0]}}}
        fresh = {'page:2': {'text': 'The diffraction results for cured specimen Amber are listed in Table 1.'}}
        packed = extract._pan_repair_inventory_context(fresh, inventory, {**SOURCES, **fresh})
        self.assertIn('page:1', packed)
        self.assertIn('page:2', packed)
        self.assertEqual(packed['page:1']['text'], SOURCES['page:1']['text'])

    def test_batch_binding_reuses_acquisition_and_preserves_per_object_reviews(self):
        rows = [observation('explicit cured specimen ' + str(i)) for i in range(8)]
        for i, row in enumerate(rows):
            row['dataset']['dataset_description'] = 'Description paraphrase ' + str(i)
        schema = deepcopy(SCHEMA)
        schema['datasets']['item_schema']['dataset_description'] = {'type': 'string|null'}
        base = InventoryModel(rows)
        batch_sources = {'page:1': {'text': 'Diffraction measured ' + ', '.join(row['sample']['sample_name'] for row in rows) + ' in the same acquisition.'}}
        trace = extract.run_pan_evidence_repair(record=EMPTY, anchors={}, detector={'target_judgments': []},
            schema=schema, call_model=base, retrieve=lambda *args: {'sources': batch_sources})
        result, _ = extract.apply_pan_reviewed_transactions(EMPTY, trace, schema)
        self.assertEqual(len(result['samples']), 8)
        self.assertEqual(len(result['datasets']), 1)
        self.assertEqual(trace['summary']['inventory_binding_request_count'], 2)
        self.assertEqual(trace['summary']['inventory_binding_batch_request_count'], 2)
        self.assertTrue(all(a['execution']['decision_response_received'] for a in trace['actions'] if a.get('inventory_row')))
        replayed, _, checked = extract.replay_pan_evidence_repair(record=EMPTY, trace=trace,
            detector={'target_judgments': []}, schema=schema)
        self.assertEqual(replayed, result)
        self.assertEqual(checked['summary']['applied_patch_count'], 17)

    def test_gaps_have_priority_over_recertifying_an_existing_seed(self):
        rows = [observation('specimen A'), observation('new recovered specimen')]
        base = InventoryModel(rows, {'sample': 'new', 'dataset': 'datasets:dataset_1', 'facility': None})
        with patch.object(extract, 'PAN_REPAIR_MAX_PACKETS', 1):
            trace = engine(base, RECORD)
        first = next(a for a in trace['actions'] if a.get('inventory_row'))
        self.assertEqual(first['inventory_row']['sample']['sample_name'], 'new recovered specimen')
        result, _ = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertIn('new recovered specimen', [sample['sample_name'] for sample in result['samples']])


class DeletionAndCapabilityTests(unittest.TestCase):
    def test_hierarchy_only_edge_removal_is_rejected_by_promoted_policy(self):
        record = deepcopy(RECORD)
        record['facilities'] = [{'id': 'facility_1', 'name': 'Meridian instrument'}, {'id': 'facility_2', 'name': 'Umbrella laboratory'}]
        record['datasets'][0]['facility_refs'] = ['facility_1', 'facility_2']
        target = 'datasets:dataset_1:facility_refs:facility_2'
        det = {'target_judgments': [{'target_id': 'datasets:dataset_1', 'judgment': 'supported'},
            {'target_id': target, 'judgment': 'insufficient_evidence'}]}
        def model(prompt, payload, kind, limit):
            if kind == 'pan_repair_decision':
                self.assertIn('remove', payload['capabilities']['targets'][target]['allowed_actions'])
                return {'assessments': [{'target_id': target, 'verdict': 'unsupported', 'evidence_basis': 'facility_hierarchy',
                    'requested_action': 'remove', 'evidence': EVIDENCE}], 'edits': []}
            return FakeModel([])(prompt, payload, kind, limit)
        trace = extract.run_pan_evidence_repair(record=record, anchors={}, detector=det, schema=SCHEMA,
            enable_coverage=False, call_model=model, retrieve=lambda *args: {'sources': SOURCES})
        result, applied = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 0)
        self.assertEqual(result['datasets'][0]['facility_refs'], ['facility_1', 'facility_2'])
        extract._pan_repair_assert_protected(result, extract._pan_repair_protected_targets(record, det))

    def test_permission_is_for_the_edge_not_its_parent_dataset(self):
        record = deepcopy(RECORD)
        det = detector('datasets:dataset_1', 'supported')
        target = 'datasets:dataset_1:sample_refs:sample_1'
        task = {'task_id': 'datasets:dataset_1', 'dataset_id': 'dataset_1', 'entity_group': None,
                'risks': [{'target_id': target}]}
        protected = extract._pan_repair_protected_targets(record, det)
        allowed = extract._pan_repair_capabilities(task, record, protected)['targets'][target]
        self.assertTrue(allowed['parent_dataset_fields_protected'])
        self.assertIn('remove', allowed['allowed_actions'])
        decision = {'assessments': [{'target_id': target, 'verdict': 'unsupported', 'evidence_basis': 'facility_hierarchy',
            'evidence': EVIDENCE, 'requested_action': 'remove'}], 'edits': []}
        compiled = extract._pan_repair_complete_actions(decision, task, record, SOURCES, protected)
        self.assertEqual(compiled['edits'], [{'target_id': target, 'action': 'remove'}])
        # The unsupported label alone cannot request a deletion.
        decision['assessments'][0].pop('requested_action')
        self.assertFalse(extract._pan_repair_complete_actions(decision, task, record, SOURCES, protected)['edits'])
        decision['assessments'][0]['requested_action'] = 'remove'
        protected[target] = True
        self.assertFalse(extract._pan_repair_complete_actions(decision, task, record, SOURCES, protected)['edits'])

    def scenario(self, contrary=False, review_flags=True):
        target = 'datasets:dataset_1:sample_refs:sample_2'
        record = deepcopy(RECORD)
        record['samples'][1]['sample_name'] = 'untreated Cobalt'
        record['datasets'][0]['sample_refs'].append('sample_2')
        first = 'Diffraction data were collected from specimen Amber. Microscopy was also performed on untreated Cobalt.'
        second = ('Table 2 reports diffraction data for untreated Cobalt, confirming its inclusion in this diffraction acquisition.' if contrary else
                  'Only specimen Amber was used for diffraction; untreated Cobalt was measured exclusively by microscopy in a separate experiment.')
        seen = []
        def retrieve(task, plan, queries, pages):
            seen.append(deepcopy(plan));page = min(len(seen), 2)
            return {'queries': plan.get('queries', []), 'pages': [page], 'sources': {f'page:{page}': {'text': first if page == 1 else second}}}
        def model(prompt, payload, kind, limit):
            if kind == 'pan_repair_planner':
                return {'plans': []}
            if kind == 'pan_repair_protocol_correction':
                return {}
            if kind == 'pan_repair_decision':
                later = payload['paper_round'] > 1
                evidence = [{'source_id': 'page:2' if later else 'page:1', 'span_ids': ['s1']}]
                if later and contrary:
                    return FakeModel([], verdict='supported', basis='explicit_measurement')(prompt, payload, kind, limit) | {
                        'assessments': [{'target_id': target, 'verdict': 'supported', 'evidence_basis': 'explicit_measurement', 'evidence': evidence}]}
                response = FakeModel([edit('remove_relation', target, 'different_campaign', evidence=evidence)])(prompt, payload, kind, limit)
                if later:
                    response['assessments'][0]['exclusion'] = {'kind': 'explicit_exclusion', 'source_statement': second, 'evidence': evidence}
                return response
            response = FakeModel([])(prompt, payload, kind, limit)
            if kind == 'pan_repair_challenge':
                for item in response['reviews']:
                    item.update(exclusion_confirmed=review_flags, counterevidence_considered=review_flags, contradiction_found=False)
            return response
        trace = extract.run_pan_evidence_repair(record=record, anchors={}, detector=detector(target, 'insufficient_evidence'),
            schema=SCHEMA, enable_coverage=False, call_model=model, retrieve=retrieve)
        return record, trace, seen, target

    def test_exclusion_requires_actual_countersearch_positive_witness_and_review(self):
        record, trace, plans, target = self.scenario()
        self.assertIn(target, trace['actions'][0]['deferred_deletions'])
        self.assertEqual(plans[1]['counterevidence_for'], [target])
        self.assertEqual(trace['transactions'][0]['paper_round'], 2)
        self.assertEqual(trace['summary']['audited_deletion_count'], 1)
        result, checked = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
        self.assertEqual(result['datasets'][0]['sample_refs'], ['sample_1'])
        replayed, _, _ = extract.replay_pan_evidence_repair(record=record, trace=trace,
            detector=detector(target, 'insufficient_evidence'), schema=SCHEMA)
        self.assertEqual(replayed, result)

    def test_later_table_support_prevents_absence_based_deletion(self):
        record, trace, plans, target = self.scenario(contrary=True)
        result, _ = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
        self.assertEqual(result, record)
        self.assertFalse(trace['transactions'])
        self.assertTrue(plans[1]['counterevidence_for'])

    def test_after_preference_without_exclusion_review_flags_is_not_approval(self):
        record, trace, plans, target = self.scenario(review_flags=False)
        result, _ = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
        self.assertEqual(result, record)
        self.assertFalse(trace['transactions'])

    def test_audit_tampering_or_removing_review_flag_cannot_authorize_cached_deletion(self):
        record, trace, plans, target = self.scenario()
        for mode in ('search', 'witness', 'approval'):
            changed = deepcopy(trace)
            transaction = changed['transactions'][0]
            if mode == 'search':
                transaction['edits'][0]['deletion_check']['counterevidence_search']['target_ids'] = []
            elif mode == 'witness':
                transaction['edits'][0]['assessment']['exclusion']['source_statement'] = 'invented exclusion'
            else:
                transaction['edits'][0]['review']['exclusion_confirmed'] = False
            output, applied = extract.apply_pan_reviewed_transactions(record, changed, SCHEMA)
            self.assertEqual(output, record)
            self.assertTrue(applied['skipped'])
            replayed, _, _ = extract.replay_pan_evidence_repair(record=record, trace=changed,
                detector=detector(target, 'insufficient_evidence'), schema=SCHEMA)
            self.assertEqual(replayed, record)

    def test_countersearch_ranks_target_results_tables_not_only_method_mentions(self):
        pages = {1: 'Methods\nDiffraction was performed on Amber. Microscopy used Cobalt.',
                 2: 'Table 2. Diffraction results for Cobalt and Amber.',
                 3: 'Unrelated introductory background.'}
        class Executor:
            def call(self, name, paper_id, args):
                return {'hits': [], 'ranked_pages': [{'page_number': 3}]}
        task = {'task_id': 'datasets:dataset_1', 'risks': [], 'entity_group': None}
        plan = {'counterevidence_for': ['datasets:dataset_1:sample_refs:sample_2'], 'ranking_queries': ['diffraction', 'Cobalt'], 'force_evidence_expansion': True}
        with patch.object(extract, 'record_tool_call'), patch.object(extract, 'record_tool_error'):
            result = extract._retrieve_pan_repair_task(SimpleNamespace(page_texts=pages, paper_id='synthetic'), Path('unused'),
                Executor(), {}, {}, task, plan, set(), set())
        self.assertEqual(result['pages'][0], 2)


class DownloadedProtocolRegressionTests(unittest.TestCase):
    def test_downloaded_inventory_responses_are_adapted_without_gold_or_approval_invention(self):
        root = Path(__file__).resolve().parents[1]
        files = list((root / 'outputs/run').glob('*/pan_agent_risk_refinement.json'))
        if not files:
            self.skipTest('Optional downloaded response fixtures are absent.')
        schema = json.loads((root / 'SOURCES/schemas/photon_and_neutron.json').read_text(encoding='utf-8'))
        raw_count = admitted = adapted = 0
        for path in files:
            trace = json.loads(path.read_text(encoding='utf-8'))
            if trace.get('agent_trace_version') != 'pan-agent-risk-refinement-v11':
                continue
            for extraction in trace.get('measurement_inventory', {}).get('extractions', []):
                response = extraction['response']; before = deepcopy(response)
                rows, errors = extract._pan_repair_inventory_rows(response, extraction['sources'], schema)
                raw_count += len(response.get('observations', [])); admitted += len(rows)
                adapted += len(extract._pan_repair_inventory_normalize(response, schema)[1])
                self.assertEqual(response, before)
                for row in rows:
                    self.assertTrue(extract._pan_repair_ground_evidence(row['evidence'], extraction['sources']))
                    for part in ('sample', 'dataset', 'facility'):
                        if row.get(part): self.assertFalse({'id', 'sample_refs', 'facility_refs'} & row[part].keys())
            # Existing approvals cannot bypass the newly required exclusion audit.
            for transaction in trace.get('transactions', []):
                for item in transaction['edits']:
                    if item['evidence_basis'] == 'different_campaign':
                        self.assertIsNone(extract._pan_repair_deletion_check(item, transaction['sources'], {
                            'target_ids': [item['target_id']], 'queries': ['counterevidence'], 'source_ids': list(transaction['sources'])}))
        if not raw_count:
            self.skipTest('Optional v11 inventory responses are absent.')
        self.assertGreater(admitted, 0)
        self.assertGreater(adapted, 0)
        self.assertLess(admitted, raw_count)  # Missing evidence/scope still fails closed.


if __name__ == '__main__':
    unittest.main()
