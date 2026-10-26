"""Synthetic repair tests; no gold files, PDFs, model servers or evaluator calls."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack, nullcontext, redirect_stdout
from io import StringIO
from types import SimpleNamespace

import extract


SCHEMA = {
    "samples": {"item_schema": {"id": {"type": "string"}, "sample_name": {"type": "string"},
                                "sample_state": {"type": "string|null"}}},
    "facilities": {"item_schema": {"id": {"type": "string"}, "name": {"type": "string"}}},
    "datasets": {"item_schema": {"id": {"type": "string"}, "dataset_name": {"type": "string"},
                                 "sample_refs": {"type": "list[string]"}, "facility_refs": {"type": "list[string]"}}},
}
RECORD = {
    "facilities": [],
    "samples": [{"id": "sample_1", "sample_name": "specimen A", "sample_state": "solid"},
                {"id": "sample_2", "sample_name": "model object", "sample_state": None}],
    "datasets": [{"id": "dataset_1", "dataset_name": "diffraction", "sample_refs": ["sample_1"], "facility_refs": []},
                 {"id": "dataset_2", "dataset_name": "numerical simulation", "sample_refs": ["sample_2"], "facility_refs": []}],
}
TEXT = "The model object is an artificial input used only in numerical simulations, not an experimental specimen."
SOURCES = {"page:1": {"page_number": 1, "text": TEXT}}
EVIDENCE = [{"source_id": "page:1", "quote": TEXT}]
MEASUREMENT = "Diffraction data were collected for specimen A and the newly measured specimen B in the same campaign."
MEASURED_SOURCES = {"page:1": {"page_number": 1, "text": MEASUREMENT}}
MEASURED_EVIDENCE = [{"source_id": "page:1", "quote": MEASUREMENT}]


def measurement_context(evidence):
    return {"object_role": "measured_specimen", "identity_match": "exact", "campaign_match": "same",
            "identity_evidence": deepcopy(evidence), "measurement_evidence": deepcopy(evidence),
            "link_evidence": deepcopy(evidence)}


def detector(target="samples:sample_2", status="unsupported"):
    return {"detector_report_version": "pan-anchor-llm-detector-v10", "target_judgments": [
        {"target_id": target, "target_type": "sample", "judgment": status},
    ]}


def edit(operation="remove_entity", target="samples:sample_2", basis="simulation_only", **kwargs):
    return {"operation": operation, "target_id": target,
            "verdict": "unsupported" if operation.startswith("remove") else "supported",
            "evidence_basis": basis, "evidence": deepcopy(EVIDENCE), "reason": "source explicitly establishes role",
            "measurement_campaign": "diffraction campaign", **kwargs}


def admission(evidence, group="samples"):
    return {"identity": "distinct", "scope": "in_scope",
            "role": {"samples": "specimen", "datasets": "acquisition", "facilities": "facility"}[group],
            "identity_evidence": deepcopy(evidence), "measurement_evidence": deepcopy(evidence),
            "scope_evidence": deepcopy(evidence),
            "facility_classification": {"kind": "facility", "measurement_location": True, "evidence": deepcopy(evidence)},
            "reported_data": {"kind": "explicit_report", "description": "The supplied fictional source reports this object's measurements",
                              "evidence": deepcopy(evidence)}}


class FakeModel:
    def __init__(self, edits, verdict="unsupported", basis="simulation_only", approve=True, sources=None):
        self.edits = edits
        self.verdict = verdict
        self.basis = basis
        self.approve = approve
        self.calls = []

    def __call__(self, prompt, payload, kind, limit):
        self.calls.append(kind)
        if kind == "pan_repair_planner":
            return {"plans": [{"task_id": t["task_id"], "queries": ["model object"]} for t in payload["tasks"]]}
        if kind == "pan_repair_coverage":
            return {"assessments": [], "edits": []}
        if kind.startswith("pan_repair_inventory_extract"):
            return {"observations": [], "gaps": []}
        if kind in {"pan_repair_decision", "pan_repair_protocol_correction"}:
            assessments = {
                r["target_id"]:
                {"target_id": r["target_id"], "verdict": self.verdict, "evidence_basis": self.basis,
                 "evidence": deepcopy(self.edits[0].get("evidence", EVIDENCE) if self.edits else EVIDENCE),
                 "reason": "structured source assessment"} for r in payload["task"]["risks"]}
            if self.edits:
                for assessment in assessments.values():
                    assessment.update(verdict="unresolved", evidence_basis="unresolved", evidence=[])
            for e in self.edits:
                assessments[e["target_id"]] = {
                    "target_id": e["target_id"], "verdict": self.verdict,
                    "evidence_basis": e.get("evidence_basis", self.basis),
                    "evidence": deepcopy(e.get("evidence", EVIDENCE)), "reason": "structured source assessment"}
            for assessment in assessments.values():
                if assessment["verdict"] == "supported":
                    assessment["context"] = measurement_context(assessment["evidence"])
            return {"edits": deepcopy(self.edits), "assessments": list(assessments.values())}
        if kind == "pan_repair_challenge":
            return {"reviews": [{"edit_id": e["edit_id"], "decision": "approve" if self.approve else "reject",
                                 "target_verdict": e["verdict"], "evidence_basis": e["evidence_basis"],
                                 "checks": {"identity": True, "campaign": True, "schema": True, "exclusion": True},
                                 "admission": admission(e["evidence"], e["target_id"].split(":")[0]),
                                 "facility_identity": {"relation": "umbrella_of", "old_kind": "institution", "keep_kind": "facility", "evidence": deepcopy(e["evidence"])},
                                 "evidence": deepcopy(e["evidence"]), "reason": "independent source check"} for e in payload["edits"]]}
        raise AssertionError(kind)


def run(model, record=None, det=None, sources=None):
    return extract.run_pan_evidence_repair(record=deepcopy(record or RECORD), anchors={}, detector=det or detector(), schema=SCHEMA,
                             enable_coverage=False,
                             call_model=model, retrieve=lambda *args: {
                                 "queries": ["model object"], "pages": [1], "sources": deepcopy(sources or SOURCES),
                             })


class RepairTests(unittest.TestCase):
    def test_repair_json_truncation_is_not_repaired_or_partially_accepted(self):
        for content, finish in [('{"reviews":[]}', 'length'), ('{"reviews":[', 'stop')]:
            with self.subTest(content=content, finish=finish):
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
                response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)])
                with patch.object(client.chat.completions, 'create', return_value=response) as create, \
                        patch.object(extract, 'record_llm_usage') as usage:
                    state = extract.AgentState(sources_dir=Path('.'), paper_id='synthetic')
                    with self.assertRaises(extract.PanRepairResponseError):
                        extract.call_llm_json(client, 'synthetic', 'Review paper evidence.', {}, 1200,
                                             state=state, stage='pan_agent_refinement',
                                             call_kind='pan_repair_challenge', event_log_path=None)
                    self.assertEqual(create.call_count, 1)
                    self.assertEqual(usage.call_count, 1)

    def test_truncated_decision_splits_targets_within_the_shared_budget(self):
        calls = []
        det = {"target_judgments": [{"target_id": 'samples:sample_1', "judgment": 'unsupported'},
                                    {"target_id": 'samples:sample_2', "judgment": 'unsupported'}]}
        model = FakeModel([])
        def ask(prompt, payload, kind, limit):
            calls.append(kind)
            if kind == 'pan_repair_decision':
                raise extract.PanRepairResponseError('repair_json_truncated', 'length')
            if kind.startswith('pan_repair_decision'):
                self.assertEqual(len(payload['task']['risks']), 1)
                model.edits = [edit(target=payload['task']['risks'][0]['target_id'])]
                return model(prompt, payload, 'pan_repair_decision', limit)
            return model(prompt, payload, kind, limit)
        trace = run(ask, det=det)
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 2)
        self.assertEqual(result['samples'], [])
        self.assertEqual(trace['summary']['model_call_count'], len(calls))
        self.assertEqual(trace['summary']['response_recovery_count'], 1)
        self.assertEqual(calls.count('pan_repair_decision_split_retry'), 2)

    def test_persistent_truncation_and_exhausted_budget_keep_the_whole_task(self):
        for budget in (3, 32):
            calls = []
            def ask(prompt, payload, kind, limit):
                calls.append(kind)
                if kind == 'pan_repair_planner':
                    return {'plans': []}
                raise extract.PanRepairResponseError('repair_json_truncated', 'length')
            with self.subTest(budget=budget), patch.object(extract, 'PAN_REPAIR_MAX_CALLS', budget):
                trace = run(ask)
                result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
                self.assertEqual(result, RECORD)
                self.assertEqual(applied['summary']['applied_patch_count'], 0)
                self.assertLessEqual(len(calls), budget)
                self.assertFalse(any('json_repair' in kind for kind in calls))

    def test_truncated_review_never_commits_a_partially_reviewed_transaction(self):
        for fail_child in (False, True):
            record = deepcopy(RECORD)
            record['samples'].append({'id': 'sample_3', 'sample_name': 'model object C', 'sample_state': None})
            text = 'Objects A, B and C are artificial inputs used only in numerical simulations, not experimental specimens.'
            evidence = [{'source_id': 'page:1', 'quote': text}]
            proposals = [edit(target='samples:sample_' + str(i), evidence=evidence) for i in range(1, 4)]
            model = FakeModel(proposals)
            def ask(prompt, payload, kind, limit):
                if kind == 'pan_repair_challenge':
                    raise extract.PanRepairResponseError('repair_json_truncated', 'length')
                if kind == 'pan_repair_challenge_split_retry':
                    if fail_child and len(payload['edits']) == 1:
                        raise extract.PanRepairResponseError('repair_json_truncated', 'length')
                    return model(prompt, payload, 'pan_repair_challenge', limit)
                return model(prompt, payload, kind, limit)
            det = {'target_judgments': [{'target_id': p['target_id'], 'judgment': 'unsupported'} for p in proposals]}
            with self.subTest(fail_child=fail_child):
                trace = run(ask, record=record, det=det, sources={'page:1': {'text': text}})
                result, applied = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
                if fail_child:
                    # These are independent entity removals, not a split/replacement.
                    self.assertEqual([s['id'] for s in result['samples']], ['sample_3'])
                    self.assertEqual(applied['summary']['applied_patch_count'], 2)
                else:
                    self.assertEqual(result['samples'], [])
                    self.assertEqual(applied['summary']['applied_patch_count'], 3)

    def test_protocol_correction_returns_only_a_patch_for_failed_targets(self):
        proposal = edit(replacement_id='sample_2', evidence=[{'source_id': 'page:1', 'span_ids': ['s1']}])
        model = FakeModel([proposal])
        def ask(prompt, payload, kind, limit):
            if kind == 'pan_repair_protocol_correction':
                self.assertEqual(limit, 1600)
                self.assertNotIn('quote', payload['previous_decision']['assessments'][0]['evidence'][0])
                return {'edit_updates': [{'target_id': 'samples:sample_2', 'action': 'remove'}]}
            return model(prompt, payload, kind, limit)
        trace = run(ask)
        _, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 1)
        self.assertEqual(trace['summary']['protocol_correction_count'], 1)

    def test_target_shape_compiles_remove_intent_and_generates_reference_cascade(self):
        proposal = edit('remove_relation', 'datasets:dataset_2')
        trace = run(FakeModel([proposal]), det=detector('datasets:dataset_2'))
        _, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied['applied'][0]['operation'], 'remove_entity')
        self.assertEqual(applied['summary']['applied_patch_count'], 1)

    def test_reviewer_can_correct_a_category_without_changing_the_transaction(self):
        model = FakeModel([edit(basis='cited_work_only')])
        def ask(prompt, payload, kind, limit):
            out = model(prompt, payload, kind, limit)
            if kind == 'pan_repair_challenge':
                out['reviews'][0].pop('evidence_basis')
                out['reviews'][0].update(basis_valid=False, corrected_basis='simulation_only')
            return out
        trace = run(ask)
        _, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 1)
        self.assertEqual(applied['applied'][0]['evidence_basis'], 'simulation_only')
        self.assertEqual(trace['transactions'][0]['edits'][0]['proposal_evidence_basis'], 'cited_work_only')

    def test_approve_with_measurement_basis_cannot_authorize_deletion(self):
        model = FakeModel([edit()])
        def ask(prompt, payload, kind, limit):
            out = model(prompt, payload, kind, limit)
            if kind == 'pan_repair_challenge':
                out['reviews'][0]['evidence_basis'] = 'explicit_measurement'
            return out
        self.assertFalse(run(ask)['transactions'])

    def test_source_spans_do_not_mix_neighboring_measurement_sections(self):
        text = '2.1 Surface measurements\nSpecimens A and B were measured for surface response.\n2.2 Bulk measurements\nBulk response was measured only for specimen C.\n'
        spans = extract._pan_repair_source_spans(text)
        self.assertEqual(''.join(s['text'] for s in spans), text)
        self.assertGreater(len({s['section'] for s in spans}), 1)
        grounded = extract._pan_repair_ground_evidence([
            {'source_id': 'page:1', 'span_ids': [s['span_id'] for s in spans]}], {'page:1': {'text': text}})
        self.assertEqual(len(grounded), 2)
        self.assertNotIn('Bulk measurements', grounded[0]['quote'])
        self.assertNotIn('Surface measurements', grounded[1]['quote'])

    def test_component_only_context_never_freezes_a_pure_sample_reference(self):
        proposal = edit('add_relation', 'datasets:dataset_1:sample_refs:sample_1', 'explicit_measurement', evidence=MEASURED_EVIDENCE)
        model = FakeModel([proposal], verdict='supported', basis='explicit_measurement')
        def ask(prompt, payload, kind, limit):
            out = model(prompt, payload, kind, limit)
            for a in out.get('assessments', []):
                if a.get('context'):
                    a['context']['object_role'] = 'component_only'
            return out
        trace = run(ask, det=detector('datasets:dataset_1:sample_refs:sample_1', 'insufficient_evidence'), sources=MEASURED_SOURCES)
        self.assertEqual(trace['summary']['freeze_count'], 0)
        self.assertFalse(trace['transactions'])

    def test_retention_rejection_keeps_value_but_does_not_freeze_as_correct(self):
        proposal = edit('add_relation', 'datasets:dataset_1:sample_refs:sample_1', 'explicit_measurement', evidence=MEASURED_EVIDENCE)
        model = FakeModel([proposal], verdict='supported', basis='explicit_measurement', approve=False)
        trace = run(model, det=detector('datasets:dataset_1:sample_refs:sample_1', 'insufficient_evidence'), sources=MEASURED_SOURCES)
        self.assertEqual(trace['summary']['freeze_count'], 0)
        self.assertEqual(trace['unresolved'][0]['verdict'], 'unresolved')
        self.assertTrue(any(r['reason'] == 'retention_not_approved' for r in trace['rejected']))

    def test_coverage_discovers_a_missing_measurement_without_detector_targets(self):
        record = {'samples': [], 'facilities': [], 'datasets': []}
        sample = 'samples:sample_new_coverage_1'
        dataset = 'datasets:dataset_new_coverage_1'
        link = dataset + ':sample_refs:sample_new_coverage_1'
        def ask(prompt, payload, kind, limit):
            if kind == 'pan_repair_inventory_extract':
                return {'observations': [{'sample': {'sample_name': 'specimen B', 'sample_state': 'solid'},
                    'dataset': {'dataset_name': 'diffraction'}, 'facility': None, 'campaign': 'diffraction',
                    'object_role': 'measured_specimen', 'relation_scope': 'specific', 'evidence': MEASURED_EVIDENCE}], 'gaps': []}
            if kind == 'pan_repair_inventory_binding':
                return {'verdict': 'supported', 'bindings': {'sample': 'new', 'dataset': 'new', 'facility': None},
                        'evidence': MEASURED_EVIDENCE}
            return FakeModel([])(prompt, payload, kind, limit)
        trace = extract.run_pan_evidence_repair(record=record, anchors={}, detector={'target_judgments': []}, schema=SCHEMA,
                                               call_model=ask, retrieve=lambda *args: {'sources': MEASURED_SOURCES})
        result, applied = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 3)
        self.assertEqual(result['datasets'][0]['sample_refs'], [result['samples'][0]['id']])
        self.assertEqual(record, {'samples': [], 'facilities': [], 'datasets': []})

    def test_identity_split_requires_measured_states_and_complete_replacement_links(self):
        for complete in (True, False):
            with self.subTest(complete=complete):
                record = {'samples': [{'id': 'sample_1', 'sample_name': 'alloy', 'sample_state': 'solid'}],
                          'facilities': [], 'datasets': [{'id': 'dataset_1', 'dataset_name': 'diffraction',
                                                        'sample_refs': ['sample_1'], 'facility_refs': []}]}
                if not complete:
                    record['datasets'].append({**record['datasets'][0], 'id': 'dataset_2'})
                split_schema = deepcopy(SCHEMA)
                split_schema['samples']['item_schema']['sample_description'] = {'type': 'string|null'}
                text = 'As-prepared alloy A and annealed alloy B were separately measured by diffraction in the reported campaign.'
                evidence = [{'source_id': 'page:1', 'quote': text}]
                def ask(prompt, payload, kind, limit):
                    if kind == 'pan_repair_decision':
                        ids = ['sample_new_coverage_1', 'sample_new_coverage_2']
                        targets = ['samples:' + x for x in ids] + ['datasets:dataset_1:sample_refs:' + x for x in ids]
                        assessments = [{'target_id': t, 'verdict': 'supported', 'evidence_basis': 'explicit_measurement',
                                        'evidence': evidence, 'context': measurement_context(evidence)} for t in targets]
                        assessments.append({'target_id': 'samples:sample_1', 'verdict': 'unsupported', 'evidence_basis': 'identity_refinement',
                                            'identity_axes': ['sample_description'], 'evidence': evidence})
                        edits = [{'action': 'add', 'target_id': 'samples:' + x, 'measurement_campaign': 'diffraction',
                                  'entity': {'sample_name': name, 'sample_state': 'solid', 'sample_description': name}}
                                 for x, name in zip(ids, ['as-prepared alloy A', 'annealed alloy B'])]
                        edits += [{'action': 'add', 'target_id': t, 'measurement_campaign': 'diffraction'} for t in targets[2:]]
                        edits.append({'action': 'remove', 'target_id': 'samples:sample_1', 'replacement_ids': ids, 'reopen_supported': True})
                        return {'assessments': assessments, 'edits': edits}
                    if kind == 'pan_repair_protocol_correction':
                        return {}
                    return FakeModel([])(prompt, payload, kind, limit)
                trace = extract.run_pan_evidence_repair(record=record, anchors={}, detector={'target_judgments': [
                    {'target_id': 'samples:sample_1', 'judgment': 'insufficient_evidence'}]}, schema=split_schema,
                    call_model=ask, retrieve=lambda *args: {'sources': {'page:1': {'text': text}}}, enable_coverage=False)
                result, applied = extract.apply_pan_reviewed_transactions(record, trace, split_schema)
                if complete:
                    self.assertEqual(applied['summary']['applied_patch_count'], 5)
                    self.assertEqual(len(result['samples']), 2)
                    self.assertNotIn('sample_1', result['datasets'][0]['sample_refs'])
                else:
                    self.assertEqual(result, record)
                    self.assertEqual(applied['summary']['applied_patch_count'], 0)

    def test_seed_removal_is_executable_with_review_and_cascade(self):
        original = deepcopy(RECORD)
        trace = run(FakeModel([edit()]))
        result, applied = extract.apply_pan_reviewed_transactions(original, trace, SCHEMA)
        self.assertEqual(applied["summary"]["applied_patch_count"], 1)
        self.assertEqual(result["samples"], [RECORD["samples"][0]])
        self.assertEqual(result["datasets"][1]["sample_refs"], [])
        self.assertEqual(original, RECORD)

    def test_insufficient_can_be_resolved_by_new_counterevidence(self):
        trace = run(FakeModel([edit()]), det=detector(status="insufficient_evidence"))
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied["summary"]["applied_patch_count"], 1)
        self.assertEqual(len(result["samples"]), 1)

    def test_missing_evidence_never_justifies_deletion(self):
        trace = run(FakeModel([edit(basis="insufficient_evidence")]))
        self.assertFalse(trace["transactions"])
        self.assertEqual(trace["rejected"][0]["reason"], "missing_positive_evidence_basis")

    def test_legacy_edit_verdict_does_not_override_grounded_assessment(self):
        trace = run(FakeModel([edit(verdict="supported")]))
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(len(result["samples"]), 1)
        self.assertEqual(applied["summary"]["applied_patch_count"], 1)

    def test_supported_assessment_never_authorizes_removal(self):
        trace = run(FakeModel([edit()], verdict="supported", basis="explicit_measurement"))
        self.assertFalse(trace["transactions"])
        self.assertEqual(trace["summary"]["protocol_correction_count"], 1)

    def test_simulation_reason_cannot_freeze_as_supported(self):
        trace = run(FakeModel([], verdict="supported", basis="simulation_only"))
        self.assertEqual(trace["summary"]["freeze_count"], 0)

    def test_challenge_rejection_keeps_original(self):
        trace = run(FakeModel([edit()], approve=False))
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(result, RECORD)
        self.assertEqual(applied["summary"]["applied_patch_count"], 0)

    def test_quote_layout_matching_preserves_chemical_numbers_and_signs(self):
        sources = {"page:1": {"text": "The specimen Cu0.73Zn0.27 was measured using diffraction.\nThis is an experi-\nmental measurement."}}
        good = [{"source_id": "page:1", "quote": "The specimen Cu0.73Zn0.27 was measured using diffraction. This is an experimental measurement."}]
        self.assertTrue(extract._pan_repair_ground_evidence(good, sources))
        for old, new in [("0.73", "0.63"), ("Cu", "CU"), ("Cu0.73", "Cu-0.73")]:
            altered = [{**good[0], "quote": good[0]["quote"].replace(old, new)}]
            self.assertFalse(extract._pan_repair_ground_evidence(altered, sources))

    def test_normalized_quote_can_apply_and_is_rechecked(self):
        sources = {"page:1": {"page_number": 1, "text": TEXT.replace("artificial input", "artificial\ninput")}}
        trace = run(FakeModel([edit()]), sources=sources)
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied["applied"][0]["evidence"][0]["match_mode"], "pdf_layout")
        trace["transactions"][0]["sources"]["page:1"]["text"] = "Unrelated text"
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(result, RECORD)
        self.assertEqual(applied["skipped"][0]["reason"], "transaction_rolled_back")

    def test_add_relation_to_existing_endpoint_from_insufficient(self):
        add = edit("add_relation", "datasets:dataset_1:sample_refs:sample_2", "explicit_measurement", evidence=MEASURED_EVIDENCE)
        trace = run(FakeModel([add], verdict="supported", basis="explicit_measurement"),
                    det=detector("datasets:dataset_1", "insufficient_evidence"), sources=MEASURED_SOURCES)
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(result["datasets"][0]["sample_refs"], ["sample_1", "sample_2"])
        self.assertEqual(applied["summary"]["applied_patch_count"], 1)

    def test_add_entity_and_reference_are_one_transaction(self):
        add = edit("add_entity", "samples:sample_new_task_1", "explicit_measurement",
                   entity={"sample_name": "measured specimen B", "sample_state": "solid"}, evidence=MEASURED_EVIDENCE)
        link = edit("add_relation", "datasets:dataset_1:sample_refs:sample_new_task_1", "explicit_measurement", evidence=MEASURED_EVIDENCE)
        trace = run(FakeModel([link, add], verdict="supported", basis="explicit_measurement"),
                    det=detector("datasets:dataset_1", "insufficient_evidence"), sources=MEASURED_SOURCES)
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied["summary"]["applied_patch_count"], 2)
        self.assertFalse(extract._pan_repair_structural_issues(result))

    def test_invalid_relation_rolls_back_all_task_edits(self):
        remove = edit("remove_entity", "datasets:dataset_2")
        link = edit("remove_relation", "datasets:dataset_2:sample_refs:missing", "different_campaign")
        trace = run(FakeModel([remove, link]), det=detector("datasets:dataset_2"))
        result, _ = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(result, RECORD)
        self.assertTrue(any(r["reason"] == "invalid_transaction" for r in trace["rejected"]))

    def test_duplicate_removal_redirects_existing_references(self):
        trace = run(FakeModel([edit(basis="duplicate_identity", replacement_id="sample_1")]))
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(result["datasets"][1]["sample_refs"], ["sample_1"])
        self.assertEqual(applied["summary"]["applied_patch_count"], 1)
        unsafe = run(FakeModel([edit(basis="duplicate_identity")]))
        self.assertFalse(unsafe["transactions"])

    def test_preconditions_prevent_stale_application(self):
        trace = run(FakeModel([edit()]))
        changed = deepcopy(RECORD)
        changed["samples"][1]["sample_name"] = "different specimen"
        result, applied = extract.apply_pan_reviewed_transactions(changed, trace, SCHEMA)
        self.assertEqual(result, changed)
        self.assertEqual(applied["skipped"][0]["reason"], "repair_input_or_version_mismatch")

    def test_no_replanning_without_new_sources(self):
        model = FakeModel([])
        original_call = model.__call__
        def ask(prompt, payload, kind, limit):
            out = original_call(prompt, payload, kind, limit)
            if kind == "pan_repair_decision":
                out["next_retrieval"] = {"queries": ["different query"], "missing_fact": "measurement evidence"}
            return out
        trace = run(ask)
        self.assertEqual(trace["summary"]["replan_attempt_count"], 0)
        self.assertEqual(model.calls.count("pan_repair_decision"), 1)

    def test_shared_dataset_risks_are_grouped(self):
        det = {"target_judgments": [
            {"target_id": "datasets:dataset_1", "judgment": "insufficient_evidence"},
            {"target_id": "datasets:dataset_1:sample_refs:sample_1", "judgment": "insufficient_evidence"},
        ]}
        tasks, _ = extract._pan_repair_make_tasks(RECORD, det, {})
        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(tasks[0]["risks"]), 2)

    def test_guard_and_final_selector_keep_approved_seed_correction(self):
        state = extract.AgentState(sources_dir=Path('.'), paper_id="synthetic", schemas={"pan": SCHEMA},
                                   record=deepcopy(RECORD), pan_seed_record=deepcopy(RECORD))
        trace = run(FakeModel([edit()]))
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'events.jsonl'
            applied = extract.apply_pan_agent_refinement_patches(state, log, trace)
            state.record["samples"] = deepcopy(RECORD["samples"])
            extract.apply_pan_seed_guard(state, log, stage="later")
            self.assertEqual(len(state.record["samples"]), 1)
            state.record["samples"] = []
            extract.apply_seed_backbone_final_selector(state, log, stage="final")
            self.assertEqual(len(state.record["samples"]), 1)
            self.assertEqual(state.pan_original_seed_record, RECORD)
            self.assertTrue(applied["seed_changed"])
            payload = extract.checkpoint_payload(state)
            self.assertEqual(payload["pan_repair_record"], state.pan_repair_record)

    def test_independent_relations_do_not_share_reviewer_rejection(self):
        record = deepcopy(RECORD)
        record["datasets"][0]["sample_refs"] = ["sample_1", "sample_2"]
        model = FakeModel([edit("remove_relation", "datasets:dataset_1:sample_refs:sample_1", "simulation_only"),
                           edit("remove_relation", "datasets:dataset_1:sample_refs:sample_2", "simulation_only")])
        def ask(prompt, payload, kind, limit):
            result = model(prompt, payload, kind, limit)
            if kind == "pan_repair_challenge":
                self.assertEqual(len(result["reviews"]), 2)
                result["reviews"][0]["decision"] = "reject"
            return result
        trace = run(ask, record=record, det=detector("datasets:dataset_1", "insufficient_evidence"))
        result, applied = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 1)
        self.assertEqual(result['datasets'][0]['sample_refs'], ['sample_1'])

    def test_supported_target_stays_frozen_even_with_explicit_reopening(self):
        det = detector("datasets:dataset_1", "insufficient_evidence")
        det["target_judgments"].append({"target_id": "datasets:dataset_1:sample_refs:sample_1", "judgment": "supported"})
        proposal = edit("remove_relation", "datasets:dataset_1:sample_refs:sample_1", "different_campaign")
        trace = run(FakeModel([proposal]), det=det)
        self.assertFalse(trace["transactions"])
        proposal["reopen_supported"] = True
        trace = run(FakeModel([proposal]), det=det)
        self.assertFalse(trace["transactions"])
        self.assertEqual(extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)[0], RECORD)

    def test_call_budget_leaves_unresolved_without_partial_edits(self):
        with patch.object(extract, "PAN_REPAIR_MAX_CALLS", 2):
            trace = run(FakeModel([edit()]))
        self.assertLessEqual(trace["summary"]["model_call_count"], 2)
        self.assertFalse(trace["transactions"])
        self.assertEqual(trace["unresolved"][0]["verdict"], "unresolved")

    def test_span_citations_apply_without_a_second_verdict_or_model_written_quote(self):
        model = FakeModel([edit(evidence=[{"source_id": "page:1", "span_ids": ["s1"]}])])
        def ask(prompt, payload, kind, limit):
            if kind in {"pan_repair_decision", "pan_repair_challenge"}:
                self.assertNotIn("text", payload["sources"]["page:1"])
                self.assertEqual(payload["sources"]["page:1"]["spans"][0]["text"], TEXT)
            result = model(prompt, payload, kind, limit)
            for proposal in result.get("edits", []):
                for key in ("verdict", "evidence_basis", "evidence"):
                    proposal.pop(key, None)
            return result
        trace = run(ask)
        _, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied["summary"]["applied_patch_count"], 1)
        evidence = applied["applied"][0]["evidence"][0]
        self.assertEqual(evidence["quote"], TEXT)
        self.assertEqual(evidence["match_mode"], "source_span")
        trace["transactions"][0]["sources"]["page:1"]["text"] = TEXT.replace("only", "never")
        unchanged, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(unchanged, RECORD)
        self.assertEqual(applied["skipped"][0]["reason"], "transaction_rolled_back")

    def test_span_citations_allow_disjoint_but_require_existing_ordered_unique_spans(self):
        text = ("A measured Cu0.73Zn0.27 specimen was examined.\n" * 70)
        sources = {"page:2": {"text": text}}
        citation = {"source_id": "page:2", "span_ids": ["s1", "s2"]}
        self.assertTrue(extract._pan_repair_ground_evidence([citation], sources))
        disjoint = extract._pan_repair_ground_evidence([{**citation, 'span_ids': ['s1', 's3']}], sources)
        self.assertEqual(len(disjoint), 2)
        self.assertEqual([x['span_ids'] for x in disjoint], [['s1'], ['s3']])
        self.assertEqual(extract._pan_repair_ground_evidence(disjoint, sources), disjoint)
        for ids in ([], ["s2", "s1"], ["s1", "s1"], ["s99"]):
            with self.subTest(ids=ids):
                self.assertFalse(extract._pan_repair_ground_evidence([{**citation, "span_ids": ids}], sources))
        altered = {**citation, "quote": text.replace("0.73", "0.63")}
        self.assertFalse(extract._pan_repair_ground_evidence([altered], sources))

    def test_one_protocol_correction_uses_same_sources_before_independent_review(self):
        model = FakeModel([edit()])
        retrieval_calls = []
        def retrieve(*args):
            retrieval_calls.append(args)
            return {"queries": ["model"], "pages": [1], "sources": SOURCES}
        def ask(prompt, payload, kind, limit):
            out = model(prompt, payload, kind, limit)
            if kind == "pan_repair_decision":
                out["assessments"][0]["evidence"] = [{"source_id": "page:1", "quote": "The model object ... only in simulations."}]
            if kind == "pan_repair_protocol_correction":
                self.assertIn("validation_errors", payload)
                out["assessments"][0]["evidence"] = [{"source_id": "page:1", "span_ids": ["s1"]}]
            return out
        trace = extract.run_pan_evidence_repair(record=RECORD, anchors={}, detector=detector(), schema=SCHEMA, enable_coverage=False,
                                               call_model=ask, retrieve=retrieve)
        _, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied["summary"]["applied_patch_count"], 1)
        self.assertEqual(len(retrieval_calls), 1)
        self.assertEqual(model.calls, ["pan_repair_planner", "pan_repair_decision",
                                       "pan_repair_protocol_correction", "pan_repair_challenge"])
        self.assertEqual(len(trace["actions"][0]["protocol_corrections"]), 1)

    def test_correction_can_retain_supported_target_instead_of_forcing_deletion(self):
        model = FakeModel([edit(basis="explicit_measurement", evidence=MEASURED_EVIDENCE)],
                          verdict="supported", basis="explicit_measurement")
        def ask(prompt, payload, kind, limit):
            out = model(prompt, payload, kind, limit)
            if kind == "pan_repair_protocol_correction":
                out["edits"] = []
            return out
        trace = run(ask, sources=MEASURED_SOURCES)
        unchanged, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(unchanged, RECORD)
        self.assertEqual(applied["summary"]["applied_patch_count"], 0)
        self.assertEqual(trace["summary"]["freeze_count"], 1)
        self.assertEqual(trace["unresolved"], [])
        self.assertIn("pan_repair_challenge", model.calls)

    def test_schema_rule_cannot_be_used_as_paper_evidence(self):
        model = FakeModel([edit(basis="facility_hierarchy", evidence=[{
            "source_id": "page:1", "quote": "The schema requires recording only specific facilities, never umbrella institutions."}])])
        trace = run(model)
        self.assertFalse(trace["transactions"])
        self.assertEqual(model.calls.count("pan_repair_protocol_correction"), 1)
        self.assertNotIn("pan_repair_challenge", model.calls)
        self.assertTrue(any(r["reason"] == "ungrounded_evidence" for r in trace["rejected"]))

    def test_invalid_replacement_and_wrong_target_shape_are_caught_before_review(self):
        for proposal in [edit(replacement_id="sample_2"),
                         edit(target="datasets:dataset_2:sample_refs")]:
            with self.subTest(proposal=proposal):
                model = FakeModel([proposal])
                trace = run(model, det=detector("samples:sample_2" if proposal["target_id"].startswith("samples:") else "datasets:dataset_2"))
                self.assertFalse(trace["transactions"])
                self.assertNotIn("pan_repair_challenge", model.calls)
                self.assertEqual(model.calls.count("pan_repair_protocol_correction"), 1)

    def test_entity_replacement_generates_cascade_and_preserves_canonical_measurements(self):
        proposals = [edit(basis="duplicate_identity", replacement_id="sample_1"),
                     edit("remove_relation", "datasets:dataset_2:sample_refs:sample_2", "duplicate_identity")]
        trace = run(FakeModel(proposals))
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(result["datasets"][1]["sample_refs"], ["sample_1"])
        self.assertEqual(applied["summary"]["applied_patch_count"], 1)
        self.assertEqual(trace["actions"][0]["notes"][0]["reason"], "reference_change_generated_by_entity_cascade")

    def test_existing_supported_relation_is_a_noop_with_independent_retention_review(self):
        proposal = edit("add_relation", "datasets:dataset_1:sample_refs:sample_1", "explicit_measurement", evidence=MEASURED_EVIDENCE)
        model = FakeModel([proposal], verdict="supported", basis="explicit_measurement")
        trace = run(model, det=detector("datasets:dataset_1:sample_refs:sample_1", "insufficient_evidence"), sources=MEASURED_SOURCES)
        self.assertFalse(trace["transactions"])
        self.assertEqual(trace["summary"]["protocol_correction_count"], 0)
        self.assertEqual(trace["summary"]["freeze_count"], 1)
        self.assertIn("pan_repair_challenge", model.calls)

    def test_identity_candidates_cannot_reopen_supported_samples(self):
        record = deepcopy(RECORD)
        record["samples"][1] = {**record["samples"][0], "id": "sample_2"}
        det = {"target_judgments": [{"target_id": "samples:sample_2", "judgment": "supported"}]}
        tasks, _ = extract._pan_repair_make_tasks(record, det, {})
        self.assertEqual(tasks[0]["identity_candidates"][0]["canonical_candidate"], "sample_1")
        self.assertEqual(det["target_judgments"][0]["judgment"], "supported")
        proposal = edit(basis="duplicate_identity", replacement_id="sample_1")
        trace = run(FakeModel([proposal]), record=record, det=det)
        self.assertFalse(trace["transactions"])
        proposal["reopen_supported"] = True
        trace = run(FakeModel([proposal]), record=record, det=det)
        self.assertFalse(trace["transactions"])

    def test_protocol_correction_budget_is_shared_across_tasks(self):
        det = {"target_judgments": [
            {"target_id": "datasets:dataset_2", "judgment": "unsupported"},
            {"target_id": "samples:sample_2", "judgment": "unsupported"}]}
        model = FakeModel([])
        with patch.object(extract, "PAN_REPAIR_MAX_CORRECTIONS", 1):
            trace = run(model, det=det)
        self.assertEqual(model.calls.count("pan_repair_protocol_correction"), 1)
        self.assertFalse(trace["transactions"])

    def test_no_change_completion_does_not_certify_seed_and_survives_checkpoint(self):
        state = extract.AgentState(sources_dir=Path('.'), paper_id="synthetic", schemas={"pan": SCHEMA},
                                   record=deepcopy(RECORD), pan_seed_record=deepcopy(RECORD))
        model = FakeModel([], verdict="unresolved", basis="unresolved")
        trace = run(model)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            applied = extract.apply_pan_agent_refinement_patches(state, root/'events.jsonl', trace)
            self.assertTrue(state.pan_repair_status["completed"])
            self.assertFalse(state.pan_repair_status["changed"])
            self.assertEqual(state.pan_repair_status["unresolved_count"], 1)
            self.assertEqual(state.pan_repair_record, {})
            self.assertFalse(applied["seed_changed"])
            state.tool_calls = [
                {"stage": "pan_agent_refinement", "tool": "search_pdf_text", "status": "success"},
                {"stage": "generic_extracted", "tool": "search_pdf_text", "status": "success"}]
            metrics = extract.verification_repair_metrics(state)
            self.assertEqual(metrics["new_evidence_retrieval_count"], 1)
            self.assertEqual(metrics["repairs_with_new_evidence_count"], 0)
            self.assertEqual(metrics["repair_unresolved_count"], 1)
            self.assertTrue(extract.should_skip_stage_from_seed_audit(state, "samples_extracted")[0])
            extract.save_checkpoint(state, root/'checkpoint.json')
            restored = extract.AgentState(sources_dir=Path('.'), paper_id="synthetic")
            self.assertTrue(extract.restore_checkpoint(restored, root/'checkpoint.json'))
            self.assertEqual(restored.pan_repair_status, state.pan_repair_status)
            self.assertEqual(restored.pan_repair_record, {})

    def test_mismatched_apply_input_is_not_marked_completed(self):
        state = extract.AgentState(sources_dir=Path('.'), paper_id="synthetic", schemas={"pan": SCHEMA},
                                   record=deepcopy(RECORD), pan_seed_record=deepcopy(RECORD))
        trace = run(FakeModel([edit()]))
        state.record["samples"][0]["sample_name"] = "other specimen"
        with tempfile.TemporaryDirectory() as tmp:
            extract.apply_pan_agent_refinement_patches(state, Path(tmp)/'events.jsonl', trace)
        self.assertFalse(state.pan_repair_status["completed"])
        self.assertFalse(state.pan_repair_record)

    def test_tool_results_are_cached_across_tasks(self):
        class Tools:
            def __init__(self): self.calls = []
            def call(self, name, paper_id, arguments):
                self.calls.append(name)
                return {"hits": [{"page_number": 1, "snippet": TEXT}]} if name == "search_pdf_text" else {}
        state = extract.AgentState(sources_dir=Path('.'), paper_id="synthetic", page_texts={1: TEXT})
        tools, cache = Tools(), {}
        task = {"risks": [], "entity_group": "samples"}
        with patch.object(extract, "record_tool_call"):
            for _ in range(2):
                output = extract._retrieve_pan_repair_task(state, Path('unused'), tools, {}, cache, task,
                                                          {"queries": ["model object"]}, set(), set())
        self.assertEqual(tools.calls, ["search_pdf_text", "read_pdf_pages"])
        self.assertIn("page:1", output["sources"])

    def test_cached_inputs_reject_mismatched_fingerprint_or_status(self):
        state = extract.AgentState(sources_dir=Path('.'), paper_id="synthetic", input_fingerprint={"paper_pdf": "abc"})
        targets = extract.build_pan_anchor_targets(RECORD)
        cached = {"paper_id": "synthetic", "detector_report_version": "pan-anchor-llm-detector-v10",
                  "target_judgments": [{"target_id": t["target_id"], "judgment": "supported"} for t in targets]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = {"checkpoint.json": {"input_fingerprint": state.input_fingerprint},
                      "global_seed_prediction.json": RECORD,
                      "global_seed_anchors.json": {"anchor_report": {"paper_id": "synthetic", "targets": targets}},
                      "global_seed_anchor_detector.json": cached}
            for name, value in values.items():
                (root/name).write_text(json.dumps(value), encoding='utf-8')
            restored = extract._load_cached_pan_repair_inputs(state, root)
            self.assertEqual(restored, cached)
            self.assertEqual(state.stage_status["pan_global_seeded"], "done")
            self.assertEqual(state.pan_original_seed_record, RECORD)
            cached["target_judgments"][0]["judgment"] = "unsupported_by_schema"
            (root/'global_seed_anchor_detector.json').write_text(json.dumps(cached), encoding='utf-8')
            with self.assertRaises(RuntimeError):
                extract._load_cached_pan_repair_inputs(state, root)
            state.input_fingerprint = {"paper_pdf": "changed"}
            with self.assertRaises(RuntimeError):
                extract._load_cached_pan_repair_inputs(state, root)

    def test_cached_pipeline_runs_only_repair_and_preserves_upstream_files(self):
        self._check_cached_pipeline(discover=False)

    def test_supported_evidence_does_not_require_three_copies_of_the_same_citation(self):
        proposal = edit('add_relation', 'datasets:dataset_1:sample_refs:sample_1', 'explicit_measurement',
                        evidence=MEASURED_EVIDENCE)
        model = FakeModel([proposal], verdict='supported', basis='explicit_measurement')
        for context in (None, {**measurement_context(MEASURED_EVIDENCE), 'identity_evidence': [], 'link_evidence': []}):
            def ask(prompt, payload, kind, limit):
                out = model(prompt, payload, kind, limit)
                for assessment in out.get('assessments', []):
                    assessment.pop('context', None)
                    if context is not None:
                        assessment['context'] = deepcopy(context)
                return out
            with self.subTest(context=context):
                trace = run(ask, det=detector(proposal['target_id'], 'insufficient_evidence'), sources=MEASURED_SOURCES)
                self.assertEqual(trace['summary']['freeze_count'], 1)
                self.assertEqual(trace['summary']['protocol_correction_count'], 0)

    def test_bare_identifiers_are_resolved_only_when_unique_or_explicitly_declared(self):
        response = {'assessments': [{'target_id': 'sample_2'}], 'edits': None}
        self.assertEqual(extract._pan_repair_normalize_response(response, RECORD)['assessments'][0]['target_id'],
                         'samples:sample_2')
        ambiguous = deepcopy(RECORD)
        ambiguous['facilities'] = [{'id': 'sample_2', 'name': 'ambiguous'}]
        self.assertEqual(extract._pan_repair_canonical_target('sample_2', ambiguous), 'sample_2')
        self.assertEqual(extract._pan_repair_canonical_target('sample_999', RECORD), 'sample_999')
        response = {'edits': [{'action': 'add', 'target_id': 'sample_new_case_1', 'entity': {'sample_name': 'B'}}],
                    'assessments': [{'target_id': 'sample_new_case_1'}]}
        normalized = extract._pan_repair_normalize_response(response, RECORD)
        self.assertEqual(normalized['assessments'][0]['target_id'], 'samples:sample_new_case_1')

    def test_correction_null_arrays_and_unique_bare_drop_are_safe(self):
        previous = {'assessments': [], 'edits': [{'target_id': 'samples:sample_2', 'action': 'remove'}]}
        errors = [{'target_id': 'samples:sample_2', 'reason': 'invalid'}]
        empty = extract._pan_repair_patch_decision(previous, {'assessment_updates': None, 'drop_edits': None}, errors, RECORD)
        self.assertEqual(empty, previous)
        dropped = extract._pan_repair_patch_decision(previous, {'drop_edits': ['sample_2']}, errors, RECORD)
        self.assertEqual(dropped['edits'], [])
        with self.assertRaises(ValueError):
            extract._pan_repair_patch_decision(previous, {'drop_edits': ['sample_1']}, errors, RECORD)

    def test_semantically_equal_responses_merge_evidence_without_rewriting_verdict(self):
        a = {'target_id': 'samples:sample_2', 'verdict': 'unsupported', 'evidence_basis': 'simulation_only',
             'reason': 'first explanation', 'evidence': EVIDENCE}
        b = {**a, 'reason': 'another explanation', 'evidence': [{'source_id': 'page:1', 'span_ids': ['s1']}]}
        result = extract._pan_repair_merge_responses([{'assessments': [a]}, {'assessments': [b]}])
        self.assertEqual(len(result['assessments']), 1)
        self.assertEqual(len(result['assessments'][0]['evidence']), 2)
        c = {**a, 'verdict': 'supported', 'evidence_basis': 'explicit_measurement'}
        conflict = extract._pan_repair_merge_responses([{'assessments': [a]}, {'assessments': [c]}])
        self.assertEqual(conflict['assessments'][0]['verdict'], 'unresolved')
        self.assertTrue(conflict['assessments'][0]['response_conflict'])

    def test_invalid_patch_is_logged_and_does_not_discard_independent_valid_edit(self):
        proposals = [edit(target='samples:sample_1', evidence=[{'source_id': 'page:1', 'quote': 'An invented observation not in this source.'}]), edit()]
        model = FakeModel(proposals)
        def ask(prompt, payload, kind, limit):
            if kind == 'pan_repair_protocol_correction':
                return {'drop_edits': ['unknown_target']}
            return model(prompt, payload, kind, limit)
        det = {'target_judgments': [detector('samples:sample_1')['target_judgments'][0], detector()['target_judgments'][0]]}
        trace = run(ask, det=det)
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual([s['id'] for s in result['samples']], ['sample_1'])
        self.assertEqual(applied['summary']['applied_patch_count'], 1)
        correction = trace['actions'][0]['protocol_corrections'][0]
        self.assertEqual(correction['raw_patch']['drop_edits'], ['unknown_target'])
        self.assertIn('invalid correction drops', correction['error'])

    def test_dependent_addition_is_not_partially_committed_after_review_rejection(self):
        sample = edit('add_entity', 'samples:sample_new_case_1', 'explicit_measurement',
                      entity={'sample_name': 'specimen B', 'sample_state': 'solid'}, evidence=MEASURED_EVIDENCE)
        link = edit('add_relation', 'datasets:dataset_1:sample_refs:sample_new_case_1', 'explicit_measurement', evidence=MEASURED_EVIDENCE)
        model = FakeModel([sample, link], verdict='supported', basis='explicit_measurement')
        def ask(prompt, payload, kind, limit):
            out = model(prompt, payload, kind, limit)
            if kind == 'pan_repair_challenge':
                out['reviews'][1]['decision'] = 'reject'
            return out
        trace = run(ask, det=detector('datasets:dataset_1', 'insufficient_evidence'), sources=MEASURED_SOURCES)
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(result, RECORD)
        self.assertEqual(applied['summary']['applied_patch_count'], 0)
        self.assertTrue(any(r['reason'] == 'transaction_peer_not_approved' for r in trace['rejected']))

    def test_comparison_review_sees_canonical_specimen_and_redirected_reference(self):
        model = FakeModel([edit(basis='duplicate_identity', replacement_id='sample_1')])
        def ask(prompt, payload, kind, limit):
            if kind == 'pan_repair_challenge':
                comparison = payload['comparisons'][0]
                self.assertEqual(len(comparison['before']['samples']), 2)
                self.assertEqual([s['id'] for s in comparison['after']['samples']], ['sample_1'])
                self.assertEqual(comparison['after']['datasets'][1]['sample_refs'], ['sample_1'])
                return {'reviews': [{'edit_id': payload['edits'][0]['edit_id'], 'preferred': 'after',
                                     'evidence': EVIDENCE, 'reason': 'Retains canonical specimen; only duplicate record is removed.'}]}
            return model(prompt, payload, kind, limit)
        trace = run(ask)
        _, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 1)

    def test_reason_text_never_overrides_negative_or_contradictory_review(self):
        proposal = edit()
        for check in ({'preferred': 'before'}, {'preferred': 'uncertain'},
                      {'preferred': 'after', 'decision': 'reject'},
                      {'preferred': 'after', 'checks': {'exclusion': False}},
                      {'preferred': 'after', 'corrected_basis': 'explicit_measurement'}):
            with self.subTest(check=check):
                review = {**check, 'reason': 'This is simulation only and should be removed.', 'evidence': EVIDENCE}
                self.assertIsNone(extract._pan_repair_checked_review(proposal, review, SOURCES))

    def test_invalid_citation_is_not_silently_dropped_from_an_otherwise_valid_list(self):
        citations = EVIDENCE + [{'source_id': 'page:999', 'quote': TEXT}]
        self.assertEqual(extract._pan_repair_ground_evidence(citations, SOURCES), [])

    def test_disjoint_review_citations_apply_and_are_revalidated_as_separate_passages(self):
        text = TEXT + '\n\nUnrelated details on data processing are reported in this paragraph.\n\n' + TEXT
        citations = [{'source_id': 'page:1', 'span_ids': ['s1', 's3']}]
        model = FakeModel([edit(evidence=citations)])
        def ask(prompt, payload, kind, limit):
            if kind == 'pan_repair_challenge':
                return {'reviews': [{'edit_id': payload['edits'][0]['edit_id'], 'preferred': 'after',
                                     'evidence': citations, 'reason': 'Both passages establish simulation-only use.'}]}
            return model(prompt, payload, kind, limit)
        trace = run(ask, sources={'page:1': {'text': text}})
        _, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 1)
        self.assertEqual([e['span_ids'] for e in applied['applied'][0]['review']['evidence']], [['s1'], ['s3']])
        trace['transactions'][0]['sources']['page:1']['text'] = text.replace('only', 'never')
        unchanged, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(unchanged, RECORD)
        self.assertEqual(applied['summary']['applied_patch_count'], 0)

    def test_malformed_review_batch_does_not_discard_other_independent_reviews(self):
        record = deepcopy(RECORD)
        record['samples'] += [{'id': f'sample_{i}', 'sample_name': f'model object {i}', 'sample_state': None} for i in range(3, 5)]
        proposals = [edit(target=f'samples:sample_{i}') for i in range(1, 5)]
        model = FakeModel(proposals)
        def ask(prompt, payload, kind, limit):
            if kind.startswith('pan_repair_challenge') and len(payload['edits']) == 1:
                return {'reviews': 'invalid array'}
            return model(prompt, payload, kind, limit)
        det = {'target_judgments': [detector(p['target_id'])['target_judgments'][0] for p in proposals]}
        with patch.object(extract, 'PAN_REPAIR_REVIEW_BATCH', 3):
            trace = run(ask, record=record, det=det)
        result, applied = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
        self.assertEqual([s['id'] for s in result['samples']], ['sample_4'])
        self.assertEqual(applied['summary']['applied_patch_count'], 3)
        self.assertEqual(trace['summary']['protocol_failed_task_count'], 1)

    def test_raw_truncated_content_is_retained_and_response_recovery_is_bounded(self):
        def ask(prompt, payload, kind, limit):
            if kind == 'pan_repair_planner':
                return {'plans': []}
            raise extract.PanRepairResponseError('repair_json_truncated', 'length', '{"assessments":[')
        trace = run(ask)
        self.assertTrue(any(e.get('raw_content') == '{"assessments":[' for e in trace['model_responses']))
        self.assertLessEqual(trace['summary']['response_recovery_count'], extract.PAN_REPAIR_MAX_RECOVERIES)
        self.assertLessEqual(trace['summary']['model_call_count'], 12)

    def test_runtime_thinking_is_explicit_and_format_correction_is_non_thinking(self):
        state = extract.AgentState(sources_dir=Path('.'), paper_id='synthetic')
        model = FakeModel([edit()])
        requests = []
        def call(**kwargs):
            requests.append(kwargs)
            state.llm_calls.append({'total_tokens': 10})
            out = model(kwargs['system_prompt'], kwargs['payload'], kwargs['call_kind'], kwargs['max_tokens'])
            if kwargs['call_kind'] == 'pan_repair_decision':
                out['edits'][0]['replacement_id'] = 'sample_2'
            return out
        with patch.dict(extract.os.environ, {'REPAIR_THINKING': '1', 'REPAIR_THINKING_TOKENS': '2048',
                                            'REPAIR_MAX_TOTAL_TOKENS': '0'}), \
                patch.object(extract, 'call_llm_json', side_effect=call), \
                patch.object(extract, '_retrieve_pan_repair_task', return_value={'sources': SOURCES}):
            trace = extract.run_pan_agent_risk_refinement(None, 'synthetic-model', state, None, None, RECORD, {}, detector(), 10000)
        self.assertTrue(requests)
        self.assertTrue(any(r['call_kind'] == 'pan_repair_protocol_correction' for r in requests))
        for request in requests:
            correction = 'protocol_correction' in request['call_kind']
            self.assertEqual(request['enable_thinking'], not correction)
            self.assertEqual(request['thinking_token_budget'], None if correction else 2048)
        self.assertEqual(trace['runtime']['total_tokens'], len(requests) * 10)
        self.assertEqual(trace['runtime']['model'], 'synthetic-model')

    def test_no_total_token_cutoff_and_current_usage_counts_reasoning_once(self):
        state = extract.AgentState(sources_dir=Path('.'), paper_id='synthetic', llm_calls=[{'total_tokens': 999999}])
        def call(**kwargs):
            state.llm_calls.append({'total_tokens': 75000, 'reasoning_tokens': 4000})
            if kwargs['call_kind'] == 'pan_repair_planner':
                return {'plans': []}
            return {'assessments': [], 'edits': []}
        with patch.dict(extract.os.environ, {'REPAIR_MAX_TOTAL_TOKENS': '0'}), \
                patch.object(extract, 'call_llm_json', side_effect=call) as invoke, \
                patch.object(extract, '_retrieve_pan_repair_task', return_value={'sources': SOURCES}):
            trace = extract.run_pan_agent_risk_refinement(None, 'synthetic', state, None, None, RECORD, {}, detector(), 10000)
        self.assertGreater(invoke.call_count, 2)
        self.assertEqual(trace['runtime']['total_tokens'], invoke.call_count * 75000)
        self.assertFalse(trace['runtime']['token_budget_reached'])
        self.assertNotIn('repair_total_token_budget_exhausted', trace['runtime']['stop_reasons'])
        self.assertFalse(trace['transactions'])

    def test_missing_usage_stops_further_inference_instead_of_counting_it_as_free(self):
        state = extract.AgentState(sources_dir=Path('.'), paper_id='synthetic')
        def call(**kwargs):
            state.llm_calls.append({'total_tokens': None})
            return {'plans': []}
        with patch.object(extract, 'call_llm_json', side_effect=call) as invoke, \
                patch.object(extract, '_retrieve_pan_repair_task', return_value={'sources': SOURCES}):
            trace = extract.run_pan_agent_risk_refinement(None, 'synthetic', state, None, None, RECORD, {}, detector(), 10000)
        self.assertEqual(invoke.call_count, 1)
        self.assertFalse(trace['runtime']['usage_complete'])
        self.assertIn('repair_usage_unavailable', trace['runtime']['stop_reasons'])

    def test_completely_missing_usage_entry_also_stops_further_requests(self):
        state = extract.AgentState(sources_dir=Path('.'), paper_id='synthetic')
        with patch.object(extract, 'call_llm_json', return_value={'plans': []}) as invoke, \
                patch.object(extract, '_retrieve_pan_repair_task', return_value={'sources': SOURCES}):
            trace = extract.run_pan_agent_risk_refinement(None, 'synthetic', state, None, None, RECORD, {}, detector(), 10000)
        self.assertEqual(invoke.call_count, 1)
        self.assertFalse(trace['runtime']['usage_complete'])
        self.assertIn('repair_usage_unavailable', trace['runtime']['stop_reasons'])

    def test_repair_http_body_reserves_thinking_without_hidden_context_retry(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'), finish_reason='stop')])
        with patch.object(client.chat.completions, 'create', return_value=response) as create, \
                patch.object(extract, 'record_llm_usage'):
            extract.call_llm_json(client, 'synthetic', 'Judge evidence.', {}, 5000,
                                  state=extract.AgentState(sources_dir=Path('.'), paper_id='synthetic'),
                                  stage='pan_agent_refinement', call_kind='pan_repair_decision', event_log_path=None,
                                  enable_thinking=True, thinking_token_budget=2048)
            body = create.call_args.kwargs
            self.assertTrue(body['extra_body']['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(body['extra_body']['thinking_token_budget'], 2048)
            self.assertEqual(body['max_tokens'], 5000)
        with patch.object(client.chat.completions, 'create', side_effect=RuntimeError('maximum context length exceeded')) as create:
            with self.assertRaises(RuntimeError):
                extract.call_llm_json(client, 'synthetic', 'Judge.', {}, 5000,
                                      state=extract.AgentState(sources_dir=Path('.'), paper_id='synthetic'),
                                      stage='pan_agent_refinement', call_kind='pan_repair_decision', event_log_path=None)
            self.assertEqual(create.call_count, 1)

    def test_infrastructure_failure_stops_subsequent_model_requests(self):
        state = extract.AgentState(sources_dir=Path('.'), paper_id='synthetic')
        with patch.object(extract, 'call_llm_json', side_effect=RuntimeError('HTTP 400: unsupported parameter')) as invoke, \
                patch.object(extract, '_retrieve_pan_repair_task', return_value={'sources': SOURCES}):
            trace = extract.run_pan_agent_risk_refinement(None, 'synthetic', state, None, None, RECORD, {}, detector(), 10000)
        self.assertEqual(invoke.call_count, 1)
        self.assertIn('repair_model_request_failed', trace['runtime']['stop_reasons'])
        self.assertEqual(trace['summary']['model_call_count'], 1)
        self.assertFalse(trace['runtime']['usage_complete'])
        self.assertFalse(trace['transactions'])

    def test_repair_disables_sdk_retries_without_changing_the_upstream_client(self):
        selected = []
        repair_client = object()
        original_client = SimpleNamespace(with_options=lambda **options: selected.append(options) or repair_client)
        state = extract.AgentState(sources_dir=Path('.'), paper_id='synthetic')
        def call(**kwargs):
            self.assertIs(kwargs['client'], repair_client)
            state.llm_calls.append({'total_tokens': 10})
            return {'assessments': [], 'edits': []}
        with patch.object(extract, 'call_llm_json', side_effect=call), \
                patch.object(extract, '_retrieve_pan_repair_task', return_value={'sources': SOURCES}):
            extract.run_pan_agent_risk_refinement(original_client, 'synthetic', state, None, None, RECORD, {}, detector(), 10000)
        self.assertEqual(selected, [{'max_retries': 0}])

    def test_reasoning_json_is_never_mistaken_for_a_final_approval(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
        content = '<think>{"reviews":[{"preferred":"after"}]}</think>'
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason='stop')])
        with patch.object(client.chat.completions, 'create', return_value=response), patch.object(extract, 'record_llm_usage'):
            with self.assertRaisesRegex(extract.PanRepairResponseError, 'reasoning_in_final_content'):
                extract.call_llm_json(client, 'synthetic', 'Judge.', {}, 5000,
                                      state=extract.AgentState(sources_dir=Path('.'), paper_id='synthetic'),
                                      stage='pan_agent_refinement', call_kind='pan_repair_challenge', event_log_path=None)

    def test_runner_preflight_rejects_wrong_model_and_unsupported_reasoning_budget(self):
        import re
        import sys
        runner = (Path(__file__).resolve().parents[1] / 'run.sh').read_text(encoding='utf-8')
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", runner, re.S)
        models = next(b for b in blocks if 'served = [' in b)
        budget = next(b for b in blocks if 'supported = any(' in b)
        cases = [(models, {'data': [{'id': 'chosen-model'}]}, True),
                 (models, {'data': [{'id': 'stale-model'}]}, False),
                 (budget, {'components': {'schemas': {'ChatCompletionRequest': {
                     'properties': {'thinking_token_budget': {}}}}}}, True),
                 (budget, {'components': {'schemas': {}}}, False)]
        for code, payload, accepted in cases:
            with self.subTest(accepted=accepted, payload=payload), \
                    patch.object(sys, 'argv', ['preflight', 'http://unused', 'chosen-model']), \
                    patch('urllib.request.urlopen', return_value=StringIO(json.dumps(payload))):
                if accepted:
                    exec(compile(code, '<runner-preflight>', 'exec'), {})
                else:
                    with self.assertRaises(SystemExit):
                        exec(compile(code, '<runner-preflight>', 'exec'), {})

    def test_runner_uses_the_same_gemma_default_for_full_and_cached_repair(self):
        runner = (Path(__file__).resolve().parents[1] / 'run.sh').read_text(encoding='utf-8')
        selection = runner.split('# Model selection.', 1)[1].split('PAPER_LIST=', 1)[0]
        self.assertIn('MODEL_PROFILE="${MODEL_PROFILE:-gemma31b}"', selection)
        self.assertNotIn('REPAIR_FROM_CACHE', selection)
        small = selection.split('default|qwen9b|small)', 1)[1].split(';;', 1)[0]
        self.assertIn('MODEL="${MODEL:-Qwen/Qwen3.5-9B}"', small)

    def test_downloaded_v7_response_shapes_can_be_replayed_without_annotations(self):
        """Optional real-trace regression: paper text/model output only, no evaluator."""
        traces = []
        for path in (Path(__file__).resolve().parents[1] / 'outputs' / 'run').glob('*/pan_agent_risk_refinement.json'):
            trace = json.loads(path.read_text(encoding='utf-8'))
            if trace.get('agent_trace_version') == 'pan-agent-risk-refinement-v7':
                traces.append(trace)
        if not traces:
            self.skipTest('Downloaded v7 response traces are not available')
        recovered_disjoint, accepted_empty_legacy = 0, 0
        for trace in traces:
            for action in trace.get('actions', []):
                sources = action.get('sources', {})
                for round_ in action.get('rounds', []):
                    for assessment in round_.get('decision', {}).get('assessments', []):
                        context = assessment.get('context', {})
                        if assessment.get('verdict') == 'supported' and any(context.get(k) == [] for k in ('identity_evidence', 'link_evidence')):
                            accepted, _ = extract._pan_repair_assessments([assessment], sources)
                            accepted_empty_legacy += bool(accepted)
            for rejected in trace.get('rejected', []):
                review = rejected.get('review', {})
                if review.get('decision') != 'approve':
                    continue
                for action in trace.get('actions', []):
                    grounded = extract._pan_repair_ground_evidence(review.get('evidence'), action.get('sources', {}))
                    if grounded and len(grounded) > len(review.get('evidence', [])):
                        recovered_disjoint += 1
                        break
        self.assertGreater(recovered_disjoint, 0)
        self.assertGreater(accepted_empty_legacy, 0)

    def test_cached_pipeline_keeps_discovered_entities_and_normalizes_references(self):
        self._check_cached_pipeline(discover=True)

    def _check_cached_pipeline(self, discover):
        schemas = extract.load_schemas(Path('SOURCES'))
        record = extract.clean_record(deepcopy(RECORD), schemas['generic'], schemas['pan'])
        record = extract.normalize_local_ids(record)
        fingerprint = {"paper_pdf": "synthetic_pdf", "generic_schema": "g", "pan_schema": "p"}
        targets = extract.build_pan_anchor_targets(record)
        risky = {"samples:sample_2", "datasets:dataset_2", "datasets:dataset_2:sample_refs:sample_2"}
        cached = {"paper_id": "synthetic", "detector_report_version": "pan-anchor-llm-detector-v10",
                  "target_judgments": [{"target_id": t['target_id'], "target_type": t['target_type'],
                                        "judgment": "unsupported" if t['target_id'] in risky else "supported"} for t in targets]}
        class Span:
            def set_attribute(self, *args): pass
        class Tracer:
            def start_as_current_span(self, *args): return nullcontext(Span())
        model = FakeModel([])
        def call_model(**kwargs):
            kwargs['state'].llm_calls.append({'total_tokens': 10})
            if kwargs['call_kind'] == 'pan_repair_inventory_extract' and discover:
                evidence = [{'source_id': 'page:2', 'span_ids': ['s1']}]
                return {'observations': [{'sample': {'sample_name': 'specimen B', 'sample_state': None},
                    'dataset': {'dataset_name': 'diffraction'}, 'facility': None, 'campaign': 'diffraction',
                    'object_role': 'measured_specimen', 'relation_scope': 'specific', 'evidence': evidence}], 'gaps': []}
            if kwargs['call_kind'] == 'pan_repair_inventory_binding':
                return {'verdict': 'supported', 'bindings': {'sample': 'new', 'dataset': 'datasets:dataset_1', 'facility': None},
                        'evidence': [{'source_id': 'page:2', 'span_ids': ['s1']}]}
            if kwargs['call_kind'] == 'pan_repair_decision':
                task = kwargs['payload']['task']
                model.edits = [edit(target='datasets:dataset_2' if task['dataset_id'] else 'samples:sample_2')]
            return model(kwargs['system_prompt'], kwargs['payload'], kwargs['call_kind'], kwargs['max_tokens'])
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            values = {'checkpoint.json': {'input_fingerprint': fingerprint}, 'global_seed_prediction.json': record,
                      'global_seed_anchors.json': {'anchor_report': {'paper_id': 'synthetic', 'targets': targets}},
                      'global_seed_anchor_detector.json': cached}
            for name, value in values.items():
                (root/name).write_text(json.dumps(value), encoding='utf-8')
            original_bytes = {name: (root/name).read_bytes() for name in values if name != 'checkpoint.json'}
            stack.enter_context(patch.object(extract, 'load_pdf_pages', return_value={1: TEXT, 2: MEASUREMENT}))
            stack.enter_context(patch.object(extract, 'compute_input_fingerprint', return_value=fingerprint))
            stack.enter_context(patch.object(extract, 'build_document_memory', return_value={'body_pages': [1]}))
            stack.enter_context(patch.object(extract, 'call_llm_json', side_effect=call_model))
            stack.enter_context(patch.object(extract, '_retrieve_pan_repair_task', return_value={
                'queries': ['simulation'], 'pages': [1, 2], 'sources': {
                    **SOURCES, 'page:2': {'page_number': 2, 'text': MEASUREMENT}}}))
            stack.enter_context(redirect_stdout(StringIO()))
            final, state = extract.run_extraction(
                'synthetic', None, Tracer(), 'synthetic-model', 120000, 4500,
                root/'checkpoint.json', root/'events.jsonl', None, repair_from_cache=True,
            )
            self.assertEqual(len(final['samples']), 2 if discover else 1)
            self.assertEqual(len(final['datasets']), 1)
            self.assertFalse(extract._pan_repair_structural_issues(final))
            if discover:
                self.assertIn('specimen B', [s['sample_name'] for s in final['samples']])
                self.assertEqual(len(final['datasets'][0]['sample_refs']), 2)
            self.assertEqual(state.pan_original_seed_record, record)
            # Reviewed IDs are immutable. The unchanged generic validator still
            # reports its old contiguous-numbering preference for discoveries.
            self.assertTrue(set(state.validation_issues) <= {'sample IDs are not sequential.'})
            self.assertTrue(set(model.calls) <= {'pan_repair_planner', 'pan_repair_decision', 'pan_repair_protocol_correction',
                                               'pan_repair_challenge', 'pan_repair_inventory_extract',
                                               'pan_repair_fact_facilities', 'pan_repair_fact_missing_samples',
                                               'pan_repair_fact_links', 'pan_repair_fact_sample_identity',
                                               'pan_repair_fact_dataset_identity'})
            for name, original in original_bytes.items():
                self.assertEqual((root/name).read_bytes(), original)
            self.assertEqual(json.loads((root/'pan_agent_refinement_apply.json').read_text())['summary']['applied_patch_count'], 4 if discover else 2)


class RepairThreeRoundTests(unittest.TestCase):
    def engine(self, model, retrieve, det=None, record=None, coverage=False):
        return extract.run_pan_evidence_repair(record=deepcopy(record or RECORD), anchors={},
            detector=det or detector(), schema=SCHEMA, call_model=model,
            retrieve=retrieve, enable_coverage=coverage)

    def test_unresolved_fact_gets_new_retrieval_and_second_decision_without_model_queries(self):
        seen, payloads = [], []
        def retrieve(task, plan, queries, pages):
            seen.append(deepcopy(plan))
            page = len(seen)
            text = 'The methods section identifies a model but does not yet establish its measurement role.' if page == 1 else TEXT
            return {'queries': plan.get('queries', []), 'pages': [page],
                    'sources': {f'page:{page}': {'text': text, 'page_number': page}}}
        def model(prompt, payload, kind, limit):
            if kind == 'pan_repair_decision':
                payloads.append(deepcopy(payload))
                if payload['paper_round'] == 1:
                    return {'assessments': [{'target_id': 'samples:sample_2', 'verdict': 'unresolved',
                                            'evidence_basis': 'unresolved', 'evidence': []}], 'edits': []}
                proposal = edit(evidence=[{'source_id': 'page:2', 'quote': TEXT}])
                return FakeModel([proposal])(prompt, payload, kind, limit)
            return FakeModel([])(prompt, payload, kind, limit)
        trace = self.engine(model, retrieve)
        self.assertEqual(trace['summary']['paper_round_count'], 2)
        self.assertEqual(trace['summary']['replan_attempt_count'], 1)
        self.assertTrue(seen[1]['force_evidence_expansion'])
        self.assertTrue(seen[1]['queries'])
        self.assertTrue(payloads[1]['previous_assessments'])
        self.assertEqual(len(trace['transactions']), 1)
        result, _, applied = extract.replay_pan_evidence_repair(record=RECORD, trace=trace, detector=detector(), schema=SCHEMA)
        self.assertEqual([s['id'] for s in result['samples']], ['sample_1'])
        self.assertFalse(applied['skipped'])

    def test_three_round_limit_is_global_and_all_tasks_get_each_sweep(self):
        det = {'target_judgments': [detector()['target_judgments'][0],
                                  detector('datasets:dataset_1', 'insufficient_evidence')['target_judgments'][0]]}
        counts, order = {}, []
        def retrieve(task, plan, queries, pages):
            tid = task['task_id']
            counts[tid] = counts.get(tid, 0) + 1
            n = counts[tid]
            return {'queries': plan.get('queries', []), 'pages': [n], 'sources': {
                f'page:{n}': {'text': f'Additional method observation number {n} about this experiment and specimen.'}}}
        def model(prompt, payload, kind, limit):
            if kind == 'pan_repair_decision':
                order.append((payload['paper_round'], payload['task']['task_id']))
                return {'assessments': [{'target_id': r['target_id'], 'verdict': 'unresolved',
                    'evidence_basis': 'unresolved', 'evidence': []} for r in payload['task']['risks']], 'edits': []}
            return FakeModel([])(prompt, payload, kind, limit)
        trace = self.engine(model, retrieve, det=det)
        self.assertEqual([n for n, _ in order], [1, 1, 2, 2, 3, 3])
        self.assertEqual(set(counts.values()), {3})
        self.assertTrue(trace['summary']['round_limit_reached'])
        self.assertEqual(trace['summary']['max_paper_rounds'], 3)
        self.assertEqual(trace['summary']['replan_attempt_count'], 4)
        self.assertFalse(trace['transactions'])

    def test_completed_task_stops_after_one_round_and_relabelled_text_does_not_replan(self):
        good = self.engine(FakeModel([edit()]), lambda *a: {'sources': SOURCES})
        self.assertEqual(good['summary']['paper_round_count'], 1)
        model = FakeModel([], verdict='unresolved', basis='unresolved')
        count = []
        def retrieve(*args):
            count.append(1)
            return {'sources': {f'page:{len(count)}': {'text': TEXT}}, 'pages': [len(count)]}
        trace = self.engine(model, retrieve)
        self.assertEqual(model.calls.count('pan_repair_decision'), 1)
        self.assertEqual(len(count), 2)
        self.assertEqual(trace['actions'][-1]['stop_reason'], 'no_new_evidence')
        self.assertEqual(trace['summary']['replan_attempt_count'], 0)

    def test_review_feedback_requires_new_evidence_and_a_new_independent_review(self):
        decisions, reviews = [], []
        texts = [TEXT, 'Further analysis confirms the model object was used exclusively in simulation and was never measured.']
        def retrieve(task, plan, queries, pages):
            n = len(decisions) + 1
            return {'pages': [n], 'queries': plan.get('queries', []),
                    'sources': {f'page:{n}': {'text': texts[n - 1]}}}
        def model(prompt, payload, kind, limit):
            if kind == 'pan_repair_decision':
                decisions.append(deepcopy(payload))
                n = len(decisions)
                return FakeModel([edit(evidence=[{'source_id': f'page:{n}', 'quote': texts[n - 1]}])])(prompt, payload, kind, limit)
            if kind.startswith('pan_repair_challenge'):
                reviews.append(deepcopy(payload))
                return FakeModel([], approve=len(reviews) > 1)(prompt, payload, kind, limit)
            return FakeModel([])(prompt, payload, kind, limit)
        trace = self.engine(model, retrieve)
        self.assertEqual(len(decisions), 2)
        self.assertEqual(len(reviews), 2)
        self.assertTrue(decisions[1]['review_feedback'])
        self.assertEqual([t['paper_round'] for t in trace['transactions']], [2])

    def test_new_unrelated_evidence_cannot_retry_an_old_rejected_claim(self):
        rounds = []
        model = FakeModel([edit()], approve=False)
        def retrieve(*args):
            rounds.append(1)
            return {'sources': {**SOURCES, f'hit:{len(rounds)}': {
                'text': f'Independent background passage {len(rounds)} about some unrelated synthesis methods.'}}}
        trace = self.engine(model, retrieve)
        self.assertEqual(model.calls.count('pan_repair_challenge'), 1)
        self.assertTrue(any(r['reason'] == 'rejected_intent_has_no_new_support' for r in trace['rejected']))
        self.assertFalse(trace['transactions'])

    def test_supported_reference_blocks_indirect_cascades_from_either_endpoint(self):
        for target in ('samples:sample_2', 'datasets:dataset_2'):
            det = {'target_judgments': [detector(target)['target_judgments'][0],
                detector('datasets:dataset_2:sample_refs:sample_2', 'supported')['target_judgments'][0]]}
            trace = run(FakeModel([edit(target=target)]), det=det)
            result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
            self.assertEqual(result, RECORD)
            self.assertFalse(applied['applied'])

    def test_supported_dataset_fields_frozen_but_unsupported_reference_can_be_removed(self):
        det = {'target_judgments': [detector('datasets:dataset_2', 'supported')['target_judgments'][0],
            detector('datasets:dataset_2:sample_refs:sample_2')['target_judgments'][0]]}
        trace = run(FakeModel([edit('remove_relation', 'datasets:dataset_2:sample_refs:sample_2')]), det=det)
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(len(applied['applied']), 1)
        self.assertEqual(result['datasets'][1], {**RECORD['datasets'][1], 'sample_refs': []})
        self.assertEqual(result['samples'], RECORD['samples'])

    def test_supported_sample_replacement_and_split_are_immutable_even_if_forced(self):
        det = detector('samples:sample_2', 'supported')
        protected = extract._pan_repair_protected_targets(RECORD, det)
        task = {'task_id': 'samples', 'entity_group': 'samples', 'dataset_id': None, 'risks': []}
        for proposal in (edit(replacement_id='sample_1', reopen_supported=True),
                         edit(replacement_ids=['sample_1', 'sample_new_a'], reopen_supported=True)):
            decision = FakeModel([proposal])('', {'task': task}, 'pan_repair_decision', 3000)
            _, _, errors, _ = extract._pan_repair_validate_decision(decision, task, RECORD, SOURCES,
                {'samples:sample_2': 'supported'}, SCHEMA, 0, protected)
            self.assertTrue(any(e['reason'] == 'detector_supported_content_is_frozen' for e in errors))
        renamed = deepcopy(RECORD)
        renamed['samples'][1]['sample_name'] = 'changed name'
        with self.assertRaisesRegex(ValueError, 'frozen'):
            extract._pan_repair_assert_protected(renamed, protected)

    def test_final_formatting_cannot_renumber_a_supported_survivor(self):
        det = {'target_judgments': [detector('datasets:dataset_1')['target_judgments'][0],
                                  detector('datasets:dataset_2', 'supported')['target_judgments'][0]]}
        trace = run(FakeModel([edit(target='datasets:dataset_1')]), det=det)
        state = extract.AgentState(sources_dir=Path('.'), paper_id='synthetic', record=deepcopy(RECORD),
            pan_seed_record=deepcopy(RECORD), schemas={'pan': SCHEMA})
        with patch.object(extract, 'log_runtime_event'), patch.object(extract, 'note_repair_attempt'):
            extract.apply_pan_agent_refinement_patches(state, None, trace)
        state.record = extract.normalize_local_ids(state.record)
        self.assertEqual(state.record['datasets'][0]['id'], 'dataset_1')
        extract._restore_reviewed_pan_output(state)
        self.assertEqual(state.record['datasets'], [RECORD['datasets'][1]])
        self.assertFalse(extract._pan_repair_structural_issues(state.record))

    def test_old_positive_token_cap_is_rejected_before_any_request(self):
        with patch.dict(extract.os.environ, {'REPAIR_MAX_TOTAL_TOKENS': '100000'}), \
                patch.object(extract, 'call_llm_json') as model, self.assertRaisesRegex(ValueError, 'bounded paper-wide'):
            extract._pan_repair_runtime_config()
        model.assert_not_called()

    def test_missing_sample_and_link_are_discovered_in_normal_task_not_only_coverage(self):
        proposals = [edit('add_entity', 'samples:sample_new_measured_1', 'explicit_measurement',
                         entity={'sample_name': 'specimen B', 'sample_state': 'solid'}, evidence=MEASURED_EVIDENCE),
                     edit('add_relation', 'datasets:dataset_1:sample_refs:sample_new_measured_1',
                          'explicit_measurement', evidence=MEASURED_EVIDENCE)]
        model = FakeModel(proposals, verdict='supported')
        trace = run(model, sources=MEASURED_SOURCES)
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(applied['summary']['applied_patch_count'], 2)
        self.assertIn('sample_new_measured_1', result['datasets'][0]['sample_refs'])
        self.assertNotIn('pan_repair_coverage', model.calls)
        self.assertEqual(len(trace['transactions'][0]['edits']), 2)

    def test_followup_queries_survive_multiple_decision_batches(self):
        responses = [{'assessments': [], 'edits': [], 'next_retrieval': {
            'missing_fact': 'Find the measured state', 'queries': ['annealed alloy', {'bad': 'query'}], 'pages_hint': [3]}},
            {'assessments': [], 'edits': [], 'next_retrieval': {
            'missing_fact': 'Find the technique campaign', 'queries': ['diffraction methods'], 'pages_hint': [4]}}]
        merged = extract._pan_repair_merge_responses(responses)
        self.assertEqual(merged['next_retrieval']['queries'], ['annealed alloy', 'diffraction methods'])
        self.assertEqual(merged['next_retrieval']['pages_hint'], [3, 4])

    def test_expansion_searches_beyond_already_read_top_pages_and_shares_tool_cache(self):
        class Tools:
            def __init__(self): self.calls = []
            def call(self, name, paper_id, arguments):
                self.calls.append((name, deepcopy(arguments)))
                if name == 'search_pdf_text':
                    return {'hits': [{'page_number': 1, 'snippet': TEXT}]}
                if name == 'rank_pages_for_metadata':
                    return {'ranked_pages': [{'page_number': p} for p in range(1, arguments['top_k'] + 1)]}
                return {}
        state = extract.AgentState(sources_dir=Path('.'), paper_id='synthetic',
            page_texts={p: f'Measurement information on page number {p} about the specimen.' for p in range(1, 13)})
        tools, cache = Tools(), {}
        task = {'risks': [], 'entity_group': 'samples'}
        plan = {'queries': ['new state'], 'ranking_queries': ['specimen'], 'force_evidence_expansion': True}
        with patch.object(extract, 'record_tool_call'):
            output = extract._retrieve_pan_repair_task(state, None, tools, {}, cache, task,
                                                     plan, set(), set(range(1, 7)))
            before = deepcopy(tools.calls)
            extract._retrieve_pan_repair_task(state, None, tools, {}, cache, task, plan, set(), set(range(1, 7)))
        self.assertEqual(output['pages'], list(range(7, 13)))
        self.assertEqual(tools.calls, before)
        ranked = next(a for n, a in tools.calls if n == 'rank_pages_for_metadata')
        self.assertEqual(ranked['top_k'], 12)
        self.assertEqual(ranked['queries'], ['specimen', 'new state'])

    def test_supported_indirect_block_does_not_discard_independent_transaction(self):
        record = deepcopy(RECORD)
        det = {'target_judgments': [detector()['target_judgments'][0],
            detector('datasets:dataset_1:sample_refs:sample_1', 'supported')['target_judgments'][0]]}
        proposals = [edit(target='samples:sample_1', reopen_supported=True), edit()]
        trace = run(FakeModel(proposals), det=det, record=record)
        result, applied = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
        self.assertEqual(result['samples'], [record['samples'][0]])
        self.assertEqual(result['datasets'][0], record['datasets'][0])
        self.assertEqual(len(applied['applied']), 1)


class RepairProtocolReplayTests(unittest.TestCase):
    def fixture(self, proposals=None, *, sources=None, det=None, verdict='unsupported'):
        model = FakeModel(proposals if proposals is not None else [edit()], verdict=verdict)
        def ask(prompt, payload, kind, limit):
            out = model(prompt, payload, kind, limit)
            if kind.startswith('pan_repair_challenge'):
                # The real server also returns a root review object, not just an envelope.
                return out['reviews'][0]
            return out
        trace = run(ask, sources=sources, det=det)
        return trace, det or detector()

    def legacy(self, trace):
        old = deepcopy(trace)
        # v8 was a single sweep; do not pretend multi-round actions were v8 data.
        old['actions'] = [a for a in old['actions'] if a.get('paper_round', 1) == 1]
        old['model_responses'] = [e for e in old['model_responses'] if e.get('paper_round', 1) == 1]
        old['agent_trace_version'] = 'pan-agent-risk-refinement-v8'
        old['rejected'] = [{'edit_id': e['edit_id'], 'target_id': e['target_id'],
                            'reason': 'challenge_not_approved', 'review': {}}
                           for t in old['transactions'] for e in t['edits']]
        old['transactions'], old['proposed_patches'] = [], []
        for entry in old['model_responses']:
            entry.pop('review_context', None)
            entry.pop('task_id', None)
        return old

    def replay(self, trace, det=None, record=None):
        return extract.replay_pan_evidence_repair(record=record or RECORD, trace=trace,
                                                 detector=det or detector(), schema=SCHEMA)

    def test_root_review_and_envelope_have_the_same_effect(self):
        root, _ = self.fixture()
        wrapped = run(FakeModel([edit()]))
        for trace in (root, wrapped):
            result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
            self.assertEqual(len(result['samples']), 1)
            self.assertEqual(applied['summary']['applied_patch_count'], 1)
            self.assertEqual(trace['summary']['protocol_failed_task_count'], 0)
            self.assertEqual(trace['summary']['model_valid_response_count'], 3)

    def test_review_ids_must_be_explicit_unique_and_requested(self):
        checks = [{"preferred": "after"}, {"edit_id": "stale:1", "preferred": "after"},
                  {"edit_id": "samples:1", "reviews": []},
                  {"reviews": [{"edit_id": "samples:1"}, {"edit_id": "samples:1"}]}]
        for response in checks:
            with self.subTest(response=response), self.assertRaises(extract.PanRepairResponseError):
                extract._pan_repair_normalize_response(response, RECORD, ['samples:1'])
        root = {"edit_id": "samples:1", "preferred": "before", "reason": "Actually remove it."}
        self.assertEqual(extract._pan_repair_normalize_response(root, RECORD), root)
        self.assertEqual(extract._pan_repair_normalize_response(root, RECORD, ['samples:1'])['reviews'][0], root)

    def test_single_review_never_approves_missing_independent_or_dependent_peers(self):
        pairs = [[edit(target='samples:sample_1'), edit()],
                 [edit('add_entity', 'samples:sample_new_case_1', 'explicit_measurement',
                       entity={'sample_name': 'B', 'sample_state': 'solid'}, evidence=MEASURED_EVIDENCE),
                  edit('add_relation', 'datasets:dataset_1:sample_refs:sample_new_case_1',
                       'explicit_measurement', evidence=MEASURED_EVIDENCE)]]
        for dependent, proposals in enumerate(pairs):
            det = detector('datasets:dataset_1', 'insufficient_evidence') if dependent else {
                'target_judgments': [detector(p['target_id'])['target_judgments'][0] for p in proposals]}
            trace, _ = self.fixture(proposals, det=det, sources=MEASURED_SOURCES if dependent else SOURCES,
                                    verdict='supported' if dependent else 'unsupported')
            result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
            self.assertEqual(applied['summary']['applied_patch_count'], 0 if dependent else 1)
            self.assertEqual(trace['summary']['protocol_failed_task_count'], 1)
            if dependent:
                self.assertEqual(result, RECORD)

    def test_correction_drop_accepts_explicit_edit_or_target_id_but_not_guesses(self):
        previous = {'edits': [{'target_id': 'samples:sample_2', 'action': 'remove'}]}
        errors = [{'edit_id': 'samples:7', 'target_id': 'samples:sample_2', 'reason': 'invalid'}]
        for identity in ('samples:7', 'samples:sample_2', 'sample_2'):
            self.assertFalse(extract._pan_repair_patch_decision(previous, {'drop_edits': [identity]}, errors, RECORD)['edits'])
        for identity in ('samples:6', 'samples:sample_1', 'unknown'):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                extract._pan_repair_patch_decision(previous, {'drop_edits': [identity]}, errors, RECORD)
        ambiguous = errors + [{'edit_id': 'samples:7', 'target_id': 'samples:sample_1'}]
        with self.assertRaises(ValueError):
            extract._pan_repair_patch_decision(previous, {'drop_edits': ['samples:7']}, ambiguous, RECORD)
        previous['assessments'] = [{'target_id': 'samples:sample_2', 'verdict': 'supported'}]
        with self.assertRaises(ValueError):
            extract._pan_repair_patch_decision(previous, {'assessment_updates': [
                {'target_id': 'samples:sample_2', 'verdict': 'unsupported'}]}, errors, RECORD)
        with self.assertRaises(ValueError):
            extract._pan_repair_patch_decision(previous, {'assessment_updates': [
                {'target_id': 'samples:sample_2', 'verdict': 'unresolved'},
                {'target_id': 'samples:sample_2', 'verdict': 'unsupported'}]}, errors, RECORD)

    def test_budget_and_infrastructure_stops_are_not_protocol_errors(self):
        for error, counter in [('repair_total_token_budget_exhausted', 'budget_stopped_task_count'),
                               ('repair_model_request_failed', 'infrastructure_stopped_task_count')]:
            model = FakeModel([edit()])
            def ask(prompt, payload, kind, limit):
                if kind.startswith('pan_repair_challenge'):
                    raise RuntimeError(error)
                return model(prompt, payload, kind, limit)
            trace = run(ask)
            self.assertEqual(trace['summary'][counter], 1)
            self.assertEqual(trace['summary']['protocol_failed_task_count'], 0)
            self.assertEqual(trace['summary']['model_response_received_count'], 2)
            self.assertEqual(trace['summary']['model_valid_response_count'], 2)
            self.assertEqual(trace['summary']['logical_model_call_count'], 3)
            self.assertFalse(trace['transactions'])

    def test_no_evidence_and_negative_review_are_separate_failure_categories(self):
        trace = extract.run_pan_evidence_repair(record=RECORD, anchors={}, detector=detector(), schema=SCHEMA,
                    enable_coverage=False, call_model=FakeModel([]), retrieve=lambda *args: {'sources': {}})
        self.assertEqual(trace['summary']['evidence_failed_task_count'], 1)
        self.assertEqual(trace['summary']['protocol_failed_task_count'], 0)
        model = FakeModel([edit()])
        def ask(prompt, payload, kind, limit):
            if kind.startswith('pan_repair_challenge'):
                return {'edit_id': payload['edits'][0]['edit_id'], 'preferred': 'before'}
            return model(prompt, payload, kind, limit)
        trace = run(ask)
        self.assertEqual(trace['summary']['semantic_rejected_task_count'], 1)
        self.assertEqual(trace['summary']['evidence_failed_task_count'], 0)

    def test_coverage_distinguishes_response_validation_and_budget_stop(self):
        for mode in ('budget', 'empty_valid', 'invalid', 'malformed'):
            model = FakeModel([edit()])
            def ask(prompt, payload, kind, limit):
                if mode == 'budget':
                    raise RuntimeError('repair_total_token_budget_exhausted')
                if mode == 'empty_valid':
                    return {'observations': [], 'gaps': []}
                if mode == 'malformed':
                    return {}
                return {'observations': [{'sample': {}, 'dataset': {}}], 'gaps': []}
            with patch.object(extract, 'PAN_REPAIR_MAX_CORRECTIONS', 0):
                trace = extract.run_pan_evidence_repair(record=RECORD, anchors={}, detector={'target_judgments': []},
                    schema=SCHEMA, call_model=ask, retrieve=lambda *args: {'sources': SOURCES})
            summary = trace['summary']
            self.assertEqual(summary['coverage_response_received'], mode != 'budget')
            self.assertEqual(summary['coverage_validation_completed'], mode in {'empty_valid', 'invalid'})
            self.assertEqual(summary['coverage_evaluated'], mode == 'empty_valid')
            if mode == 'budget':
                self.assertEqual(summary['protocol_failed_task_count'], 0)
                self.assertEqual(summary['model_request_count'], 0)

    def test_partially_stopped_decision_batch_does_not_invent_format_errors(self):
        model = FakeModel([], verdict='unresolved', basis='unresolved')
        def ask(prompt, payload, kind, limit):
            if kind == 'pan_repair_decision' and payload['task']['risks'][0]['target_id'] == 'samples:sample_2':
                raise RuntimeError('repair_total_token_budget_exhausted')
            return model(prompt, payload, kind, limit)
        det = {'target_judgments': [detector('samples:sample_1')['target_judgments'][0], detector()['target_judgments'][0]]}
        with patch.object(extract, 'PAN_REPAIR_DECISION_BATCH', 1):
            trace = run(ask, det=det)
        self.assertEqual(trace['summary']['budget_stopped_task_count'], 1)
        self.assertEqual(trace['summary']['protocol_failed_task_count'], 0)
        self.assertFalse(trace['actions'][0]['execution']['validation_passed'])
        self.assertEqual(trace['summary']['unresolved_count'], 2)

    def test_offline_replay_recovers_only_recorded_approved_claims_without_calls(self):
        trace, det = self.fixture()
        for original in (trace, self.legacy(trace)):
            before = deepcopy(original)
            with patch.object(extract, 'call_llm_json', side_effect=AssertionError('no model')), \
                    patch.object(extract, '_retrieve_pan_repair_task', side_effect=AssertionError('no retrieval')), \
                    patch.object(extract, 'load_json', side_effect=AssertionError('no external data')):
                result, replay, applied = self.replay(original, det)
            self.assertEqual(len(result['samples']), 1)
            self.assertEqual(applied['summary']['applied_patch_count'], 1)
            self.assertEqual(replay['summary']['model_request_count'], 0)
            self.assertEqual(original, before)

    def test_offline_replay_rejects_missing_negative_foreign_and_ungrounded_reviews(self):
        trace, _ = self.fixture()
        for mode in ('missing', 'before', 'foreign', 'ungrounded', 'binding', 'claim', 'graph'):
            old = self.legacy(trace) if mode in {'binding', 'ungrounded'} else deepcopy(trace)
            entry = next(e for e in old['model_responses'] if e['call_kind'] == 'pan_repair_challenge')
            if mode == 'missing':
                old['model_responses'].remove(entry)
            elif mode == 'before':
                entry['raw_response']['preferred'] = 'before'
                entry['raw_response']['decision'] = 'reject'
                entry['raw_response']['reason'] = 'Definitely remove the simulation.'
            elif mode == 'foreign':
                entry['raw_response']['edit_id'] = 'samples:999'
            elif mode == 'ungrounded':
                old['actions'][0]['sources']['page:1']['text'] = 'Unrelated text.'
            elif mode == 'binding':
                old['rejected'].append({**old['rejected'][0], 'target_id': 'samples:sample_1'})
            elif mode == 'claim':
                entry['review_context']['edits'][0]['replacement_id'] = 'sample_1'
            else:
                entry['review_context']['comparisons'][0]['after'] = RECORD
            with self.subTest(mode=mode):
                result, replay, applied = self.replay(old)
                self.assertEqual(result, RECORD)
                self.assertEqual(applied['summary']['applied_patch_count'], 0)
                self.assertTrue(replay['unresolved'])

    def test_offline_replay_checks_seed_detector_and_protocol_identity(self):
        trace, _ = self.fixture()
        for mode in ('seed', 'detector', 'paper', 'version', 'risk'):
            old, det = deepcopy(trace), detector()
            if mode == 'seed': old['input_record']['samples'] = []
            if mode == 'detector': det['detector_report_version'] = 'other'
            if mode == 'paper': det['paper_id'] = 'other'
            if mode == 'version': old['agent_trace_version'] = 'old-unknown'
            if mode == 'risk': old['actions'][0]['risks'][0]['judgment'] = 'supported'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.replay(old, det)

    def test_offline_replay_never_partially_commits_dependent_cached_edits(self):
        proposals = [edit('add_entity', 'samples:sample_new_case_1', 'explicit_measurement',
                          entity={'sample_name': 'B', 'sample_state': 'solid'}, evidence=MEASURED_EVIDENCE),
                     edit('add_relation', 'datasets:dataset_1:sample_refs:sample_new_case_1',
                          'explicit_measurement', evidence=MEASURED_EVIDENCE)]
        det = detector('datasets:dataset_1', 'insufficient_evidence')
        trace = run(FakeModel(proposals, verdict='supported'), det=det, sources=MEASURED_SOURCES)
        self.assertEqual(len(trace['transactions'][0]['edits']), 2)
        old = self.legacy(trace)
        review = next(e for e in old['model_responses'] if e['call_kind'] == 'pan_repair_challenge')
        review['raw_response']['reviews'] = review['raw_response']['reviews'][:1]
        result, _, applied = self.replay(old, det)
        self.assertEqual(result, RECORD)
        self.assertEqual(applied['summary']['applied_patch_count'], 0)

    def test_offline_retention_root_review_does_not_change_metadata(self):
        proposal = edit('add_entity', 'samples:sample_1', 'explicit_measurement', evidence=MEASURED_EVIDENCE)
        det = detector('samples:sample_1', 'insufficient_evidence')
        trace, _ = self.fixture([proposal], det=det, verdict='supported', sources=MEASURED_SOURCES)
        self.assertEqual(trace['summary']['freeze_count'], 1)
        result, replay, applied = self.replay(self.legacy(trace), det)
        self.assertEqual(result, RECORD)
        self.assertEqual(len(replay['freeze_targets']), 1)
        self.assertEqual(applied['summary']['applied_patch_count'], 0)

    def test_replayed_correction_cannot_reuse_review_for_changed_edit(self):
        trace, _ = self.fixture()
        old = self.legacy(trace)
        action = old['actions'][0]
        action['protocol_corrections'] = [{
            'error': 'invalid correction drops', 'previous_decision': deepcopy(action['decision']),
            'validation_errors': [{'target_id': 'samples:sample_2', 'edit_id': 'samples:1'}],
            'raw_patch': {'edit_updates': [{**action['decision']['edits'][0], 'reopen_supported': True}]}}]
        result, replay, applied = self.replay(old)
        self.assertEqual(result, RECORD)
        self.assertEqual(applied['summary']['applied_patch_count'], 0)
        self.assertTrue(any(r['reason'] == 'cached_patch_changed_reviewed_claims' for r in replay['rejected']))

    def test_replay_cli_requires_new_disjoint_output_and_preserves_inputs(self):
        trace, det = self.fixture()
        trace['paper_id'] = det['paper_id'] = 'synthetic'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root/'run'
            paper_dir = run_dir/'synthetic'
            paper_dir.mkdir(parents=True)
            schema_dir = root/'sources'/'schemas'
            schema_dir.mkdir(parents=True)
            (schema_dir/'photon_and_neutron.json').write_text(json.dumps(SCHEMA), encoding='utf-8')
            for name, value in [('global_seed_prediction.json', RECORD), ('global_seed_anchor_detector.json', det),
                                ('pan_agent_risk_refinement.json', self.legacy(trace))]:
                (paper_dir/name).write_text(json.dumps(value), encoding='utf-8')
            before = {p: p.read_bytes() for p in paper_dir.iterdir()}
            for output in (run_dir, root, paper_dir/'replay'):
                with self.assertRaises(ValueError), patch.object(extract, 'SOURCES_DIR', root/'sources'):
                    extract.repair_replay_command(SimpleNamespace(run_dir=run_dir, output_dir=output, paper_id=None))
            output = root/'replay'
            with patch.object(extract, 'SOURCES_DIR', root/'sources'), redirect_stdout(StringIO()):
                extract.repair_replay_command(SimpleNamespace(run_dir=run_dir, output_dir=output, paper_id=None))
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            self.assertTrue((output/'synthetic'/'prediction.json').exists())
            self.assertEqual(json.loads((output/'manifest.json').read_text())['model_request_count'], 0)
            with self.assertRaises(ValueError):
                extract.repair_replay_command(SimpleNamespace(run_dir=run_dir, output_dir=output, paper_id=None))

    def test_downloaded_v8_traces_replay_without_gold_or_requests_when_present(self):
        paths = sorted(Path('outputs/run').glob('*/pan_agent_risk_refinement.json'))
        traces = [(p, json.loads(p.read_text(encoding='utf-8'))) for p in paths]
        traces = [(p, t) for p, t in traces if t.get('agent_trace_version') == 'pan-agent-risk-refinement-v8']
        if not traces:
            self.skipTest('Optional downloaded v8 artifacts are not present')
        schema = extract.load_schemas(Path('SOURCES'))['pan']
        for path, trace in traces:
            det = json.loads((path.parent/'global_seed_anchor_detector.json').read_text(encoding='utf-8'))
            with self.subTest(paper=path.parent.name), \
                    patch.object(extract, 'call_llm_json', side_effect=AssertionError('no model')), \
                    patch.object(extract, '_retrieve_pan_repair_task', side_effect=AssertionError('no retrieval')):
                result, replay, applied = extract.replay_pan_evidence_repair(
                    record=trace['input_record'], trace=trace, detector=det, schema=schema)
                self.assertEqual(replay['summary']['model_request_count'], 0)
                self.assertFalse(applied['skipped'])
                self.assertFalse(extract._pan_repair_structural_issues(result))


if __name__ == "__main__":
    unittest.main()
