"""Standalone main-baseline regression tests after experiment entry removal."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import extract

class MainPolicyTests(unittest.TestCase):
    def test_candidate_priority_does_not_change_entities_or_create_links(self):
        from test_pan_fact_repair import record,SOURCES
        rec=record(); before=deepcopy(rec)
        tasks=extract._pan_measurement_tasks(rec,SOURCES)
        self.assertEqual(rec,before)
        self.assertTrue(all(c['candidate_only'] for c in tasks))
        datasets={d['id']:d for d in rec['datasets']}
        for c in tasks:
            self.assertTrue(all(c['parent_id'] not in datasets[d]['sample_refs'] for d in c['missing_dataset_ids']))

    def test_windows_ignore_references_and_preserve_literal_text(self):
        pages={1:'Table Amber measured diffraction. '*220,2:'References\nAmber diffraction'}
        for discovery in (True,False):
            windows=extract._pan_repair_evidence_windows(pages,['Amber diffraction'],discovery)
            self.assertTrue(windows)
            for _,page,start,text in windows:
                self.assertEqual(page,1)
                self.assertEqual(pages[page][start:start+len(text)],text)
                self.assertLessEqual(len(text),3000)

    def test_retrieval_caps_and_no_repeated_passages(self):
        def original(*args): return {'sources':{},'pages':[],'queries':[],'new_evidence':False}
        state=SimpleNamespace(page_texts={i:('Table '+str(i)+' Amber diffraction measured. ')*200 for i in range(1,20)})
        cache={}; seen=set()
        with patch.object(extract,'_retrieve_pan_repair_task_base',original):
            for i in range(10):
                result=extract._retrieve_pan_repair_task(state,None,None,{},cache,
                    {'inventory_discovery':i%2==0},{'queries':['Amber diffraction']},set(),set())
                keys=set(result['sources'])
                self.assertFalse(keys & seen)
                self.assertLessEqual(len(keys),2)
                seen.update(keys)
        self.assertEqual(len(seen),12)

    def test_evidence_gap_does_not_override_exclusion_or_identity(self):
        for identity in ('distinct','existing','measurement_state','uncertain'):
            for scope in ('in_scope','out_of_scope','uncertain'):
                for op in ('add_sample','add_facility','split_sample'):
                    expected=op=='add_sample' and identity=='distinct' and scope=='in_scope'
                    self.assertEqual(extract._pan_fact_extra_evidence_gap({'op':op},{'identity':identity,'scope':scope},{}),expected)

    def test_binding_only_unwraps_one_matching_row(self):
        payload={'task':{'inventory_row':{'observation_id':'a'}}}
        row={'observation_id':'a','verdict':'unresolved'}
        self.assertEqual(extract._pan_repair_normalize_binding({'decisions':[row]},payload,'pan_repair_inventory_binding'),row)
        for bad in [{'decisions':[row,row]},{'decisions':[dict(row,observation_id='b')]},{'decisions':[row],'verdict':'supported'}]:
            self.assertIs(extract._pan_repair_normalize_binding(bad,payload,'pan_repair_inventory_binding'),bad)

    def test_main_version_and_no_experiment_import(self):
        self.assertEqual(extract.PAN_REPAIR_VERSION,'pan-agent-risk-refinement-gpu3-v1')
        source=__import__('inspect').getsource(extract)
        for n in (1,2,3): self.assertNotIn('import extract_gpu'+str(n),source)

if __name__=='__main__': unittest.main()
