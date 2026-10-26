"""Source-only synthetic graphs; no gold, paper IDs, evaluator, files or model server."""
from copy import deepcopy
from unittest.mock import patch
import unittest
from types import SimpleNamespace

import extract
from test_pan_repair import SCHEMA, admission
from test_pan_inventory import InventoryModel, SOURCES as INVENTORY_SOURCES, EMPTY


TEXT = ("Cedar Measurement Centre (CMC) acquired diffraction and spectroscopy of specimen Amber. "
        "The untreated and cured specimens were prepared separately and both were measured. "
        "Table 1 lists Amber and Birch reference specimens, both measured by diffraction. "
        "Diffraction was measured separately in transmission and reflection geometry. "
        "The institute address is not an acquisition site.")
SOURCES = {"page:1": {"page_number": 1, "text": TEXT}}
EV = ["e1"]


def record():
    return {"facilities": [{"id": "facility_1", "name": "Cedar Measurement Centre"},
                           {"id": "facility_2", "name": "CMC"}],
            "samples": [{"id": "sample_1", "sample_name": "Amber", "sample_state": "solid"},
                        {"id": "sample_2", "sample_name": "Birch", "sample_state": "solid"}],
            "datasets": [{"id": "dataset_1", "dataset_name": "diffraction",
                          "sample_refs": ["sample_1"], "facility_refs": ["facility_2"]},
                         {"id": "dataset_2", "dataset_name": "spectroscopy",
                          "sample_refs": ["sample_1", "sample_2"], "facility_refs": ["facility_2"]}]}


def merge():
    return {"op": "merge_facility", "old_id": "facility_2", "keep_id": "facility_1",
            "basis": "duplicate_identity", "evidence": EV,
            "links": [{"dataset_id": d, "evidence": EV} for d in ("dataset_1", "dataset_2")]}


def partition(group):
    labels = ["untreated", "cured"] if group == "samples" else ["transmission", "reflection"]
    return {"evidence": EV, "parts": [
        {"key": str(i), "kind": "prepared_or_recovered" if group == "samples" else "acquisition_configuration",
         "qualifier": label, "evidence": EV,
         "entity": {"sample_name": label + " Amber", "sample_state": label + " solid"} if group == "samples"
         else {"dataset_name": label + " diffraction"}}
        for i, label in enumerate(labels)]}


def frozen(rec):
    targets = [f'{g}:{x["id"]}' for g in rec for x in rec[g]]
    targets += [f'datasets:{d["id"]}:{f}:{i}' for d in rec["datasets"]
                for f in ("sample_refs", "facility_refs") for i in d[f]]
    return {"target_judgments": [{"target_id": t, "judgment": "supported"} for t in targets]}


class Model:
    def __init__(self, split=None, merge_site=False, reject=False):
        self.split, self.merge_site, self.reject = split, merge_site, reject
        self.calls = []

    def __call__(self, prompt, payload, kind, limit):
        self.calls.append((kind, deepcopy(payload)))
        mode, context = payload["mode"], payload["context"]
        if mode == "facilities":
            return {"proposals": [merge()] if self.merge_site else []}
        if mode == "missing_samples":
            return {"proposals": []}
        if mode in {"sample_identity", "dataset_identity"}:
            group = context["group"]
            if self.split == group and context["parent"]["id"] in {"sample_1", "dataset_1"}:
                return partition(group)
            return {"parts": []}
        if mode == "map_part":
            if "choices" in context:
                choices = context["choices"]
                if context["group"] == "samples":
                    return {"selected": [{"choice": c["choice"], "evidence": EV} for c in choices]}
                return {"selected": [{"choice": c["choice"], "evidence": EV} for c in choices
                                     if c["endpoint_id"] in context["parent"].get(c["field"], [])]}
            if context["group"] == "samples":
                return {"links": [{"field": "sample_refs", "endpoint_id": d["id"], "evidence": EV}
                                  for d in payload["metadata"]["datasets"] if "sample_1" in d["sample_refs"]]}
            return {"links": [{"field": f, "endpoint_id": i, "evidence": EV}
                              for f in ("sample_refs", "facility_refs") for i in context["parent"][f]]}
        if mode == "links":
            return {"proposals": [{"op": "add_link", "dataset_id": d, "evidence": EV}
                                  for d in context["candidate_dataset_ids"]]}
        if mode == "review":
            return {"decision": "reject" if self.reject else "approve", "evidence": EV, "reason": "source",
                    "entity_decision": "reject" if self.reject else "approve",
                    "admission": admission(EV, "facilities" if context["proposal"]["op"] == "add_facility" else "samples"),
                    "admissions": {p["key"]: admission(EV, "samples" if context["proposal"]["op"] == "split_sample" else "datasets") for p in context["proposal"].get("parts", [])},
                    "facility_identity": {"relation": "same_identity", "old_kind": "facility", "keep_kind": "facility", "evidence": EV},
                    "pair_checks": [{"pair_id": p["pair_id"], "scope": "explicit_pair", "evidence": EV}
                                    for p in context.get("pairs", [])], "identity": "new_specimen",
                    "existing_sample_id": None, "preparation_evidence": EV, "measurement_evidence": EV}
        if mode == "candidate_identity":
            return {"admission": admission(EV), "object_identity": {
                "relation": "distinct_object", "existing_sample_id": None, "evidence": EV}}
        raise AssertionError(kind)


class FactRepairTests(unittest.TestCase):
    def setUp(self):
        self.rec = record()
        self.det = frozen(self.rec)
        self.protected = extract._pan_repair_protected_targets(self.rec, self.det)

    def engine(self, model, rec=None):
        rec = deepcopy(rec or self.rec)
        with patch("builtins.open", side_effect=AssertionError("no file access")), \
             patch("io.open", side_effect=AssertionError("no annotation access")):
            return extract.run_pan_evidence_repair(
                record=rec, anchors={}, detector=frozen(rec), schema=SCHEMA,
                call_model=model, retrieve=lambda *args: {"sources": deepcopy(SOURCES), "pages": [1]},
                enable_coverage=False, enable_measurement_reconciliation=True, max_rounds=2)

    def test_merge_redirects_all_protected_edges_without_mutating_detector(self):
        before = deepcopy(self.rec)
        trace = self.engine(Model(merge_site=True))
        output, applied = extract.apply_pan_reviewed_transactions(self.rec, trace, SCHEMA)
        self.assertFalse(applied["skipped"])
        self.assertEqual(output["facilities"], [self.rec["facilities"][0]])
        self.assertTrue(all(d["facility_refs"] == ["facility_1"] for d in output["datasets"]))
        self.assertEqual(output["samples"], before["samples"])
        self.assertEqual(self.rec, before)
        self.assertEqual(trace["detector_judgments"], self.det["target_judgments"])
        extract._pan_repair_assert_protected(output, applied["effective_protected_targets"])

    def test_merge_refuses_incomplete_mapping_and_unknown_endpoints(self):
        proposal = merge()
        proposal["links"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete_facility"):
            extract._pan_fact_compile(self.rec, proposal, SOURCES, SCHEMA)
        for field in ("old_id", "keep_id"):
            proposal = merge()
            proposal[field] = "facility_999"
            with self.subTest(field=field), self.assertRaises(ValueError):
                extract._pan_fact_compile(self.rec, proposal, SOURCES, SCHEMA)

    def test_sample_partition_full_controller_and_replay(self):
        trace = self.engine(Model(split="samples"))
        output, applied = extract.apply_pan_reviewed_transactions(self.rec, trace, SCHEMA)
        self.assertFalse(applied["skipped"])
        self.assertEqual(trace["summary"]["fact_applied_by_operation"].get("split_sample"), 1)
        self.assertNotIn("sample_1", [s["id"] for s in output["samples"]])
        self.assertEqual(len(output["samples"]), 3)
        replayed, _, _ = extract.replay_pan_evidence_repair(
            record=self.rec, trace=trace, detector=self.det, schema=SCHEMA)
        self.assertEqual(replayed, output)

    def test_dataset_partition_preserves_facilities_and_samples(self):
        trace = self.engine(Model(split="datasets"))
        output, applied = extract.apply_pan_reviewed_transactions(self.rec, trace, SCHEMA)
        self.assertFalse(applied["skipped"])
        self.assertEqual(trace["summary"]["fact_applied_by_operation"].get("split_dataset"), 1)
        self.assertEqual(output["samples"], self.rec["samples"])
        self.assertEqual(output["facilities"], self.rec["facilities"])
        self.assertNotIn("dataset_1", [d["id"] for d in output["datasets"]])
        replayed, _, _ = extract.replay_pan_evidence_repair(
            record=self.rec, trace=trace, detector=self.det, schema=SCHEMA)
        self.assertEqual(output, replayed)

    def test_rejected_review_does_not_change_any_protected_value(self):
        trace = self.engine(Model(merge_site=True, split="samples", reject=True))
        output, _ = extract.apply_pan_reviewed_transactions(self.rec, trace, SCHEMA)
        self.assertEqual(output, self.rec)
        self.assertFalse(trace["transactions"])

    def test_tampered_review_provenance_rejected_by_apply_and_replay(self):
        trace = self.engine(Model(merge_site=True))
        trace["model_responses"] = [r for r in trace["model_responses"] if r["call_kind"] != "pan_repair_fact_review"]
        output, applied = extract.apply_pan_reviewed_transactions(self.rec, trace, SCHEMA)
        self.assertEqual(output, self.rec)
        self.assertTrue(applied["skipped"])
        with self.assertRaises(ValueError):
            extract.replay_pan_evidence_repair(record=self.rec, trace=trace, detector=self.det, schema=SCHEMA)

    def test_evidence_ids_must_exist_and_missing_evidence_is_not_repaired(self):
        for evidence in ([], ["e999"], ["e1", "e1"], "e1", [{"quote": TEXT}]):
            proposal = merge()
            proposal["evidence"] = evidence
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                extract._pan_fact_compile(self.rec, proposal, SOURCES, SCHEMA)

    def test_unsupported_operation_cannot_enter_through_discovery(self):
        proof = {"mode": "missing_samples", "context": {}, "index": 0,
                 "response": {"proposals": [merge()]}}
        with self.assertRaisesRegex(ValueError, "outside_task"):
            extract._pan_fact_derive(proof)

    def test_missing_sample_requires_explicit_measurement_and_no_invented_graph_fields(self):
        proposal = {"op": "add_sample", "entity": {"sample_name": "Cedar", "sample_state": "solid"},
                    "evidence": EV, "links": [{"dataset_id": "dataset_1", "evidence": EV}]}
        output, _, _, _ = extract._pan_fact_compile(self.rec, proposal, SOURCES, SCHEMA)
        self.assertEqual(len(output["samples"]), 3)
        proposal["entity"]["id"] = "sample_999"
        with self.assertRaises(ValueError):
            extract._pan_fact_compile(self.rec, proposal, SOURCES, SCHEMA)
        proposal["entity"].pop("id")
        proposal["links"] = []
        with self.assertRaises(ValueError):
            extract._pan_fact_compile(self.rec, proposal, SOURCES, SCHEMA)

    def test_dataset_split_refuses_lost_reference_and_unmeasured_part(self):
        proposal = {"op": "split_dataset", "parent_id": "dataset_1", **partition("datasets")}
        for part in proposal["parts"]:
            part["links"] = [{"field": "sample_refs", "endpoint_id": "sample_1", "evidence": EV}]
        with self.assertRaisesRegex(ValueError, "incomplete_dataset"):
            extract._pan_fact_compile(self.rec, proposal, SOURCES, SCHEMA)
        for part in proposal["parts"]:
            part["links"] = [{"field": "facility_refs", "endpoint_id": "facility_2", "evidence": EV}]
        with self.assertRaisesRegex(ValueError, "without_measured_sample"):
            extract._pan_fact_compile(self.rec, proposal, SOURCES, SCHEMA)

    def test_in_situ_parameter_and_duplicate_partition_are_rejected(self):
        proposal = {"op": "split_sample", "parent_id": "sample_1", **partition("samples")}
        for part in proposal["parts"]:
            part["links"] = [{"field": "sample_refs", "endpoint_id": d, "evidence": EV}
                             for d in ("dataset_1", "dataset_2")]
        proposal["parts"][0]["kind"] = "in_situ_parameter"
        with self.assertRaises(ValueError):
            extract._pan_fact_compile(self.rec, proposal, SOURCES, SCHEMA)
        proposal["parts"][0]["kind"] = "prepared_or_recovered"
        proposal["parts"][1]["key"] = proposal["parts"][0]["key"]
        with self.assertRaises(ValueError):
            extract._pan_fact_compile(self.rec, proposal, SOURCES, SCHEMA)

    def test_no_automatic_cartesian_link_copy(self):
        proposal = {"op": "split_sample", "parent_id": "sample_1", **partition("samples")}
        for i, part in enumerate(proposal["parts"], 1):
            part["links"] = [{"field": "sample_refs", "endpoint_id": "dataset_" + str(i), "evidence": EV}]
        output, _, _, _ = extract._pan_fact_compile(self.rec, proposal, SOURCES, SCHEMA)
        first = set(output["datasets"][0]["sample_refs"])
        second = set(output["datasets"][1]["sample_refs"])
        self.assertEqual(len(first), 1)
        self.assertFalse(first & second)

    def test_final_restore_uses_evolved_protection_and_keeps_ids(self):
        trace = self.engine(Model(merge_site=True, split="samples"))
        output, applied = extract.apply_pan_reviewed_transactions(self.rec, trace, SCHEMA)
        state = SimpleNamespace(record=deepcopy(self.rec), pan_repair_status={
            "version": extract.PAN_REPAIR_VERSION, "completed": True,
            "reviewed_pan_record": deepcopy(output), "protected_targets": applied["effective_protected_targets"]})
        extract._restore_reviewed_pan_output(state)
        self.assertEqual(state.record, output)

    def test_mapping_response_tampering_is_not_replayable(self):
        trace = self.engine(Model(split="samples"))
        entry = next(r for r in trace["model_responses"] if r["call_kind"] == "pan_repair_fact_map_part")
        entry["raw_response"]["links"] = []
        with self.assertRaises(ValueError):
            extract.replay_pan_evidence_repair(record=self.rec, trace=trace, detector=self.det, schema=SCHEMA)

    def test_fact_queue_does_not_trigger_second_whole_inventory_scan(self):
        inventory = InventoryModel()
        def model(prompt, payload, kind, limit):
            if kind.startswith("pan_repair_fact_"):
                return {"parts": []} if kind.endswith("identity") else {"proposals": []}
            return inventory(prompt, payload, kind, limit)
        trace = extract.run_pan_evidence_repair(record=EMPTY, anchors={}, detector={"target_judgments": []},
            schema=SCHEMA, call_model=model, retrieve=lambda *args: {"sources": INVENTORY_SOURCES, "pages": [1]},
            max_rounds=2, enable_measurement_reconciliation=True)
        self.assertTrue(trace["measurement_inventory"]["extractions"])
        self.assertEqual({e["paper_round"] for e in trace["measurement_inventory"]["extractions"]}, {1})

    def test_independent_bad_link_does_not_block_valid_link(self):
        class Partial(Model):
            def __call__(self, prompt, payload, kind, limit):
                if payload["mode"] == "links":
                    return {"proposals": ["malformed row", {"op": "add_link", "dataset_id": "invented", "evidence": EV},
                                          {"op": "add_link", "dataset_id": "dataset_1", "evidence": EV}]}
                return super().__call__(prompt, payload, kind, limit)
        trace = self.engine(Partial())
        output, applied = extract.apply_pan_reviewed_transactions(self.rec, trace, SCHEMA)
        self.assertFalse(applied["skipped"])
        self.assertEqual(set(output["datasets"][0]["sample_refs"]), {"sample_1", "sample_2"})

    def test_shared_call_bound_and_no_repeated_voting(self):
        with patch.object(extract, "PAN_REPAIR_FACT_CALLS", 3):
            model = Model(merge_site=True)
            trace = self.engine(model)
        self.assertLessEqual(len(model.calls), 3)
        self.assertGreater(trace["summary"]["measurement_deferred_count"], 0)

    def test_renaming_and_order_do_not_change_compiler_semantics(self):
        renamed = deepcopy(self.rec)
        renamed["facilities"][0]["name"] = "Larch Centre"
        renamed["facilities"][1]["name"] = "LC"
        renamed["datasets"].reverse()
        renamed["samples"].reverse()
        renamed_sources = deepcopy(SOURCES)
        renamed_sources["page:1"]["text"] = TEXT.replace("Cedar Measurement Centre (CMC)", "Larch Centre (LC)")
        output, _, _, _ = extract._pan_fact_compile(renamed, merge(), renamed_sources, SCHEMA)
        self.assertEqual(len(output["facilities"]), 1)
        self.assertEqual(output["samples"], renamed["samples"])
        self.assertTrue(all(d["facility_refs"] == ["facility_1"] for d in output["datasets"]))


if __name__ == "__main__":
    unittest.main()
