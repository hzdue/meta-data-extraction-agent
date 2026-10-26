"""Synthetic measurement graph regressions; no annotation or remote inference."""
from copy import deepcopy
from unittest.mock import patch
import unittest

import extract
from test_pan_repair import SCHEMA, RECORD, FakeModel, detector
from test_pan_inventory import observation, SOURCES as INVENTORY_SOURCES, InventoryModel, engine, EMPTY


TEXT = ("Specimen A was prepared in untreated and cured states. "
        "Diffraction measured untreated specimen A and cured specimen A separately.")
SOURCES = {"page:1": {"page_number": 1, "text": TEXT}}
EV = [{"source_id": "page:1", "quote": TEXT}]
CHECKS = dict.fromkeys(("identity", "campaign", "schema", "complete_remapping",
                       "prepared_not_in_situ", "no_cooccurrence_only"), True)


def facts():
    return {"decision": "refine", "partition_evidence": deepcopy(EV),
            "states": [{"key": label, "state_kind": "prepared_or_recovered", "state_label": label,
                        "sample": {"sample_name": label + " specimen A", "sample_state": "solid",
                                   "sample_description": label + " specimen A"},
                        "identity_evidence": deepcopy(EV)} for label in ("untreated", "cured")],
            "links": [{"dataset_id": "dataset_1", "state_key": label,
                       **{k: deepcopy(EV) for k in ("identity_evidence", "measurement_evidence", "link_evidence")}}
                      for label in ("untreated", "cured")]}


def review(response, request_id):
    return {"request_id": request_id, "decision": "approve", "checks": deepcopy(CHECKS),
            "partition_evidence": deepcopy(EV), "states": deepcopy(response["states"]),
            "links": deepcopy(response["links"])}


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.schema = deepcopy(SCHEMA)
        self.schema["samples"]["item_schema"]["sample_description"] = {"type": "string|null"}
        self.record = deepcopy(RECORD)
        self.detector = detector("samples:sample_1", "supported")
        self.detector["target_judgments"].append(
            {"target_id": "datasets:dataset_1:sample_refs:sample_1", "judgment": "supported"})
        self.protected = extract._pan_repair_protected_targets(self.record, self.detector)

    def transaction(self, response=None):
        response = response or facts()
        candidate, edits, preview = extract._pan_measurement_compile(
            self.record, "sample_1", response, SOURCES, self.schema)
        rid = extract._pan_measurement_request_id(self.record, "sample_1", response, SOURCES)
        return {"kind": "measurement_reconciliation", "parent_id": "sample_1",
                "before": deepcopy(self.record), "fact_table": response, "sources": SOURCES,
                "request_id": rid, "review": review(response, rid), "edits": edits, "preview": preview}

    def test_atomic_refinement_preserves_unrelated_graph_and_input(self):
        before = deepcopy(self.record)
        with patch("builtins.open", side_effect=AssertionError("No files in inference")):
            output, _, protected = extract._pan_measurement_apply(
                self.record, self.transaction(), self.schema, self.protected)
        self.assertEqual(self.record, before)
        self.assertEqual(output["datasets"][1], before["datasets"][1])
        self.assertEqual(len(output["datasets"][0]["sample_refs"]), 2)
        self.assertNotIn("sample_1", output["datasets"][0]["sample_refs"])
        self.assertNotIn("samples:sample_1", protected)
        extract._pan_repair_assert_protected(output, protected)
        self.assertEqual(self.detector["target_judgments"][0]["judgment"], "supported")

    def test_incomplete_partition_or_evidence_rejected(self):
        for field in ("links", "partition_evidence"):
            response = facts()
            response[field] = []
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.transaction(response)
        for field in ("identity_evidence", "measurement_evidence", "link_evidence"):
            response = facts()
            response["links"][0][field] = []
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.transaction(response)

    def test_in_situ_duplicate_state_unknown_endpoint_rejected(self):
        variants = []
        response = facts()
        response["states"][0]["state_kind"] = "in_situ"
        variants.append(response)
        response = facts()
        response["states"][1]["state_label"] = "untreated"
        variants.append(response)
        response = facts()
        response["links"][0]["dataset_id"] = "invented"
        variants.append(response)
        response = facts()
        response["links"].append(deepcopy(response["links"][0]))
        variants.append(response)
        for response in variants:
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.transaction(response)

    def test_independent_review_must_cover_all_states_and_links(self):
        for field in ("states", "links"):
            tx = self.transaction()
            tx["review"][field].pop()
            with self.subTest(field=field), self.assertRaises(ValueError):
                extract._pan_measurement_apply(self.record, tx, self.schema, self.protected)
        for check in CHECKS:
            tx = self.transaction()
            tx["review"]["checks"][check] = False
            with self.subTest(check=check), self.assertRaises(ValueError):
                extract._pan_measurement_apply(self.record, tx, self.schema, self.protected)

    def test_changed_context_or_approval_rejected(self):
        for key in ("request_id", "before"):
            tx = self.transaction()
            tx[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                extract._pan_measurement_apply(self.record, tx, self.schema, self.protected)

    def test_retain_adds_only_missing_edge(self):
        response = {"decision": "retain", "states": [], "links": [facts()["links"][0]]}
        response["links"][0].update(dataset_id="dataset_2", state_key="current")
        output, _, _ = extract._pan_measurement_apply(
            self.record, self.transaction(response), self.schema, self.protected)
        self.assertEqual(output["samples"], self.record["samples"])
        self.assertEqual(set(output["datasets"][1]["sample_refs"]), {"sample_1", "sample_2"})

    def test_full_controller_replay_and_tamper_rejection(self):
        def model(prompt, payload, kind, limit):
            if kind == "pan_repair_measurement_table":
                return facts() if payload["parent"]["id"] == "sample_1" else {"decision": "unresolved"}
            if kind == "pan_repair_measurement_review":
                return review(payload["fact_table"], payload["request_id"])
            return FakeModel([])(prompt, payload, kind, limit)
        # Preserve the v13 protocol as a replay compatibility fixture. New protocol has separate tests.
        with patch.object(extract, '_run_pan_fact_reconciliation', extract._run_pan_measurement_reconciliation):
            trace = extract.run_pan_evidence_repair(
                record=self.record, anchors={}, detector=self.detector, schema=self.schema,
                call_model=model, retrieve=lambda *a: {"sources": deepcopy(SOURCES), "pages": [1]},
                enable_coverage=False, max_rounds=2, enable_measurement_reconciliation=True)
        self.assertEqual(trace["summary"]["measurement_applied_count"], 1)
        output, _ = extract.apply_pan_reviewed_transactions(self.record, trace, self.schema)
        trace['agent_trace_version'] = 'pan-agent-risk-refinement-v13'
        replayed, _, _ = extract.replay_pan_evidence_repair(
            record=self.record, trace=trace, detector=self.detector, schema=self.schema)
        self.assertEqual(output, replayed)
        damaged = deepcopy(trace)
        damaged["model_responses"] = [r for r in damaged["model_responses"]
                                    if r["call_kind"] != "pan_repair_measurement_review"]
        with self.assertRaises(ValueError):
            extract.replay_pan_evidence_repair(
                record=self.record, trace=damaged, detector=self.detector, schema=self.schema)

    def test_runtime_defaults_and_round_ceiling(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(extract._pan_repair_runtime_config()["max_paper_rounds"], 2)
            self.assertTrue(extract._pan_repair_runtime_config()["measurement_reconciliation"])
        with patch.dict("os.environ", {"REPAIR_MAX_ROUNDS": "4"}), self.assertRaises(ValueError):
            extract._pan_repair_runtime_config()

    def test_partial_binding_keeps_valid_keyed_rows(self):
        rows, errors = extract._pan_repair_inventory_rows(
            {"observations": [observation(), observation("another specimen")]},
            INVENTORY_SOURCES, self.schema)
        self.assertFalse(errors)
        valid_row = {"observation_id": rows[0]["observation_id"], "verdict": "unresolved"}
        valid, errors = extract._pan_repair_binding_batch(
            {"decisions": [valid_row, {"observation_id": []}]}, rows, self.record, INVENTORY_SOURCES)
        self.assertEqual(valid, [valid_row])
        self.assertTrue(errors)
        valid, _ = extract._pan_repair_binding_batch(
            {"decisions": [valid_row, valid_row]}, rows, self.record, INVENTORY_SOURCES)
        self.assertFalse(valid)

    def test_batch_protocol_failure_recovers_individual_rows_and_replays(self):
        class BrokenBatch(InventoryModel):
            def __call__(self, prompt, payload, kind, limit):
                if kind == "pan_repair_inventory_binding_batch":
                    self.calls.append((kind, deepcopy(payload)))
                    return {"wrong_envelope": []}
                return super().__call__(prompt, payload, kind, limit)
        model = BrokenBatch([observation(), observation("another specimen")])
        trace = engine(model)
        recoveries = [r for r in trace["model_responses"]
                      if r["call_kind"] == "pan_repair_inventory_binding_single_recovery"]
        self.assertEqual(len(recoveries), 2)
        self.assertEqual(len({r["task_id"] for r in recoveries}), 2)
        output, applied = extract.apply_pan_reviewed_transactions(EMPTY, trace, SCHEMA)
        self.assertFalse(applied["skipped"])
        replayed, _, _ = extract.replay_pan_evidence_repair(
            record=EMPTY, trace=trace, detector={"target_judgments": []}, schema=SCHEMA)
        self.assertEqual(output, replayed)


if __name__ == "__main__":
    unittest.main()
