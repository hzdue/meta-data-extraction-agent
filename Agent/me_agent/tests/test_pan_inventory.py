"""Source-only synthetic measurement inventories: no gold, PDFs, evaluator or server."""

from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import unittest

import extract
from test_pan_repair import SCHEMA, FakeModel, RECORD, detector, edit, run, admission


TEXT = "Diffraction measured the cured specimen Amber at the Meridian instrument. The untreated specimen was not used."
SOURCES = {"page:1": {"page_number": 1, "text": TEXT}}
EVIDENCE = [{"source_id": "page:1", "span_ids": ["s1"]}]
EMPTY = {"samples": [], "facilities": [], "datasets": []}


def observation(name="cured specimen Amber", facility=False, evidence=None):
    return {"sample": {"sample_name": name, "sample_state": "solid"},
            "dataset": {"dataset_name": "diffraction"},
            "facility": {"name": "Meridian instrument"} if facility else None,
            "campaign": "reported diffraction acquisition", "object_role": "measured_specimen",
            "relation_scope": "specific", "evidence": deepcopy(evidence or EVIDENCE)}


class InventoryModel:
    def __init__(self, rows=None, bindings=None, reject=None):
        self.rows = rows if rows is not None else [observation()]
        self.bindings = bindings or {"sample": "new", "dataset": "new", "facility": None}
        self.reject = reject
        self.calls = []

    def __call__(self, prompt, payload, kind, limit):
        self.calls.append((kind, deepcopy(payload)))
        if kind.startswith("pan_repair_inventory_extract"):
            assert "metadata" not in payload and "inventory_sources" not in payload
            assert "detector" not in payload and "protected_targets" not in payload
            return {"observations": deepcopy(self.rows), "gaps": []}
        if kind.startswith("pan_repair_inventory_binding"):
            if payload['task'].get('inventory_rows'):
                return {'decisions': [{'observation_id': row['observation_id'], 'verdict': 'supported',
                    'bindings': deepcopy(self.bindings), 'evidence': deepcopy(EVIDENCE)} for row in payload['task']['inventory_rows']]}
            return {"verdict": "supported", "bindings": deepcopy(self.bindings), "evidence": deepcopy(EVIDENCE)}
        if kind.startswith("pan_repair_challenge"):
            return {"reviews": [{"edit_id": claim["edit_id"], "preferred":
                "before" if self.reject and self.reject(claim) else "after",
                "admission": admission(claim["evidence"], claim["target_id"].split(":")[0]),
                "evidence": deepcopy(claim["evidence"])} for claim in payload["edits"]]}
        return FakeModel([])(prompt, payload, kind, limit)


def engine(model, record=None, det=None, retrieve=None):
    return extract.run_pan_evidence_repair(record=deepcopy(record if record is not None else EMPTY),
        anchors={}, detector=det or {"target_judgments": []}, schema=SCHEMA, call_model=model,
        retrieve=retrieve or (lambda *args: {"sources": deepcopy(SOURCES), "pages": [1]}))


class MeasurementInventoryTests(unittest.TestCase):
    def test_new_measured_state_and_edges_use_complete_compiled_transaction(self):
        model = InventoryModel([observation(facility=True)], {"sample": "new", "dataset": "new", "facility": "new"})
        with patch("builtins.open", side_effect=AssertionError("Inference must not open files")), \
                patch("io.open", side_effect=AssertionError("Inference must not open gold/evaluation/files")):
            trace = engine(model)
        result, applied = extract.apply_pan_reviewed_transactions(EMPTY, trace, SCHEMA)
        self.assertEqual(applied["summary"]["applied_patch_count"], 5)
        self.assertEqual(len(trace["transactions"]), 1)
        self.assertEqual(result["datasets"][0]["sample_refs"], [result["samples"][0]["id"]])
        self.assertEqual(result["datasets"][0]["facility_refs"], [result["facilities"][0]["id"]])
        self.assertEqual(trace["summary"]["inventory_represented_count"], 1)
        self.assertFalse(extract._pan_repair_structural_issues(result))

    def test_more_than_eight_edits_are_packetized_without_losing_rows(self):
        rows = [observation("measured specimen " + str(i)) for i in range(8)]
        trace = engine(InventoryModel(rows))
        result, applied = extract.apply_pan_reviewed_transactions(EMPTY, trace, SCHEMA)
        self.assertEqual(len(result["samples"]), 8)
        self.assertEqual(len(result["datasets"]), 1)
        self.assertEqual(len(result["datasets"][0]["sample_refs"]), 8)
        self.assertEqual(applied["summary"]["applied_patch_count"], 17)
        self.assertEqual(trace["summary"]["inventory_pending_count"], 0)
        self.assertTrue(all(len(t["edits"]) <= 5 for t in trace["transactions"]))

    def test_unprocessed_packets_survive_empty_later_scans(self):
        model = InventoryModel([observation("measured one"), observation("measured two"), observation("measured three")])
        with patch.object(extract, "PAN_REPAIR_MAX_PACKETS", 1):
            trace = engine(model)
        self.assertEqual(trace["summary"]["paper_round_count"], 3)
        self.assertEqual(trace["summary"]["inventory_represented_count"], 3)
        self.assertEqual([t["paper_round"] for t in trace["transactions"]], [1, 2, 3])
        self.assertEqual(sum(k.startswith("pan_repair_inventory_extract") for k, _ in model.calls), 1)

    def test_protected_dataset_allows_new_independent_site_edge(self):
        record = deepcopy(RECORD)
        rows = [observation("specimen A", facility=True)]
        bindings = {"sample": "samples:sample_1", "dataset": "datasets:dataset_1", "facility": "new"}
        det = {"target_judgments": [{"target_id": t, "judgment": "supported"} for t in
               ("samples:sample_1", "datasets:dataset_1", "datasets:dataset_1:sample_refs:sample_1")]}
        trace = engine(InventoryModel(rows, bindings), record, det)
        result, applied = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
        self.assertEqual(applied["summary"]["applied_patch_count"], 2)
        self.assertEqual(result["samples"], record["samples"])
        extract._pan_repair_assert_protected(result, extract._pan_repair_protected_targets(record, det))
        self.assertEqual(len(result["datasets"][0]["facility_refs"]), 1)

    def test_wrong_reused_state_blocks_all_dependent_edges_and_entities(self):
        bindings = {"sample": "samples:sample_1", "dataset": "datasets:dataset_1", "facility": "new"}
        model = InventoryModel([observation(facility=True)], bindings,
                               reject=lambda c: c["edit_id"] == "retain:samples:sample_1")
        trace = engine(model, RECORD)
        result, applied = extract.apply_pan_reviewed_transactions(RECORD, trace, SCHEMA)
        self.assertEqual(result, RECORD)
        self.assertEqual(applied["summary"]["applied_patch_count"], 0)
        self.assertEqual(trace["summary"]["inventory_pending_count"], 1)
        self.assertEqual(sum(k == "pan_repair_inventory_binding" for k, _ in model.calls), 1)

    def test_negative_link_review_cannot_leave_orphan_new_sample(self):
        model = InventoryModel(reject=lambda c: ":sample_refs:" in c["target_id"])
        trace = engine(model)
        result, applied = extract.apply_pan_reviewed_transactions(EMPTY, trace, SCHEMA)
        self.assertEqual(result, EMPTY)
        self.assertEqual(applied["summary"]["applied_patch_count"], 0)
        self.assertTrue(any(r["reason"] == "transaction_peer_not_approved" for r in trace["rejected"]))

    def test_absent_or_uncertain_endpoint_cannot_be_invented_or_partially_committed(self):
        for binding in ({"sample": None, "dataset": "new", "facility": None},
                        {"sample": "new", "dataset": "new", "facility": "facilities:facility_1"}):
            with self.subTest(binding=binding):
                trace = engine(InventoryModel(bindings=binding))
                self.assertFalse(trace["transactions"])

    def test_invalid_inventory_role_scope_and_non_schema_fields_fail_closed(self):
        for changes in ({"object_role": "simulation_only"}, {"relation_scope": "paper_cooccurrence"},
                        {"sample": {"sample_name": "guessed", "gold_label": "inference forbidden"}},
                        {"evidence": [{"source_id": "missing", "span_ids": ["s1"]}]}):
            with self.subTest(changes=changes):
                trace = engine(InventoryModel([{**observation(), **changes}]))
                self.assertFalse(trace["transactions"])
                self.assertEqual(trace["summary"]["inventory_observation_count"], 0)

    def test_complete_existing_observation_is_not_duplicated(self):
        rows = [observation("specimen A")]
        model = InventoryModel(rows, {"sample": "samples:sample_1", "dataset": "datasets:dataset_1", "facility": None})
        trace = engine(model, RECORD)
        self.assertFalse(trace["transactions"])
        self.assertEqual(trace["summary"]["inventory_represented_count"], 1)
        retained = [e for k, p in model.calls if k == "pan_repair_challenge" for e in p["edits"]]
        self.assertEqual(len(retained), 3)
        self.assertTrue(all("inventory_identity" in c for c in retained))

    def test_new_spelling_of_existing_name_is_not_permission_to_duplicate(self):
        row = observation("specimen A")
        trace = engine(InventoryModel([row], {"sample": "new", "dataset": "datasets:dataset_1", "facility": None}), RECORD)
        self.assertFalse(trace["transactions"])
        self.assertEqual(trace["summary"]["inventory_pending_count"], 1)

    def test_gap_survives_omission_and_closes_only_after_reviewed_observation(self):
        question = "Which recovered specimen was actually measured?"
        pages = {1: "The recovery procedure is described, but the measured specimen is not yet identified here.",
                 2: "Further methods refer to a measurement table on the next page.",
                 3: TEXT}
        seen = []
        base = InventoryModel()
        def retrieve(task, plan, queries, previous):
            page = len(seen) + 1
            seen.append(deepcopy(plan))
            return {"sources": {f"page:{page}": {"text": pages[page], "page_number": page}}, "pages": [page]}
        def model(prompt, payload, kind, limit):
            if kind == "pan_repair_inventory_extract":
                page = int(next(iter(payload["sources"])).split(":")[1])
                if page == 1:
                    return {"observations": [], "gaps": [{"question": question, "queries": ["recovered specimen"]}]}
                self.assertIn(question, [g["question"] for g in payload["open_gaps"]])
                if page == 2:
                    return {"observations": [], "gaps": []}
                return {"observations": [observation(evidence=[{"source_id": "page:3", "span_ids": ["s1"]}])],
                        "gaps": [], "gap_resolutions": [{"question": question, "observation_indexes": [0]}]}
            if kind == "pan_repair_inventory_binding":
                return {"verdict": "supported", "bindings": base.bindings,
                        "evidence": [{"source_id": "page:3", "span_ids": ["s1"]}]}
            return base(prompt, payload, kind, limit)
        trace = engine(model, retrieve=retrieve)
        self.assertEqual(trace["summary"]["paper_round_count"], 3)
        self.assertEqual(trace["summary"]["inventory_open_gap_count"], 0)
        self.assertIn("recovered specimen", seen[1]["queries"])
        self.assertEqual(next(iter(trace["measurement_inventory"]["gaps"].values()))["status"], "resolved")

    def test_none_is_not_a_missing_fact_but_real_unresolved_text_is_retained(self):
        for value in ("none", "None identified.", "No gaps found", "N/A", None):
            self.assertFalse(extract._pan_repair_gap_text(value))
        self.assertTrue(extract._pan_repair_gap_text("None identified for the first specimen; the second remains unresolved."))
        action = {"decision": {"next_retrieval": {"missing_fact": "None identified."}}}
        self.assertIsNone(extract._pan_repair_followup_plan({"risks": []}, action, [], {}, {"queries": set(), "pages": set()}))

    def test_full_task_history_not_just_last_pack_controls_novelty(self):
        count = []
        def retrieve(*args):
            count.append(1)
            numbers = range(1, 9) if len(count) in (1, 3) else range(9, 17)
            return {"sources": {f"page:{i}": {"text": ("An unrelated but distinct measurement context number " + str(i) + ".") * 20}
                                for i in numbers}}
        model = FakeModel([], verdict="unresolved", basis="unresolved")
        trace = extract.run_pan_evidence_repair(record=RECORD, anchors={}, detector=detector(), schema=SCHEMA,
                    enable_coverage=False, call_model=model, retrieve=retrieve)
        self.assertEqual(model.calls.count("pan_repair_decision"), 2)
        self.assertEqual(trace["actions"][-1]["stop_reason"], "no_new_evidence")

    def test_duplicate_parameter_completion_needs_grounded_judgment_and_unique_identical_peer(self):
        record = deepcopy(RECORD)
        record["samples"][1] = {**record["samples"][0], "id": "sample_2"}
        trace = run(FakeModel([edit(basis="duplicate_identity")]), record=record)
        self.assertEqual(trace["transactions"][0]["edits"][0]["replacement_id"], "sample_1")
        result, _ = extract.apply_pan_reviewed_transactions(record, trace, SCHEMA)
        self.assertEqual(result["datasets"][1]["sample_refs"], ["sample_1"])
        for mutate in ("different_fields", "ambiguous_peer", "no_duplicate_judgment"):
            changed = deepcopy(record)
            if mutate == "different_fields":
                changed["samples"][1]["sample_state"] = "liquid"
            if mutate == "ambiguous_peer":
                changed["samples"].append({**changed["samples"][0], "id": "sample_3"})
            proposal = edit(basis="duplicate_identity" if mutate != "no_duplicate_judgment" else "unresolved")
            failed = run(FakeModel([proposal]), record=changed)
            self.assertFalse(failed["transactions"])

    def test_v11_replay_binds_inventory_compilation_and_independent_identity_reviews(self):
        trace = engine(InventoryModel([observation("specimen A", facility=True)],
            {"sample": "samples:sample_1", "dataset": "datasets:dataset_1", "facility": "new"}), RECORD)
        det = {"target_judgments": []}
        result, replay, applied = extract.replay_pan_evidence_repair(record=RECORD, trace=trace, detector=det, schema=SCHEMA)
        self.assertEqual(applied["summary"]["applied_patch_count"], 2)
        self.assertEqual(replay["summary"]["model_request_count"], 0)
        self.assertEqual(len(result["facilities"]), 1)
        for mutation in ("row", "binding", "extraction", "retention_review", "claim_context"):
            changed = deepcopy(trace)
            action = next(a for a in changed["actions"] if a.get("inventory_row"))
            if mutation == "row":
                action["inventory_row"]["sample"]["sample_name"] = "another physical object"
            elif mutation == "binding":
                action["inventory_response"]["bindings"]["sample"] = "samples:sample_2"
            elif mutation == "extraction":
                changed["measurement_inventory"]["extractions"][0]["response"]["observations"] = []
            else:
                entries = [e for e in changed["model_responses"] if e["call_kind"] == "pan_repair_challenge"]
                for e in entries:
                    if mutation == "retention_review":
                        e["raw_response"]["reviews"] = [r for r in e["raw_response"]["reviews"]
                            if r["edit_id"] != "retain:samples:sample_1"]
                    else:
                        e["review_context"]["edits"][0]["inventory_identity"]["campaign"] = "not the reviewed campaign"
            try:
                output, _, checked = extract.replay_pan_evidence_repair(record=RECORD, trace=changed, detector=det, schema=SCHEMA)
            except ValueError:
                continue
            self.assertEqual(output, RECORD, mutation)
            self.assertEqual(checked["summary"]["applied_patch_count"], 0, mutation)

    def test_exact_v10_round_cache_remains_replayable(self):
        # Reconstruct the legacy graph framing, not a v18 trace with just its label changed.
        from unittest.mock import patch
        def legacy(before, after, edits):
            return {"before": extract._pan_repair_local_graph(before, edits),
                    "after": extract._pan_repair_local_graph(after, edits)}
        with patch.object(extract, '_pan_repair_graph_comparison', legacy):
            trace = run(FakeModel([edit()]))
        trace["agent_trace_version"] = "pan-agent-risk-refinement-v10"
        result, replay, applied = extract.replay_pan_evidence_repair(record=RECORD, trace=trace, detector=detector(), schema=SCHEMA)
        self.assertEqual(applied["summary"]["applied_patch_count"], 1)
        self.assertEqual(replay["replay_provenance"]["original_agent_version"], "pan-agent-risk-refinement-v10")
        self.assertEqual(len(result["samples"]), 1)

    def test_source_discovery_prefers_methods_results_tables_not_seed_names(self):
        pages = {1: "Abstract\nGeneric overview of a study without any experimental object names.",
                 2: "2 Experimental methods\nSamples were prepared and diffraction data were collected.",
                 3: "3 Results\nMeasurements of cured specimens are shown in the diffraction data.",
                 4: "Table 1. Measured specimen identities and their reference states.",
                 5: "References\nA long unrelated citation listing.",
                 6: "Generic discussion of possible future work."}
        class Executor:
            calls = []
            def call(self, name, paper_id, args):
                self.calls.append((name, args))
                return {"hits": [], "ranked_pages": []}
        executor = Executor()
        task = {"task_id": "coverage", "coverage": True, "inventory_discovery": True, "risks": []}
        with patch.object(extract, "record_tool_call"), patch.object(extract, "record_tool_error"):
            result = extract._retrieve_pan_repair_task(SimpleNamespace(page_texts=pages, paper_id="synthetic"),
                Path("unused"), executor, {}, {}, task, {"document_structure_first": True}, set(), set())
        self.assertEqual(set(result["pages"]), {2, 3, 4})
        self.assertTrue(all("specimen Amber" not in str(args) for _, args in executor.calls))
        self.assertEqual(len(result["sources"]), 3)

    def test_new_unrelated_page_cannot_reopen_rejected_observation(self):
        model = InventoryModel(reject=lambda c: c["operation"] == "add_relation")
        scans = []
        def retrieve(*args):
            scans.append(1)
            if len(scans) == 1:
                return {"sources": SOURCES, "pages": [1]}
            return {"sources": {"page:2": {"text": "The appendix provides unrelated computational implementation details."}}, "pages": [2]}
        def call(prompt, payload, kind, limit):
            if kind == "pan_repair_inventory_extract" and "page:2" in payload["sources"]:
                return {"observations": [], "gaps": []}
            return model(prompt, payload, kind, limit)
        trace = engine(call, retrieve=retrieve)
        self.assertFalse(trace["transactions"])
        self.assertEqual(sum(k == "pan_repair_inventory_binding" for k, _ in model.calls), 1)

    def test_new_relevant_evidence_reopens_unresolved_identity_within_three_rounds(self):
        scans, bindings = [], []
        def retrieve(*args):
            scans.append(1)
            page = min(len(scans), 2)
            return {"sources": {f"page:{page}": {"text": TEXT if page == 1 else
                "The recovered solid Amber is the specimen used in the reported diffraction acquisition."}}, "pages": [page]}
        base = InventoryModel()
        def call(prompt, payload, kind, limit):
            if kind == "pan_repair_inventory_extract":
                source = next(iter(payload["sources"]))
                return {"observations": [observation(evidence=[{"source_id": source, "span_ids": ["s1"]}])], "gaps": []}
            if kind == "pan_repair_inventory_binding":
                bindings.append(deepcopy(payload))
                if len(bindings) == 1:
                    return {"verdict": "unresolved", "bindings": {}, "next_retrieval": {
                        "missing_fact": "Confirm the recovered specimen identity", "queries": ["recovered solid"], "pages_hint": [2]}}
                return {"verdict": "supported", "bindings": base.bindings,
                        "evidence": [{"source_id": "page:2", "span_ids": ["s1"]}]}
            return base(prompt, payload, kind, limit)
        trace = engine(call, retrieve=retrieve)
        self.assertEqual(len(bindings), 2)
        self.assertEqual(trace["transactions"][0]["paper_round"], 2)
        self.assertEqual(trace["summary"]["inventory_represented_count"], 1)
        self.assertIn("page:2", bindings[1]["sources"])

    def test_truncated_inventory_is_not_partially_committed(self):
        count = []
        def model(prompt, payload, kind, limit):
            count.append(kind)
            raise extract.PanRepairResponseError("repair_json_truncated", "length", '{"observations":[')
        trace = engine(model)
        self.assertFalse(trace["transactions"])
        self.assertEqual(trace["summary"]["inventory_observation_count"], 0)
        self.assertLessEqual(trace["summary"]["response_recovery_count"], extract.PAN_REPAIR_MAX_RECOVERIES)
        self.assertLessEqual(len(count), 5)
        self.assertTrue(any(e.get("raw_content") for e in trace["model_responses"]))

    def test_edit_limit_is_not_misreported_as_a_missing_paper_fact(self):
        with patch.object(extract, "PAN_REPAIR_MAX_EDITS", 2):
            trace = engine(InventoryModel())
        self.assertFalse(trace["transactions"])
        row = next(iter(trace["measurement_inventory"]["observations"].values()))
        self.assertEqual(row["last_failure"], "edit_safety_limit")
        self.assertFalse(trace["measurement_inventory"]["gaps"])

    def test_binding_protocol_correction_uses_same_evidence_without_retrieval(self):
        base = InventoryModel()
        kinds = []
        def model(prompt, payload, kind, limit):
            kinds.append(kind)
            if kind == "pan_repair_inventory_binding":
                return {"verdict": "supported", "bindings": {"sample": "samples:missing", "dataset": "new"}, "evidence": EVIDENCE}
            return base(prompt, payload, kind, limit)
        trace = engine(model)
        self.assertEqual(trace["summary"]["inventory_represented_count"], 1)
        self.assertIn("pan_repair_inventory_binding_compact_retry", kinds)
        self.assertEqual(sum(k.startswith("pan_repair_inventory_binding") for k in kinds), 2)

    def test_inventory_infrastructure_failure_stops_further_model_requests(self):
        calls = []
        def model(prompt, payload, kind, limit):
            calls.append(kind)
            raise RuntimeError("repair_model_request_failed")
        trace = engine(model)
        self.assertEqual(len(calls), 1)
        self.assertFalse(trace["transactions"])
        self.assertEqual(trace["summary"]["controller_stop_reason"], "repair_model_request_failed")


if __name__ == "__main__":
    unittest.main()
