"""Identity, exact-pair review, compact protocol and bounded retrieval regressions."""
from copy import deepcopy
import unittest

import extract
from test_pan_fact_repair import Model, record, frozen, SOURCES, EV, partition
from test_pan_repair import SCHEMA


class CandidateModel(Model):
    def __init__(self, identity="new_specimen", review_identity="new_specimen"):
        super().__init__()
        self.identity, self.review_identity = identity, review_identity

    def __call__(self, prompt, payload, kind, limit):
        if payload["mode"] == "missing_samples":
            return {"proposals": [{"op": "add_sample", "entity": {"sample_name": "Maple", "sample_state": "solid"},
                                   "evidence": EV, "links": [{"dataset_id": "dataset_1", "evidence": EV}]}]}
        response = super().__call__(prompt, payload, kind, limit)
        if payload["mode"] == "candidate_identity":
            response["admission"]["identity"] = "distinct" if self.identity == "new_specimen" else self.identity
        if payload["mode"] == "review" and payload["context"]["proposal"]["op"] == "add_sample":
            response["admission"]["identity"] = "distinct" if self.review_identity == "new_specimen" else self.review_identity
        return response


class FactV15Tests(unittest.TestCase):
    def setUp(self):
        self.record = record()
        self.detector = frozen(self.record)
        self.sources = deepcopy(SOURCES)
        self.sources["page:1"]["text"] += " Table 2: Maple is a separately prepared reference specimen measured by diffraction."

    def run_model(self, model, retrieve=None):
        trace = extract.run_pan_evidence_repair(record=self.record, anchors={}, detector=self.detector,
            schema=SCHEMA, call_model=model, retrieve=retrieve or (lambda *args: {"sources": self.sources}),
            max_rounds=2, enable_coverage=False, enable_measurement_reconciliation=True)
        result, applied = extract.apply_pan_reviewed_transactions(self.record, trace, SCHEMA)
        self.assertFalse(applied["skipped"])
        return result, trace

    def test_alias_measurement_state_and_uncertainty_do_not_add_entities(self):
        for identity in ("existing", "measurement_state", "uncertain"):
            with self.subTest(identity=identity):
                result, trace = self.run_model(CandidateModel(identity=identity))
                self.assertEqual(result["samples"], self.record["samples"])
                self.assertTrue(any(a.get("identity_decisions") for a in trace["actions"]))
                self.assertFalse(any(t.get("proposal", {}).get("op") == "add_sample" for t in trace["transactions"]))

    def test_new_specimen_requires_independent_identity_approval(self):
        result, _ = self.run_model(CandidateModel(review_identity="uncertain"))
        self.assertEqual(result["samples"], self.record["samples"])
        result, trace = self.run_model(CandidateModel())
        self.assertEqual(len(result["samples"]), 3)
        replayed, _, _ = extract.replay_pan_evidence_repair(record=self.record, trace=trace,
            detector=self.detector, schema=SCHEMA)
        self.assertEqual(result, replayed)
        trace["model_responses"] = [r for r in trace["model_responses"] if r["call_kind"] != "pan_repair_fact_candidate_identity"]
        with self.assertRaises(ValueError):
            extract.replay_pan_evidence_repair(record=self.record, trace=trace, detector=self.detector, schema=SCHEMA)

    def test_generic_pair_claim_cannot_pass_even_if_review_says_approve(self):
        class Generic(Model):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload["mode"] == "review":
                    for pair in response["pair_checks"]:
                        pair["scope"] = "generic"
                return response
        result, trace = self.run_model(Generic())
        self.assertEqual(result, self.record)
        self.assertGreater(trace["summary"]["fact_validation_failures"].get("fact_pair_not_explicit", 0), 0)

    def test_identity_context_contains_only_parent_not_all_materials(self):
        model = Model(split="samples")
        _, trace = self.run_model(model)
        requests = [r[1] for r in model.calls if r[1]["mode"] == "sample_identity"]
        self.assertTrue(requests)
        for request in requests:
            self.assertEqual(request["metadata"], {"samples": [request["context"]["parent"]]})
        maps = [r[1] for r in model.calls if r[1]["mode"] == "map_part"]
        self.assertTrue(maps)
        self.assertTrue(all(r["metadata"] == {} for r in maps))
        self.assertTrue(all(t["proof"].get("protocol") == 19 for t in trace["transactions"]))

    def test_code_assigns_keys_and_normalizes_equivalent_declared_roles(self):
        response = partition("samples")
        for part in response["parts"]:
            part["key"] = "duplicate_key"
            part["kind"] = "prepared"
            part["entity"]["sample_name"] = "Amber"
        parts = extract._pan_fact_parts(response, "samples", self.sources)
        self.assertEqual([p["key"] for p in parts], ["part_1", "part_2"])
        self.assertEqual(len({p["entity"]["sample_name"] for p in parts}), 2)
        response["parts"][0]["kind"] = "in_situ_parameter"
        with self.assertRaises(ValueError):
            extract._pan_fact_parts(response, "samples", self.sources)

    def test_choice_menu_rejects_foreign_ids_and_raw_endpoint_guessing(self):
        choices = extract._pan_fact_choices(self.record, "samples")
        for response in ({"selected": [{"choice": "c999", "evidence": EV}]},
                         {"links": [{"endpoint_id": "dataset_1", "evidence": EV}]}):
            with self.assertRaises(ValueError):
                extract._pan_fact_selected(response, choices, self.sources)

    def test_second_review_requires_new_evidence_and_replays(self):
        class NeedEvidence(Model):
            def __call__(self, prompt, payload, kind, limit):
                response = super().__call__(prompt, payload, kind, limit)
                if payload["mode"] == "review" and not any(e["page"] == 2 for e in payload["excerpts"]):
                    response.update(decision="unresolved", reason_code="missing_pair_evidence")
                return response
        def retrieve(task, plan, *args):
            if any("Birch diffraction" in q for q in plan["queries"]):
                return {"sources": {"page:2": {"page_number": 2, "text": "Diffraction was independently acquired for Birch, as shown in Figure 3."}}}
            return {"sources": self.sources}
        model = NeedEvidence()
        result, trace = self.run_model(model, retrieve)
        self.assertIn("sample_2", result["datasets"][0]["sample_refs"])
        self.assertEqual(sum(k == "pan_repair_fact_review" for k, _ in model.calls), 2)
        replayed, _, _ = extract.replay_pan_evidence_repair(record=self.record, trace=trace,
            detector=self.detector, schema=SCHEMA)
        self.assertEqual(result, replayed)
        model = NeedEvidence()
        result, _ = self.run_model(model)
        self.assertEqual(result, self.record)
        self.assertEqual(sum(k == "pan_repair_fact_review" for k, _ in model.calls), 1)

    def test_structural_correction_once_preserves_claims_and_replays(self):
        class WrongEnvelope(Model):
            def __call__(self, prompt, payload, kind, limit):
                if payload["mode"] == "links":
                    response = super().__call__(prompt, payload, kind, limit)
                    return {"items": response["proposals"]}
                if payload["mode"] == "protocol_correction":
                    return {"proposals": payload["context"]["previous_response"]["items"]}
                return super().__call__(prompt, payload, kind, limit)
        result, trace = self.run_model(WrongEnvelope())
        self.assertIn("sample_2", result["datasets"][0]["sample_refs"])
        replayed, _, _ = extract.replay_pan_evidence_repair(record=self.record, trace=trace,
            detector=self.detector, schema=SCHEMA)
        self.assertEqual(result, replayed)

    def test_pair_scheduler_does_not_expand_missing_cartesian_product(self):
        unrelated = {"page:1": {"text": "Amber was prepared. Diffraction measured Amber.", "page_number": 1}}
        self.assertEqual(extract._pan_fact_pair_candidates(self.record, self.record["samples"][1], unrelated), [])

    def test_correction_cannot_swap_endpoint_semantics_or_add_claims(self):
        original = {"items": [{"sample_id": "s1", "dataset_id": "d1"}]}
        self.assertTrue(extract._pan_fact_lossless_correction(original,
            {"proposals": original["items"]}, "links"))
        for bad in ({"proposals": [{"sample_id": "d1", "dataset_id": "s1"}]},
                    {"proposals": original["items"], "reason": "new claim"}):
            self.assertFalse(extract._pan_fact_lossless_correction(original, bad, "links"))

    def test_missing_preparation_evidence_is_not_new_specimen(self):
        with self.assertRaises(ValueError):
            extract._pan_fact_new_identity({"identity": "new_specimen", "existing_sample_id": None,
                "preparation_evidence": [], "measurement_evidence": EV}, self.sources)

    def test_renamed_source_or_shorter_excerpt_is_not_new_evidence(self):
        for text in (self.sources["page:1"]["text"], "Maple is a separately prepared reference specimen"):
            self.assertFalse(extract._pan_repair_new_evidence(self.sources,
                {"renamed": {"text": text}}))


if __name__ == "__main__":
    unittest.main()
