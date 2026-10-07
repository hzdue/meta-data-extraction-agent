"""Regressions for stale inputs/resources, report provenance, and invalid YAML."""
import json
from pathlib import Path

import pytest
import yaml

from meta_data_extraction import alignment as a, evaluation as e
from scripts import run_evaluation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def write_plan(path, fields=None):
    path.write_text(yaml.safe_dump({"version": "test", "scoring_plan": fields or {
        "bibliographic.title": {"metric": "exact"}
    }}, sort_keys=False), encoding="utf8")


@pytest.mark.parametrize("changed", ["unchanged", "gold", "prediction", "damaged_report"])
def test_every_invocation_recomputes_from_current_inputs(tmp_path, monkeypatch, changed):
    plan = tmp_path / "plan.yaml"
    write_plan(plan)
    gold = tmp_path / "gold" / "paper.json"
    pred = tmp_path / "experiment" / "model" / "paper.json"
    for path in (gold, pred):
        write_json(path, {"bibliographic": {"title": "same"}})
    args = dict(
        gold_dir=gold.parent, output_dir=tmp_path / "reports", papers=["paper"],
        plan_path=plan, expected_evaluator=e.build_evaluator_metadata(plan),
    )
    report_path = tmp_path / "reports" / "model" / "paper_evaluation.json"
    assert run_evaluation.run_condition(pred.parent, **args) == (1, 0)
    if changed == "damaged_report":
        report_path.write_text("invalid JSON", encoding="utf-8")
    elif changed in {"gold", "prediction"}:
        write_json(gold if changed == "gold" else pred, {"bibliographic": {"title": "different"}})
    calls = []
    original = run_evaluation.evaluate
    def track(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(run_evaluation, "evaluate", track)
    assert run_evaluation.run_condition(pred.parent, **args) == (1, 0)
    assert len(calls) == 1
    after = json.loads(report_path.read_text(encoding="utf-8"))
    assert after["fields"]["bibliographic.title"]["score"] == (0 if changed in {"gold", "prediction"} else 1)


def test_identity_sources_record_the_separate_gold_path(tmp_path):
    plan = tmp_path / "plan.yaml"
    write_plan(plan)
    gold, pred, identity = [tmp_path / name for name in ("gold.json", "pred.json", "identity.json")]
    for path in (gold, pred, identity):
        write_json(path, {})
    report = e.evaluate(gold, pred, plan, identity)
    assert all(source == {"path": str(identity)} for source in report["identity_sources"].values())


def helper_data(canonical):
    alias = [{"alias": "probe", "canonical_name": canonical}]
    return {"version": "same-version", "entities": {
        "datasets": {"aliases": alias, "generic_name_terms": [], "curated_forms": {"2405.14324": {
            "dataset_4": {"accepted_technique_forms": [canonical]}}}},
        "samples": {"aliases": alias, "curated_forms": {"2405.14324": {
            "sample_1": {"accepted_name_forms": [canonical]}}}},
        "facilities": {"aliases": alias},
        "bibliographic.funding": {"aliases": alias},
    }}


def test_helper_edits_are_visible_without_restarting(tmp_path, monkeypatch):
    helper = tmp_path / "helper.yaml"
    monkeypatch.setattr(a, "_ALIGNMENT_HELPER_PATH", helper)
    for word in ("before", "after"):
        helper.write_text(yaml.safe_dump(helper_data(word)), encoding="utf8")
        for normalizer in (a.normalize_funder, a.normalize_dataset,
                           a.normalize_sample, a.normalize_facility):
            assert normalizer("probe") == word
        assert a._facility_identity_tokens("probe") == a._facility_identity_tokens(word)
        datasets = a._prepare_dataset_identities([{"id": "dataset_4"}], "2405.14324")
        samples = a._prepare_sample_identities([{"id": "sample_1"}], "2405.14324")
        assert datasets[0]["accepted_technique_forms"] == [word]
        assert samples[0]["accepted_name_forms"] == [word]


@pytest.mark.parametrize("text, message", [
    ("version: test\nscroing_plan: {}", "unsupported keys"),
    ("version: test", "scoring_plan"),
    ("version: test\nscoring_plan: {}", "scoring_plan"),
    ("version: test\nscoring_plan:\n  title: {metric: exact}\n  title: {metric: rouge_l}", "Duplicate YAML key"),
    ("version: test\nscoring_plan:\n  title: {metric: exact, normalizre: casefold}", "unsupported keys"),
    ("version: test\nscoring_plan:\n  refs: {metric: reference, target: absent}", "reference target"),
])
def test_bad_plan_fails_before_scoring(tmp_path, text, message):
    path = tmp_path / "plan.yaml"
    path.write_text(text, encoding="utf8")
    with pytest.raises(ValueError, match=message):
        e.load_plan(path)
    with pytest.raises(ValueError, match=message):
        e.build_evaluator_metadata(path)


def test_helper_rejects_duplicate_keys(tmp_path, monkeypatch):
    path = tmp_path / "helper.yaml"
    path.write_text("version: test\nentities:\n  datasets: {}\n  datasets: {}", encoding="utf8")
    monkeypatch.setattr(a, "_ALIGNMENT_HELPER_PATH", path)
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        a.load_alignment_helper()


@pytest.mark.parametrize("option", [{"tau": 0.1}, {"key_attrs": ["dataset_name"]}])
def test_registry_matcher_rejects_ineffective_similarity_options(option):
    with pytest.raises(ValueError, match="unsupported keys"):
        e._spec_from_yaml({"metric": "entity", "matcher": "dataset_registry",
                           "attrs": {}, **option})


def test_alignment_aliases_are_not_available_as_scoring_normalizers():
    with pytest.raises(ValueError, match="unknown normalizer"):
        e._spec_from_yaml({"metric": "rouge_l", "normalizer": "facility_alias"})


def test_reference_scores_do_not_depend_on_plan_order(tmp_path):
    path = tmp_path / "plan.yaml"
    write_plan(path, {"refs": {"metric": "reference", "target": "entities"},
                      "entities": {"metric": "entity", "key_attrs": ["name"], "attrs": {}}})
    assert list(e.load_plan(path)) == ["refs", "entities"]
    gold, pred = tmp_path / "gold.json", tmp_path / "pred.json"
    write_json(gold, {"entities": [{"id": "g", "name": "same"}], "refs": ["g"]})
    write_json(pred, {"entities": [{"id": "p", "name": "same"}], "refs": ["p"]})
    assert e.evaluate(gold, pred, path)["fields"]["refs"]["f1"] == 1.0


def test_entity_attributes_can_reference_their_own_collection(tmp_path):
    path = tmp_path / "plan.yaml"
    write_plan(path, {"entities": {"metric": "entity", "key_attrs": ["name"], "attrs": {
        "refs": {"metric": "reference", "target": "entities"}}}})
    gold, pred = tmp_path / "gold.json", tmp_path / "pred.json"
    write_json(gold, {"entities": [{"id": "g", "name": "same", "refs": ["g"]}]})
    write_json(pred, {"entities": [{"id": "p", "name": "same", "refs": ["p"]}]})
    assert e.evaluate(gold, pred, path)["fields"]["entities.attrs.refs"]["f1"] == 1.0


def test_reversing_the_published_plan_preserves_all_field_scores(tmp_path):
    plan = yaml.safe_load(Path("configs/evaluation_metrics.yaml").read_text(encoding="utf-8"))
    plan["scoring_plan"] = dict(reversed(list(plan["scoring_plan"].items())))
    reversed_plan = tmp_path / "reversed.yaml"
    reversed_plan.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    gold = "data/gold/1804.07163.json"
    pred = "data/experiment/pan_gpt4o_full_0shot/1804.07163.json"
    assert e.evaluate(gold, pred, reversed_plan)["fields"] == e.evaluate(gold, pred)["fields"]


def test_entity_attributes_reject_nested_entity_metrics():
    with pytest.raises(ValueError, match="entity attributes"):
        e._spec_from_yaml({
            "metric": "entity", "key_attrs": ["name"], "attrs": {
                "children": {"metric": "entity", "key_attrs": ["name"], "attrs": {}}
            },
        })
