"""Synthetic M06 model judgments; no measured accuracy or expert-review claims."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from extraction_fixtures import synthetic_bundle, synthetic_record
from meta_data_extraction.agent.contracts import digest, snapshot
from meta_data_extraction.agent.evidence import evidence_ref, sha256
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.agent.providers import ChatAIProvider, LiveRunSpec, LiveSource, MockProvider, ProviderResult
from meta_data_extraction.agent.support_checks import (
    SupportReport, SupportTarget, check_support, diagnose_reports,
    interpret_support_response, render_support_prompt, verify_report,
)


@pytest.fixture
def support_case(workspace):
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "schemas/prompt_template.md", workspace / "schemas/prompt_template.md")
    profile = load_profile(workspace)
    text = ("Invented fixture. Sample A and Sample B share CaCO3 but are distinct. "
            "Sample A was prepared at 500 K and measured at 300 K in Dataset A at Synthetic Beam Source. "
            "Sample B was measured at 400 K in Dataset B.\n")
    bundle = synthetic_bundle(text)
    record = synthetic_record(profile)
    record["datasets"][0]["sample_refs"] = ["sample_2"]
    record["datasets"][0]["dataset_description"] = "Sample A measured at 500 K"
    state = snapshot(record, bundle, profile)
    ref = evidence_ref(bundle, bundle.regions[0].region_id)
    targets = (SupportTarget(path="/datasets/0/dataset_description", evidence=(ref,)),
               SupportTarget(path="/datasets/0/sample_refs/0", evidence=(ref,)))
    return workspace, profile, bundle, state, targets


def wire(targets, *, verdict="contradicted", diagnosis="preparation_acquisition_confusion", indices=(0,)):
    return {"judgments": [dict(path=t.path, verdict=verdict, diagnosis=diagnosis,
                              summary="The cited statement assigns 500 K to preparation, not acquisition.",
                              evidence_indices=list(indices)) for t in targets]}


def run(case, data=None, *, raw_retention="retain", finish="stop", **fields):
    root, profile, bundle, record, targets = case
    if data is None:
        data = wire(targets)
    raw = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    provider = MockProvider([ProviderResult(status="ok", content=raw, finish_reason=finish, **fields)])
    report = check_support(root, profile, record, bundle, targets, provider, raw_retention=raw_retention)
    return report, provider, raw


def rehash(report, update):
    data = report.model_dump(mode="json")
    data.update(update)
    data["report_id"] = "support:" + digest({key: value for key, value in data.items() if key != "report_id"})
    return SupportReport.model_validate(data)


def test_bound_identity_sensitive_prompt_and_nonmutation(support_case):
    root, profile, bundle, record, targets = support_case
    original = record.canonical_json
    result, provider, raw = run(support_case)
    assert result.status == "succeeded" and result.parse_status == "valid"
    assert len(provider.requests) == 1
    assert result.coverage_status == "selected_assertions_only"
    assert result.judgment_status == "automated_unreviewed"
    assert all(j.verdict == "contradicted" and j.approval_status == "unreviewed" for j in result.judgments)
    assert all(j.provenance == "mock_model_judgment" for j in result.judgments)
    assert result.judgments[0].role == "measurement_acquisition"
    assert result.judgments[1].role == "attachment"
    target = next(e for e in record.entities if e.local_id == "sample_2")
    assert result.judgments[1].target_handles == (target.handle,)
    prompt = provider.requests[0]
    assert prompt == render_support_prompt(profile, record, bundle, targets)
    payload = json.loads(prompt.split("SUPPLIED_DATA_JSON\n")[1])
    linked = payload["assertions"][1]["linked_entities"]
    assert any(e["record"].get("sample_name") == "Sample B" and e["record"].get("chemical_composition") == "CaCO3" for e in linked)
    assert "Sample A" in prompt and "Sample B" in prompt
    assert "preparation temperature is not an acquisition temperature" in prompt
    assert result.raw_output == raw and "raw_output" not in result.model_dump()
    assert result.provider.input_tokens is result.provider.output_tokens is None
    assert result.provider.response_sha256 == sha256(raw.encode())
    assert record.canonical_json == original
    verify_report(result, profile, record, bundle, root=root)


@pytest.mark.parametrize("verdict,diagnosis", [
    ("supported", "supported"), ("contradicted", "unsupported_value"),
    ("contradicted", "wrong_sample"), ("contradicted", "wrong_dataset"),
    ("contradicted", "preparation_acquisition_confusion"), ("contradicted", "context_conflict"),
    ("insufficient_evidence", "missing_evidence"), ("insufficient_evidence", "ambiguous_expression"),
    ("insufficient_evidence", "context_conflict"),
])
def test_explicit_verdict_states(support_case, verdict, diagnosis):
    _, profile, bundle, record, targets = support_case
    result, _, _ = run(support_case, wire(targets, verdict=verdict, diagnosis=diagnosis))
    assert result.status == "succeeded"
    assert {j.verdict for j in result.judgments} == {verdict}
    verify_report(result, profile, record, bundle)


@pytest.mark.parametrize("verdict,diagnosis", [
    ("supported", "wrong_sample"), ("contradicted", "supported"),
    ("insufficient_evidence", "unsupported_value"), ("false", "unsupported_value"),
    ("supported", "invented"),
])
def test_invalid_verdict_combinations(support_case, verdict, diagnosis):
    result, _, _ = run(support_case, wire(support_case[-1], verdict=verdict, diagnosis=diagnosis))
    assert result.status == "failed" and not result.judgments


@pytest.mark.parametrize("raw", [
    "not JSON", "[]", "null", "true", "42", "{}", '{"judgments":[]}',
    '{"judgments":NaN}', '{"judgments":Infinity}', '{"judgments":',
    '{"judgments":[],"judgments":[]}', "markdown-wrapped JSON",
    'Here is the answer: {"judgments":[]}',
])
def test_bad_json_never_successful_empty(support_case, raw):
    _, profile, bundle, record, _ = support_case
    result, provider, _ = run(support_case, raw)
    assert result.status == "failed" and result.parse_status == "failed"
    assert not result.judgments and result.raw_output == raw and len(provider.requests) == 1
    verify_report(result, profile, record, bundle)


@pytest.mark.parametrize("mutation", ["unknown_path", "duplicate_path", "missing_path", "reordered", "extra_envelope", "extra_field", "no_verdict", "no_diagnosis", "no_summary", "blank_summary", "long_summary", "chain_of_thought", "made_up_offset"])
def test_strict_envelope(support_case, mutation):
    data = wire(support_case[-1])
    first = data["judgments"][0]
    if mutation == "unknown_path": first["path"] = "/datasets/99/dataset_description"
    elif mutation == "duplicate_path": data["judgments"][1] = deepcopy(first)
    elif mutation == "missing_path": data["judgments"].pop()
    elif mutation == "reordered": data["judgments"].reverse()
    elif mutation == "extra_envelope": data["patch"] = []
    elif mutation == "extra_field": first["approved"] = True
    elif mutation == "no_verdict": del first["verdict"]
    elif mutation == "no_diagnosis": del first["diagnosis"]
    elif mutation == "no_summary": del first["summary"]
    elif mutation == "blank_summary": first["summary"] = "   "
    elif mutation == "long_summary": first["summary"] = "x" * 501
    elif mutation == "chain_of_thought": first["reasoning"] = "private reasoning"
    else: first["evidence"] = [{"start": 1, "end": 2}]
    report, provider, _ = run(support_case, data)
    assert report.status == "failed" and not report.judgments
    assert len(provider.requests) == 1


@pytest.mark.parametrize("indices", [[1], [-1], [0, 0], [True], [0.0], ["0"], None, "0"])
def test_reply_indices_strict(support_case, indices):
    data = wire(support_case[-1])
    data["judgments"][0]["evidence_indices"] = indices
    report, _, _ = run(support_case, data)
    assert report.status == "failed" and report.issues[0].code == "invalid_support_response"


@pytest.mark.parametrize("verdict,diagnosis", [("supported", "supported"), ("contradicted", "wrong_sample")])
def test_definite_verdict_needs_evidence(support_case, verdict, diagnosis):
    report, _, _ = run(support_case, wire(support_case[-1], verdict=verdict, diagnosis=diagnosis, indices=()))
    assert report.status == "failed"


def test_no_evidence_only_insufficient(support_case):
    root, profile, bundle, record, targets = support_case
    targets = tuple(t.model_copy(update={"evidence": ()}) for t in targets)
    result, _, _ = run((root, profile, bundle, record, targets), wire(targets, verdict="insufficient_evidence", diagnosis="missing_evidence", indices=()))
    assert result.status == "succeeded" and all(not j.evidence for j in result.judgments)
    verify_report(result, profile, record, bundle)


@pytest.mark.parametrize("field,value", [("source_sha256", "0" * 64), ("representation_sha256", "0" * 64), ("source_id", "other"), ("region_id", "missing"), ("start", 1), ("end", 999999), ("quotation", "fabricated")])
def test_bad_input_evidence_prevents_dispatch(support_case, field, value):
    root, profile, bundle, record, targets = support_case
    ref = targets[0].evidence[0].model_copy(update={field: value})
    targets = (SupportTarget(path=targets[0].path, evidence=(ref,)),)
    provider = MockProvider([])
    result = check_support(root, profile, record, bundle, targets, provider, raw_retention="retain")
    assert result.status == "failed" and result.issues[0].code == "invalid_evidence"
    assert not provider.requests and result.provider is None
    verify_report(result, profile, record, bundle)


@pytest.mark.parametrize("kind", ["unknown_assertion", "duplicate_target", "invalid_evidence"])
def test_bad_target_sets_prevent_dispatch(support_case, kind):
    root, profile, bundle, record, targets = support_case
    if kind == "unknown_assertion": targets = (SupportTarget(path="/datasets/0/id"),)
    elif kind == "duplicate_target": targets = (targets[0], targets[0])
    else: targets = (SupportTarget(path=targets[0].path, evidence=(targets[0].evidence[0],) * 2),)
    provider = MockProvider([])
    report = check_support(root, profile, record, bundle, targets, provider, raw_retention="hash_only")
    assert report.status == "failed" and report.issues[0].code == kind
    assert not provider.requests
    verify_report(report, profile, record, bundle)


@pytest.mark.parametrize("targets", [(), [], (SupportTarget(path="/x"),) * 33])
def test_target_budget(support_case, targets):
    root, profile, bundle, record, _ = support_case
    provider = MockProvider([])
    with pytest.raises(ValueError):
        check_support(root, profile, record, bundle, targets, provider, raw_retention="retain")
    assert not provider.requests


@pytest.mark.parametrize("finish", ["length", "tool_calls", "content_filter", "unknown"])
def test_bad_finish_preserves_observation(support_case, finish):
    result, provider, raw = run(support_case, finish=finish)
    assert result.status == "failed" and result.raw_output == raw
    assert result.response_truncated == (finish == "length") and len(provider.requests) == 1
    assert result.provider.finish_reason == finish


def test_unknown_finish_review(support_case):
    _, profile, bundle, record, _ = support_case
    result, _, _ = run(support_case, finish=None)
    assert result.status == "needs_review" and result.issues[0].code == "finish_reason_unknown"
    verify_report(result, profile, record, bundle)


def test_truncated_source_review(support_case):
    root, profile, _, record, _ = support_case
    bundle = synthetic_bundle("An invented synthetic document. More unavailable content.\n", max_chars=35)
    state = snapshot(record.canonical, bundle, profile)
    target = SupportTarget(path="/datasets/0/dataset_description", evidence=(evidence_ref(bundle, bundle.regions[0].region_id),))
    report, _, _ = run((root, profile, bundle, state, (target,)))
    assert report.source_incomplete and report.status == "needs_review"
    verify_report(report, profile, state, bundle)


@pytest.mark.parametrize("status", ["timeout", "failed", "blocked", "rate_limited"])
def test_failed_call_once_no_retry(support_case, status):
    root, profile, bundle, record, targets = support_case
    provider = MockProvider([ProviderResult(status=status, error_code="synthetic_failure", latency_ms=5.0)])
    result = check_support(root, profile, record, bundle, targets, provider, raw_retention="retain")
    assert result.status == "failed" and result.provider.status == status
    assert result.provider.input_tokens is result.provider.output_tokens is None
    assert len(provider.requests) == 1
    verify_report(result, profile, record, bundle)


@pytest.mark.parametrize("error,expected", [(TimeoutError("sensitive"), "provider_timeout"), (RuntimeError("sensitive"), "adapter_exception")])
def test_adapter_exceptions_safe(support_case, error, expected):
    class Broken:
        model_name, mode = "mock", "mock"
        def generate(self, prompt):
            raise error
    root, profile, bundle, record, targets = support_case
    result = check_support(root, profile, record, bundle, targets, Broken(), raw_retention="retain")
    assert result.status == "failed" and result.provider.error_code == expected
    assert "sensitive" not in result.model_dump_json()


def test_mode_mismatch(support_case):
    class Broken:
        model_name, mode = "mock", "mock"
        def generate(self, prompt):
            return ProviderResult(status="ok", content="{}", mode="live")
    root, profile, bundle, record, targets = support_case
    result = check_support(root, profile, record, bundle, targets, Broken(), raw_retention="retain")
    assert result.status == "failed" and result.provider.error_code == "provider_mode_mismatch"


@pytest.mark.parametrize("retention", ["retain", "hash_only"])
def test_pure_replay_identical_without_calls(support_case, retention):
    _, profile, bundle, record, targets = support_case
    result, provider, raw = run(support_case, raw_retention=retention)
    replay = interpret_support_response(profile, record, bundle, targets,
        ProviderResult(status="ok", content=raw, finish_reason="stop"), model_name="mock", raw_retention=retention)
    assert replay == result and len(provider.requests) == 1
    assert result.raw_output == (raw if retention == "retain" else None)
    verify_report(result, profile, record, bundle)


def test_retention_must_be_explicit(support_case):
    root, profile, bundle, record, targets = support_case
    provider = MockProvider([])
    with pytest.raises(ValueError):
        check_support(root, profile, record, bundle, targets, provider, raw_retention="everything")
    assert not provider.requests


def test_changed_revision_or_source(support_case):
    root, profile, bundle, record, targets = support_case
    result, _, _ = run(support_case)
    canonical = record.canonical
    canonical["datasets"][0]["dataset_description"] = "Sample A measured at 300 K"
    changed = snapshot(canonical, bundle, profile)
    with pytest.raises(ValueError): verify_report(result, profile, changed, bundle)
    with pytest.raises(ValueError):
        check_support(root, profile, record, synthetic_bundle("Another document"), targets, MockProvider([]), raw_retention="retain")


def test_changed_profile_prevents_dispatch(support_case):
    root, profile, bundle, record, targets = support_case
    schema = root / "schemas/prompt_template.md"
    schema.write_text(schema.read_text() + "\nChanged")
    provider = MockProvider([])
    with pytest.raises(ValueError): check_support(root, profile, record, bundle, targets, provider, raw_retention="retain")
    assert not provider.requests


@pytest.mark.parametrize("field,value", [("entity_handle", "invented"), ("target_handles", ["invented"]), ("role", "other"), ("value_json", '"invented"'), ("record_revision", "revision:invented"), ("provenance", "live_model_judgment")])
def test_rehashed_forged_binding_rejected(support_case, field, value):
    _, profile, bundle, record, _ = support_case
    result, _, _ = run(support_case)
    judgments = result.model_dump(mode="json")["judgments"]
    judgments[0][field] = value
    forged = rehash(result, {"judgments": judgments})
    with pytest.raises(ValueError): verify_report(forged, profile, record, bundle)


def test_digest_raw_and_rehashed_verdict_forgery(support_case):
    _, profile, bundle, record, _ = support_case
    result, _, _ = run(support_case)
    with pytest.raises(ValueError): verify_report(result.model_copy(update={"status": "needs_review"}), profile, record, bundle)
    with pytest.raises(ValueError): verify_report(result.model_copy(update={"raw_output": "fake"}), profile, record, bundle)
    judgments = result.model_dump(mode="json")["judgments"]
    judgments[0].update(verdict="supported", diagnosis="supported")
    forged = rehash(result, {"judgments": judgments}).model_copy(update={"raw_output": result.raw_output})
    with pytest.raises(ValueError): verify_report(forged, profile, record, bundle)


def test_conflicting_judgments_never_majority_vote(support_case):
    _, profile, bundle, record, targets = support_case
    wrong, _, _ = run(support_case, wire(targets, verdict="contradicted", diagnosis="wrong_sample"))
    supported, _, _ = run(support_case, wire(targets, verdict="supported", diagnosis="supported"))
    alternative = wire(targets, verdict="supported", diagnosis="supported")
    alternative["judgments"][0]["summary"] = "Another synthetic opinion"
    another, _, _ = run(support_case, alternative)
    diagnosis = diagnose_reports((wrong, supported, another), profile, record, bundle)
    assert len(diagnosis.issues) == 2
    assert all(i.category == "diagnosis_conflict" and i.recommended_action == "human_review" for i in diagnosis.issues)
    assert diagnosis.approval_status == "unreviewed"


@pytest.mark.parametrize("diagnosis,category,action", [
    ("wrong_sample", "attachment", "inspect_source"),
    ("wrong_dataset", "attachment", "inspect_source"),
    ("preparation_acquisition_confusion", "attachment", "inspect_source"),
    ("unsupported_value", "support", "inspect_source"),
    ("context_conflict", "ambiguity", "human_review"),
])
def test_diagnostic_actions(support_case, diagnosis, category, action):
    _, profile, bundle, record, targets = support_case
    report, _, _ = run(support_case, wire(targets, diagnosis=diagnosis))
    result = diagnose_reports((report,), profile, record, bundle)
    assert all(i.category == category and i.recommended_action == action for i in result.issues)


def test_failed_checker_and_duplicate_reports(support_case):
    _, profile, bundle, record, _ = support_case
    failed, _, _ = run(support_case, "invalid")
    result = diagnose_reports((failed,), profile, record, bundle)
    assert result.issues[0].category == "checker_failure"
    assert result.issues[0].recommended_action == "human_review"
    with pytest.raises(ValueError): diagnose_reports((failed, failed), profile, record, bundle)
    with pytest.raises(ValueError): diagnose_reports((), profile, record, bundle)


def test_unguarded_live_provider_rejected(support_case):
    class Unguarded:
        model_name, mode = "qwen3.8-27b", "live"
        def generate(self, prompt):
            raise AssertionError("Must never dispatch")
    root, profile, bundle, record, targets = support_case
    with pytest.raises(ValueError, match="guarded ChatAI"):
        check_support(root, profile, record, bundle, targets, Unguarded(), raw_retention="hash_only")


@pytest.mark.parametrize("mismatch", ["source", "profile", "prompt"])
def test_live_scope_checked_without_credentials(support_case, mismatch, monkeypatch):
    root, profile, bundle, record, targets = support_case
    prompt_hash = sha256(render_support_prompt(profile, record, bundle, targets).encode())
    spec = LiveRunSpec(endpoint="https://example.invalid/v1", sources=(LiveSource(
        path=bundle.source_id if mismatch != "source" else "other", sha256=bundle.source_sha256),),
        split="synthetic", profile_sha256=profile.profile_id.removeprefix("profile:") if mismatch != "profile" else "0" * 64,
        approved_prompt_sha256=(prompt_hash if mismatch != "prompt" else "0" * 64,), max_calls=1,
        max_output_tokens=100, max_total_output_tokens=100)
    provider = object.__new__(ChatAIProvider)
    provider._spec = spec
    monkeypatch.setattr(ChatAIProvider, "generate", lambda *_: pytest.fail("No live dispatch"))
    with pytest.raises(ValueError, match="scope"):
        check_support(root, profile, record, bundle, targets, provider, raw_retention="hash_only")


def test_nested_duplicate_keys(support_case):
    raw = json.dumps(wire(support_case[-1])).replace('"verdict": "contradicted"', '"verdict": "supported", "verdict": "contradicted"')
    report, _, _ = run(support_case, raw)
    assert report.status == "failed"


def test_response_and_prompt_budgets(support_case, monkeypatch):
    report, _, _ = run(support_case, "x" * 100_001)
    assert report.status == "failed"
    monkeypatch.setattr("meta_data_extraction.agent.support_checks.MAX_PROMPT_BYTES", 10)
    root, profile, bundle, record, targets = support_case
    provider = MockProvider([])
    report = check_support(root, profile, record, bundle, targets, provider, raw_retention="retain")
    assert report.status == "failed" and report.issues[0].code == "prompt_size_limit" and not provider.requests
    verify_report(report, profile, record, bundle)


def test_facility_and_sample_roles(support_case):
    root, profile, bundle, record, targets = support_case
    ref = targets[0].evidence[0]
    targets = (SupportTarget(path="/datasets/0/facility_refs/0", evidence=(ref,)),
               SupportTarget(path="/samples/0/sample_description", evidence=(ref,)))
    canonical = record.canonical
    canonical["samples"][0]["sample_description"] = "Prepared at 500 K"
    record = snapshot(canonical, bundle, profile)
    report, _, _ = run((root, profile, bundle, record, targets), wire(targets, verdict="supported", diagnosis="supported"))
    assert report.judgments[0].role == "attachment"
    facility = next(e for e in record.entities if e.local_id == "facility_1")
    assert report.judgments[0].target_handles == (facility.handle,)
    assert report.judgments[1].role == "sample_identity_or_preparation"
    verify_report(report, profile, record, bundle)


@pytest.mark.parametrize("boundary", ["render", "interpret", "verify"])
def test_same_id_changed_profile_rejected_at_pure_boundaries(support_case, boundary):
    _, profile, bundle, record, targets = support_case
    result, _, raw = run(support_case)
    schema = profile.schema
    schema["datasets"]["description"] = "Changed meaning with same claimed ID"
    forged = profile.model_copy(update={"schema_payload_json": json.dumps(schema, ensure_ascii=False, separators=(",", ":"))})
    with pytest.raises(ValueError):
        if boundary == "render":
            render_support_prompt(forged, record, bundle, targets)
        elif boundary == "interpret":
            interpret_support_response(forged, record, bundle, targets,
                ProviderResult(status="ok", content=raw, finish_reason="stop"), model_name="mock", raw_retention="hash_only")
        else:
            verify_report(result, forged, record, bundle)
