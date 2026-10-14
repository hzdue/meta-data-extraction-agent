"""Synthetic M07 tool behavior; supplied mock judgments are not expert gold."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from extraction_fixtures import synthetic_bundle, synthetic_record
from test_nexus_registry import synthetic_snapshot, request, review
from meta_data_extraction.agent.contracts import canonical_json, snapshot
from meta_data_extraction.agent.evidence import evidence_ref
from meta_data_extraction.agent.nexus_registry import build_registry, propose_entry
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.agent.providers import MockProvider, ProviderResult
from meta_data_extraction.agent.record_patches import PatchPolicy, PatchProposal, ReplaceScalar, build_candidate
from meta_data_extraction.agent.workflow_tools import ToolContext, execute_tool
from meta_data_extraction.agent.workflow_types import (
    ConsultDefinition, Finish, ProposeUpdate, RequestReview, RetrieveSource, WorkflowIssue, WorkflowState,
)

DESCRIPTION = "/datasets/0/dataset_description"
REFERENCE = "/datasets/0/sample_refs/0"
TEXT = (
    "Explicitly invented workflow fixture. Sample A and Sample B are distinct CaCO3 specimens.\n\n"
    "Sample A was prepared at 500 K. Dataset A measured Sample A at 300 K using X-ray diffraction "
    "at Synthetic Beam Source. Sample B was measured in Dataset B.\n"
)


def response(paths, *, verdict="supported", diagnosis=None, status="ok", tokens=True):
    diagnosis = diagnosis or ("supported" if verdict == "supported" else "preparation_acquisition_confusion")
    values = {"judgments": [dict(path=p, verdict=verdict, diagnosis=diagnosis,
               summary="Invented model judgment for bounded workflow testing.", evidence_indices=[0]) for p in paths]}
    return ProviderResult(status=status, content=json.dumps(values), finish_reason="stop",
                          input_tokens=10 if tokens else None, output_tokens=10 if tokens else None)


@pytest.fixture
def tool_case(workspace):
    shutil.copyfile(Path(__file__).resolve().parents[2] / "schemas/prompt_template.md",
                    workspace / "schemas/prompt_template.md")
    profile = load_profile(workspace)
    bundle = synthetic_bundle(TEXT)
    canonical = synthetic_record(profile)
    canonical["datasets"][0].update(dataset_description="Measured at 500 K", sample_refs=["sample_2"])
    record = snapshot(canonical, bundle, profile)
    proposal = PatchProposal(expected_revision=record.revision_id, operations=(
        ReplaceScalar(path=DESCRIPTION, expected_value_json=canonical_json("Measured at 500 K"),
                      replacement_value_json=canonical_json("Measured at 300 K")),
        ReplaceScalar(path=REFERENCE, expected_value_json=canonical_json("sample_2"),
                      replacement_value_json=canonical_json("sample_1")),
    ), provenance="synthetic_fixture", summary="Invented paired correction")
    prepared = build_candidate(profile, bundle, record, proposal).prepared
    checker = MockProvider([
        response(prepared.changed_paths, verdict="contradicted"),
        response(prepared.required_support_paths),
    ], model_name="mock-workflow-checker")
    ref = evidence_ref(bundle, bundle.regions[1].region_id)
    issues = tuple(WorkflowIssue(id=f"issue-{i}", path=p, diagnosis="wrong_sample" if "refs" in p else
        "preparation_acquisition_confusion", summary="Synthetic open issue", query="Dataset A measured",
        proposal_ids=("correction",), definition_keys=("mass",), evidence=(ref,)) for i, p in enumerate((DESCRIPTION, REFERENCE)))
    state = WorkflowState(initial_revision=record.revision_id, record=record, issues=issues)
    context = ToolContext(workspace, profile, bundle, {"correction": proposal}, checker,
                          PatchPolicy(accept_model_judgments=True))
    return context, state, prepared


def propose():
    return ProposeUpdate(issue_id="issue-0", proposal_id="correction")


def modify_issue(state, index=0, **values):
    issues = list(state.issues)
    issues[index] = issues[index].model_copy(update=values)
    return state.model_copy(update={"issues": tuple(issues)})


def test_source_tool_inspects_only_supplied_bundle(tool_case):
    context, state, _ = tool_case
    outcome = execute_tool(context, RetrieveSource(issue_id="issue-0", query="Dataset A"), state, model_budget=0)
    assert outcome.status == "ok" and outcome.evidence
    assert all(ref.source_id == context.bundle.source_id for ref in outcome.evidence)
    assert outcome.current == state.record and not context.checker.requests
    assert outcome.definitions == () and outcome.resolved_paths == ()


def test_no_match_and_clipped_selection_preserve_real_offsets(tool_case):
    context, state, _ = tool_case
    missing = execute_tool(context, RetrieveSource(issue_id="issue-0", query="unfindableword"), state, model_budget=0)
    assert missing.code == "source_no_match" and not missing.evidence
    clipped = execute_tool(context, RetrieveSource(issue_id="issue-0", query="Dataset A", max_chars=12, limit=1), state, model_budget=0)
    assert clipped.status == "ok" and len(clipped.evidence[0].quotation) <= 12
    ref = clipped.evidence[0]
    assert context.bundle.text[ref.start:ref.end] == ref.quotation


@pytest.mark.parametrize("kind", ["unknown", "absent_issue", "not_allowlisted", "bad_args"])
def test_invalid_tool_requests_rejected(tool_case, kind):
    context, state, _ = tool_case
    if kind == "unknown": action = {"kind": "execute_python", "code": "pass"}
    elif kind == "absent_issue": action = RetrieveSource(issue_id="unknown", query="A")
    elif kind == "not_allowlisted": action = ProposeUpdate(issue_id="issue-0", proposal_id="other")
    else: action = {"kind": "retrieve_source", "issue_id": "issue-0", "query": "A", "max_chars": 999999}
    result = execute_tool(context, action, state, model_budget=2)
    assert result.status == "blocked" and result.code == "invalid_tool_request"
    assert not context.checker.requests


@pytest.mark.parametrize("action", [RequestReview(issue_id="issue-0", summary="Needs review"),
                                   Finish(outcome="unresolved", summary="Remain unresolved")])
def test_coordinator_actions_not_executed_as_tools(tool_case, action):
    context, state, _ = tool_case
    result = execute_tool(context, action, state, model_budget=2)
    assert result.code == "coordinator_action" and not context.checker.requests


@pytest.mark.parametrize("budget", [True, -1, 1.0, "2"])
def test_model_budget_strict(tool_case, budget):
    context, state, _ = tool_case
    result = execute_tool(context, propose(), state, model_budget=budget)
    assert result.code == "invalid_model_budget" and not context.checker.requests


@pytest.mark.parametrize("budget", [0, 1])
def test_two_slots_reserved_before_any_checker_call(tool_case, budget):
    context, state, _ = tool_case
    result = execute_tool(context, propose(), state, model_budget=budget)
    assert result.code == "model_budget_exhausted" and not context.checker.requests


def test_two_checks_apply_atomic_supported_correction(tool_case):
    context, state, prepared = tool_case
    result = execute_tool(context, propose(), state, model_budget=2)
    assert result.status == "ok" and result.current == prepared.candidate
    assert result.receipt_id and result.resolved_paths == prepared.changed_paths
    assert len(context.checker.requests) == len(result.providers) == len(result.reports) == 2
    assert all(report.raw_retention == "hash_only" and report.raw_output is None for report in result.reports)
    assert result.definitions == () and not result.definition_keys
    assert state.record == prepared.original
    assert all(obs.mode == "mock" for obs in result.providers)


@pytest.mark.parametrize("status", ["resolved", "pending_review"])
def test_closed_issue_never_corrected(tool_case, status):
    context, state, _ = tool_case
    state = modify_issue(state, status=status)
    result = execute_tool(context, propose(), state, model_budget=2)
    assert result.code == "invalid_tool_request" and not context.checker.requests


def test_patch_cannot_correct_other_closed_issue(tool_case):
    context, state, _ = tool_case
    state = modify_issue(state, index=1, status="resolved")
    result = execute_tool(context, propose(), state, model_budget=2)
    assert result.code == "proposal_outside_open_issues" and not context.checker.requests


def test_missing_inspected_source_or_disabled_policy_blocks(tool_case):
    context, state, _ = tool_case
    missing = execute_tool(context, propose(), modify_issue(state, evidence=()), model_budget=2)
    assert missing.code == "source_evidence_required" and not context.checker.requests
    disabled = replace(context, patch_policy=PatchPolicy())
    blocked = execute_tool(disabled, propose(), state, model_budget=2)
    assert blocked.code == "model_judgment_policy_disabled" and not context.checker.requests


def test_forged_publication_location_blocks_before_checker(tool_case):
    context, state, _ = tool_case
    ref = state.issues[0].evidence[0].model_copy(update={"quotation": "Not in the publication"})
    result = execute_tool(context, propose(), modify_issue(state, evidence=(ref,)), model_budget=2)
    assert result.code == "candidate_binding_invalid" and not context.checker.requests


@pytest.mark.parametrize("verdict,diagnosis", [("supported", "supported"), ("insufficient_evidence", "missing_evidence"),
                                            ("contradicted", "context_conflict")])
def test_before_gate_prevents_second_call(tool_case, verdict, diagnosis):
    context, state, prepared = tool_case
    checker = MockProvider([response(prepared.changed_paths, verdict=verdict, diagnosis=diagnosis)])
    result = execute_tool(replace(context, checker=checker), propose(), state, model_budget=2)
    assert result.code == "original_not_contradicted" and result.current == state.record
    assert len(checker.requests) == len(result.providers) == 1


@pytest.mark.parametrize("status", ["timeout", "failed", "rate_limited", "blocked"])
def test_provider_failures_observed_no_retry(tool_case, status):
    context, state, _ = tool_case
    checker = MockProvider([ProviderResult(status=status, error_code="fixture_failure")])
    result = execute_tool(replace(context, checker=checker), propose(), state, model_budget=2)
    assert result.code == "support_check_incomplete" and result.current == state.record
    assert result.providers[0].status == status and result.providers[0].input_tokens is None
    assert len(checker.requests) == 1


def test_explicit_bounded_unknown_usage_stays_unknown(tool_case):
    context, state, prepared = tool_case
    checker = MockProvider([response(prepared.changed_paths, verdict="contradicted", tokens=False),
                            response(prepared.required_support_paths, tokens=False)])
    result = execute_tool(replace(context, checker=checker), propose(), state, model_budget=2, unknown_usage="allow_bounded_calls")
    assert len(result.providers) == 2
    assert all(obs.input_tokens is None and obs.output_tokens is None for obs in result.providers)


def test_after_unsupported_candidate_cannot_apply(tool_case):
    context, state, prepared = tool_case
    checker = MockProvider([response(prepared.changed_paths, verdict="contradicted"),
                            response(prepared.required_support_paths, verdict="insufficient_evidence", diagnosis="missing_evidence")])
    result = execute_tool(replace(context, checker=checker), propose(), state, model_budget=2)
    assert result.code == "candidate_support_incomplete" and result.current == state.record
    assert not result.resolved_paths and result.receipt_id is None and len(checker.requests) == 2


@pytest.mark.parametrize("stop_at,expected_calls", [(1, 0), (2, 0), (3, 1), (4, 1), (5, 2)])
@pytest.mark.parametrize("reason", ["timeout", "cancelled"])
def test_stop_before_after_each_provider_never_applies(tool_case, stop_at, expected_calls, reason):
    context, state, _ = tool_case
    checks = 0
    def stop():
        nonlocal checks
        checks += 1
        return reason if checks == stop_at else None
    result = execute_tool(context, propose(), state, model_budget=2, should_stop=stop)
    assert result.code == reason and result.current == state.record and result.receipt_id is None
    assert len(context.checker.requests) == expected_calls == len(result.providers)


@pytest.mark.parametrize("reason", ["secret reason should not leak", [], True])
def test_unexpected_stop_reason_redacted(tool_case, reason):
    context, state, _ = tool_case
    result = execute_tool(context, propose(), state, model_budget=2, should_stop=lambda: reason)
    assert result.code == "stopped" and not context.checker.requests
    assert "secret reason" not in result.model_dump_json()


def test_stop_checker_exception_safe(tool_case):
    context, state, _ = tool_case
    def stopped():
        raise RuntimeError("sensitive")
    result = execute_tool(context, propose(), state, model_budget=2, should_stop=stopped)
    assert result.code == "stop_check_failed" and not context.checker.requests


def test_repeat_old_proposal_is_stale_not_second_mutation(tool_case):
    context, state, prepared = tool_case
    applied = execute_tool(context, propose(), state, model_budget=2)
    new_state = WorkflowState(initial_revision=prepared.candidate.revision_id, record=prepared.candidate, issues=state.issues)
    repeated = execute_tool(context, propose(), new_state, model_budget=2)
    assert applied.status == "ok" and repeated.code == "stale_revision"
    assert repeated.current == prepared.candidate and len(context.checker.requests) == 2


def test_context_copies_catalogs_and_rejects_provider_scope_drift(tool_case):
    context, state, _ = tool_case
    proposals = dict(context.proposals)
    fresh = replace(context, proposals=proposals)
    proposals.clear()
    assert "correction" in fresh.proposals
    with pytest.raises(TypeError): fresh.proposals["other"] = context.proposals["correction"]
    fresh.checker.model_name = "changed"
    result = execute_tool(fresh, propose(), state, model_budget=2)
    assert result.code == "invalid_tool_request"


def test_source_profile_binding_mismatch_rejected(tool_case):
    context, state, _ = tool_case
    other = synthetic_bundle("Different invented source")
    incompatible = replace(context, bundle=other)
    result = execute_tool(incompatible, propose(), state, model_budget=2)
    assert result.code == "invalid_tool_request" and not context.checker.requests


def test_changed_profile_file_prevents_action(tool_case):
    context, state, _ = tool_case
    filename = context.root / "schemas/prompt_template.md"
    filename.write_text(filename.read_text() + "\nChanged")
    result = execute_tool(context, RetrieveSource(issue_id="issue-0", query="Dataset"), state, model_budget=0)
    assert result.code == "invalid_tool_request"


def test_missing_configured_proposal_blocked(tool_case):
    context, state, _ = tool_case
    result = execute_tool(replace(context, proposals={}), propose(), state, model_budget=2)
    assert result.code == "proposal_unavailable"


@pytest.mark.parametrize("approval", ["absent", "unreviewed", "rejected", "needs_changes"])
def test_only_approved_definition_can_be_consulted(tool_case, approval):
    context, state, _ = tool_case
    if approval != "absent":
        resource = synthetic_snapshot(context.root)
        entry = propose_entry(context.profile, resource, request())
        reviews = () if approval == "unreviewed" else (review(entry, decision=approval),)
        registry = build_registry(context.profile, resource, (entry,), reviews)
        context = replace(context, nxdl=resource, registry=registry,
                          definition_requests={"mass": ("/samples/*/sample_mass", "specimen_mass")})
    result = execute_tool(context, ConsultDefinition(issue_id="issue-0", definition_key="mass"), state, model_budget=0)
    assert result.code == "approved_definition_unavailable" and not result.evidence and not result.definitions


def test_synthetic_reviewed_definition_stays_separate_from_source(tool_case):
    context, state, _ = tool_case
    # Fabricated UNIT TEST review, not an actual expert approval.
    resource = synthetic_snapshot(context.root, documentation="Invented definition says mass 900 mg")
    entry = propose_entry(context.profile, resource, request())
    registry = build_registry(context.profile, resource, (entry,), (review(entry),))
    context = replace(context, nxdl=resource, registry=registry,
                      definition_requests={"mass": ("/samples/*/sample_mass", "specimen_mass")})
    result = execute_tool(context, ConsultDefinition(issue_id="issue-0", definition_key="mass"), state, model_budget=0)
    assert result.status == "ok" and result.definition_keys == ("mass",)
    assert result.definitions and result.evidence == () and result.current == state.record
    assert result.definitions[0].support_status == "not_assessed"
    blocked = execute_tool(context, propose(), modify_issue(state, evidence=()), model_budget=2)
    assert blocked.code == "source_evidence_required" and not context.checker.requests


def test_changed_nxdl_file_blocks_guidance(tool_case):
    context, state, _ = tool_case
    resource = synthetic_snapshot(context.root)
    entry = propose_entry(context.profile, resource, request())
    registry = build_registry(context.profile, resource, (entry,), (review(entry),))
    context = replace(context, nxdl=resource, registry=registry,
                      definition_requests={"mass": ("/samples/*/sample_mass", "specimen_mass")})
    filename = context.root / resource.resource.files[0].path
    filename.write_bytes(filename.read_bytes() + b"\n")
    result = execute_tool(context, ConsultDefinition(issue_id="issue-0", definition_key="mass"), state, model_budget=0)
    assert result.code == "definition_binding_invalid" and not result.definitions


def test_definition_setup_requires_matching_pair_and_scope(tool_case):
    context, _, _ = tool_case
    resource = synthetic_snapshot(context.root)
    with pytest.raises(ValueError): replace(context, nxdl=resource)
    with pytest.raises(ValueError): replace(context, definition_requests={"mass": ("/samples/*/sample_mass", "invented")})


def test_fake_live_provider_is_never_accepted(tool_case):
    context, _, _ = tool_case
    class Unguarded:
        model_name, mode = "qwen3.8-27b", "live"
        def generate(self, prompt):
            pytest.fail("No live calls")
    with pytest.raises(ValueError, match="guarded"): replace(context, checker=Unguarded())


def test_no_match_retains_selection_diagnostics(tool_case):
    context, state, _ = tool_case
    result = execute_tool(context, RetrieveSource(issue_id="issue-0", query="neverpresenttoken"), state, model_budget=0)
    assert result.status == "blocked" and result.selection.status == "no_match"
    assert result.selection.candidate_count == 0


def test_clipped_source_availability_is_explicit(tool_case):
    context, state, _ = tool_case
    result = execute_tool(context, RetrieveSource(issue_id="issue-0", query="Dataset A", max_chars=12, limit=1),
                          state, model_budget=0)
    assert result.selection.status == "truncated"
    assert result.selection.passages[0].clipped
    assert result.evidence == tuple(p.evidence for p in result.selection.passages)


def test_after_timeout_preserves_both_observations_and_original_record(tool_case):
    context, state, prepared = tool_case
    checker = MockProvider([response(prepared.changed_paths, verdict="contradicted"),
                            ProviderResult(status="timeout", error_code="fixture_timeout")])
    result = execute_tool(replace(context, checker=checker), propose(), state, model_budget=2)
    assert result.code == "support_check_incomplete" and result.current == state.record
    assert [obs.status for obs in result.providers] == ["ok", "timeout"]
    assert len(checker.requests) == 2 and not result.resolved_paths


def test_eight_link_bound_prevents_unbounded_check_prompt(tool_case):
    context, state, _ = tool_case
    region = context.bundle.regions[1]
    refs = tuple(evidence_ref(context.bundle, region.region_id, start=region.start+i, end=region.start+i+1) for i in range(9))
    result = execute_tool(context, propose(), modify_issue(state, evidence=refs), model_budget=2)
    assert result.code == "evidence_budget_exceeded" and not context.checker.requests


@pytest.mark.parametrize("unknown_stage,expected_calls", [("before", 1), ("after", 2)])
def test_unknown_usage_default_stops_calls_or_application(tool_case, unknown_stage, expected_calls):
    context, state, prepared = tool_case
    checker = MockProvider([
        response(prepared.changed_paths, verdict="contradicted", tokens=unknown_stage != "before"),
        response(prepared.required_support_paths, tokens=unknown_stage != "after"),
    ])
    result = execute_tool(replace(context, checker=checker), propose(), state, model_budget=2)
    assert result.code == "unknown_provider_usage" and result.current == state.record
    assert len(result.providers) == len(checker.requests) == expected_calls
    assert result.receipt_id is None


@pytest.mark.parametrize("policy", ["invented", None, []])
def test_unknown_usage_policy_is_explicitly_validated(tool_case, policy):
    context, state, _ = tool_case
    result = execute_tool(context, propose(), state, model_budget=2, unknown_usage=policy)
    assert result.code == "invalid_usage_policy" and not context.checker.requests
