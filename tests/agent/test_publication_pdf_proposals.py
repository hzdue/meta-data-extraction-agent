"""Available-text PDF suggestions are review artifacts, never automatic approval.

All source statements and model responses here are invented software fixtures.
The integration cases generate an actual PDF inside the temporary workspace and
exercise the existing parser; they make no provider/network calls.
"""
import pymupdf
import pytest

from test_publication_checks import case as bridge_case
from test_publication_checks import report_for as _report_for, wire as _wire
from test_publication_run import scope as publication_scope
from test_publication_run import (
    amended as _amended, child_directory as _child_directory,
    json_artifact as _json_artifact, response as _response, snapshots as _snapshots,
)
from meta_data_extraction.agent.contracts import snapshot
from meta_data_extraction.agent.evidence import ParserProvenance, bundle_document, sha256
from meta_data_extraction.agent.publication_checks import (
    PublicationCheckError, eligible_patch_paths, issues_from_support,
    parse_patch_response, plan_support, render_patch_prompt,
)
from meta_data_extraction.agent.record_patches import (
    PatchPolicy, apply_supported_patch, build_candidate,
)
from meta_data_extraction.agent.review_store import ReviewStore
from meta_data_extraction.data_models import Document
import meta_data_extraction.agent.publication_run as runner


def _pdf_case(case, *, warnings=(), max_chars=None):
    profile, record, previous = case
    bundle = bundle_document(
        Document(id="synthetic-pdf", text=previous.text, warnings=list(warnings)),
        source_id="tests/agent/fixtures/invented.pdf",
        source_bytes=b"Invented unit-fixture PDF bytes, never sent to a parser",
        source_format="pdf", parser=ParserProvenance(name="synthetic", version="1"),
        max_chars=max_chars,
    )
    return profile, snapshot(record.canonical, bundle, profile), bundle


@pytest.fixture
def pdf_case(bridge_case):
    return _pdf_case(bridge_case)


def test_pdf_text_contradictions_can_produce_only_unreviewed_proposals(pdf_case):
    profile, record, bundle = pdf_case
    original = record.canonical_json
    plan = plan_support(*pdf_case)
    report = _report_for(pdf_case, plan)
    assert report.status == "needs_review" and report.source_incomplete
    assert tuple(issue.code for issue in report.issues) == ("source_incomplete",)
    assert {warning.code for warning in bundle.warnings} == {
        "page_mapping_unavailable", "images_unavailable",
    }
    assert set(eligible_patch_paths(*pdf_case, report)) == {
        "/datasets/0/dataset_description", "/datasets/0/sample_refs/0",
    }
    proposal = parse_patch_response(_wire(), *pdf_case, report)
    assert proposal.approval_status == "unreviewed"
    assert proposal.proposal.provenance == "model_proposal"
    assert proposal.record_revision == record.revision_id
    assert proposal.support_report_id == report.report_id
    assert record.canonical_json == original
    prompt = render_patch_prompt(*pdf_case, report)
    assert "images_unavailable" in prompt
    assert "unreviewed" in prompt.lower()
    issues = issues_from_support(*pdf_case, plan, report, proposals=(proposal.proposal,))
    assert any(issue.diagnosis == "source_incomplete" for issue in issues)
    assert not any(issue.diagnosis == "checker_failure" for issue in issues)
    assert any(proposal.proposal.proposal_id in issue.proposal_ids for issue in issues)


@pytest.mark.parametrize("limitation", ["parser_warning", "truncated", "unknown_finish"])
def test_other_pdf_limits_still_block_proposal_generation(bridge_case, limitation):
    case = _pdf_case(bridge_case,
        warnings=("Invented parser warning",) if limitation == "parser_warning" else (),
        max_chars=180 if limitation == "truncated" else None)
    plan = plan_support(*case)
    report = _report_for(case, plan, finish=None if limitation == "unknown_finish" else "stop")
    assert report.status == "needs_review"
    assert eligible_patch_paths(*case, report) == ()
    with pytest.raises(PublicationCheckError, match="no_patch_candidates"):
        render_patch_prompt(*case, report)


@pytest.mark.parametrize("status", ["failed", "timeout", "blocked", "rate_limited"])
def test_failed_pdf_check_never_yields_proposal(pdf_case, status):
    plan = plan_support(*pdf_case)
    report = _report_for(pdf_case, plan, status=status)
    assert eligible_patch_paths(*pdf_case, report) == ()
    assert any(issue.diagnosis == "checker_failure"
               for issue in issues_from_support(*pdf_case, plan, report))


def test_pdf_conflicting_or_supported_judgments_are_not_patch_candidates(pdf_case):
    plan = plan_support(*pdf_case)
    conflict = _report_for(pdf_case, plan, diagnosis="context_conflict")
    supported = _report_for(pdf_case, plan, contradicted=())
    assert eligible_patch_paths(*pdf_case, conflict) == ()
    assert eligible_patch_paths(*pdf_case, supported) == ()


def test_pdf_nonallowlisted_assertions_cannot_be_proposed(pdf_case):
    plan = plan_support(*pdf_case, max_targets=28)
    report = _report_for(pdf_case, plan, contradicted=("/samples/0/sample_name",))
    assert any(judgment.verdict == "contradicted" for judgment in report.judgments)
    assert eligible_patch_paths(*pdf_case, report) == ()


def test_pdf_stale_record_and_forged_evidence_fail_before_proposals(pdf_case):
    profile, record, bundle = pdf_case
    plan = plan_support(*pdf_case)
    report = _report_for(pdf_case, plan)
    canonical = record.canonical
    canonical["datasets"][0]["dataset_description"] = "Different revision"
    changed = snapshot(canonical, bundle, profile)
    with pytest.raises(ValueError):
        eligible_patch_paths(profile, changed, bundle, report)
    judgment = next(item for item in report.judgments if item.verdict == "contradicted")
    forged_ref = judgment.evidence[0].model_copy(update={"quotation": "Invented offset mismatch"})
    forged = judgment.model_copy(update={"evidence": (forged_ref,)})
    tampered = report.model_copy(update={"judgments": tuple(
        forged if item.path == judgment.path else item for item in report.judgments)})
    with pytest.raises(ValueError):
        eligible_patch_paths(*pdf_case, tampered)


def test_pdf_proposal_cannot_bypass_atomic_application_gate(pdf_case):
    profile, record, bundle = pdf_case
    plan = plan_support(*pdf_case)
    before = _report_for(pdf_case, plan)
    proposal = parse_patch_response(_wire(), *pdf_case, before).proposal
    built = build_candidate(profile, bundle, record, proposal)
    assert built.status == "ready"
    after_case = profile, built.prepared.candidate, bundle
    after = _report_for(after_case, plan_support(*after_case, max_targets=28), contradicted=())
    result = apply_supported_patch(profile, bundle, record, built.prepared, before, after,
                                   policy=PatchPolicy(accept_model_judgments=True))
    assert result.status == "rejected" and result.code == "support_check_incomplete"
    assert result.snapshot == record and result.receipt is None


@pytest.fixture
def actual_pdf_scope(publication_scope):
    root, spec = publication_scope
    text = (root / spec.source.path).read_text(encoding="utf-8")
    relative = "tests/agent/fixtures/publication_generated.pdf"
    path = root / relative
    with pymupdf.open() as document:
        page = document.new_page(width=612, height=792)
        assert page.insert_textbox(pymupdf.Rect(48, 48, 564, 744), text, fontsize=11) > 0
        document.save(path)
    spec = _amended(spec, source={"path": relative, "sha256": sha256(path.read_bytes())})
    # No invented PDF locations: prepare the actual parser result, then build
    # authored responses against its exact source representation and target list.
    bundle, profile, _, _ = runner._prepare(root, spec)
    assert {warning.code for warning in bundle.warnings} == {
        "page_mapping_unavailable", "images_unavailable",
    }
    original = _json_artifact(root, spec.mock_responses[0].path)["output"]
    spec = _response(root, spec, 0, {"output": original, "evidence": []})
    record = snapshot(original, bundle, profile)
    plan = plan_support(profile, record, bundle, max_targets=spec.max_targets,
                        evidence_chars=spec.evidence_chars)
    scripted = _report_for((profile, record, bundle), plan)
    check = {"judgments": [{
        "path": item.path, "verdict": item.verdict, "diagnosis": item.diagnosis,
        "summary": "Invented software diagnostic; no empirical support claim.",
        "evidence_indices": [0] if item.evidence else [],
    } for item in scripted.judgments]}
    spec = _response(root, spec, 1, check)
    return root, spec, original


@pytest.mark.parametrize("automatic,max_calls", [(False, 3), (True, 5)])
def test_actual_pdf_proposal_persists_for_review_without_automatic_edits(
        actual_pdf_scope, monkeypatch, automatic, max_calls):
    root, spec, original = actual_pdf_scope
    spec = _amended(spec, allow_automatic_patches=automatic, max_calls=max_calls)
    report = runner.run_publication(root, spec)
    assert report["status"] == "pending_review"
    assert report["new_model_calls"] == report["total_model_calls"] == 3
    assert report["new_live_requests"] == report["total_live_requests"] == 0
    assert _json_artifact(root, report["canonical_relative"]) == original
    sidecar = _json_artifact(root, report["sidecar_relative"])
    assert sidecar["support_report"]["status"] == "needs_review"
    assert sidecar["support_report"]["source_incomplete"]
    assert sidecar["patch_generation"]["approval_status"] == "unreviewed"
    assert sidecar["patch_generation"]["status"] == "proposed"
    assert sidecar["automatic_patch_status"] == (
        "blocked_source_incomplete" if automatic else "disabled")
    assert sidecar["initial_record_revision"] == report["record_revision"]
    assert any(issue["diagnosis"] == "source_incomplete"
               for issue in sidecar["workflow_state"]["issues"])
    assert not any(issue["diagnosis"] == "checker_failure"
                   for issue in sidecar["workflow_state"]["issues"])
    assert not _child_directory(root, spec, "wc1").exists()
    assert not _child_directory(root, spec, "wc2").exists()
    with ReviewStore.open(root, report["workflow_run_id"]) as review:
        assert review.state.current.canonical == original
        assert review.state.status == "pending"
        assert review.state.decisions == ()
        assert len(review.state.proposals) == 1
        assert review.state.proposals[0].proposal.model_dump(mode="json") == sidecar["patch_generation"]["proposal"]
    before = _snapshots(root)
    monkeypatch.setattr(runner, "_provider", lambda *a, **k: pytest.fail("Replay created a provider"))
    replay = runner.run_publication(root, spec)
    current = runner.read_publication(root, spec.run_id)
    assert replay["execution"] == "cached_replay"
    assert replay["new_model_calls"] == current["new_model_calls"] == 0
    assert replay["record_revision"] == current["record_revision"] == report["record_revision"]
    assert _snapshots(root) == before
