"""Versioned retrieval integration using existing invented publication fixtures."""
import json

import pytest
from pydantic import ValidationError

from test_publication_checks import case as bridge_case
from test_publication_run import scope as publication_scope
from test_publication_run import amended, snapshots
from test_publication_types import payload

from meta_data_extraction.agent import publication_run as runner
from meta_data_extraction.agent.contracts import AssertionEvidence, at_pointer, canonical_json, digest, owner
from meta_data_extraction.agent.evidence import evidence_ref, resolve_evidence
from meta_data_extraction.agent.pilot_core import PilotError
from meta_data_extraction.agent.publication_checks import (
    PublicationCheckError, SupportPlan, plan_support, verify_plan,
)
from meta_data_extraction.agent.publication_service import PublicationService
from meta_data_extraction.agent.publication_types import PublicationSpec
from meta_data_extraction.agent.retrieval import SourceCorpus
from meta_data_extraction.agent.storage import RunConflictError


def test_support_plan_records_strategy_and_v2_version_with_exact_evidence(bridge_case):
    _, _, bundle = bridge_case
    for strategy in ("lexical_v1", "focused_v2"):
        plan = plan_support(*bridge_case, retrieval_strategy=strategy)
        assert plan.version == "publication-checks-2"
        assert plan.retrieval_strategy == strategy
        assert plan == SupportPlan.model_validate_json(plan.model_dump_json())
        verify_plan(plan, *bridge_case)
        assert plan.targets and any(t.evidence for t in plan.targets)
        for target in plan.targets:
            for ref in target.evidence:
                assert resolve_evidence(bundle, ref) == bundle.text[ref.start:ref.end]
                assert ref.source_sha256 == bundle.source_sha256
                assert ref.representation_sha256 == bundle.representation_sha256


def test_default_support_plan_is_explicit_legacy(bridge_case):
    assert plan_support(*bridge_case) == plan_support(*bridge_case, retrieval_strategy="lexical_v1")


def test_v1_support_plan_is_rejected_instead_of_silently_migrated(bridge_case):
    saved = plan_support(*bridge_case).model_dump(mode="json", exclude={"plan_id", "retrieval_strategy"})
    saved["version"] = "publication-checks-1"
    saved["plan_id"] = "support-plan:" + digest(saved)
    with pytest.raises(ValidationError, match="publication-checks-2"):
        SupportPlan.model_validate(saved)


def _saturated_seed(case):
    _, record, bundle = case
    path = plan_support(*case, max_targets=1).targets[0].path
    entity = owner(record.entities, path)
    refs = tuple(evidence_ref(bundle, region.region_id) for region in bundle.regions[:3])
    assert len(refs) == 3 and sum(len(ref.quotation) for ref in refs) < 1000
    seed = AssertionEvidence(record_revision=record.revision_id, entity_handle=entity.handle,
        path=path, field_path=path[len(entity.path):],
        value_json=canonical_json(at_pointer(record.canonical, path)),
        link_status="location_verified", evidence=refs)
    return seed


def test_strategy_changes_plan_identity_even_when_seed_evidence_is_identical(bridge_case, monkeypatch):
    seed = _saturated_seed(bridge_case)
    monkeypatch.setattr(SourceCorpus, "search", lambda *a, **kw: pytest.fail("All evidence slots are seeded"))
    settings = dict(assertions=(seed,), max_targets=1, evidence_chars=1000, per_target_chars=1000)
    legacy = plan_support(*bridge_case, **settings, retrieval_strategy="lexical_v1")
    focused = plan_support(*bridge_case, **settings, retrieval_strategy="focused_v2")
    assert legacy.targets == focused.targets and legacy.retrieval == focused.retrieval
    assert legacy.plan_id != focused.plan_id
    assert legacy.retrieval[0].seed_links_used == 3
    with pytest.raises(ValidationError):
        SupportPlan.model_validate(legacy.model_dump() | {"retrieval_strategy": "focused_v2"})


@pytest.mark.parametrize("invalid", ["unversioned", "focused_v3", None, True, 2])
def test_invalid_strategy_rejected_before_seed_slots_can_skip_search(bridge_case, monkeypatch, invalid):
    seed = _saturated_seed(bridge_case)
    monkeypatch.setattr(SourceCorpus, "search", lambda *a, **kw: pytest.fail("No retrieval should dispatch"))
    with pytest.raises(PublicationCheckError, match="invalid_retrieval_strategy"):
        plan_support(*bridge_case, assertions=(seed,), max_targets=1, evidence_chars=1000,
                     per_target_chars=1000, retrieval_strategy=invalid)


@pytest.mark.parametrize("invalid", ["unversioned", "focused_v3", None, True, 2])
def test_publication_scope_rejects_unknown_strategy(invalid):
    with pytest.raises(ValidationError):
        PublicationSpec.model_validate(payload(retrieval_strategy=invalid))


def test_publication_scope_legacy_default_and_explicit_selection():
    default = PublicationSpec.model_validate(payload())
    assert default.retrieval_strategy == "lexical_v1"
    assert default == PublicationSpec.model_validate(payload(retrieval_strategy="lexical_v1"))
    focused = PublicationSpec.model_validate(payload(retrieval_strategy="focused_v2"))
    assert focused.retrieval_strategy == "focused_v2"
    assert digest(default.model_dump(mode="json")) != digest(focused.model_dump(mode="json"))


def test_strategy_is_bound_to_parent_and_stage_requests(publication_scope):
    root, spec = publication_scope
    focused = amended(spec, retrieval_strategy="focused_v2")
    legacy_request = runner._prepare(root, spec)[3]
    focused_request = runner._prepare(root, focused)[3]
    assert legacy_request["spec"]["retrieval_strategy"] == "lexical_v1"
    assert focused_request["spec"]["retrieval_strategy"] == "focused_v2"
    assert legacy_request["source_snapshot_sha256"] == focused_request["source_snapshot_sha256"]
    assert legacy_request["profile"] == focused_request["profile"]
    assert legacy_request["spec_sha256"] != focused_request["spec_sha256"]
    assert digest(legacy_request) != digest(focused_request)
    # Even identical source excerpts/prompts cannot reuse a differently scoped stage.
    stage = runner._stage(root, spec, legacy_request, "check", "Same frozen synthetic prompt")
    saved = json.loads((stage.run.path / "request.json").read_bytes())
    assert saved["request"]["parent_sha256"] == digest(legacy_request)
    with pytest.raises(RunConflictError):
        runner._stage(root, focused, focused_request, "check", "Same frozen synthetic prompt")


def test_focused_run_persists_selection_and_rejects_changed_strategy_replay(publication_scope, monkeypatch):
    root, spec = publication_scope
    focused = amended(spec, retrieval_strategy="focused_v2", allow_automatic_patches=False,
                      generate_proposal=False, max_calls=2)
    first = runner.run_publication(root, focused)
    assert first["new_model_calls"] == 2 and first["new_live_requests"] == 0
    sidecar = json.loads((root / first["sidecar_relative"]).read_bytes())
    assert sidecar["support_plan"]["retrieval_strategy"] == "focused_v2"
    assert sidecar["support_plan"]["version"] == "publication-checks-2"
    before = snapshots(root)
    monkeypatch.setattr(runner, "run_fixed", lambda *a, **kw: pytest.fail("Replay must not extract"))
    monkeypatch.setattr(runner, "_provider", lambda *a, **kw: pytest.fail("Replay must not create a provider"))
    replay = runner.run_publication(root, focused)
    assert replay["execution"] == "cached_replay" and replay["new_model_calls"] == 0
    with pytest.raises(RunConflictError):
        runner.run_publication(root, amended(focused, retrieval_strategy="lexical_v1"))
    assert snapshots(root) == before
    service = PublicationService(root, amended(focused, retrieval_strategy="lexical_v1"))
    with pytest.raises(PilotError, match="publication_scope_changed"):
        service.status(focused.run_id)
