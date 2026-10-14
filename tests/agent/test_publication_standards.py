"""Offline integration diagnostics; all standards and review fixtures are invented."""
from pathlib import Path
import shutil

import pytest

from extraction_fixtures import synthetic_bundle, synthetic_record
from test_nexus_registry import request, review, synthetic_snapshot
from test_panet_mapping import FAMILY, NS, synthetic_ontology
from meta_data_extraction.agent.contracts import AssertionEvidence, canonical_json, owner, snapshot
from meta_data_extraction.agent.evidence import evidence_ref, resolve_evidence, sha256
from meta_data_extraction.agent.nexus_registry import build_registry, propose_entry
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.agent.publication_standards import (
    NeXusRuntimeSelection, NeXusSettings, PaNETRecordReport, PaNETSettings,
    load_reviewed_nexus, map_record_techniques, standards_input_pins,
)


@pytest.fixture
def profile(workspace):
    shutil.copyfile(Path(__file__).resolve().parents[2] / "schemas/prompt_template.md",
                    workspace / "schemas/prompt_template.md")
    return load_profile(workspace)


@pytest.fixture
def setup(workspace, profile):
    ontology = synthetic_ontology(workspace)
    bundle = synthetic_bundle()
    record = snapshot(synthetic_record(profile), bundle, profile)
    settings = PaNETSettings(resource=ontology.resource, families=(FAMILY,))
    return bundle, record, settings


def assertion_for(record, refs):
    path = "/datasets/0/measurement_technique"
    entity = owner(record.entities, path)
    return AssertionEvidence(
        record_revision=record.revision_id, entity_handle=entity.handle, path=path,
        field_path=path[len(entity.path):],
        value_json=canonical_json(record.canonical["datasets"][0]["measurement_technique"]),
        link_status="location_verified", evidence=refs,
    )


def test_fresh_record_gets_bound_candidates_without_scientific_approval(workspace, setup):
    bundle, record, settings = setup
    before = record.model_dump()
    report = map_record_techniques(workspace, bundle, record, settings)
    technique = report.techniques[0]
    decision = technique.decisions[0]
    assert technique.status == "inspected"
    assert technique.evidence_origin == "source_literal_search"
    assert technique.candidates.candidates[0].concept_id == NS + "xrd"
    assert technique.original_expression == "X-ray diffraction"
    assert decision.binding.revision_id == record.revision_id
    assert decision.binding.entity_handle == record.entities[-1].handle
    assert decision.status == "unresolved" and decision.reason == "usage_uncertain"
    assert decision.proposed_id is None and decision.approval_status == "unreviewed"
    assert resolve_evidence(bundle, decision.mention.mention) == "X-ray diffraction"
    assert record.model_dump() == before
    assert report.canonical_modified is False and report.model_calls == 0
    assert PaNETRecordReport.model_validate_json(report.model_dump_json()) == report
    assert map_record_techniques(workspace, bundle, record, settings) == report


def test_linked_assertions_restrict_search_to_their_actual_regions(workspace, profile, setup):
    _, _, settings = setup
    bundle = synthetic_bundle("Related work uses X-ray diffraction.\n\nThis study uses X-ray diffraction.\n")
    record = snapshot(synthetic_record(profile), bundle, profile)
    ref = evidence_ref(bundle, bundle.regions[1].region_id)
    assertion = assertion_for(record, (ref,))
    report = map_record_techniques(workspace, bundle, record, settings, assertions=(assertion,))
    item = report.techniques[0]
    assert item.evidence_origin == "assertion_evidence" and len(item.decisions) == 1
    assert item.decisions[0].mention.context == ref
    assert item.decisions[0].mention.context_use == "uncertain"


def test_linked_assertion_without_literal_never_falls_back_to_unrelated_source(workspace, profile, setup):
    _, _, settings = setup
    bundle = synthetic_bundle("Related work uses X-ray diffraction.\n\nThis study uses another technique.\n")
    record = snapshot(synthetic_record(profile), bundle, profile)
    assertion = assertion_for(record, (evidence_ref(bundle, bundle.regions[1].region_id),))
    item = map_record_techniques(workspace, bundle, record, settings, assertions=(assertion,)).techniques[0]
    assert item.status == "not_located" and item.decisions == ()
    assert item.candidates.candidates[0].concept_id == NS + "xrd"


def test_overlapping_evidence_deduplicates_and_mentions_are_bounded(workspace, profile, setup):
    _, _, settings = setup
    bundle = synthetic_bundle("X-ray diffraction. " * 10)
    record = snapshot(synthetic_record(profile), bundle, profile)
    ref = evidence_ref(bundle, bundle.regions[0].region_id)
    item = map_record_techniques(workspace, bundle, record, settings,
                                  assertions=(assertion_for(record, (ref, ref)),)).techniques[0]
    assert len(item.decisions) == 4 and item.mentions_truncated
    assert len({item.mention.mention.start for item in item.decisions}) == 4


@pytest.mark.parametrize("expression,status", [("XRD", "not_located"), ("   ", "not_searchable"),
                                               ("?", "not_searchable"), ("X" * 4097, "not_searchable")],
                         ids=["nonliteral", "blank", "punctuation", "oversized"])
def test_nonliteral_and_unsearchable_fields_remain_explicit(workspace, profile, setup, expression, status):
    bundle, record, settings = setup
    canonical = record.canonical
    canonical["datasets"][0]["measurement_technique"] = expression
    changed = snapshot(canonical, bundle, profile)
    item = map_record_techniques(workspace, bundle, changed, settings).techniques[0]
    assert item.status == status and item.decisions == ()
    assert item.original_expression == expression


def test_missing_technique_is_not_invented(workspace, profile, setup):
    bundle, record, settings = setup
    canonical = record.canonical
    canonical["datasets"][0]["measurement_technique"] = None
    changed = snapshot(canonical, bundle, profile)
    assert map_record_techniques(workspace, bundle, changed, settings).techniques == ()


@pytest.mark.parametrize("field", ["record_revision", "entity_handle", "field_path", "value_json"])
def test_stale_assertion_bindings_are_rejected(workspace, setup, field):
    bundle, record, settings = setup
    assertion = assertion_for(record, (evidence_ref(bundle, bundle.regions[0].region_id),))
    changed = assertion.model_copy(update={field: "stale"})
    with pytest.raises(ValueError):
        map_record_techniques(workspace, bundle, record, settings, assertions=(changed,))


def test_duplicate_or_forged_assertion_evidence_is_rejected(workspace, setup):
    bundle, record, settings = setup
    ref = evidence_ref(bundle, bundle.regions[0].region_id)
    assertion = assertion_for(record, (ref,))
    with pytest.raises(ValueError, match="Duplicate"):
        map_record_techniques(workspace, bundle, record, settings, assertions=(assertion, assertion))
    changed = assertion.model_copy(update={"evidence": (ref.model_copy(update={"quotation": "forged"}),)})
    with pytest.raises(ValueError):
        map_record_techniques(workspace, bundle, record, settings, assertions=(changed,))


def test_edited_record_requires_new_revision_sidecar(workspace, profile, setup):
    bundle, record, settings = setup
    old = map_record_techniques(workspace, bundle, record, settings)
    canonical = record.canonical
    canonical["datasets"][0]["dataset_description"] = "Changed record revision"
    changed = snapshot(canonical, bundle, profile)
    new = map_record_techniques(workspace, bundle, changed, settings)
    assert new.report_id != old.report_id and new.revision_id != old.revision_id
    assert new.techniques[0].entity_handle == old.techniques[0].entity_handle
    corrupted = old.model_dump()
    corrupted["revision_id"] = new.revision_id
    with pytest.raises(ValueError):
        PaNETRecordReport.model_validate(corrupted)


def test_different_source_resource_change_and_bad_family_fail_closed(workspace, setup):
    bundle, record, settings = setup
    with pytest.raises(ValueError, match="source snapshot"):
        map_record_techniques(workspace, synthetic_bundle("Another source."), record, settings)
    with pytest.raises(ValueError, match="Families"):
        map_record_techniques(workspace, bundle, record, settings.model_copy(update={"families": ("unknown",)}))
    (workspace / settings.resource.path).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        map_record_techniques(workspace, bundle, record, settings)


def nexus_settings(workspace, profile, *, decision=None, unresolved=False):
    nxdl = synthetic_snapshot(workspace, extends="NXmissing" if unresolved else "")
    entry = propose_entry(profile, nxdl, request())
    registry = build_registry(profile, nxdl, (entry,),
                               reviews=() if decision is None else (review(entry, decision=decision),))
    relative = "tests/agent/fixtures/publication-registry.json"
    raw = registry.model_dump_json().encode("utf-8")
    (workspace / relative).write_bytes(raw)
    return NeXusSettings(resource=nxdl.resource, registry_path=relative, registry_sha256=sha256(raw))


@pytest.mark.parametrize("decision", [None, "rejected", "needs_changes"])
def test_nexus_unapproved_registry_is_explicitly_blocked(workspace, profile, decision):
    settings = nexus_settings(workspace, profile, decision=decision)
    result = load_reviewed_nexus(workspace, profile, settings)
    assert result.status == "blocked_no_approved_guidance" and result.definition_requests == {}
    assert result.publication_support == result.application_compliance == "not_assessed"


def test_nexus_approved_fixture_selection_retains_explicit_synthetic_provenance(workspace, profile):
    settings = nexus_settings(workspace, profile, decision="approved")
    result = load_reviewed_nexus(workspace, profile, settings)
    assert result.status == "ready"
    assert result.definition_requests == {"sample_mass": ("/samples/*/sample_mass", "specimen_mass")}
    assert result.snapshot.resource.kind == "synthetic"
    assert result.registry.entries[0].review.reviewer == "Synthetic fixture reviewer"
    assert NeXusRuntimeSelection.model_validate_json(result.model_dump_json()) == result


def test_nexus_unresolved_resource_never_supplies_runtime_guidance(workspace, profile):
    settings = nexus_settings(workspace, profile, unresolved=True)
    result = load_reviewed_nexus(workspace, profile, settings)
    assert result.status == "blocked_no_approved_guidance"
    assert result.registry.entries[0].entry.resolution_status == "unresolved"


def test_nexus_modified_registry_or_profile_fails_closed(workspace, profile):
    settings = nexus_settings(workspace, profile, decision="approved")
    changed = profile.model_copy(update={"profile_id": "forged"})
    with pytest.raises(ValueError):
        load_reviewed_nexus(workspace, changed, settings)
    path = workspace / settings.registry_path
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="pinned content"):
        load_reviewed_nexus(workspace, profile, settings)


def test_nexus_duplicate_registry_keys_rejected_even_with_matching_hash(workspace, profile):
    settings = nexus_settings(workspace, profile)
    path = workspace / settings.registry_path
    raw = path.read_bytes().replace(b'{', b'{"registry_version":"m05-registry-1",', 1)
    path.write_bytes(raw)
    changed = settings.model_copy(update={"registry_sha256": sha256(raw)})
    with pytest.raises(ValueError, match="Duplicate"):
        load_reviewed_nexus(workspace, profile, changed)


def test_nexus_cannot_enable_a_definition_outside_reviewed_registry(workspace, profile):
    settings = nexus_settings(workspace, profile)
    result = load_reviewed_nexus(workspace, profile, settings)
    payload = result.model_dump()
    payload.update(status="ready", definition_requests={"forged": ("/samples/*/sample_mass", "specimen_mass")})
    with pytest.raises(ValueError):
        NeXusRuntimeSelection.model_validate(payload)


def test_freeze_pins_lists_only_explicit_standards_inputs_and_detects_changes(workspace, profile, setup):
    _, _, panet = setup
    nexus = nexus_settings(workspace, profile)
    pins = standards_input_pins(workspace, panet=panet, nexus=nexus)
    assert pins == dict(sorted({panet.resource.path: panet.resource.sha256,
                               nexus.resource.files[0].path: nexus.resource.files[0].sha256,
                               nexus.registry_path: nexus.registry_sha256}.items()))
    assert standards_input_pins(workspace) == {}
    assert PaNETSettings.model_validate_json(panet.model_dump_json()) == panet
    assert NeXusSettings.model_validate_json(nexus.model_dump_json()) == nexus
    (workspace / nexus.registry_path).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        standards_input_pins(workspace, nexus=nexus)


def test_registry_cannot_read_offline_gold_paths(workspace, profile):
    settings = nexus_settings(workspace, profile)
    changed = settings.model_copy(update={"registry_path": "data/gold/forbidden.json"})
    with pytest.raises(ValueError):
        load_reviewed_nexus(workspace, profile, changed)
