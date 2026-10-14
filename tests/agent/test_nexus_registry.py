"""Synthetic registry/review fixtures; no actual human approval or live calls."""
from copy import deepcopy
from pathlib import Path
import json
import shutil

import pytest
from pydantic import ValidationError

from meta_data_extraction.agent.contracts import canonical_json, digest
from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.agent.nexus_registry import (
    Guidance, MappingEntry, MappingRequest, ProfileRegistry, RegistryError,
    ReviewRecord, ReviewedMapping, build_registry, lookup_guidance,
    propose_entry, verify_registry,
)
from meta_data_extraction.agent.nexus_resources import NXDLFileSpec, NXDLResourceSpec, load_nexus
from meta_data_extraction.agent.profiles import load_profile


def synthetic_snapshot(workspace, *, mass_units="NX_MASS", extends="", extra="", documentation="Synthetic mass"):
    """Only invented definitions; these fixtures are not verified NeXus content."""
    inheritance = f' extends="{extends}"' if extends else ""
    raw = f'''<definition xmlns="urn:synthetic:m05:registry"
      name="NXsample" category="base"{inheritance}>
      <doc>Explicitly synthetic registry fixture.</doc>
      <field name="name" type="NX_CHAR" minOccurs="0"><doc>Synthetic sample name</doc></field>
      <field name="chemical_formula" type="NX_CHAR" minOccurs="0"><doc>Synthetic composition</doc></field>
      <field name="mass" type="NX_FLOAT" units="{mass_units}" minOccurs="0"><doc>{documentation}</doc></field>
      <field name="temperature" type="NX_FLOAT" units="NX_TEMPERATURE" minOccurs="1">
        <doc>Synthetic acquisition temperature</doc></field>
      <field name="preparation_temperature" type="NX_FLOAT" units="NX_TEMPERATURE" minOccurs="0">
        <doc>Synthetic preparation temperature</doc></field>
      <field name="temperature_without_units" type="NX_FLOAT" minOccurs="0"><doc>Synthetic unspecified unit</doc></field>
      {extra}
    </definition>'''.encode("utf-8")
    relative = "tests/agent/fixtures/synthetic_registry.nxdl.xml"
    destination = workspace / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    spec = NXDLResourceSpec(kind="synthetic", commit="synthetic-m05", release="synthetic-v1",
                            files=(NXDLFileSpec(path=relative, sha256=sha256(raw),
                                                source_url="urn:synthetic:m05:registry"),),
                            license_name="synthetic-only", license_url="urn:synthetic:m05:license")
    return load_nexus(workspace, spec)


@pytest.fixture
def profile(workspace):
    source = Path(__file__).resolve().parents[2] / "schemas/prompt_template.md"
    shutil.copyfile(source, workspace / "schemas/prompt_template.md")
    return load_profile(workspace)


@pytest.fixture
def nxdl(workspace):
    return synthetic_snapshot(workspace)


def request(**changes):
    values = dict(mapping_key="sample_mass", canonical_field="/samples/*/sample_mass",
                  semantic_role="specimen_mass", relation="partial", target_class="NXsample",
                  target_path="mass", target_role="specimen_mass",
                  scope="Only the sample used for measurement; source expression preserved.")
    values.update(changes)
    return MappingRequest(**values)


def temperature_request(*, preparation=False, **changes):
    role = "preparation_temperature" if preparation else "acquisition_temperature"
    values = dict(mapping_key=role,
                  canonical_field="/samples/*/sample_description" if preparation else "/datasets/*/dataset_description",
                  semantic_role=role, relation="partial", target_class="NXsample",
                  target_path="preparation_temperature" if preparation else "temperature",
                  target_role=role, scope="One temperature aspect of the existing free-text description.")
    values.update(changes)
    return MappingRequest(**values)


def review(entry, *, decision="approved", **changes):
    # This is a fabricated UNIT TEST artifact, not a real expert judgment.
    values = dict(review_id="synthetic-test-review", entry_id=entry.entry_id,
                  entry_sha256=entry.entry_sha256, reviewer="Synthetic fixture reviewer",
                  provenance="supplied_human_review", decision=decision,
                  reviewed_at="2026-10-07T10:00:00Z", rationale="Synthetic fixture to test content binding only.")
    values.update(changes)
    return ReviewRecord(**values)


def test_proposal_binds_exact_inherited_field_and_resource_without_approval(profile, nxdl):
    entry = propose_entry(profile, nxdl, request())
    assert entry.profile_id == profile.profile_id
    assert entry.resource_snapshot_id == nxdl.snapshot_id
    assert json.loads(entry.field_definition_json) == profile.schema["samples"]["item_schema"]["sample_mass"]
    assert entry.field_definition_sha256 == digest(json.loads(entry.field_definition_json))
    assert entry.target.nx_type == "NX_FLOAT" and entry.target.units == "NX_MASS"
    assert entry.target.class_name == "NXsample"
    assert entry.approval_status == "unreviewed" and entry.support_status == "not_assessed"
    assert entry.scientific_value_policy == "source_evidence_required"
    assert entry.application_compliance == "not_assessed"
    assert entry.request.conversion_policy == "preserve_raw"


def test_guidance_excludes_unreviewed_by_default_and_labels_explicit_opt_in(profile, nxdl):
    entry = propose_entry(profile, nxdl, request())
    registry = build_registry(profile, nxdl, (entry,))
    assert lookup_guidance(profile, nxdl, registry, entry.request.canonical_field, "specimen_mass") == ()
    proposed = lookup_guidance(profile, nxdl, registry, entry.request.canonical_field, "specimen_mass", include_proposals=True)
    assert len(proposed) == 1 and proposed[0].approval_status == "unreviewed"
    assert proposed[0].review is None and proposed[0].support_status == "not_assessed"


def test_supplied_content_bound_review_unlocks_guidance_without_claiming_source_support(profile, nxdl):
    entry = propose_entry(profile, nxdl, request())
    artifact = review(entry)
    registry = build_registry(profile, nxdl, (entry,), (artifact,))
    result = lookup_guidance(profile, nxdl, registry, entry.request.canonical_field, "specimen_mass")
    assert len(result) == 1 and result[0].approval_status == "approved"
    assert result[0].review == artifact and result[0].support_status == "not_assessed"
    assert entry.approval_status == "unreviewed"
    verify_registry(profile, nxdl, registry)


@pytest.mark.parametrize("decision", ["rejected", "needs_changes"])
def test_rejected_and_changes_requested_are_never_guidance(profile, nxdl, decision):
    entry = propose_entry(profile, nxdl, request())
    registry = build_registry(profile, nxdl, (entry,), (review(entry, decision=decision),))
    assert lookup_guidance(profile, nxdl, registry, entry.request.canonical_field, "specimen_mass", include_proposals=True) == ()


def test_forged_approval_label_without_review_is_rejected(profile, nxdl):
    entry = propose_entry(profile, nxdl, request())
    with pytest.raises(ValidationError, match="Approval requires"):
        ReviewedMapping(entry=entry, approval_status="approved")
    with pytest.raises(ValidationError):
        MappingEntry.model_validate({**entry.model_dump(), "approval_status": "approved"})


def test_equal_quantity_in_preparation_and_acquisition_keeps_distinct_roles(profile, nxdl):
    prepared = propose_entry(profile, nxdl, temperature_request(preparation=True))
    measured = propose_entry(profile, nxdl, temperature_request())
    registry = build_registry(profile, nxdl, (prepared, measured))
    result = lookup_guidance(profile, nxdl, registry, measured.request.canonical_field,
                             "acquisition_temperature", include_proposals=True)
    assert len(result) == 1 and result[0].target.field_path == "temperature"
    assert prepared.request.semantic_role != measured.request.semantic_role
    assert prepared.target.units == measured.target.units == "NX_TEMPERATURE"
    assert prepared.entry_id != measured.entry_id


def test_role_transfer_is_rejected_even_when_units_are_the_same():
    with pytest.raises(ValidationError, match="semantic-role transfer"):
        temperature_request(preparation=True, target_role="acquisition_temperature")


def test_nxsample_temperature_cannot_be_relabeled_as_preparation(profile, nxdl):
    with pytest.raises(RegistryError, match="sample preparation"):
        propose_entry(profile, nxdl, temperature_request(preparation=True, target_path="temperature"))


@pytest.mark.parametrize("relation", ["exact", "equivalent"])
@pytest.mark.parametrize("preparation", [False, True])
def test_temperature_aspect_cannot_replace_entire_description(relation, preparation):
    with pytest.raises(ValidationError, match="partially maps"):
        temperature_request(preparation=preparation, relation=relation)


@pytest.mark.parametrize("relation", ["exact", "equivalent"])
def test_numeric_nxdl_representation_is_not_equivalent_to_raw_string(profile, nxdl, relation):
    with pytest.raises(RegistryError, match="partial mapping"):
        propose_entry(profile, nxdl, request(relation=relation))


@pytest.mark.parametrize("relation", ["exact", "equivalent", "partial"])
def test_relation_types_are_retained_for_compatible_text_proposals(profile, nxdl, relation):
    entry = propose_entry(profile, nxdl, request(mapping_key="sample_name", canonical_field="/samples/*/sample_name",
                                               semantic_role="sample_identity", target_path="name",
                                               target_role="sample_identity", relation=relation))
    assert entry.request.relation == relation and entry.approval_status == "unreviewed"


@pytest.mark.parametrize("units", ["NX_LENGTH", "NX_TEMPERATURE", "NX_UNITLESS"])
def test_dimension_mismatch_rejected(profile, workspace, units):
    nxdl = synthetic_snapshot(workspace, mass_units=units)
    with pytest.raises(RegistryError, match="units category"):
        propose_entry(profile, nxdl, request())


def test_explicit_units_policy_cannot_contradict_role():
    with pytest.raises(ValidationError, match="incompatible"):
        request(expected_units="NX_TEMPERATURE")


def test_unit_category_not_stated_stays_unresolved(profile, nxdl):
    entry = propose_entry(profile, nxdl, temperature_request(target_path="temperature_without_units"))
    assert entry.resolution_status == "unresolved" and "target_units_not_stated" in entry.unresolved
    with pytest.raises(ValidationError, match="Unresolved resource"):
        build_registry(profile, nxdl, (entry,), (review(entry),))


def test_missing_definition_has_explicit_diagnostic_not_empty_success(profile, nxdl):
    entry = propose_entry(profile, nxdl, request(target_path="missing_field"))
    assert entry.target is None and entry.resolution_status == "unresolved"
    assert entry.unresolved == ("target_definition_unavailable",)
    registry = build_registry(profile, nxdl, (entry,))
    assert lookup_guidance(profile, nxdl, registry, entry.request.canonical_field, "specimen_mass", include_proposals=True) == ()


def test_unresolved_inheritance_cannot_be_approved(profile, workspace):
    nxdl = synthetic_snapshot(workspace, extends="NXcomponent")
    entry = propose_entry(profile, nxdl, request())
    assert entry.resolution_status == "unresolved" and entry.unresolved
    registry = build_registry(profile, nxdl, (entry,))
    guidance = lookup_guidance(profile, nxdl, registry, entry.request.canonical_field, "specimen_mass", include_proposals=True)
    assert guidance[0].resolution_status == "unresolved"
    with pytest.raises(ValidationError, match="Unresolved resource"):
        build_registry(profile, nxdl, (entry,), (review(entry),))


def test_no_suitable_mapping_has_no_assigned_target(profile, nxdl):
    entry = propose_entry(profile, nxdl, request(mapping_key="origin_no_match",
        canonical_field="/samples/*/supplier_or_creator", semantic_role="sample_origin",
        relation="no_match", target_class=None, target_path=None, target_role=None,
        scope="No selected definition covers the inherited provenance instructions."))
    assert entry.resolution_status == "no_match" and entry.target is None
    registry = build_registry(profile, nxdl, (entry,))
    assert lookup_guidance(profile, nxdl, registry, entry.request.canonical_field, "sample_origin", include_proposals=True) == ()


def test_no_match_cannot_smuggle_target():
    with pytest.raises(ValidationError, match="No-match"):
        request(relation="no_match")


def test_required_experimental_data_does_not_produce_canonical_value(profile, nxdl):
    before = profile.model_dump_json()
    entry = propose_entry(profile, nxdl, temperature_request())
    assert entry.target.min_occurs == "1"
    registry = build_registry(profile, nxdl, (entry,))
    result = lookup_guidance(profile, nxdl, registry, entry.request.canonical_field,
                             "acquisition_temperature", include_proposals=True)
    assert result[0].scientific_value_policy == "source_evidence_required"
    assert "value" not in Guidance.model_fields and "canonical" not in ProfileRegistry.model_fields
    assert profile.model_dump_json() == before


def test_changed_scope_invalidates_supplied_review(profile, nxdl):
    initial = propose_entry(profile, nxdl, request())
    changed = propose_entry(profile, nxdl, request(scope="Different scope requires a different review."))
    with pytest.raises(RegistryError, match="stale review"):
        build_registry(profile, nxdl, (changed,), (review(initial),))


def test_changed_profile_field_invalidates_entry_and_review(profile, nxdl):
    entry = propose_entry(profile, nxdl, request())
    changed = profile.schema
    changed["samples"]["item_schema"]["sample_mass"]["description"] = "Changed meaning"
    updated = profile.model_copy(update={"schema_payload_json": json.dumps(changed)})
    with pytest.raises(RegistryError, match="current profile"):
        build_registry(updated, nxdl, (entry,), (review(entry),))


def test_changed_definition_content_or_version_invalidates_entry_and_review(profile, workspace, nxdl):
    entry = propose_entry(profile, nxdl, request())
    changed = synthetic_snapshot(workspace, documentation="Changed synthetic mass semantics")
    with pytest.raises(RegistryError, match="current profile or NXDL"):
        build_registry(profile, changed, (entry,), (review(entry),))


def test_stale_review_content_hash_is_rejected(profile, nxdl):
    entry = propose_entry(profile, nxdl, request())
    with pytest.raises(ValidationError, match="content hash"):
        review(entry, entry_sha256="0" * 64)


@pytest.mark.parametrize("changes", [
    {"review_version": "m05-review-0"}, {"provenance": "model_generated"},
    {"reviewer": " "}, {"rationale": " "}, {"review_id": " "},
    {"reviewed_at": "2026-10-07"}, {"reviewed_at": "invalid"},
    {"decision": "pending"}, {"approval": True},
])
def test_review_requires_typed_versioned_supplied_metadata(profile, nxdl, changes):
    entry = propose_entry(profile, nxdl, request())
    with pytest.raises(ValidationError):
        review(entry, **changes)


def test_duplicate_entry_and_duplicate_reviews_are_rejected(profile, nxdl):
    entry = propose_entry(profile, nxdl, request())
    with pytest.raises(RegistryError, match="Duplicate mapping"):
        build_registry(profile, nxdl, (entry, entry))
    with pytest.raises(RegistryError):
        build_registry(profile, nxdl, (entry,), (review(entry), review(entry)))


def test_duplicate_keys_and_review_ids_are_rejected(profile, nxdl):
    first = propose_entry(profile, nxdl, request())
    second = propose_entry(profile, nxdl, request(scope="Different partial scope"))
    with pytest.raises(ValidationError, match="Duplicate mapping keys"):
        build_registry(profile, nxdl, (first, second))
    second = propose_entry(profile, nxdl, request(mapping_key="alternate_mass", scope="Different partial scope"))
    with pytest.raises(ValidationError, match="review IDs"):
        build_registry(profile, nxdl, (first, second), (review(first), review(second)))


def test_tampered_entry_and_registry_digests_are_rejected(profile, nxdl):
    entry = propose_entry(profile, nxdl, request())
    with pytest.raises(ValidationError, match="fingerprint"):
        MappingEntry.model_validate({**entry.model_dump(), "field_definition_json": "{}"})
    registry = build_registry(profile, nxdl, (entry,))
    with pytest.raises(ValidationError, match="fingerprint"):
        ProfileRegistry.model_validate({**registry.model_dump(), "registry_id": "forged"})


def test_rehashed_forged_target_still_fails_current_resource_verification(profile, nxdl):
    entry = propose_entry(profile, nxdl, request())
    # Even a content-consistent forged entry must match the resource retrieval.
    altered = entry.model_dump(mode="json")
    altered["field_definition_json"] = canonical_json({"type": "string", "description": "invented"})
    altered["field_definition_sha256"] = digest(json.loads(altered["field_definition_json"]))
    altered["entry_sha256"] = digest({k: v for k, v in altered.items() if k not in {"entry_id", "entry_sha256"}})
    altered["entry_id"] = "nexus-mapping:" + altered["entry_sha256"]
    with pytest.raises(RegistryError, match="current profile"):
        build_registry(profile, nxdl, (MappingEntry.model_validate(altered),))


def test_json_round_trip_retains_reviews_and_revalidates_current_inputs(profile, nxdl):
    entry = propose_entry(profile, nxdl, request())
    registry = build_registry(profile, nxdl, (entry,), (review(entry),))
    replay = ProfileRegistry.model_validate_json(registry.model_dump_json())
    assert replay == registry
    verify_registry(profile, nxdl, replay)


@pytest.mark.parametrize("changes", [
    {"canonical_field": "/samples/0/sample_mass"},
    {"canonical_field": "/samples/*/new_temperature"},
    {"semantic_role": "acquisition_temperature"},
    {"target_path": None}, {"target_class": " "}, {"scope": " "},
    {"conversion_policy": "convert_automatically"}, {"scientific_value": 300},
])
def test_invalid_or_expansive_request_is_rejected(changes):
    with pytest.raises(ValidationError):
        request(**changes)


def test_free_text_lookup_cannot_cross_semantic_role(profile, nxdl):
    entry = propose_entry(profile, nxdl, temperature_request())
    registry = build_registry(profile, nxdl, (entry,))
    with pytest.raises(RegistryError, match="outside"):
        lookup_guidance(profile, nxdl, registry, entry.request.canonical_field,
                        "preparation_temperature", include_proposals=True)


def test_empty_registry_cannot_masquerade_as_completed_mapping_review(profile, nxdl):
    with pytest.raises(RegistryError):
        build_registry(profile, nxdl, ())


def test_unrelated_group_dependency_does_not_block_direct_field_guidance(profile, workspace):
    nxdl = synthetic_snapshot(workspace, extra='''<group type="NXexternal">
      <field name="external_value" type="NX_FLOAT" units="NX_LENGTH" minOccurs="0"/>
    </group>''')
    entry = propose_entry(profile, nxdl, request())
    assert entry.resolution_status == "resolved"
    registry = build_registry(profile, nxdl, (entry,), (review(entry),))
    assert len(lookup_guidance(profile, nxdl, registry, entry.request.canonical_field, "specimen_mass")) == 1


@pytest.mark.parametrize("opt_in", ["false", "true", 0, 1, None])
def test_proposal_opt_in_requires_boolean(profile, nxdl, opt_in):
    entry = propose_entry(profile, nxdl, request())
    registry = build_registry(profile, nxdl, (entry,))
    with pytest.raises(RegistryError, match="boolean"):
        lookup_guidance(profile, nxdl, registry, entry.request.canonical_field,
                        "specimen_mass", include_proposals=opt_in)


@pytest.mark.parametrize("relation", ["exact", "equivalent"])
def test_boolean_nxdl_representation_is_not_equivalent_to_raw_string(profile, workspace, relation):
    nxdl = synthetic_snapshot(workspace, extra='''<field name="synthetic_flag"
      type="NX_BOOLEAN" minOccurs="0"><doc>Synthetic boolean; not a sample name</doc></field>''')
    mapping = request(mapping_key="boolean_target", canonical_field="/samples/*/sample_name",
                      semantic_role="sample_identity", target_path="synthetic_flag",
                      target_role="sample_identity", relation=relation)
    with pytest.raises(RegistryError, match="partial mapping"):
        propose_entry(profile, nxdl, mapping)
