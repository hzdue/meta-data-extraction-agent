"""Synthetic structural checks; no annotations, evaluation helpers, or models."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest
from pydantic import ValidationError

from meta_data_extraction import utils
from meta_data_extraction.agent.profiles import (
    Profile, ProfileError, entity_addresses, load_profile, reference_addresses, render_inherited_prompt,
    validate_record, verify_profile_resources,
)


@pytest.fixture
def profile_workspace(workspace):
    repository = Path(__file__).resolve().parents[2]
    for name in ("prompt_template.md", "generic_no_instructions.json", "photon_and_neutron_no_instructions.json"):
        shutil.copyfile(repository / "schemas" / name, workspace / "schemas" / name)
    return workspace


@pytest.fixture
def profile(profile_workspace):
    return load_profile(profile_workspace)


def empty_record(profile):
    record = utils.build_skeleton(profile.schema)
    # Synthetic source explicitly states these are absent. Do not turn the
    # inherited null prompt placeholders into an unknown -> false conversion.
    for field in ("has_references", "has_figures", "has_tables"):
        record["document_overview"][field] = False
    return record


def populated_record(profile):
    record = empty_record(profile)
    record["bibliographic"]["title"] = "Synthetic sample A at 80 K"
    record["bibliographic"]["affiliations"] = [{"id": "aff_1", "label": None, "text": "Synthetic Institute"}]
    record["bibliographic"]["authors"] = [{
        "id": "author_1", "full_name": "A. Example", "orcid": None,
        "email": None, "is_corresponding": False, "affiliation_refs": ["aff_1"],
    }]
    record["facilities"] = [{"id": "facility_1", "name": "Synthetic Beam Source"}]
    sample = utils.build_skeleton(profile.schema["samples"]["item_schema"])
    sample.update(id="sample_1", sample_name="Sample A")
    record["samples"] = [sample]
    dataset = utils.build_skeleton(profile.schema["datasets"]["item_schema"])
    dataset.update(id="dataset_1", dataset_name="Dataset A", sample_refs=["sample_1"],
                   facility_refs=["facility_1"], measurement_technique="X-ray diffraction")
    record["datasets"] = [dataset]
    return record


def test_freezes_resources_with_original_versions(profile, profile_workspace):
    assert profile.schema_names == ("generic.json", "photon_and_neutron.json")
    assert [f.version for f in profile.schema_files] == ["1.1.0", "1.1.0"]
    assert profile.prompt_version == "1.1.0"
    assert all(len(f.sha256) == 64 for f in profile.schema_files)
    assert len(profile.schema_sha256) == len(profile.prompt_sha256) == 64
    assert profile == load_profile(profile_workspace)
    assert profile == Profile.model_validate_json(profile.model_dump_json())
    verify_profile_resources(profile, profile_workspace)
    with pytest.raises(ValidationError):
        profile.schema_json = "{}"
    schema = profile.schema
    schema.clear()
    assert "samples" in profile.schema


def test_reuses_schema_loading_and_matches_inherited_prompt(profile, profile_workspace, monkeypatch):
    monkeypatch.setattr(utils, "SCHEMAS_DIR", profile_workspace / "schemas")
    monkeypatch.setattr(utils, "PROMPT_TEMPLATE", profile_workspace / "schemas/prompt_template.md")
    assert profile.schema == utils.load_merged_schema(list(profile.schema_names))
    text = "Sample café has 80 K. {{SCHEMA_JSON}} {{OUTPUT_SKELETON}} {{DOCUMENT_TEXT}}"
    assert render_inherited_prompt(profile, text) == utils.build_prompt(list(profile.schema_names), text)
    assert render_inherited_prompt(profile, text).endswith(text)


@pytest.mark.parametrize("filename", ["generic.json", "photon_and_neutron.json", "prompt_template.md"])
def test_resource_changes_invalidate_profile_but_not_snapshot(profile, profile_workspace, filename):
    prompt_before = render_inherited_prompt(profile, "Synthetic source")
    path = profile_workspace / "schemas" / filename
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    updated = load_profile(profile_workspace)
    assert updated.profile_id != profile.profile_id
    assert render_inherited_prompt(profile, "Synthetic source") == prompt_before
    with pytest.raises(ProfileError, match="differs"):
        verify_profile_resources(profile, profile_workspace)


def test_ablated_resources_have_same_shapes_and_distinct_identity(profile, profile_workspace):
    ablated = load_profile(profile_workspace, ("generic_no_instructions.json", "photon_and_neutron_no_instructions.json"))
    assert ablated.profile_id != profile.profile_id
    assert validate_record(populated_record(profile), ablated) == ()
    assert "instructions" not in ablated.schema["samples"]


@pytest.mark.parametrize("names", [
    (), ("generic.json", "generic.json"), ("../data/gold/synthetic.json",),
    ("unknown.json",), ("generic.json", "generic_no_instructions.json"),
])
def test_rejects_unknown_duplicate_or_conflicting_resources(profile_workspace, names):
    with pytest.raises(ProfileError):
        load_profile(profile_workspace, names)


def test_empty_extraction_is_distinct_from_empty_object(profile):
    record = empty_record(profile)
    assert validate_record(record, profile) == ()
    issues = validate_record({}, profile)
    assert {i.code for i in issues} == {"missing_field"}
    assert {i.path for i in issues} == {"/bibliographic", "/document_overview", "/facilities", "/samples", "/datasets"}


@pytest.mark.parametrize("record", [None, [], "{}", 0, True])
def test_nonobjects_never_succeed(profile, record):
    issues = validate_record(record, profile)
    assert len(issues) == 1
    assert (issues[0].code, issues[0].path) == ("type_mismatch", "")


def test_nullable_and_empty_lists_keep_canonical_meaning(profile):
    record = populated_record(profile)
    before = deepcopy(record)
    assert validate_record(record, profile) == ()
    assert record == before
    record["datasets"][0]["sample_refs"] = None
    record["samples"][0]["sample_name"] = None
    issues = validate_record(record, profile)
    assert {(i.code, i.path) for i in issues} == {
        ("type_mismatch", "/datasets/0/sample_refs"),
        ("type_mismatch", "/samples/0/sample_name"),
    }


def test_skeleton_boolean_ambiguity_is_visible_without_coercion(profile):
    record = utils.build_skeleton(profile.schema)
    issues = validate_record(record, profile)
    assert len(issues) == 3
    assert all(i.code == "type_mismatch" for i in issues)
    assert any("Unknown-boolean missingness is unresolved" in note for note in profile.validation_notes)
    assert record["document_overview"]["has_tables"] is None


@pytest.mark.parametrize("invalid", [0, 1, "true", None])
def test_booleans_are_not_coerced(profile, invalid):
    record = empty_record(profile)
    record["document_overview"]["has_figures"] = invalid
    assert [(i.code, i.path) for i in validate_record(record, profile)] == [
        ("type_mismatch", "/document_overview/has_figures")
    ]


def test_reports_unknown_and_missing_fields_without_dropping_them(profile):
    record = populated_record(profile)
    record["samples"][0]["temperature"] = "80 K"
    record["extra/key~"] = "unapproved"
    del record["datasets"][0]["purpose"]
    issues = validate_record(record, profile)
    assert {(i.code, i.path) for i in issues} == {
        ("unknown_field", "/samples/0/temperature"),
        ("unknown_field", "/extra~1key~0"),
        ("missing_field", "/datasets/0/purpose"),
    }
    assert record["samples"][0]["temperature"] == "80 K"


def test_declared_allowed_values_and_cardinality(profile):
    record = populated_record(profile)
    record["datasets"][0]["purpose"] = "guessed"
    record["datasets"][0]["measurement_mode"] = ["Snapshot", "MadeUp"]
    record["bibliographic"]["keywords"] = ["test"] * 51
    assert {(i.code, i.path) for i in validate_record(record, profile)} == {
        ("invalid_value", "/datasets/0/purpose"),
        ("invalid_value", "/datasets/0/measurement_mode/1"),
        ("cardinality", "/bibliographic/keywords"),
    }


def test_bad_list_item_shapes_are_visible(profile):
    record = populated_record(profile)
    record["samples"] = ["sample_1"]
    record["datasets"][0]["sample_refs"] = [3]
    assert {(i.code, i.path) for i in validate_record(record, profile)} == {
        ("type_mismatch", "/samples/0"), ("type_mismatch", "/datasets/0/sample_refs/0"),
    }


def test_local_reference_checks_do_not_reassign_same_composition(profile):
    record = populated_record(profile)
    record["samples"].append({**record["samples"][0], "id": "sample_2"})
    assert validate_record(record, profile) == ()
    record["datasets"][0]["sample_refs"] = ["sample_3"]
    record["datasets"][0]["facility_refs"] = ["sample_1"]
    record["bibliographic"]["authors"][0]["affiliation_refs"] = ["missing"]
    before = deepcopy(record)
    assert {(i.code, i.path) for i in validate_record(record, profile)} == {
        ("dangling_reference", "/datasets/0/sample_refs/0"),
        ("dangling_reference", "/datasets/0/facility_refs/0"),
        ("dangling_reference", "/bibliographic/authors/0/affiliation_refs/0"),
    }
    assert record == before


def test_duplicate_ids_make_references_ambiguous(profile):
    record = populated_record(profile)
    record["samples"].append(deepcopy(record["samples"][0]))
    assert {(i.code, i.path) for i in validate_record(record, profile)} == {
        ("duplicate_local_id", "/samples/1/id"),
        ("ambiguous_reference", "/datasets/0/sample_refs/0"),
    }


def test_wrong_id_prefix_is_reported_without_rewriting(profile):
    record = populated_record(profile)
    record["samples"][0]["id"] = "facility_1"
    record["datasets"][0]["sample_refs"] = ["facility_1"]
    assert [(i.code, i.path) for i in validate_record(record, profile)] == [
        ("invalid_local_id", "/samples/0/id")
    ]
    assert record["samples"][0]["id"] == "facility_1"


def test_entity_addresses_preserve_distinct_entries_and_idless_funding(profile):
    record = populated_record(profile)
    record["bibliographic"]["funding"] = [
        {"funder": "Synthetic funder", "grant_id": "Test A"},
        {"funder": "Synthetic funder", "grant_id": "Test A"},
    ]
    addresses = entity_addresses(record)
    assert {a.path for a in addresses} == {
        "/bibliographic/affiliations/0", "/bibliographic/authors/0", "/facilities/0",
        "/samples/0", "/datasets/0", "/bibliographic/funding/0", "/bibliographic/funding/1",
    }
    funding = [a for a in addresses if a.collection == "bibliographic/funding"]
    assert all(a.local_id is None for a in funding)
    assert "id" not in record["bibliographic"]["funding"][0]


def test_patterns_preserved_and_checked_verbatim(profile):
    record = empty_record(profile)
    record["bibliographic"]["doi"] = "10.1234/synthetic"
    assert validate_record(record, profile) == ()
    assert profile.schema["bibliographic"]["doi"]["constraints"]["pattern"].startswith(r"^10\.")
    record["bibliographic"]["doi"] = "not a DOI"
    assert [(i.code, i.path) for i in validate_record(record, profile)] == [
        ("invalid_value", "/bibliographic/doi")
    ]
    assert any("prose format constraints" in note for note in profile.validation_notes)


def test_reference_addresses_follow_canonical_relations_without_resolving(profile):
    record = populated_record(profile)
    record["datasets"][0]["sample_refs"] = ["sample_1", "sample_1", "missing", 7]
    before = deepcopy(record)
    refs = reference_addresses(record)
    assert [(ref.path, ref.target_collection, ref.local_id) for ref in refs] == [
        ("/bibliographic/authors/0/affiliation_refs/0", "bibliographic/affiliations", "aff_1"),
        ("/datasets/0/sample_refs/0", "samples", "sample_1"),
        ("/datasets/0/sample_refs/1", "samples", "sample_1"),
        ("/datasets/0/sample_refs/2", "samples", "missing"),
        ("/datasets/0/facility_refs/0", "facilities", "facility_1"),
    ]
    assert record == before
    assert reference_addresses({"datasets": [None, {"sample_refs": None}]}) == ()
