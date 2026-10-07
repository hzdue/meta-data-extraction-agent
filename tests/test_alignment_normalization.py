"""Alignment normalization tests for global helper aliases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from meta_data_extraction import alignment
from meta_data_extraction.alignment import (
    load_alignment_helper,
    normalize_surface,
    grant_identity,
    normalize_dataset,
    normalize_funder,
    normalize_sample,
)


ROOT = Path(__file__).resolve().parents[1]


def test_shipped_alignment_helper_structure_and_references() -> None:
    helper = load_alignment_helper()
    declared = yaml.safe_load((ROOT / "configs/alignment_helper.yaml").read_text(encoding="utf-8"))
    assert helper == declared
    assert isinstance(helper["version"], str) and helper["version"].strip()
    assert set(helper) == {"version", "artifact_type", "scope", "entities"}
    assert set(helper["entities"]) == {
        "bibliographic.funding", "samples", "datasets", "facilities"
    }
    for entity, section in helper["entities"].items():
        expected_keys = {"aliases"}
        if entity in {"samples", "datasets"}:
            expected_keys.add("curated_forms")
        if entity == "datasets":
            expected_keys.add("generic_name_terms")
        assert set(section) == expected_keys
        assert section["aliases"]
        for item in section["aliases"]:
            assert set(item) == {"alias", "canonical_name"}
            assert all(isinstance(value, str) and value.strip() for value in item.values())
        keys = [normalize_surface(item["alias"]) for item in section["aliases"]]
        assert len(keys) == len(set(keys)), entity
        for paper_id, paper in section.get("curated_forms", {}).items():
            gold = json.loads((ROOT / "data/gold" / f"{paper_id}.json").read_text(encoding="utf-8"))
            assert set(paper) <= {item["id"] for item in gold[entity]}
            allowed_forms = {"accepted_name_forms", (
                "accepted_composition_forms" if entity == "samples" else "accepted_technique_forms"
            )}
            for forms in paper.values():
                assert forms and set(forms) <= allowed_forms
                for values in forms.values():
                    assert isinstance(values, list)
                    assert all(isinstance(value, str) and value.strip() for value in values)
    datasets = helper["entities"]["datasets"]
    assert datasets["generic_name_terms"] == [
        "data",
        "dataset",
        "experiment",
        "measurement",
        "result",
        "study",
    ]
    assert "rheological experiments" not in datasets["curated_forms"][
        "1804.07163"
    ]["dataset_4"]["accepted_technique_forms"]


def test_helper_content_revision_is_not_pinned_in_code(tmp_path, monkeypatch) -> None:
    version = "9.8.7"
    path = tmp_path / "helper.yaml"
    path.write_text(yaml.safe_dump({"version": version, "entities": {}}), encoding="utf-8")
    monkeypatch.setattr(alignment, "_ALIGNMENT_HELPER_PATH", path)
    assert alignment.load_alignment_helper()["version"] == version


def test_helper_excludes_rejected_or_output_specific_dataset_forms() -> None:
    helper = load_alignment_helper()
    curated = helper["entities"]["datasets"]["curated_forms"]
    rejected = {
        ("2202.11658", "dataset_4", "accepted_technique_forms"):
            "transmission electron microscopy imaging and atomic-resolution "
            "high-angle annular dark-field scanning transmission electron "
            "microscopy imaging",
        ("2404.06853", "dataset_3", "accepted_technique_forms"):
            "scanning electron microscopy and energy dispersive X-ray spectroscopy",
        ("1901.08819", "dataset_1", "accepted_technique_forms"):
            "polarized neutron reflectivity",
        ("2603.01364", "dataset_5", "accepted_name_forms"): "currentvoltage",
    }
    for (paper, gold_id, field), value in rejected.items():
        assert value not in curated[paper][gold_id][field]


@pytest.mark.parametrize(
    ("normalizer", "helper_section"),
    [
        (alignment.normalize_funder, "bibliographic.funding"),
        (alignment.normalize_sample, "samples"),
        (alignment.normalize_dataset, "datasets"),
    ],
)
def test_every_reviewed_alias_expands(
    normalizer, helper_section: str
) -> None:
    data = load_alignment_helper()
    aliases = data["entities"][helper_section]["aliases"]
    for entry in aliases:
        assert normalizer(entry["alias"].upper()) == normalize_surface(entry["canonical_name"])


@pytest.mark.parametrize(
    ("plural", "singular"),
    [
        ("datasets", "dataset"),
        ("experiments", "experiment"),
        ("measurements", "measurement"),
        ("results", "result"),
        ("scans", "scan"),
        ("studies", "study"),
    ],
)
def test_dataset_identity_normalizes_generic_noun_inflections(
    plural: str, singular: str
) -> None:
    assert alignment.normalize_dataset(f"rheological {plural}") == (
        f"rheological {singular}"
    )


def test_generic_noun_inflections_remain_dataset_field_specific() -> None:
    assert normalize_sample("rheological experiments") == (
        "rheological experiments"
    )


def test_nd_in_sans_sample_label_is_not_neutron_diffraction() -> None:
    normalized = normalize_dataset("SANS (nd-Ni)")
    assert normalized == "small angle neutron scattering nd ni"
    assert "neutron diffraction" not in normalized


def test_aliases_are_field_scoped_and_token_bounded() -> None:
    # GO is a sample alias but neither an English-word substring nor a dataset
    # alias. ND must not affect the sample chemical symbol Nd.
    assert normalize_sample("going") == "going"
    assert normalize_dataset("GO") == "go"
    assert normalize_sample("Nd2O3") == "nd2o3"
    assert (
        normalize_dataset("ND")
        == "neutron diffraction"
    )


def test_hyphen_delimited_compound_expands_by_component() -> None:
    assert normalize_dataset("HERFD-XANES") == (
        "high energy resolution fluorescence detection "
        "x ray absorption near edge structure"
    )


def test_excluded_compound_does_not_expand_component_alias() -> None:
    assert (
        normalize_dataset("in situ DE-XAS")
        == "in situ de xas"
    )


def test_parenthetical_acronym_does_not_duplicate_full_name() -> None:
    assert (
        normalize_funder("European Research Council (ERC)")
        == "european research council"
    )
    assert (
        normalize_dataset("grazing-incidence small-angle X-ray scattering (GISAXS)")
        == "grazing incidence small angle x ray scattering"
    )


@pytest.mark.parametrize("unsafe", ["NPD", "CT", "PDF", "NSF", "PSD", "NRIXS"])
def test_excluded_or_ambiguous_dataset_aliases_do_not_expand(unsafe: str) -> None:
    assert (
        normalize_dataset(unsafe)
        == unsafe.casefold()
    )


def test_surface_and_grant_id_normalization() -> None:
    assert normalize_surface("  Small–Angle  X-ray (Scattering) ") == (
        "small angle x ray scattering"
    )
    assert (
        grant_identity("Grant No.  ABC–123 ")
        == "abc123"
    )
    assert normalize_surface("Fe2+ / Fe3+") == "fe2+ / fe3+"
