import json
from itertools import permutations
from pathlib import Path

import yaml

from meta_data_extraction.alignment import (
    _match_samples,
    _prepare_sample_identities,
    normalize_sample,
    _sample_variant_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def _entries(document_id: str) -> list[dict]:
    gold = json.loads(
        (ROOT / "data" / "gold" / f"{document_id}.json").read_text(encoding="utf-8")
    )
    return _prepare_sample_identities(gold.get("samples") or [], document_id)


def _sample_helpers() -> dict:
    helper = yaml.safe_load(
        (ROOT / "configs" / "alignment_helper.yaml").read_text(encoding="utf-8")
    )
    return helper["entities"]["samples"]


def test_runtime_registry_covers_assigned_gold_samples():
    corpus = json.loads(
        (ROOT / "data" / "corpus.json").read_text(encoding="utf-8")
    )
    paper_ids = [paper["arxiv_id"] for paper in corpus["papers"]]
    registry = {paper_id: _entries(paper_id) for paper_id in paper_ids}
    assert len(registry) == 40
    assert sum(len(entries) for entries in registry.values()) == 127


def test_pdf_grounded_sample_forms_are_generated():
    helpers = _sample_helpers()
    assert (
        sum(len(paper) for paper in helpers["curated_forms"].values()) == 28
    )
    co2cr = _entries("1909.08292")[1]
    assert "co2cr0.75ti0.25al" in co2cr["accepted_composition_forms"]
    synthetic = _entries("2605.14792")[1]
    assert "synthetic fewo4" in synthetic["accepted_name_forms"]
    assert "iron oxide mnp" in _entries("1901.08819")[0]["accepted_name_forms"]
    assert (
        "monolayer of magnetic iron oxide nanoparticles"
        in _entries("1901.08819")[0]["accepted_name_forms"]
    )
    thick, thin = _entries("2501.08122")[1:3]
    assert "thick hm layer" in thick["accepted_name_forms"]
    assert "thin hm layer" in thin["accepted_name_forms"]
    lettered = _entries("2603.01364")
    assert [entry["accepted_name_forms"][-1] for entry in lettered] == ["a", "b", "c"]
    assert (
        "granular system consists of bi-disperse plastic beads"
        in _entries("2010.12832")[0]["accepted_name_forms"]
    )


def test_parameterized_series_list_gold_and_expanded_forms_consistently():
    curated = _sample_helpers()["curated_forms"]
    expected = {
        "1909.08292": {
            "sample_1": ("Co2Cr1-xTixAl x=0", "Co2CrAl"),
            "sample_2": ("Co2Cr1-xTixAl x=0.25", "Co2Cr0.75Ti0.25Al"),
            "sample_3": ("Co2Cr1-xTixAl x=0.5", "Co2Cr0.5Ti0.5Al"),
        },
        "2001.00707": {
            "sample_1": ("Mn2V1-xCoxGa x=0", "Mn2VGa"),
            "sample_2": ("Mn2V1-xCoxGa x=0.25", "Mn2V0.75Co0.25Ga"),
            "sample_3": ("Mn2V1-xCoxGa x=0.5", "Mn2V0.5Co0.5Ga"),
            "sample_4": ("Mn2V1-xCoxGa x=0.75", "Mn2V0.25Co0.75Ga"),
            "sample_5": ("Mn2V1-xCoxGa x=1", "Mn2CoGa"),
        },
        "2007.00972": {
            "sample_1": ("BaTiO3", "Ba1-xSrxTiO3 x=0.0"),
            "sample_2": ("Ba0.5Sr0.5TiO3", "Ba1-xSrxTiO3 x=0.5"),
            "sample_3": ("SrTiO3", "Ba1-xSrxTiO3 x=1.0"),
        },
    }
    for paper, samples in expected.items():
        gold = json.loads(
            (ROOT / "data" / "gold" / f"{paper}.json").read_text(encoding="utf-8")
        )
        gold_by_id = {sample["id"]: sample for sample in gold["samples"]}
        for gold_id, forms in samples.items():
            assert curated[paper][gold_id]["accepted_name_forms"] == list(forms)
            assert curated[paper][gold_id]["accepted_composition_forms"] == list(
                forms
            )
            gold_sample = gold_by_id[gold_id]
            assert normalize_sample(forms[0]) == normalize_sample(
                gold_sample["sample_name"]
            )
            assert normalize_sample(
                forms[0]
            ) == normalize_sample(gold_sample["chemical_composition"])

    expanded_2010 = {
        "sample_1": "SrTiO3",
        "sample_2": "Sr0.94Gd0.06TiO3",
        "sample_3": "Sr0.88Gd0.12TiO3",
        "sample_4": "Sr0.82Gd0.18TiO3",
        "sample_5": "Sr0.76Gd0.24TiO3",
        "sample_6": "Sr0.70Gd0.30TiO3",
    }
    gold = json.loads(
        (ROOT / "data" / "gold" / "2010.00094.json").read_text(encoding="utf-8")
    )
    gold_by_id = {sample["id"]: sample for sample in gold["samples"]}
    for gold_id, expanded in expanded_2010.items():
        forms = curated["2010.00094"][gold_id]
        sample = gold_by_id[gold_id]
        assert forms["accepted_name_forms"][0] == sample["sample_name"]
        assert normalize_sample(
            forms["accepted_composition_forms"][0]
        ) == normalize_sample(sample["chemical_composition"])
        assert forms["accepted_composition_forms"][1] == expanded


def test_expanded_series_names_align_and_ambiguous_or_wrong_forms_do_not():
    expanded = [
        {"sample_name": name, "chemical_composition": None}
        for name in (
            "Mn2VGa",
            "Mn2V0.75Co0.25Ga",
            "Mn2V0.5Co0.5Ga",
            "Mn2V0.25Co0.75Ga",
            "Mn2CoGa",
        )
    ]
    assert _match_samples(expanded, _entries("2001.00707")) == {
        index: (index, "exact_name") for index in range(5)
    }
    entries = _entries("2001.00707")
    assert _match_samples(
        [{"sample_name": "Mn2V1-xCoxGa", "chemical_composition": None}], entries
    ) == {}
    assert _match_samples(
        [
            {
                "sample_name": "Mn2V0.25Co0.25Ga",
                "chemical_composition": "Mn2V0.25Co0.25Ga",
            }
        ],
        entries,
    ) == {}


def test_source_reviewed_sample_names_align_to_frozen_gold_identities():
    assert _match_samples(
        [
            {
                "sample_name": "monolayer of magnetic iron oxide nanoparticles",
                "chemical_composition": "iron oxide",
            }
        ],
        _entries("1901.08819"),
    ) == {0: (0, "exact_name")}
    assert _match_samples(
        [
            {"sample_name": "thick HM layer", "chemical_composition": "CoFeB/Mo"},
            {"sample_name": "thin HM layer", "chemical_composition": "CoFeB/Mo"},
        ],
        _entries("2501.08122"),
    ) == {0: (1, "exact_name"), 1: (2, "exact_name")}
    assert _match_samples(
        [
            {"sample_name": label, "chemical_composition": None}
            for label in ("A", "B", "C")
        ],
        _entries("2603.01364"),
    ) == {
        0: (0, "exact_name"),
        1: (1, "exact_name"),
        2: (2, "exact_name"),
    }


def test_barium_strontium_titanate_series_forms_align():
    predictions = [
        {
            "sample_name": f"Ba1-xSrxTiO3 x = {value}",
            "chemical_composition": f"Ba1-xSrxTiO3 x = {value}",
        }
        for value in ("0.0", "0.5", "1.0")
    ]
    assert _match_samples(predictions, _entries("2007.00972")) == {
        0: (0, "exact_name"),
        1: (1, "exact_name"),
        2: (2, "exact_name"),
    }


def test_non_calibration_pdf_grounded_paraphrases_align():
    assert _match_samples(
        [
            {
                "sample_name": "granular system consists of bi-disperse plastic beads",
                "chemical_composition": None,
            }
        ],
        _entries("2010.12832"),
    ) == {0: (0, "exact_name")}
    assert _match_samples(
        [
            {
                "sample_name": "iron oxide MNP",
                "chemical_composition": "iron oxide nanoparticles",
            }
        ],
        _entries("1901.08819"),
    ) == {0: (0, "exact_name")}


def test_grounded_forms_do_not_resolve_ambiguous_plain_samples():
    tin_entries = _entries("2212.04169")
    assert (
        _match_samples(
            [{"sample_name": "TiN", "chemical_composition": "TiN"}], tin_entries
        )
        == {}
    )
    al_entries = _entries("2407.10282")
    assert (
        _match_samples(
            [{"sample_name": "Al6061", "chemical_composition": "Al6061"}], al_entries
        )
        == {}
    )


def test_composition_collision_is_explicit():
    entries = _entries("2603.01364")
    assert [entry["normalized_name"] for entry in entries] == [
        "sample a",
        "sample b",
        "sample c",
    ]
    assert all(entry["composition_is_unique"] is False for entry in entries)
    assert all(entry["discriminator"] == "sample_name" for entry in entries)


def test_numeric_variant_can_be_the_group_discriminator():
    entries = [
        entry
        for entry in _entries("2309.13729")
        if entry["composition_is_unique"] is False
    ]
    assert all(
        entry["discriminator"] == "temperatures" for entry in entries
    )
    assert {tuple(entry["variant_identity"]["temperatures"]) for entry in entries} == {
        ("350°c",),
        ("400°c",),
        ("450°c",),
    }


def test_sample_alignment_uses_the_first_distinguishing_property():
    entries = _prepare_sample_identities([
        {"id": "sample_a", "sample_name": "alloy x=0.1 at 350 °C", "chemical_composition": "Fe"},
        {"id": "sample_b", "sample_name": "alloy x=0.2 at 400 °C", "chemical_composition": "Fe"},
    ], "synthetic-discriminator-priority")
    # Both variable and temperature distinguish the gold records. Variables
    # take priority, so the conflicting temperature remains an attribute error.
    assert _match_samples([
        {"sample_name": "alloy x=0.1 at 400 °C", "chemical_composition": "Fe"}
    ], entries) == {0: (0, "shared_composition_discriminator")}


def test_nullable_composition_is_not_a_collision_group():
    null_entries = [
        entry
        for path in (ROOT / "data" / "gold").glob("*.json")
        for entry, gold in zip(
            _entries(path.stem),
            json.loads(path.read_text(encoding="utf-8")).get("samples") or [],
        )
        if gold["chemical_composition"] is None
    ]
    assert null_entries
    assert all(entry["normalized_composition"] is None for entry in null_entries)
    assert all(entry["composition_is_unique"] is None for entry in null_entries)
    assert all(entry["discriminator"] is None for entry in null_entries)


def test_variant_identity_is_structured_and_printed_only():
    assert _sample_variant_identity(
        "LSCNO x=0.06 and 5 at.%",
        "multilayer N = 100 at 350 °C and 5 Å; 2 atom% and 5 at.%",
    ) == {
        "variables": {"x": ["0.06"]},
        "counts": ["100"],
        "temperatures": ["350°c"],
        "thicknesses": ["5å"],
        "concentrations": ["2atom%", "5at%"],
    }


def test_sample_variants_preserve_complete_scientific_notation():
    assert _sample_variant_identity("x=1e-3 and 2.5E-2 nm") == {
        "variables": {"x": ["1e-3"]},
        "thicknesses": ["2.5e-2nm"],
    }


def test_sample_variants_reject_partial_numeric_and_unit_tokens():
    for value in ("N=10.5", "N=10e2", "10 nmol", "x=1e", "x=0.1suffix"):
        assert _sample_variant_identity(value) == {}, value


def test_sample_scientific_notation_cannot_align_on_the_mantissa_alone():
    entries = _prepare_sample_identities([
        {"id": "a", "sample_name": "alloy x=1e-3", "chemical_composition": "Fe"},
        {"id": "b", "sample_name": "alloy x=2e-3", "chemical_composition": "Fe"},
    ], "synthetic")
    pred = [{"sample_name": "alloy x=1e-2", "chemical_composition": "Fe"}]
    assert _match_samples(pred, entries) == {}


def test_sample_scientific_notation_still_identifies_the_correct_variant():
    entries = _prepare_sample_identities([
        {"id": "a", "sample_name": "alloy x=1e-3", "chemical_composition": "Fe"},
        {"id": "b", "sample_name": "alloy x=2e-3", "chemical_composition": "Fe"},
    ], "synthetic")
    pred = [{"sample_name": "alloy prepared with x=1e-3", "chemical_composition": "Fe"}]
    assert _match_samples(pred, entries) == {0: (0, "shared_composition_discriminator")}


def test_safe_sample_presentation_normalization():
    assert normalize_sample("CoFeB / Mo") == "cofeb/mo"
    assert normalize_sample("NaCl:urea∙H₂O") == "nacl:urea·h2o"
    assert normalize_sample("NaCl:urea.H2O") == "nacl:urea·h2o"
    assert normalize_sample("Al/AlO_x/Al") == "al/alox/al"


def test_staged_alignment_respects_shared_composition():
    entries = _prepare_sample_identities(
        [
            {
                "id": "sample_a",
                "sample_name": "Sample A",
                "chemical_composition": "Al/AlOx/Al",
            },
            {
                "id": "sample_b",
                "sample_name": "Sample B",
                "chemical_composition": "Al/AlOx/Al",
            },
        ],
        "synthetic-shared-composition",
    )
    assert _match_samples(
        [{"sample_name": "Sample A", "chemical_composition": "Al/AlOx/Al"}],
        entries,
    ) == {0: (0, "exact_name")}
    assert _match_samples(
        [{"sample_name": "unknown", "chemical_composition": "Al/AlOx/Al"}],
        entries,
    ) == {}


def test_exact_name_remains_evidence_when_composition_differs():
    entries = _prepare_sample_identities(
        [
            {
                "id": "sample_film",
                "sample_name": "CoFeB film",
                "chemical_composition": "Co40Fe40B20",
            }
        ],
        "synthetic-composition-disagreement",
    )
    assert _match_samples(
        [{"sample_name": "CoFeB film", "chemical_composition": "CoFeB"}],
        entries,
    ) == {0: (0, "exact_name")}


def test_staged_alignment_rejects_missing_required_name():
    entries = _entries("2309.13729")
    assert (
        _match_samples(
            [{"sample_name": None, "chemical_composition": "CoFe2O4"}], entries
        )
        == {}
    )


def test_staged_alignment_uses_unique_composition_after_exact_name():
    entries = _entries("2309.13729")
    matches = _match_samples(
        [{"sample_name": "starting material", "chemical_composition": "CoFe2O4"}],
        entries,
    )
    assert matches == {0: (0, "unique_composition")}


def test_exact_sample_names_reserve_targets_before_composition_fallback():
    entries = _prepare_sample_identities([
        {"id": "gold_a", "sample_name": "alpha", "chemical_composition": "Fe2O3"},
        {"id": "gold_b", "sample_name": "beta", "chemical_composition": "CoO"},
    ], "synthetic-stage-order")
    predictions = [
        {"id": "exact", "sample_name": "alpha", "chemical_composition": "CoO"},
        {"id": "blocked", "sample_name": "unreported", "chemical_composition": "Fe2O3"},
        {"id": "fallback", "sample_name": "beta extra", "chemical_composition": "CoO"},
    ]
    for order in permutations(predictions):
        matches = _match_samples(list(order), entries)
        assert {
            (order[pred]["id"], entries[gold]["gold_id"], stage)
            for pred, (gold, stage) in matches.items()
        } == {
            ("exact", "gold_a", "exact_name"),
            ("fallback", "gold_b", "unique_composition"),
        }


def test_staged_alignment_resolves_shared_composition_by_temperature():
    entries = _entries("2309.13729")
    matches = _match_samples(
        [
            {
                "sample_name": "nanocomposite prepared at 400 °C",
                "chemical_composition": "CoFe2O4/Co-Fe",
            }
        ],
        entries,
    )
    assert matches == {0: (2, "shared_composition_discriminator")}


def test_numeric_variant_alone_does_not_override_missing_or_conflicting_composition():
    entries = _entries("2309.13729")
    predictions = [
        {
            "sample_name": "nanocomposite prepared at 350 °C",
            "chemical_composition": "Co0.56Fe2.44O4",
        },
        {
            "sample_name": "CoFe2O4/Co-Fe nanocomposites (400 °C)",
            "chemical_composition": None,
        },
        {
            "sample_name": "450 °C",
            "chemical_composition": "CoFe2O4/Co-Fe composite",
        },
    ]
    assert _match_samples(predictions, entries) == {}


def test_numeric_variant_rejects_unrelated_material():
    entries = _entries("2309.13729")
    assert (
        _match_samples(
            [
                {
                    "sample_name": "unknown material at 400 °C",
                    "chemical_composition": "Fe",
                }
            ],
            entries,
        )
        == {}
    )


def test_shared_composition_discriminator_rejects_unknown_value():
    entries = _entries("2309.13729")
    assert (
        _match_samples(
            [
                {
                    "sample_name": "nanocomposite prepared at 375 °C",
                    "chemical_composition": "CoFe2O4/Co-Fe",
                }
            ],
            entries,
        )
        == {}
    )


def test_staged_alignment_uses_unique_name_containment_and_rejects_duplicate():
    entries = _entries("2410.00268")
    prediction = {
        "sample_name": "PMMA thin film",
        "chemical_composition": None,
    }
    matches = _match_samples([prediction, prediction], entries)
    assert matches == {0: (0, "unique_name_containment")}
