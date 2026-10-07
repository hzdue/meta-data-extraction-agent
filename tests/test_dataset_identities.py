import json
from itertools import permutations
from pathlib import Path

from meta_data_extraction import alignment
from meta_data_extraction.alignment import (
    _match_datasets,
    _dataset_alignment_text,
    _prepare_dataset_identities,
)


ROOT = Path(__file__).resolve().parents[1]


def test_overlapping_technique_forms_group_transitively(monkeypatch):
    helper = alignment.load_alignment_helper()
    document_id = "synthetic-transitive-techniques"
    helper["entities"]["datasets"]["curated_forms"][document_id] = {
        "gold_a": {"accepted_technique_forms": ["bridge one"]},
        "gold_b": {"accepted_technique_forms": ["bridge one", "bridge two"]},
        "gold_c": {"accepted_technique_forms": ["bridge two"]},
    }
    monkeypatch.setattr(alignment, "load_alignment_helper", lambda: helper)
    gold = [
        {"id": "gold_a", "dataset_name": "alpha", "measurement_technique": "method a"},
        {"id": "gold_b", "dataset_name": "beta", "measurement_technique": "method b"},
        {"id": "gold_c", "dataset_name": "gamma", "measurement_technique": "method c"},
    ]
    for order in permutations(gold):
        entries = _prepare_dataset_identities(list(order), document_id)
        assert {entry["technique_group"] for entry in entries} == {"technique_group_1"}
        assert _match_datasets([
            {"dataset_name": "unknown", "measurement_technique": "method a"}
        ], entries) == {}
        matches = _match_datasets([
            {"dataset_name": "alpha signal", "measurement_technique": "method a"}
        ], entries)
        target, stage = matches[0]
        assert entries[target]["gold_id"] == "gold_a"
        assert stage == "shared_technique_name_discriminator"


def gold_datasets(paper_id: str) -> list[dict]:
    path = ROOT / "data" / "gold" / f"{paper_id}.json"
    return json.loads(path.read_text(encoding="utf-8")).get("datasets") or []


def registry():
    corpus = json.loads(
        (ROOT / "data" / "corpus.json").read_text(encoding="utf-8")
    )
    paper_ids = [paper["arxiv_id"] for paper in corpus["papers"]]
    papers = {}
    for paper_id in sorted(paper_ids):
        papers[paper_id] = _prepare_dataset_identities(
            gold_datasets(paper_id), paper_id
        )
    return {"papers": papers}


def extraction_datasets(run_name: str, paper_id: str) -> list[dict]:
    path = ROOT / "data" / "experiment" / run_name / f"{paper_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("output", data).get("datasets") or []


def test_runtime_identities_cover_assigned_gold_datasets():
    data = registry()
    assert len(data["papers"]) == 40
    assert sum(len(entries) for entries in data["papers"].values()) == 118
    assert all(
        entry["accepted_technique_forms"] and entry["accepted_name_forms"]
        for entries in data["papers"].values()
        for entry in entries
    )


def test_unique_technique_matches_with_both_keys_present():
    entries = registry()["papers"]["1806.08116"]
    prediction = {
        "measurement_technique": gold_datasets("1806.08116")[0]["measurement_technique"],
        "dataset_name": "different descriptive name",
    }
    assert _match_datasets([prediction], entries) == {0: (0, "unique_technique")}


def test_missing_name_uses_only_unique_technique():
    entries = registry()["papers"]["1806.08116"]
    assert _match_datasets(
        [
            {
                "measurement_technique": gold_datasets("1806.08116")[0]["measurement_technique"],
                "dataset_name": None,
            }
        ],
        entries,
    ) == {0: (0, "unique_technique_missing_name")}
    assert _match_datasets(
        [{"measurement_technique": ["XRD"], "dataset_name": "XRD data"}], entries
    ) == {0: (1, "generic_normalized_unique_name")}
    assert _match_datasets(
        [{"measurement_technique": ["XRD", "XAS"], "dataset_name": "XRD"}], entries
    ) == {0: (1, "exact_unique_name")}
    assert _dataset_alignment_text(["XRD"]) is None
    assert _dataset_alignment_text([]) is None
    assert _dataset_alignment_text(["XRD", "XAS"]) is None
    assert _match_datasets(
        [{"measurement_technique": ["XRD"], "dataset_name": None}], entries
    ) == {}


def test_shared_technique_requires_name_discriminator():
    entries = registry()["papers"]["1909.08292"]
    shared = [entry for entry in entries if entry["technique_group"]]
    gold = {entity["id"]: entity for entity in gold_datasets("1909.08292")}
    technique = gold[shared[0]["gold_id"]]["measurement_technique"]
    assert (
        _match_datasets(
            [{"measurement_technique": technique, "dataset_name": "neutron data"}],
            entries,
        )
        == {}
    )
    expected_index = entries.index(
        next(entry for entry in shared if "WOMBAT" in gold[entry["gold_id"]]["dataset_name"])
    )
    assert _match_datasets(
        [
            {
                "measurement_technique": technique,
                "dataset_name": "WOMBAT diffraction dataset",
            }
        ],
        entries,
    ) == {0: (expected_index, "shared_technique_name_discriminator")}


def test_pdf_grounded_shared_technique_forms():
    data = registry()["papers"]
    sans = data["2105.04167"]
    assert _match_datasets(
        [
            {"measurement_technique": "SANS", "dataset_name": "SANS (QUOKKA)"},
            {"measurement_technique": "SANS", "dataset_name": "SANS (D33)"},
        ],
        sans,
    ) == {
        0: (1, "shared_technique_name_discriminator"),
        1: (2, "shared_technique_name_discriminator"),
    }
    xrd = data["2212.04169"]
    assert _match_datasets(
        [{"measurement_technique": "XRD", "dataset_name": "GIXRD"}], xrd
    ) == {0: (3, "technique_and_name")}


def test_shared_technique_generic_name_remains_unmatched():
    entries = registry()["papers"]["2105.04167"]
    assert (
        _match_datasets(
            [{"measurement_technique": "SANS", "dataset_name": "SANS"}], entries
        )
        == {}
    )


def test_missing_name_with_repeated_technique_remains_unmatched():
    entries = registry()["papers"]["2105.04167"]
    assert (
        _match_datasets(
            [
                {
                    "measurement_technique": "SANS",
                    "dataset_name": None,
                }
            ],
            entries,
        )
        == {}
    )


def test_pdf_grounded_unique_technique_forms():
    data = registry()["papers"]
    assert _match_datasets(
        [{"measurement_technique": "EELS", "dataset_name": "EELS"}], data["2603.01364"]
    ) == {0: (3, "technique_and_name")}
    assert _match_datasets(
        [{"measurement_technique": "X-ray reflectivity", "dataset_name": "qXRR"}],
        data["1812.03956"],
    ) == {0: (0, "technique_and_name")}


def test_exact_unique_name_establishes_identity_despite_wrong_technique():
    entries = registry()["papers"]["2010.12832"]
    assert _match_datasets(
        [
            {
                "measurement_technique": "vibrating sample magnetometry",
                "dataset_name": "Shear force measurement",
            }
        ],
        entries,
    ) == {0: (1, "exact_unique_name")}


def test_reviewed_broader_technique_form_is_explicitly_accepted():
    entries = registry()["papers"]["1811.10341"]
    assert _match_datasets(
        [
            {
                "measurement_technique": "X-ray absorption spectroscopy",
                "dataset_name": "Ge K-edge EXAFS",
            }
        ],
        entries,
    ) == {0: (2, "technique_and_name")}


def test_annotation_grounded_unique_variants():
    data = registry()["papers"]
    assert _match_datasets(
        [
            {
                "measurement_technique": "rheological experiment",
                "dataset_name": "rheological experiment",
            }
        ],
        data["1804.07163"],
    ) == {0: (3, "technique_and_name")}
    assert _match_datasets(
        [
            {
                "measurement_technique": "electron backscatter diffraction",
                "dataset_name": "electron backscatter diffraction",
            }
        ],
        data["2407.10282"],
    ) == {0: (2, "technique_and_name")}


def test_split_annotation_does_not_arbitrarily_match_aggregate_gold():
    entries = registry()["papers"]["2010.00094"]
    predictions = [
        {
            "measurement_technique": "extended X-ray absorption fine structure",
            "dataset_name": "Sr K-edge EXAFS",
        },
        {
            "measurement_technique": "X-ray absorption near edge spectroscopy",
            "dataset_name": "O K-edge XANES",
        },
    ]
    assert _match_datasets(predictions, entries) == {}


def test_unique_technique_fallback_abstains_on_many_to_one_collision():
    entries = registry()["papers"]["2501.08122"]
    predictions = [
        {
            "measurement_technique": "polarized neutron reflectometry",
            "dataset_name": name,
        }
        for name in (
            "PNR of CoFeB film",
            "PNR of CoFeB / Mo bilayer structures",
            "PNR of CoFeB / Mo / Si trilayer structure",
        )
    ]
    diagnostics = {}
    assert _match_datasets(predictions, entries, diagnostics) == {}
    assert diagnostics == {"dataset_technique_collision": 3}
    reverse_diagnostics = {}
    assert (
        _match_datasets(list(reversed(predictions)), entries, reverse_diagnostics)
        == {}
    )
    assert reverse_diagnostics == diagnostics


def test_strong_exact_name_resolves_before_technique_collision():
    entries = registry()["papers"]["2501.08122"]
    predictions = [
        {
            "measurement_technique": "polarized neutron reflectometry",
            "dataset_name": "PNR of CoFeB film",
        },
        {
            "measurement_technique": "polarized neutron reflectometry",
            "dataset_name": "polarized neutron reflectometry",
        },
    ]
    assert _match_datasets(predictions, entries) == {1: (0, "technique_and_name")}


def test_ambiguous_pic_label_does_not_choose_between_pic_datasets():
    data = registry()["papers"]
    assert (
        _match_datasets(
            [
                {
                    "measurement_technique": [
                        "polarization-dependent imaging contrast mapping"
                    ],
                    "dataset_name": "PIC maps",
                }
            ],
            data["2009.01093"],
        )
        == {}
    )


def test_facility_labels_can_disambiguate_same_technique_datasets():
    entries = registry()["papers"]["2105.04167"]
    assert {"quokka", "ansto"} <= set(entries[1]["accepted_name_forms"])
    assert {"d33", "ill"} <= set(entries[2]["accepted_name_forms"])


def test_reviewed_annotation_linkage_gaps_are_grounded():
    data = registry()["papers"]
    assert _match_datasets(
        [
            {
                "measurement_technique": "X-ray absorption spectroscopy",
                "dataset_name": "in situ DE-XAS",
            }
        ],
        data["2410.00268"],
    ) == {0: (4, "exact_unique_name")}
    assert _match_datasets(
        [
            {
                "measurement_technique": "diffraction-enhanced X-ray absorption spectroscopy",
                "dataset_name": "in situ DE-XAS",
            }
        ],
        data["2410.00268"],
    ) == {0: (4, "technique_and_name")}
    assert _match_datasets(
        [
            {
                "measurement_technique": "X-ray reflectivity",
                "dataset_name": "XRR during the annealing",
            }
        ],
        data["1812.03956"],
    ) == {0: (0, "technique_and_name")}
    assert _match_datasets(
        [
            {
                "measurement_technique": "scanning transmission electron spectroscopy",
                "dataset_name": "STEM",
            }
        ],
        data["2212.04169"],
    ) == {0: (6, "technique_and_name")}
    assert _match_datasets(
        [
            {
                "measurement_technique": "scanning transmission electron microscopy",
                "dataset_name": "STEM",
            }
        ],
        data["2212.04169"],
    ) == {0: (6, "technique_and_name")}


def test_exact_unique_name_can_override_an_incompatible_technique():
    data = registry()["papers"]
    assert _match_datasets(
        [
            {
                "measurement_technique": "scanning transmission electron microscopy",
                "dataset_name": "Electron tomography",
            }
        ],
        data["2009.01093"],
    ) == {0: (4, "exact_unique_name")}


def test_nonexact_name_cannot_rescue_an_incompatible_technique():
    data = registry()["papers"]
    assert (
        _match_datasets(
            [
                {
                    "measurement_technique": "X-ray absorption spectroscopy",
                    "dataset_name": "in situ DE-XAS of DKE deconstruction",
                }
            ],
            data["2410.00268"],
        )
        == {}
    )


def test_4d_stem_technique_forms_follow_the_papers_own_wording():
    entries = registry()["papers"]["2009.01093"]
    for technique in (
        "Scanning nano-diffraction (4D-STEM)",
        "4D scanning transmission electron nano-diffraction",
    ):
        assert _match_datasets(
            [{"measurement_technique": technique, "dataset_name": "4D-STEM"}],
            entries,
        ) == {0: (3, "technique_and_name")}

    # "Four-Dimensional Scanning Transmission Electron Microscopy" occurs
    # only in a cited reference title, not as the paper's own method wording.
    # It is therefore not a technique equivalence, but the exact paper-unique
    # dataset label still establishes identity and leaves the wording error for
    # measurement_technique attribute scoring.
    assert _match_datasets(
        [
            {
                "measurement_technique": "4D scanning transmission electron microscopy",
                "dataset_name": "4D-STEM",
            }
        ],
        entries,
    ) == {0: (3, "exact_unique_name")}


def test_unprinted_eels_expansion_is_an_attribute_error_after_name_alignment():
    entries = registry()["papers"]["2603.01364"]
    assert _match_datasets(
        [
            {
                "measurement_technique": "electron energy-loss spectroscopy",
                "dataset_name": "EELS",
            }
        ],
        entries,
    ) == {0: (3, "exact_unique_name")}


def test_exact_unique_joint_name_aligns_incomplete_coupled_technique():
    entries = registry()["papers"]["2503.18475"]
    assert _match_datasets(
        [
            {
                "measurement_technique": "thermogravimetric analysis",
                "dataset_name": "TGA/DSC",
            }
        ],
        entries,
    ) == {0: (0, "exact_unique_name")}


def test_exact_unique_name_aligns_when_technique_is_missing_or_invalid():
    entries = registry()["papers"]["2010.12832"]
    for technique in (None, ["X-ray tomography", "shear force measurement"]):
        assert _match_datasets(
            [
                {
                    "measurement_technique": technique,
                    "dataset_name": "Shear force measurement",
                }
            ],
            entries,
        ) == {0: (1, "exact_unique_name")}


def test_name_shared_by_gold_datasets_is_not_an_exact_unique_identity():
    entries = [
        dict(entry, accepted_name_forms=["shared neutron label"])
        for entry in registry()["papers"]["1909.08292"][2:4]
    ]
    assert (
        _match_datasets(
            [
                {
                    "measurement_technique": "unregistered technique",
                    "dataset_name": "shared neutron label",
                }
            ],
            entries,
        )
        == {}
    )


def test_duplicate_exact_unique_names_align_once_and_are_order_stable():
    entries = registry()["papers"]["2010.12832"]
    predictions = [
        {
            "id": pred_id,
            "measurement_technique": "vibrating sample magnetometry",
            "dataset_name": "Shear force measurement",
        }
        for pred_id in ("pred_b", "pred_a")
    ]

    def selected_id(items):
        diagnostics = {}
        matches = _match_datasets(items, entries, diagnostics)
        assert diagnostics == {"dataset_exact_name_collision": 1}
        assert len(matches) == 1
        pred_index, (gold_index, stage) = next(iter(matches.items()))
        assert (gold_index, stage) == (1, "exact_unique_name")
        return items[pred_index]["id"]

    assert selected_id(predictions) == "pred_a"
    assert selected_id(list(reversed(predictions))) == "pred_a"


def test_shared_technique_uses_gold_derived_name_discriminator():
    entries = registry()["papers"]["2503.18475"]
    matches = _match_datasets(
        [
            {
                "measurement_technique": "Powder X-ray Diffraction",
                "dataset_name": "Powder X-ray Diffraction",
            },
            {
                "measurement_technique": "Powder X-ray Diffraction",
                "dataset_name": "High-resolution synchrotron PXRD patterns",
            },
        ],
        entries,
    )
    assert matches == {
        0: (1, "technique_and_name"),
        1: (2, "shared_technique_name_discriminator"),
    }


def test_same_technique_sibling_name_is_not_a_conflict():
    entries = registry()["papers"]["2503.18475"]
    assert _match_datasets(
        [
            {
                "measurement_technique": "Powder X-ray Diffraction",
                "dataset_name": "High-resolution synchrotron PXRD patterns",
            }
        ],
        entries,
    ) == {0: (2, "shared_technique_name_discriminator")}


def test_generic_xrd_name_remains_unmatched():
    data = registry()["papers"]
    assert (
        _match_datasets(
            [
                {
                    "measurement_technique": "X-ray diffraction",
                    "dataset_name": "XRD",
                }
            ],
            data["2212.04169"],
        )
        == {}
    )


def test_generic_dataset_noun_plural_aligns_without_curated_plural_form():
    entries = registry()["papers"]["1804.07163"]
    assert _match_datasets(
        [
            {
                "dataset_name": "rheological experiments",
                "measurement_technique": "rheological experiments",
            }
        ],
        entries,
    ) == {0: (3, "technique_and_name")}


def test_generic_normalized_unique_name_matches_only_the_complete_substantive_name():
    entries = _prepare_dataset_identities(
        [
            {
                "id": "dataset_1",
                "dataset_name": "EXAFS",
                "measurement_technique": "extended X-ray absorption fine structure",
            },
            {
                "id": "dataset_2",
                "dataset_name": "Dielectric constant and magnetodielectric",
                "measurement_technique": "dielectric spectroscopy",
            },
        ],
        "synthetic_generic_names",
    )

    assert _match_datasets(
        [
            {
                "dataset_name": "EXAFS Dataset",
                "measurement_technique": "incorrect technique",
            },
            {
                "dataset_name": "Dielectric constant and magnetodielectric data",
                "measurement_technique": "another incorrect technique",
            },
        ],
        entries,
    ) == {
        0: (0, "generic_normalized_unique_name"),
        1: (1, "generic_normalized_unique_name"),
    }

    for partial_name in ("Dielectric constant", "Magnetodielectric"):
        assert (
            _match_datasets(
                [
                    {
                        "dataset_name": partial_name,
                        "measurement_technique": "incorrect technique",
                    }
                ],
                entries,
            )
            == {}
        )


def test_generic_normalized_name_rejects_empty_ambiguous_and_merged_keys():
    empty_entries = _prepare_dataset_identities(
        [
            {
                "id": "dataset_1",
                "dataset_name": "Dataset",
                "measurement_technique": "technique one",
            },
            {
                "id": "dataset_2",
                "dataset_name": "Measurement",
                "measurement_technique": "technique two",
            },
        ],
        "synthetic_empty_names",
    )
    assert (
        _match_datasets(
            [{"dataset_name": "Data", "measurement_technique": "unknown"}],
            empty_entries,
        )
        == {}
    )

    ambiguous_entries = _prepare_dataset_identities(
        [
            {
                "id": "dataset_1",
                "dataset_name": "magnetization data",
                "measurement_technique": "technique one",
            },
            {
                "id": "dataset_2",
                "dataset_name": "magnetization measurement",
                "measurement_technique": "technique two",
            },
        ],
        "synthetic_ambiguous_names",
    )
    assert (
        _match_datasets(
            [
                {
                    "dataset_name": "magnetization results",
                    "measurement_technique": "unknown",
                }
            ],
            ambiguous_entries,
        )
        == {}
    )

    merged_entries = _prepare_dataset_identities(
        [
            {
                "id": "dataset_1",
                "dataset_name": "X-ray tomography",
                "measurement_technique": "X-ray tomography",
            },
            {
                "id": "dataset_2",
                "dataset_name": "shear force",
                "measurement_technique": "shear force measurement",
            },
        ],
        "synthetic_merged_name",
    )
    assert (
        _match_datasets(
            [
                {
                    "dataset_name": "X-ray tomography and shear force measurements",
                    "measurement_technique": "unknown",
                }
            ],
            merged_entries,
        )
        == {}
    )


def test_duplicate_generic_normalized_unique_names_align_once():
    entries = _prepare_dataset_identities(
        [
            {
                "id": "dataset_1",
                "dataset_name": "EXAFS",
                "measurement_technique": "extended X-ray absorption fine structure",
            }
        ],
        "synthetic_duplicate_generic_name",
    )
    predictions = [
        {
            "id": pred_id,
            "dataset_name": "EXAFS Dataset",
            "measurement_technique": "incorrect technique",
        }
        for pred_id in ("pred_b", "pred_a")
    ]
    diagnostics = {}
    matches = _match_datasets(predictions, entries, diagnostics)
    assert matches == {1: (0, "generic_normalized_unique_name")}
    assert diagnostics == {"dataset_generic_name_collision": 1}


def test_singularized_name_does_not_collapse_merged_dataset_record():
    entries = registry()["papers"]["2010.12832"]
    assert _match_datasets(
        [
            {
                "dataset_name": "shear force measurements",
                "measurement_technique": "shear force measurements",
            }
        ],
        entries,
    ) == {0: (1, "technique_and_name")}
    assert (
        _match_datasets(
            [
                {
                    "dataset_name": "X-ray tomography and shear force measurements",
                    "measurement_technique": "X-ray tomography",
                }
            ],
            entries,
        )
        == {}
    )


def test_source_grounded_dataset_forms_recover_reviewed_predictions():
    entries = registry()["papers"]
    cases = [
        ("pan_mistral_large3_full_0shot", "2407.20626", "technique_and_name"),
        ("pan_qwen3_5_9b_full_1shot", "2407.20626", "technique_and_name"),
        ("pan_gpt5_5_full_0shot", "2203.07170", "technique_and_name"),
        ("pan_gpt5_5_no_inst_0shot", "2407.10282", "unique_technique"),
        ("pan_gpt5_5_no_inst_0shot", "1909.08292", "unique_technique"),
    ]
    for run_name, paper_id, stage in cases:
        matches = _match_datasets(
            extraction_datasets(run_name, paper_id), entries[paper_id]
        )
        assert matches[0] == (0, stage)


def test_source_grounded_dataset_forms_preserve_neighbor_boundaries():
    entries = registry()["papers"]

    rixs = extraction_datasets("pan_gemma4_31b_local_full_0shot", "2407.20626")
    assert _match_datasets(rixs, entries["2407.20626"]) == {
        0: (0, "technique_and_name")
    }

    zinc = extraction_datasets("pan_mistral_large3_full_0shot", "2203.07170")
    assert _match_datasets(zinc, entries["2203.07170"]) == {
        0: (0, "technique_and_name")
    }

    imaging = extraction_datasets("pan_gpt5_5_full_0shot", "2407.10282")
    assert _match_datasets(imaging, entries["2407.10282"]) == {
        0: (0, "unique_technique"),
        1: (1, "technique_and_name"),
        2: (2, "technique_and_name"),
    }
