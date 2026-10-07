from meta_data_extraction.alignment import align_facilities


def _pairs(matches):
    return {(pair.pred_index, pair.gold_index, pair.stage) for pair in matches.pairs}


def test_exact_facility_wins_over_containing_beamline_description():
    pred = [
        {"name": "ALS BL4.0.3 MERIXS endstation"},
        {"name": "Advanced Light Source"},
    ]
    gold = [{"name": "Advanced Light Source"}]
    assert _pairs(align_facilities(pred, gold)) == {(1, 0, "raw_exact")}


def test_alias_exact_handles_acronym_and_parenthetical_full_name():
    pred = [{"name": "ESRF"}]
    gold = [{"name": "European Synchrotron Radiation Facility (ESRF)"}]
    assert _pairs(align_facilities(pred, gold)) == {
        (0, 0, "alias_exact")
    }


def test_facility_alias_is_global():
    pred = [{"name": "ESRF"}]
    gold = [{"name": "European Synchrotron Radiation Facility"}]
    assert _pairs(align_facilities(pred, gold)) == {(0, 0, "alias_exact")}


def test_containment_remains_available_as_fallback():
    pred = [{"name": "LLB (Saclay, France)"}]
    gold = [{"name": "Laboratoire Léon Brillouin"}]
    matches = align_facilities(pred, gold)
    assert len(matches.pairs) == 1
    assert matches.pairs[0].stage == "containment"


def test_partial_generic_containment_does_not_merge_distinct_facilities():
    pred = [{"name": "Diamond Light Source"}]
    gold = [{"name": "Advanced Light Source"}]
    assert align_facilities(pred, gold).pairs == []


def test_source_grounded_facility_form_survives_complete_containment():
    pred = [{"name": "SAMBA station of Soleil"}]
    gold = [{"name": "Synchrotron SOLEIL"}]
    assert _pairs(align_facilities(pred, gold)) == {
        (0, 0, "containment")
    }


def test_alias_identity_absorbs_reversed_canonical_name_words():
    pred = [
        {"name": "SAMBA beamline, Synchrotron SOLEIL, Gif sur Yvette, France"}
    ]
    gold = [{"name": "SOLEIL synchrotron"}]
    assert _pairs(align_facilities(pred, gold)) == {
        (0, 0, "containment")
    }


def test_alias_identity_absorbs_parenthetical_acronym_and_repeated_full_name():
    pred = [{"name": "Atomic Energy Centre (AECD) of BAEC, Dhaka"}]
    gold = [{"name": "Atomic Energy Centre of BAEC, Dhaka"}]
    assert _pairs(align_facilities(pred, gold)) == {
        (0, 0, "alias_exact")
    }


def test_instrument_name_alone_does_not_match_its_host_facility():
    pred = [{"name": "SAMBA"}]
    gold = [{"name": "Synchrotron SOLEIL"}]
    assert align_facilities(pred, gold).pairs == []


def test_multiple_aliases_for_one_facility_share_one_identity():
    pred = [{"name": "National Synchrotron Light Laboratory"}]
    gold = [{"name": "Brazilian Synchrotron Light Laboratory"}]
    assert _pairs(align_facilities(pred, gold)) == {
        (0, 0, "alias_exact")
    }


def test_duplicate_is_not_hidden_by_one_gold_entity():
    pred = [{"name": "Advanced Light Source"}, {"name": "ALS"}]
    gold = [{"name": "Advanced Light Source"}]
    matches = align_facilities(pred, gold)
    assert _pairs(matches) == {(0, 0, "raw_exact")}


def test_distinct_known_facilities_do_not_match_on_generic_words():
    pred = [{"name": "Advanced Light Source"}]
    gold = [{"name": "Advanced Photon Source"}]
    assert align_facilities(pred, gold).pairs == []


def test_ambiguous_facility_containment_abstains():
    pred = [{"name": "National Laboratory"}]
    gold = [
        {"name": "Alpha National Laboratory"},
        {"name": "Beta National Laboratory"},
    ]
    for ordered_gold in (gold, gold[::-1]):
        assert align_facilities(pred, ordered_gold).pairs == []


def test_facility_containment_uses_other_matches_to_resolve_ambiguity():
    pred = [
        {"name": "National Laboratory"},
        {"name": "Alpha National Laboratory north campus"},
    ]
    gold = [
        {"name": "Alpha National Laboratory"},
        {"name": "Beta National Laboratory"},
    ]
    assert _pairs(align_facilities(pred, gold)) == {
        (0, 1, "containment"), (1, 0, "containment")
    }


def test_beamline_only_prediction_cannot_match_a_host_name_containing_it():
    pred = [{"name": "SAMBA"}]
    gold = [{"name": "SAMBA beamline, Synchrotron SOLEIL"}]
    assert align_facilities(pred, gold).pairs == []


def test_facility_containment_preserves_a_name_with_its_parent_institution():
    pred = [{"name": "OPAL research reactor at ANSTO, Australia"}]
    gold = [{"name": "OPAL research reactor"}]
    assert _pairs(align_facilities(pred, gold)) == {(0, 0, "containment")}
