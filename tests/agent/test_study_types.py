"""M09 protocol declarations are synthetic and do not authorize inference."""
import pytest

from meta_data_extraction.agent.study_types import StudySpec


def declaration():
    return {"description": "Invented repair mechanics", "primary_outcome": "Reference diagnostic",
            "counting_policy": "One synthetic publication; failed cells remain visible.",
            "evaluation_protocol_sha256": "a" * 64,
            "publications": [{"publication_id": "invented-a", "workflows": {
                "fixed": {"path": "configs/agent/workflow_fixed.json", "sha256": "b" * 64}}}],
            "variants": ["fixed"], "caps": [2, 5], "max_total_model_calls": 7}


def test_frozen_protocol_is_explicit_about_mock_scope():
    spec = StudySpec.model_validate(declaration())
    assert spec.scope == "synthetic_repair_policy"
    assert spec.authorized_mode == "synthetic_offline"
    assert spec.seed_policy == "deterministic_mock"
    assert spec.provider_nondeterminism == "not_represented"
    assert spec.publications[0].split == "calibration"
    with pytest.raises(ValueError):
        spec.repeats = 2


@pytest.mark.parametrize("changes", [
    {"version": "future"}, {"scope": "end_to_end"}, {"authorized_mode": "live"},
    {"description": ""}, {"primary_outcome": ""}, {"counting_policy": ""},
    {"description": "  \n"}, {"primary_outcome": "\t"}, {"counting_policy": " "},
    {"evaluation_protocol_sha256": "not-a-hash"}, {"evaluation_protocol_path": "data/gold/x"},
    {"publications": []}, {"variants": []}, {"variants": ["fixed", "fixed"]},
    {"variants": ["A3"]}, {"caps": []}, {"caps": [2, 2]}, {"caps": [True]},
    {"caps": ["2"]}, {"caps": [2.0]}, {"caps": [-1]}, {"caps": [33]},
    {"repeats": 0}, {"repeats": 4}, {"repeats": True}, {"repeats": "1"},
    {"max_total_model_calls": 6}, {"max_total_model_calls": True},
    {"seed_policy": "random"}, {"provider_nondeterminism": "measured"},
])
def test_invalid_scope_and_budgets_are_rejected(changes):
    with pytest.raises(ValueError):
        StudySpec.model_validate(declaration() | changes)


@pytest.mark.parametrize("change", [
    {"publication_id": "../gold"}, {"publication_id": ""}, {"split": "development"},
    {"data_kind": "natural"}, {"derived_from": "invented-parent"},
    {"identity_policy": "each_chunk_is_independent"}, {"workflows": {}}, {"workflows": {"generic_llm": {
        "path": "configs/agent/workflow_generic.json", "sha256": "b" * 64}}},
])
def test_publication_identity_and_common_variant_set_are_explicit(change):
    value = declaration()
    value["publications"][0].update(change)
    with pytest.raises(ValueError):
        StudySpec.model_validate(value)


@pytest.mark.parametrize("name", ["invented-a", "INVENTED-A"])
def test_same_publication_cannot_enter_twice(name):
    value = declaration()
    value["publications"].append(value["publications"][0] | {"publication_id": name})
    value["max_total_model_calls"] = 14
    with pytest.raises(ValueError):
        StudySpec.model_validate(value)


def test_repeats_are_fully_reserved_and_zero_call_cap_is_valid():
    value = declaration() | {"repeats": 3, "caps": [0, 2, 5], "max_total_model_calls": 21}
    assert StudySpec.model_validate(value).caps == (0, 2, 5)
    with pytest.raises(ValueError):
        StudySpec.model_validate(value | {"max_total_model_calls": 20})
