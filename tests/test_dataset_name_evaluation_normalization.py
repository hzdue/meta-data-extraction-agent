"""Keep dataset identity aliases separate from dataset-name scoring."""

from meta_data_extraction.evaluation import (
    _score_value,
    _spec_from_yaml,
    load_plan,
    _rouge_l,
)
from meta_data_extraction.alignment import _match_datasets, _prepare_dataset_identities
from meta_data_extraction.normalization import normalize_sample_formula_formatting
from meta_data_extraction.alignment import normalize_dataset


def test_alignment_alias_expands_globally_within_dataset_fields():
    assert normalize_dataset("PNR of CoFeB/Mo/Si") == "polarized neutron reflectometry of cofeb/mo/si"


def test_does_not_infer_unsupported_eels_expansion():
    assert normalize_dataset("EELS") == "eels"


def test_matches_complete_tokens_only():
    assert (
        normalize_dataset("XRD and XRD2")
        == "x ray diffraction and xrd2"
    )


def test_dataset_name_scoring_remains_wording_sensitive():
    spec = _spec_from_yaml({"metric": "rouge_l"})
    result = _score_value(
        spec,
        "PNR",
        "polarized neutron reflectometry",
        {},
    )
    assert result["score"] == 0.0


def test_alignment_alias_can_establish_identity_without_changing_name_score():
    gold = [
        {
            "id": "dataset_1",
            "dataset_name": "reflectometry campaign",
            "measurement_technique": "polarized neutron reflectometry",
        }
    ]
    entries = _prepare_dataset_identities(gold, "2501.08122")
    prediction = {
        "id": "dataset_1",
        "dataset_name": "PNR campaign",
        "measurement_technique": "PNR",
    }
    assert _match_datasets([prediction], entries) == {
        0: (0, "unique_technique")
    }
    assert _rouge_l(prediction["dataset_name"], gold[0]["dataset_name"])["score"] < 1.0


def test_scoring_plan_uses_wording_sensitive_rouge_l():
    plan = load_plan("configs/evaluation_metrics.yaml")
    title = plan["bibliographic.title"]
    subtitle = plan["bibliographic.subtitle"]
    funding = plan["bibliographic.funding"][1]
    samples = plan["samples"][1]
    datasets = plan["datasets"][1]
    assert title == ("rouge_l",)
    assert subtitle == ("rouge_l",)
    assert funding.attr_scorers["funder"] == ("rouge_l",)
    assert samples.attr_scorers["sample_name"] == (
        "rouge_l", normalize_sample_formula_formatting,
    )
    assert datasets.attr_scorers["dataset_name"] == ("rouge_l",)


def test_rouge_l_credits_aligned_sample_name_qualifiers():
    result = _rouge_l("MS-TiN", "MS-TiN reference coating")
    assert 0.0 < result["score"] < 1.0


def test_rouge_l_credits_aligned_funder_qualifiers():
    result = _rouge_l(
        "Eotvos Lorand Research Network",
        "Eotvos Lorand Research Network present name HUNREN",
    )
    assert 0.0 < result["score"] < 1.0
