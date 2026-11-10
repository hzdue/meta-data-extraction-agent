"""Tests for the frozen no-update gradient interaction diagnostic."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from meta_data_extraction.finetune.gradient_interaction import (
    comparison_kind, diagnostic_train_mode, gram_matrices, pair_rows, parameter_group,
    read_gradient_config, summarize_pairs, verify_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_gradient_configuration_and_endpoints():
    path = ROOT / "configs/finetune_gradient_interaction.yaml"
    config = read_gradient_config(path)
    _, _, endpoints = verify_contract(ROOT, path, config)
    assert config.seeds == [42, 123, 456]
    assert config.tasks == ["D", "E", "R"]
    assert [item["endpoint_step"] for item in endpoints] == [6, 6, 6]
    assert config.attach_mlflow is False
    assert config.model_selection_allowed is False


def test_parameter_groups_and_float64_gram_accumulation():
    name = "base_model.model.model.language_model.layers.2.self_attn.q_proj.lora_A.default.weight"
    assert parameter_group(name) == ("block_02", "self_attn")
    gradients = [{name: torch.tensor([1.0, 2.0])}, {name: torch.tensor([3.0, 4.0])}]
    grams = gram_matrices(gradients)
    assert grams["global"].dtype == torch.float64
    assert grams["global"].tolist() == [[5.0, 11.0], [11.0, 25.0]]
    assert torch.equal(grams["global"], grams["block_02"])
    assert torch.equal(grams["global"], grams["family_self_attn"])


def test_comparison_design_excludes_same_task_same_paper_and_E_R():
    d1 = {"paper_id": "p1", "task": "D"}
    e1 = {"paper_id": "p1", "task": "E"}
    r1 = {"paper_id": "p1", "task": "R"}
    d2 = {"paper_id": "p2", "task": "D"}
    assert comparison_kind(d1, e1) == "same_paper_D_E"
    assert comparison_kind(d1, r1) == "same_paper_D_R"
    assert comparison_kind(d1, d2) == "cross_paper_D_D"
    assert comparison_kind(e1, r1) is None
    assert comparison_kind(d1, d1) is None


def test_pair_statistics_have_expected_counts_and_reject_zero_norm():
    metadata = [{"example_id": f"p{paper}:{task}", "paper_id": f"p{paper}", "task": task}
                for paper in range(2) for task in ("D", "E", "R")]
    vectors = torch.tensor([[1., 0.], [1., 1.], [1., -1.],
                            [0., 1.], [1., 2.], [-1., 1.]], dtype=torch.float64)
    rows, _ = pair_rows(metadata, {"global": vectors @ vectors.T}, 42, 1e-4, 1e-12)
    counts = {kind: sum(row["comparison"] == kind for row in rows)
              for kind in {row["comparison"] for row in rows}}
    assert counts == {"same_paper_D_E": 2, "same_paper_D_R": 2,
                      "cross_paper_D_D": 1, "cross_paper_D_E": 2,
                      "cross_paper_D_R": 2, "cross_paper_E_E": 1,
                      "cross_paper_R_R": 1}
    with pytest.raises(ValueError, match="Near-zero"):
        pair_rows(metadata, {"global": torch.zeros((6, 6), dtype=torch.float64)},
                  42, 1e-4, 1e-12)


def test_summary_uses_seed_medians_not_correlated_pairs_as_replicates():
    rows = [{"seed": seed, "comparison": "cross_paper_D_R", "cosine": value, "dot": value}
            for seed, value in zip((42, 123, 456), (-.2, .1, .4))]
    summary = summarize_pairs(rows)["across_seeds"]["cross_paper_D_R"]
    assert summary["median_of_seed_medians"] == pytest.approx(.1)
    assert summary["seed_median_range"] == pytest.approx([-.2, .4])


def test_diagnostic_mode_keeps_training_path_but_disables_lora_dropout():
    model = torch.nn.Module()
    model.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(.05)})
    model.eval()
    assert diagnostic_train_mode(model) == 1
    assert model.training is True
    assert model.lora_dropout["default"].p == 0
