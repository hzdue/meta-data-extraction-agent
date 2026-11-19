"""Frozen pseudo-stage schedules and optimizer-reset boundary."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from meta_data_extraction.finetune.pseudo_experiment import (
    plan_pseudo_experiment,
    prepare_pseudo_stage_a,
    prepare_pseudo_stage_b,
    read_pseudo_then_gold_config,
)
from meta_data_extraction.finetune.training import (
    attach_lora,
    build_trainer,
    initial_adapter_contract,
    reload_adapter,
)
from tests.test_finetune_training import (
    PREPARATION,
    ROOT,
    config,
    records,
    tiny_model,
    tokenizer,
)


STAGE_A = ROOT / "configs/finetune_pseudo_only.yaml"
STAGE_B = ROOT / "configs/finetune_pseudo_then_gold.yaml"


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_frozen_plan_has_one_pseudo_pass_then_fresh_gold_d24():
    plan = plan_pseudo_experiment(ROOT, STAGE_A, STAGE_B)
    assert plan["training_started"] is False
    assert plan["stage_a"] == {
        "condition": "pseudo_length_filtered",
        "seed": 42,
        "examples": 442,
        "optimizer_updates": 111,
        "diagnostic_checkpoints": [32, 64, 96, 111],
        "output_dir": "outputs/finetune/pseudo_label_supervision/pseudo_length_filtered_seed42",
    }
    assert plan["stage_b"]["examples"] == 24
    assert plan["stage_b"]["optimizer_updates"] == 6
    assert plan["stage_b"]["optimizer_policy"] == "fresh_adamw_torch"
    assert plan["stage_b"]["scheduler_policy"] == "fresh_constant"
    reviewed = read_pseudo_then_gold_config(STAGE_B)
    assert reviewed.reuse_pseudo_optimizer is False


def test_stage_a_applies_frozen_length_filter_and_preserves_validation():
    try:
        cfg, _, schedule, validation, summary, experiment = prepare_pseudo_stage_a(
            ROOT, STAGE_A, offline=True
        )
    except OSError:
        pytest.skip("Pinned tokenizer is not cached")

    assert cfg.max_steps == 111
    assert len(schedule) == 442 and len(validation) == 57
    assert len({row["paper_id"] for row in schedule}) == 442
    assert max(len(row["input_ids"]) for row in schedule) == 32_530
    assert summary["training_filter"]["maximum"] == 32_660
    assert summary["training_filter"]["excluded_over_limit"] == 62
    assert experiment["planned_exposure"] == {"D": 442}
    assert experiment["final_update_examples"] == 2


def test_gold_stage_matches_control_schedule_and_three_exposures_per_paper():
    try:
        cfg, _, schedule, validation, summary, experiment = prepare_pseudo_stage_b(
            ROOT, STAGE_B, offline=True
        )
    except OSError:
        pytest.skip("Pinned tokenizer is not cached")

    assert cfg.max_steps == cfg.save_steps == cfg.validation_steps == 6
    assert cfg.learning_rate == 1e-4
    assert len(schedule) == 24 and len(validation) == 2
    assert Counter(row["task"] for row in schedule) == Counter({"D": 24})
    papers = Counter(row["example_id"].split(":")[0] for row in schedule)
    assert len(papers) == 8 and set(papers.values()) == {3}
    assert summary["task_counts"] == {"D": 24}
    assert experiment["reuse_pseudo_optimizer"] is False
    assert experiment["schedule_sha256"] == (
        "1cf6bb57af1b8c4e4fe476293fda845f684153ccffd11a2e79efc51d73af5309"
    )


def test_stage_b_loads_trainable_lora_but_starts_fresh_optimizer(tmp_path):
    stage_a_config = config(max_steps=1, save_steps=1, validation_steps=1)
    stage_a = build_trainer(
        attach_lora(tiny_model(), stage_a_config, PREPARATION),
        tokenizer(), records(), stage_a_config, tmp_path / "stage_a", cpu=True,
    )
    stage_a.train()
    adapter = tmp_path / "stage_a_adapter"
    stage_a.save_model(str(adapter))
    saved_lora = {
        name: parameter.detach().clone()
        for name, parameter in stage_a.model.named_parameters()
        if ".lora_" in name
    }
    assert saved_lora
    assert not (adapter / "optimizer.pt").exists()
    assert not (adapter / "scheduler.pt").exists()
    boundary = initial_adapter_contract(tmp_path, adapter)
    assert boundary["path"] == "stage_a_adapter"
    assert len(boundary["adapter_sha256"]) == 64
    assert boundary["optimizer_policy"] == "fresh_not_restored_from_initial_adapter"
    assert boundary["scheduler_policy"] == "fresh_not_restored_from_initial_adapter"

    loaded = reload_adapter(
        tiny_model(), adapter, PREPARATION, trainable=True, config=stage_a_config
    )
    stage_b = build_trainer(
        loaded, tokenizer(), records(), stage_a_config, tmp_path / "stage_b", cpu=True
    )
    loaded_lora = {
        name: parameter.detach().clone()
        for name, parameter in stage_b.model.named_parameters()
        if ".lora_" in name
    }
    assert saved_lora.keys() == loaded_lora.keys()
    for name in saved_lora:
        torch.testing.assert_close(saved_lora[name], loaded_lora[name], rtol=0, atol=0)
    assert all(parameter.requires_grad for name, parameter in stage_b.model.named_parameters()
               if ".lora_" in name)
    assert stage_b.optimizer is None and stage_b.lr_scheduler is None
    assert stage_b.state.global_step == 0
    assert stage_b.exposure == {"examples": {}, "supervised_tokens": {}, "total_tokens": {}}

    stage_b.train()
    assert stage_b.state.global_step == 1
    assert stage_b.optimizer is not None and stage_b.lr_scheduler is not None
    assert sum(stage_b.exposure["examples"].values()) == stage_a_config.gradient_accumulation_steps


def test_final_partial_accumulation_consumes_each_retained_example_once(tmp_path):
    rows = records()
    tail = deepcopy(rows[:2])
    for index, row in enumerate(tail, start=4):
        row["example_id"] = f"paper_{index}:D"
    schedule = rows + tail
    cfg = config(max_steps=2, save_steps=2, validation_steps=2,
                 gradient_accumulation_steps=4)
    trainer = build_trainer(
        attach_lora(tiny_model(), cfg, PREPARATION), tokenizer(), schedule, cfg,
        tmp_path / "partial", cpu=True,
    )

    trainer.train()

    assert trainer.state.global_step == 2
    assert trainer.exposure["examples"] == {"D": 6}
