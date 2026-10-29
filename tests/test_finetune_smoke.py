"""Step 6 selection and real trainer probes on synthetic CPU fixtures."""
from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("mlflow")

from meta_data_extraction.finetune.smoke import CheckGradients, case_rows, exercise_case, probe_plan, weight_hashes
from meta_data_extraction.finetune.training import attach_lora, build_trainer
from tests.test_finetune_training import PREPARATION, config, records, tiny_model, tokenizer
from tests.test_finetune_validation import validation_rows


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_plan_selects_each_longest_task_without_validation_leakage():
    rows = records("structure_aware")
    for i, row in enumerate(rows):
        row["input_ids"] += [8] * i
        row["labels"] += [-100] * i
    original = deepcopy(rows)
    plan = probe_plan(rows, validation_rows(), config())
    assert plan["short"][0]["example_id"] == "paper_0:D"
    for task in "DER":
        assert plan[f"longest_{task}"][0]["example_id"] == f"paper_3:{task}"
    assert [r["example_id"] for r in plan["validation_loss"]] == ["validation_0:D", "validation_1:D"]
    assert rows == original
    with pytest.raises(ValueError, match="No training examples"):
        case_rows("longest_E", records(), validation_rows())
    bad = validation_rows()
    bad[0].update(paper_id="paper_0", example_id="paper_0:D")
    with pytest.raises(ValueError, match="held-out"):
        probe_plan(rows, bad, config())


def test_short_probe_checks_real_update_and_frozen_weights(tmp_path):
    cfg = config(max_steps=1, save_steps=1, validation_steps=1)
    model = attach_lora(tiny_model(), cfg, PREPARATION)
    chosen = case_rows("short", records(), validation_rows())
    check = CheckGradients()
    trainer = build_trainer(model, tokenizer(), chosen * cfg.gradient_accumulation_steps,
                            cfg, tmp_path, cpu=True, callbacks=[check])
    result = exercise_case("short", trainer, chosen, validation_rows(), cfg, tmp_path, {})
    assert result["weight_hashes_before"]["frozen"] == result["weight_hashes_after"]["frozen"]
    assert result["weight_hashes_before"]["adapter"] != result["weight_hashes_after"]["adapter"]
    assert result["exposure"]["examples"] == {"D": cfg.gradient_accumulation_steps}
    assert len(check.updates) == 1
    assert (tmp_path / "checkpoint-1/optimizer.pt").exists()
    assert (tmp_path / "adapter/adapter_model.safetensors").exists()


def test_validation_probe_does_not_update_weights_or_exposure(tmp_path):
    cfg = config()
    trainer = build_trainer(attach_lora(tiny_model(), cfg, PREPARATION), tokenizer(), records(),
                            cfg, tmp_path, cpu=True)
    before = weight_hashes(trainer.model)
    result = exercise_case("validation_loss", trainer, validation_rows(), validation_rows(), cfg, tmp_path, {})
    assert len(result["papers"]) == 2 and result["loss"] > 0
    assert weight_hashes(trainer.model) == before
    assert trainer.state.global_step == 0 and not trainer.exposure["examples"]


def test_gradient_check_rejects_nonfinite_or_missing_gradients():
    model = attach_lora(tiny_model(), config(), PREPARATION)
    state = type("State", (), {"global_step": 0})()
    check = CheckGradients()
    with pytest.raises(ValueError, match="No nonzero"):
        check.on_pre_optimizer_step(None, state, None, model=model)
    p = next(p for p in model.parameters() if p.requires_grad)
    p.grad = torch.full_like(p, float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        check.on_pre_optimizer_step(None, state, None, model=model)
