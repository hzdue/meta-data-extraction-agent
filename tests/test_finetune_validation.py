"""Step 4 runs on synthetic fixtures, without the 4B weights or GPU training."""
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from meta_data_extraction.finetune.training import attach_lora, build_trainer, prepare_output
from meta_data_extraction.finetune import validation as validation_module
from meta_data_extraction.finetune.validation import (
    best_entry, generate_answer, paper_mean, score_generation, validate_rows,
    validation_context, validation_losses,
)
from tests.test_finetune_training import (
    PREPARATION, ROOT, StopAfterTwo, config, records, tiny_model, tokenizer,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def validation_rows():
    result = records()[:2]
    for i, row in enumerate(result):
        pid = f"validation_{i}"
        row.update(paper_id=pid, example_id=f"{pid}:D", split="validation",
                   prompt_ids=row["input_ids"][:3],
                   gold={"facilities": [], "samples": [], "datasets": []})
    # Unequal answer lengths expose accidental pooling across the two papers.
    result[1]["input_ids"].insert(4, 12)
    result[1]["labels"].insert(4, 12)
    return result


def entry(step, first, second, **extra):
    papers = [{"paper_id": "short", "loss": first, "supervised_tokens": 2},
              {"paper_id": "long", "loss": second, "supervised_tokens": 200}]
    return {"step": step, "loss": paper_mean(papers), "papers": papers, **extra}


def test_equal_paper_mean_and_earliest_tie_ignore_f1():
    history = [entry(0, 2.0, 4.0, f1=1.0), entry(2, 1.0, 2.0, f1=0.0),
               entry(4, 1.0, 2.0, f1=1.0)]
    assert history[0]["loss"] == 3.0
    assert best_entry(history)["step"] == 2
    assert best_entry([entry(0, 1.0, 1.0), entry(2, 1.0, 1.0)])["step"] == 0


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -0.1])
def test_invalid_loss_cannot_select_a_checkpoint(loss):
    with pytest.raises(ValueError, match="finite"):
        entry(0, loss, 1.0)


def test_missing_paper_baseline_or_inconsistent_aggregation_rejected():
    with pytest.raises(ValueError, match="two distinct"):
        paper_mean(entry(0, 1.0, 2.0)["papers"][:1])
    with pytest.raises(ValueError, match="step-zero"):
        best_entry([entry(2, 1.0, 2.0)])
    bad = entry(0, 1.0, 2.0)
    bad["loss"] = 0.1
    with pytest.raises(ValueError, match="equal-paper"):
        best_entry([bad])


@pytest.mark.parametrize("change", ["train_paper", "test_split", "auxiliary", "gold_in_prompt", "overflow"])
def test_validation_membership_boundaries_and_context_guard(change):
    rows = validation_rows()
    training_ids = {"paper_0"}
    if change == "train_paper":
        rows[0]["paper_id"] = "paper_0"
    elif change == "test_split":
        rows[0]["split"] = "test"
    elif change == "auxiliary":
        rows[0]["task"] = "R"
    elif change == "gold_in_prompt":
        rows[0]["prompt_ids"] = rows[0]["input_ids"]
    with pytest.raises(ValueError):
        validate_rows(rows, training_ids, 6 if change == "overflow" else 128, 4)


def test_checkpoint_interval_must_match_validation():
    with pytest.raises(ValueError, match="must match"):
        config(validation_steps=3)


def test_forward_validation_matches_paper_losses_and_preserves_training_state(tmp_path):
    cfg = config(lora_dropout=0.0)
    model = attach_lora(tiny_model(), cfg, PREPARATION)
    trainer = build_trainer(model, tokenizer(), records(), cfg, tmp_path, cpu=True)
    rows = validation_rows()
    model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    exposure = deepcopy(trainer.exposure)
    rng = torch.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with validation_context(model):
        assert not model.training and not torch.is_grad_enabled()
        actual = validation_losses(trainer, rows)
        expected = []
        for row in rows:
            batch = trainer.data_collator([row])
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            expected.append(float(torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 128), batch["labels"][:, 1:].reshape(-1))))
        random.random()
        np.random.rand()
        torch.rand(4)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert [r["loss"] for r in actual] == pytest.approx(expected)
    assert paper_mean(actual) == pytest.approx(sum(expected) / 2)
    assert trainer.exposure == exposure
    assert all(p.grad is None and torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_step_zero_disables_nonzero_adapter_and_restores_after_failure(tmp_path):
    cfg = config()
    model = attach_lora(tiny_model(), cfg, PREPARATION)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.3)
    inputs = torch.tensor([[4, 5, 6]])
    model.eval()
    with torch.no_grad():
        adapted = model(input_ids=inputs).logits
        with validation_context(model, base=True):
            baseline = model(input_ids=inputs).logits
        original = tiny_model().eval()(input_ids=inputs).logits
    torch.testing.assert_close(baseline, original)
    assert not torch.allclose(adapted, baseline)
    model.train()
    with pytest.raises(RuntimeError):
        with validation_context(model, base=True):
            raise RuntimeError("simulated validation error")
    assert model.training
    assert any(p.requires_grad for n, p in model.named_parameters() if "lora_B" in n)


def test_generation_uses_only_d_prompt_and_fixed_greedy_settings(monkeypatch):
    model, tok, row = tiny_model(), tokenizer(), validation_rows()[0]
    captured = {}
    def generate(**kwargs):
        captured.update(kwargs)
        return torch.cat([kwargs["input_ids"], torch.tensor([[3]])], dim=1)
    monkeypatch.setattr(model, "generate", generate)
    result = generate_answer(model, tok, row, 4)
    assert captured["input_ids"].tolist() == [row["prompt_ids"]]
    assert "labels" not in captured and "gold" not in captured
    assert captured["generation_config"].do_sample is False
    assert captured["generation_config"].max_new_tokens == 4
    assert captured["logits_to_keep"] == 1
    assert result == {"text": "", "generated_tokens": 1, "ended_with_eos": True, "hit_output_limit": False}


def test_shared_evaluator_recovers_perfect_entities_and_references(tmp_path):
    pid = "2010.02965"
    gold = json.loads((ROOT / "data/gold" / f"{pid}.json").read_text(encoding="utf-8"))
    row = {"paper_id": pid, "gold": gold}
    generated = {"text": json.dumps(gold), "generated_tokens": 1, "ended_with_eos": True, "hit_output_limit": False}
    scored = score_generation(row, generated, tmp_path)
    assert scored["status"] == "scored"
    assert set(scored["entity_f1"].values()) == {1.0}
    assert all(r["f1"] == 1.0 and r["n_pairs"] == 6 for r in scored["conditional_reference_f1"].values())
    assert (tmp_path / scored["raw_file"]).exists()
    assert (tmp_path / scored["report_file"]).exists()
    empty = score_generation(row, {**generated, "text": "{}"}, tmp_path / "empty")
    assert empty["entity_f1"]["datasets"] == 0.0
    assert empty["conditional_reference_f1"]["sample_refs"] == {"f1": None, "n_pairs": 0}


@pytest.mark.parametrize("text", ["broken JSON", '["not an object"]', '{"x": NaN}', '{"x": 1, "x": 2}'])
def test_invalid_generations_are_recorded_with_no_fabricated_f1(tmp_path, text):
    result = score_generation(validation_rows()[0], {"text": text, "generated_tokens": 4,
                              "ended_with_eos": False, "hit_output_limit": True}, tmp_path)
    assert result["status"] == "invalid_json" and result["hit_output_limit"]
    assert result["entity_f1"] is None and result["conditional_reference_f1"] is None
    assert (tmp_path / result["raw_file"]).read_text(encoding="utf-8") == text


@pytest.mark.parametrize("method", ["json_only", "structure_aware"])
def test_validation_resume_matches_uninterrupted_and_final_off_interval_is_saved(tmp_path, method):
    cfg = config(method=method, max_steps=3)
    def trainer_at(path, callbacks=None, validation=True):
        model = attach_lora(tiny_model(), cfg, PREPARATION)
        return build_trainer(model, tokenizer(), records(method), cfg, path, cpu=True, callbacks=callbacks,
                             validation=validation_rows() if validation else None)
    full = trainer_at(tmp_path / "full")
    full.train()
    assert [r["step"] for r in full.validation.history] == [0, 2, 3]
    assert [r["step"] for r in full.state.log_history if "loss" in r] == [1, 2, 3]
    assert (tmp_path / "full/checkpoint-3/validation_state.json").exists()
    assert len(full.validation.history[0]["diagnostics"]) == 2
    assert sum(full.exposure["examples"].values()) == 6
    without_validation = trainer_at(tmp_path / "no_validation", validation=False)
    without_validation.train()
    for (name, p), (other, q) in zip(full.model.named_parameters(), without_validation.model.named_parameters()):
        assert name == other
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    interrupted = trainer_at(tmp_path / "resumed", [StopAfterTwo()])
    interrupted.train()
    resumed = trainer_at(tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=str(tmp_path / "resumed/checkpoint-2"))
    assert resumed.validation.history == full.validation.history
    assert resumed.exposure == full.exposure
    for (name, p), (other, q) in zip(full.model.named_parameters(), resumed.model.named_parameters()):
        assert name == other
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    selection = json.loads((tmp_path / "resumed/selection.json").read_text())
    assert selection["status"] == "complete"
    assert selection["step"] == best_entry(full.validation.history)["step"]


@pytest.mark.parametrize("losses,winning_step", [([1.0, 2.0, 3.0], 0), ([3.0, 1.0, 2.0], 2)])
def test_selected_artifact_is_base_or_best_adapter_not_last(monkeypatch, tmp_path, losses, winning_step):
    cfg = config(max_steps=3)
    values = iter(losses)
    def measured(trainer, rows):
        value = next(values)
        return [{"paper_id": r["paper_id"], "loss": value, "supervised_tokens": 3} for r in rows]
    monkeypatch.setattr(validation_module, "validation_losses", measured)
    model = attach_lora(tiny_model(), cfg, PREPARATION)
    trainer = build_trainer(model, tokenizer(), records(), cfg, tmp_path, cpu=True, validation=validation_rows())
    trainer.train()
    selected = json.loads((tmp_path / "selection.json").read_text())
    assert trainer.state.global_step == 3  # No automatic early stopping.
    assert selected["step"] == winning_step
    if winning_step == 0:
        assert selected["kind"] == "base" and selected["checkpoint"] is None
        assert selected["adapter_sha256"] is None
    else:
        assert selected["checkpoint"] == "checkpoint-2"
        assert len(selected["adapter_sha256"]) == 64
        assert (tmp_path / selected["checkpoint"] / "adapter_model.safetensors").exists()
        assert sum(selected["exposure"]["examples"].values()) == 4


def test_resume_rejects_missing_or_changed_validation_history(tmp_path):
    cfg = config()
    trainer = build_trainer(attach_lora(tiny_model(), cfg, PREPARATION), tokenizer(), records(), cfg,
                            tmp_path, cpu=True, validation=validation_rows(), callbacks=[StopAfterTwo()])
    trainer.train()
    path = tmp_path / "checkpoint-2/validation_state.json"
    saved = json.loads(path.read_text())
    saved["history"].pop()
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="incomplete"):
        trainer.validation.restore(path.parent, 2)
    saved["validation_sha256"] = "wrong"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="changed"):
        trainer.validation.restore(path.parent, 2)


def test_resume_accepts_latest_complete_checkpoint_when_later_save_is_partial(tmp_path):
    cfg = config()
    contract = {"config": cfg.model_dump()}
    prepare_output(tmp_path, contract, None)
    trainer = build_trainer(attach_lora(tiny_model(), cfg, PREPARATION), tokenizer(), records(), cfg,
                            tmp_path, cpu=True, validation=validation_rows(), callbacks=[StopAfterTwo()])
    trainer.train()
    partial = tmp_path / "checkpoint-4"
    partial.mkdir()
    (partial / "trainer_state.json").write_text('{"global_step": 4}')
    prepare_output(tmp_path, contract, tmp_path / "checkpoint-2")
    assert partial.exists()  # Discovery is read-only; no checkpoints are deleted.
