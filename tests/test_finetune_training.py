"""Step 3: actual Qwen hybrid layers/LoRA, answer loss, and resumable training."""
from pathlib import Path
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (PreTrainedTokenizerFast, Qwen3_5Config,
                          Qwen3_5ForConditionalGeneration, TrainerCallback, set_seed)

from meta_data_extraction.finetune.training import (
    attach_lora, build_trainer, language_targets, prepare_output, reload_adapter,
)
from meta_data_extraction.finetune.training_data import (
    AnswerCollator, TrainingConfig, prepare_training, read_config, tokenize_records, verify_prepared,
)

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = {"model_id": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**overrides):
    values = read_config(ROOT / "configs/finetune_json_only.yaml").model_dump()
    values.update(max_steps=4, save_steps=2, gradient_accumulation_steps=2,
                  max_seq_length=128, lora_rank=2, lora_alpha=4)
    values.update(overrides)
    return TrainingConfig.model_validate(values)


def tiny_model():
    set_seed(17)
    cfg = Qwen3_5Config(
        text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                     "head_dim": 16, "layer_types": ["linear_attention", "full_attention"],
                     "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                     "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                     "max_position_embeddings": 128, "pad_token_id": 0, "eos_token_id": 3},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 2,
                       "spatial_merge_size": 2, "num_position_embeddings": 16},
        image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123,
    )
    model = Qwen3_5ForConditionalGeneration(cfg)
    model.config._attn_implementation = "eager"
    return model


def tokenizer():
    vocab = {str(i): i for i in range(128)}
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="0")),
                                  unk_token="0", pad_token="0", eos_token="3")


def records(method="json_only"):
    tasks = ["D"] if method == "json_only" else ["D", "E", "R"]
    return [{"input_ids": [4, 5, 6, 10 + i, 7, 3, 8],
             "labels": [-100, -100, -100, 10 + i, 7, 3, -100],
             "task": task, "example_id": f"paper_{i}:{task}"}
            for i in range(4) for task in tasks]


def test_only_method_and_output_differ_between_configs():
    first = read_config(ROOT / "configs/finetune_json_only.yaml").model_dump()
    second = read_config(ROOT / "configs/finetune_structure_aware.yaml").model_dump()
    assert {k for k in first if first[k] != second[k]} == {"method", "output_dir"}
    assert first["max_steps"] is None  # No arbitrary duration for comparative runs.


@pytest.mark.parametrize("field,value", [("max_steps", 0), ("packing", True), ("dtype", "float16"),
                                        ("microbatch_size", 2), ("lora_dropout", 1.0)])
def test_invalid_configuration_rejected(field, value):
    with pytest.raises(ValueError):
        config(**{field: value})


def test_collator_keeps_answer_eos_masks_prompt_postfix_and_padding():
    rows = records()[:2]
    rows[1]["input_ids"].append(9)
    rows[1]["labels"].append(-100)
    batch = AnswerCollator(0, 128)(rows)
    assert batch["labels"].tolist() == [rows[0]["labels"] + [-100], rows[1]["labels"]]
    assert batch["attention_mask"][0].tolist() == [1] * 7 + [0]
    assert batch["input_ids"][0, -1] == 0
    assert batch["labels"][0, 5] == 3
    assert batch["tasks"] == ["D", "D"]
    with pytest.raises(ValueError, match="truncation"):
        AnswerCollator(0, 7)(rows)


@pytest.mark.parametrize("method", ["json_only", "structure_aware"])
def test_real_chat_boundaries_and_padding_if_tokenizer_cached(method):
    cfg = read_config(ROOT / f"configs/finetune_{method}.yaml")
    try:
        tok, rows, _ = prepare_training(ROOT, cfg, offline=True)
    except OSError:
        pytest.skip("Pinned tokenizer not cached")
    originals, _, _ = verify_prepared(ROOT, cfg)
    batch = AnswerCollator(tok.pad_token_id, cfg.max_seq_length)(rows[:3])
    for i, original in enumerate(originals[f"train_{method}.jsonl"][:3]):
        kept = batch["labels"][i][batch["labels"][i] != -100].tolist()
        assert kept[-1] == tok.eos_token_id
        assert tok.decode(kept[:-1]) == original["completion"][0]["content"]
    with pytest.raises(ValueError, match="oversized"):
        tokenize_records(tok, originals[f"train_{method}.jsonl"][:1], 100)


def test_prepared_artifacts_allow_environment_drift_but_not_payload_changes(tmp_path):
    cfg = read_config(ROOT / "configs/finetune_json_only.yaml")
    data, manifest, drift = verify_prepared(ROOT, cfg)
    assert [len(data[n]) for n in ("train_json_only.jsonl", "train_structure_aware.jsonl", "validation.jsonl")] == [8, 24, 2]
    assert set(drift) <= {"pyproject.toml", "uv.lock"}
    copied = tmp_path / "prepared"
    shutil.copytree(ROOT / cfg.prepared_dir, copied)
    cfg = cfg.model_copy(update={"prepared_dir": str(copied)})
    path = copied / "train_json_only.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset hash"):
        verify_prepared(ROOT, cfg)


def test_lora_targets_both_attention_types_and_no_vision_or_head():
    base = tiny_model()
    targets = language_targets(base)
    assert len(targets) == 15
    assert any("linear_attn" in t for t in targets) and any("self_attn" in t for t in targets)
    adapted = attach_lora(base, config(), PREPARATION)
    trainable = [n for n, p in adapted.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_" in n and ".language_model.layers." in n for n in trainable)
    assert all(not p.requires_grad for p in adapted.base_model.model.model.visual.parameters())


def test_fused_answer_loss_matches_explicit_cross_entropy(tmp_path):
    cfg = config(lora_dropout=0.0)
    model = attach_lora(tiny_model(), cfg, PREPARATION)
    trainer = build_trainer(model, tokenizer(), records(), cfg, tmp_path, cpu=True)
    rows = records()[:2]
    rows[1]["labels"][3] = -100  # Different answer lengths exercise token weighting.
    batch = trainer.data_collator(rows)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 128), batch["labels"][:, 1:].reshape(-1))
        actual = trainer.compute_loss(model, batch)
        count = (batch["labels"][:, 1:] != -100).sum()
        accumulated = sum(trainer.compute_loss(model, trainer.data_collator([row]), num_items_in_batch=count) for row in rows)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(accumulated, expected)
    assert not trainer.exposure["examples"]  # Diagnostics must not count as training.


class StopAfterTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
        return control


@pytest.mark.parametrize("method", ["json_only", "structure_aware"])
def test_training_resume_and_adapter_reload_match_uninterrupted(tmp_path, method):
    cfg = config(method=method)
    tok, rows = tokenizer(), records(method)

    def trainer_at(path, callbacks=None):
        base = tiny_model()
        model = attach_lora(base, cfg, PREPARATION)
        return build_trainer(model, tok, rows, cfg, path, cpu=True, callbacks=callbacks)

    uninterrupted = trainer_at(tmp_path / "full")
    frozen_before = {n: p.detach().clone() for n, p in uninterrupted.model.named_parameters() if not p.requires_grad}
    uninterrupted.train()
    assert uninterrupted.state.global_step == 4
    for n, p in uninterrupted.model.named_parameters():
        if n in frozen_before:
            assert torch.equal(p, frozen_before[n]), n
    assert any(torch.count_nonzero(p) > 0 for n, p in uninterrupted.model.named_parameters() if "lora_B" in n)
    interrupted = trainer_at(tmp_path / "resumed", [StopAfterTwo()])
    interrupted.train()
    assert interrupted.state.global_step == 2
    checkpoint = tmp_path / "resumed/checkpoint-2"
    assert (checkpoint / "optimizer.pt").exists()
    resumed = trainer_at(tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 4
    assert resumed.exposure == uninterrupted.exposure
    assert sum(resumed.exposure["examples"].values()) == 8
    if method == "structure_aware":
        assert set(resumed.exposure["examples"]) == {"D", "E", "R"}
    for (name, expected), (actual_name, actual) in zip(uninterrupted.model.named_parameters(), resumed.model.named_parameters()):
        assert name == actual_name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    adapter = tmp_path / "adapter"
    resumed.save_model(str(adapter))
    restored = reload_adapter(tiny_model(), adapter, PREPARATION)
    assert not any(p.requires_grad for p in restored.parameters())
    restored.eval()
    resumed.model.eval()
    batch = resumed.data_collator(rows[:1])
    inputs = {k: v for k, v in batch.items() if k not in {"labels", "tasks"}}
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs).logits, resumed.model(**inputs).logits, rtol=0, atol=0)
    assert not (adapter / "model.safetensors").exists()


def test_output_guard_rejects_overwrite_and_changed_resume_contract(tmp_path):
    output = tmp_path / "run"
    contract = {"config": {"max_steps": 4}, "data": "original"}
    prepare_output(output, contract, None)
    with pytest.raises(ValueError, match="not empty"):
        prepare_output(output, contract, None)
    checkpoint = output / "checkpoint-2"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="differs"):
        prepare_output(output, {**contract, "data": "changed"}, checkpoint)
    with pytest.raises(ValueError, match="Incomplete"):
        prepare_output(output, contract, checkpoint)
    with pytest.raises(ValueError, match="belong"):
        prepare_output(output, contract, tmp_path / "elsewhere/checkpoint-2")


def test_no_implicit_unbounded_training(tmp_path):
    cfg = config(max_steps=None)
    with pytest.raises(ValueError, match="finite max_steps"):
        build_trainer(attach_lora(tiny_model(), cfg, PREPARATION), tokenizer(), records(), cfg, tmp_path, cpu=True)
