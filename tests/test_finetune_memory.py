"""Full-context memory controls preserve loss/gradients and resumability."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from meta_data_extraction.finetune.memory import chunk_text_mlps
from meta_data_extraction.finetune.training import StopAfterCheckpoint, attach_lora, build_trainer, reload_adapter
from tests.test_finetune_training import PREPARATION, config, records, tiny_model, tokenizer
from tests.test_finetune_validation import validation_rows


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_chunking_and_offload_preserve_loss_gradients_and_parameter_names(tmp_path):
    results = []
    for chunk_size, offload in ((0, False), (2, True)):
        cfg = config(mlp_chunk_size=chunk_size, offload_activations=offload, lora_dropout=0.0)
        model = attach_lora(tiny_model(), cfg, PREPARATION)
        trainer = build_trainer(model, tokenizer(), records(), cfg, tmp_path / str(chunk_size), cpu=True)
        model.train()
        loss = trainer.compute_loss(model, trainer.data_collator(records()[:2]))
        loss.backward()
        results.append((loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None},
                        list(model.state_dict())))
    torch.testing.assert_close(results[0][0], results[1][0])
    assert results[0][2] == results[1][2]
    assert results[0][1].keys() == results[1][1].keys()
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("method", ["json_only", "structure_aware"])
def test_chunked_dropout_resume_matches_uninterrupted_and_reload(tmp_path, method):
    cfg = config(method=method, max_steps=3, mlp_chunk_size=2, offload_activations=True)
    def make(path, callbacks=None):
        return build_trainer(attach_lora(tiny_model(), cfg, PREPARATION), tokenizer(), records(method), cfg,
                             path, cpu=True, validation=validation_rows(), callbacks=callbacks)
    full = make(tmp_path / "full")
    full.train()
    interrupted = make(tmp_path / "resumed", [StopAfterCheckpoint(2)])
    interrupted.train()
    resumed = make(tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=str(tmp_path / "resumed/checkpoint-2"))
    assert full.validation.history == resumed.validation.history
    assert full.exposure == resumed.exposure
    for (name, expected), (actual_name, actual) in zip(full.model.named_parameters(), resumed.model.named_parameters()):
        assert name == actual_name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    adapter = tmp_path / "adapter"
    resumed.save_model(str(adapter))
    restored = reload_adapter(tiny_model(), adapter, PREPARATION, config=cfg).eval()
    resumed.model.eval()
    batch = resumed.data_collator(records()[:1])
    inputs = {k: v for k, v in batch.items() if k not in {"labels", "tasks"}}
    for (name, expected), (actual_name, actual) in zip(resumed.model.named_parameters(), restored.named_parameters()):
        assert name == actual_name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.no_grad():
        expected = resumed.model(**inputs, use_cache=False).logits
        actual = restored(**inputs, use_cache=False).logits
        # Reload freezes adapter parameters; float32 execution may differ by a
        # few ulps, while the saved parameters and token predictions must agree.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))


def test_chunk_size_is_validated_and_configuration_is_idempotent():
    with pytest.raises(ValueError):
        config(mlp_chunk_size=-1)
    model = tiny_model()
    assert chunk_text_mlps(model, 2) == 2
    assert chunk_text_mlps(model, 2) == 2
    with pytest.raises(ValueError, match="Cannot change"):
        chunk_text_mlps(model, 3)


def test_quantization_contract_rejects_mismatched_training_and_reload(tmp_path):
    import json

    with pytest.raises(ValueError, match="quantization differs"):
        attach_lora(tiny_model(), config(load_in_4bit=True), PREPARATION)
    cfg = config()
    trainer = build_trainer(attach_lora(tiny_model(), cfg, PREPARATION), tokenizer(), records(),
                            cfg, tmp_path, cpu=True)
    trainer.save_model(str(tmp_path / "adapter"))
    metadata = tmp_path / "adapter/inference_precision.json"
    assert json.loads(metadata.read_text()) == cfg.precision_policy()
    metadata.write_text(json.dumps(config(load_in_4bit=True).precision_policy()))
    with pytest.raises(ValueError, match="inference precision mismatch"):
        reload_adapter(tiny_model(), tmp_path / "adapter", PREPARATION)
    metadata.unlink()
    with pytest.raises(ValueError, match="missing its inference precision"):
        reload_adapter(tiny_model(), tmp_path / "adapter", PREPARATION, config=cfg)
