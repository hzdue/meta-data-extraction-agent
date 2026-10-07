"""Check model-specific request parameters without making API calls."""
from __future__ import annotations

import pytest

from meta_data_extraction.app_config import LLMConfig
from meta_data_extraction.components.llm import OpenAICompatibleLLM


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")


def _llm(**kwargs) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(model_name="m", api_key_env="LLM_API_KEY", **kwargs)


def test_classic_request_shape_is_unchanged():
    """Classic requests include temperature and use max_tokens."""
    req = _llm(temperature=0.0, max_tokens=12288)._build_request("PROMPT")
    assert req == {
        "model": "m",
        "messages": [{"role": "user", "content": "PROMPT"}],
        "max_tokens": 12288,
        "temperature": 0.0,
    }


def test_default_config_yields_classic_shape():
    """Default configuration selects the classic request parameters."""
    cfg = LLMConfig()
    assert cfg.temperature == 0.0
    assert cfg.max_tokens_param == "max_tokens"
    assert cfg.reasoning_effort is None


def test_temperature_none_is_omitted():
    req = _llm(temperature=None)._build_request("P")
    assert "temperature" not in req


def test_max_completion_tokens_replaces_max_tokens():
    req = _llm(max_tokens=12288, max_tokens_param="max_completion_tokens")._build_request("P")
    assert req["max_completion_tokens"] == 12288
    assert "max_tokens" not in req


def test_reasoning_effort_included_when_set():
    req = _llm(reasoning_effort="none")._build_request("P")
    assert req["reasoning_effort"] == "none"


def test_gpt5_style_request_combines_all_three():
    """The intended gpt-5.5 shape: no temperature, max_completion_tokens,
    reasoning_effort=none."""
    req = _llm(
        temperature=None,
        max_tokens=12288,
        max_tokens_param="max_completion_tokens",
        reasoning_effort="none",
    )._build_request("P")
    assert req == {
        "model": "m",
        "messages": [{"role": "user", "content": "P"}],
        "max_completion_tokens": 12288,
        "reasoning_effort": "none",
    }


def test_config_accepts_gpt5_settings():
    cfg = LLMConfig(
        temperature=None,
        max_tokens_param="max_completion_tokens",
        reasoning_effort="none",
    )
    assert cfg.temperature is None
    assert cfg.max_tokens_param == "max_completion_tokens"
    assert cfg.reasoning_effort == "none"


def test_config_rejects_unknown_max_tokens_param():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LLMConfig(max_tokens_param="max_output_tokens")
