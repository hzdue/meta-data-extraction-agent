from __future__ import annotations

import os

from openai import OpenAI

from meta_data_extraction.base.llm import BaseLLM, LLMResponse


class OpenAICompatibleLLM(BaseLLM):
    """Chat-completions client with credentials read from api_key_env."""

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "LLM_API_KEY",
        base_url: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int = 4096,
        max_tokens_param: str = "max_tokens",
        reasoning_effort: str | None = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_param = max_tokens_param
        self.reasoning_effort = reasoning_effort

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"API key not found. Set the {api_key_env!r} environment variable."
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _build_request(self, prompt: str) -> dict:
        request: dict = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            self.max_tokens_param: self.max_tokens,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        return request

    def generate(self, prompt: str) -> LLMResponse:
        response = self._client.chat.completions.create(**self._build_request(prompt))
        return LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason,
        )
