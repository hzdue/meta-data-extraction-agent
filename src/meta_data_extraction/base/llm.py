from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Shared response type for a single LLM completion."""

    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class BaseLLM(ABC):
    """LLM backend interface.

    Subclasses set model_name and return LLMResponse from generate.
    """

    model_name: str

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError
