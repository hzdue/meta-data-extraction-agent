from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ParserConfig(BaseModel):
    type: str = "pdf"


class FewShotConfig(BaseModel):
    """Ordered exemplar IDs loaded from released JSON files.

    The configured list determines the number and order of examples.
    """

    example_ids: list[str] = Field(default_factory=list)
    examples_dir: str = "data/few_shot"


class ExtractorConfig(BaseModel):
    """Select the zero-shot or few-shot template recorded in prompt provenance."""

    shot_mode: Literal["zero_shot", "few_shot"] = "zero_shot"
    few_shot: FewShotConfig | None = None

    @model_validator(mode="after")
    def _check_few_shot(self) -> "ExtractorConfig":
        if self.shot_mode == "few_shot":
            if self.few_shot is None or not self.few_shot.example_ids:
                raise ValueError(
                    "shot_mode='few_shot' requires extractor.few_shot.example_ids "
                    "to list at least one exemplar arXiv ID"
                )
        elif self.few_shot is not None and self.few_shot.example_ids:
            raise ValueError(
                "extractor.few_shot.example_ids is set but shot_mode is "
                f"{self.shot_mode!r}; the exemplars would be silently ignored"
            )
        return self


class ValidatorConfig(BaseModel):
    pass


class LLMConfig(BaseModel):
    provider: str = "openai_compatible"
    model: str = "llama-3.3-70b-instruct"
    base_url: str | None = None       # None = use provider default (e.g. OpenAI)
    api_key_env: str = "LLM_API_KEY"  # name of the env var holding the API key
    temperature: float | None = 0.0  # None omits temperature from the request.
    max_tokens: int = 12288
    # Request field used for the completion token limit.
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    # None omits reasoning_effort; other values are forwarded unchanged.
    reasoning_effort: str | None = None
    max_doc_chars: int = 400_000


class TrackerConfig(BaseModel):
    output_dir: str = "data/processed"


class AppConfig(BaseModel):
    """Extraction settings, independent of the evaluation scoring plan."""

    domain_name: str = "generic"
    schemas: list[str] = Field(default_factory=lambda: ["generic.json"])
    parser: ParserConfig = Field(default_factory=ParserConfig)
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
