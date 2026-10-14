"""Frozen one-publication integration scope; no credential values or gold inputs."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from meta_data_extraction.config_validation import load_yaml_mapping
from .config import SchemaName
from .evidence import FrozenModel
from .inspect_record import InputPin
from .paths import input_file, validate_run_id
from .providers import LiveRunSpec
from .publication_standards import PaNETSettings, NeXusSettings
from .retrieval import RetrievalStrategy


class PublicationLiveSettings(FrozenModel):
    endpoint: str
    model: Literal["qwen3.8-27b"] = "qwen3.8-27b"
    api_key_env: str = "CHATAI_API_KEY"
    max_output_tokens: int = Field(default=4096, ge=1, le=32768, strict=True)
    max_prompt_bytes: int = Field(default=100000, ge=1, le=100000, strict=True)
    timeout_seconds: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _numeric_timeout(cls, value):
        if type(value) not in {int, float}:
            raise ValueError("Timeout must be an explicit numeric duration")
        return value

    @model_validator(mode="after")
    def _validate_live(self):
        # Reuse the existing endpoint/key validation without creating a provider.
        LiveRunSpec(endpoint=self.endpoint, model=self.model, api_key_env=self.api_key_env,
                    sources=({"path":"tests/agent/fixtures/placeholder.md", "sha256":"0"*64},),
                    split="scope_validation_only", profile_sha256="0"*64,
                    approved_prompt_sha256=("0"*64,), max_calls=1,
                    max_output_tokens=self.max_output_tokens,
                    max_total_output_tokens=self.max_output_tokens,
                    max_prompt_bytes=self.max_prompt_bytes, timeout_seconds=self.timeout_seconds)
        return self


class PublicationSpec(FrozenModel):
    version: Literal["publication-run-1"] = "publication-run-1"
    run_id: str
    mode: Literal["mock", "live"]
    source: InputPin
    split: Literal["synthetic", "development", "calibration"]
    schemas: tuple[SchemaName, ...] = ("generic.json", "photon_and_neutron.json")
    policy: Literal["fixed", "issue_aware"] = "issue_aware"
    max_calls: int = Field(default=5, ge=2, le=10, strict=True)
    max_targets: int = Field(default=12, ge=1, le=12, strict=True)
    evidence_chars: int = Field(default=24000, ge=1000, le=40000, strict=True)
    retrieval_strategy: RetrievalStrategy = "lexical_v1"
    allow_automatic_patches: bool = False
    generate_proposal: bool = True
    raw_retention: Literal["retain", "hash_only"] = "hash_only"
    unknown_usage: Literal["stop", "allow_bounded_calls"] = "allow_bounded_calls"
    workflow_timeout_seconds: float = Field(default=300, gt=0, le=300, allow_inf_nan=False)
    mock_responses: tuple[InputPin, ...] = Field(default=(), max_length=10)
    live: PublicationLiveSettings | None = None
    panet: PaNETSettings | None = None
    nexus: NeXusSettings | None = None

    @field_validator("workflow_timeout_seconds", mode="before")
    @classmethod
    def _numeric_timeout(cls, value):
        if type(value) not in {int, float}:
            raise ValueError("Timeout must be an explicit numeric duration")
        return value

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value):
        validate_run_id(value)
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", value) or value.startswith("_control"):
            raise ValueError("Use a lowercase run ID of at most 32 characters")
        return value

    @field_validator("allow_automatic_patches", "generate_proposal", mode="before")
    @classmethod
    def _bool(cls, value):
        if type(value) is not bool:
            raise ValueError("An explicit Boolean is required")
        return value

    @model_validator(mode="after")
    def _scope(self):
        if not self.schemas or len(set(self.schemas)) != len(self.schemas):
            raise ValueError("Select distinct canonical schemas")
        if self.mode == "mock":
            if self.live is not None or not self.mock_responses or self.split != "synthetic":
                raise ValueError("Mocks require explicit synthetic responses and no live settings")
        elif self.live is None or self.mock_responses:
            raise ValueError("Live execution requires settings and forbids scripted responses")
        if self.allow_automatic_patches and (not self.generate_proposal or self.max_calls < 5):
            raise ValueError("Automatic corrections need proposal generation and five call slots")
        return self


def load_publication_spec(root: Path | str, relative: str) -> PublicationSpec:
    path = input_file(root, relative, allowed_roots=("configs/agent", "runs/agent"))
    if path.stat().st_size > 1000000:
        raise ValueError("Publication specification exceeds the size bound")
    return PublicationSpec.model_validate(load_yaml_mapping(path.read_text(encoding="utf-8"), source="publication specification"))
