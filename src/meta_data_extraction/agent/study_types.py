"""Frozen, runtime-only M09 synthetic repair-policy study declarations."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .evidence import FrozenModel
from .inspect_record import InputPin

Policy = Literal["fixed", "issue_aware", "generic_llm"]


class StudyPublication(FrozenModel):
    publication_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    split: Literal["calibration", "historical", "future_held_out"] = "calibration"
    data_kind: Literal["synthetic"] = "synthetic"
    identity_policy: Literal["one_record_per_parent_publication_no_derivative_rows"] = "one_record_per_parent_publication_no_derivative_rows"
    derived_from: Literal[None] = None
    workflows: dict[Policy, InputPin]


class StudySpec(FrozenModel):
    version: Literal["m09-study-spec-1"] = "m09-study-spec-1"
    scope: Literal["synthetic_repair_policy"] = "synthetic_repair_policy"
    authorized_mode: Literal["synthetic_offline"] = "synthetic_offline"
    description: str = Field(min_length=1, max_length=4000)
    primary_outcome: str = Field(min_length=1, max_length=2000)
    counting_policy: str = Field(min_length=1, max_length=4000)
    evaluation_protocol_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    publications: tuple[StudyPublication, ...] = Field(min_length=1, max_length=20)
    variants: tuple[Policy, ...] = Field(min_length=1, max_length=3)
    caps: tuple[int, ...] = Field(min_length=1, max_length=4)
    repeats: int = Field(default=1, ge=1, le=3, strict=True)
    max_total_model_calls: int = Field(ge=0, le=7680, strict=True)
    seed_policy: Literal["deterministic_mock"] = "deterministic_mock"
    provider_nondeterminism: Literal["not_represented"] = "not_represented"

    @field_validator("description", "primary_outcome", "counting_policy")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Study declarations must contain non-whitespace text")
        return value

    @field_validator("caps", mode="before")
    @classmethod
    def _caps(cls, value):
        if not isinstance(value, (list, tuple)) or any(type(cap) is not int or not 0 <= cap <= 32 for cap in value):
            raise ValueError("Each model-call cap must be an integer from zero through 32")
        return value

    @model_validator(mode="after")
    def _scope(self) -> StudySpec:
        if len(set(self.variants)) != len(self.variants) or len(set(self.caps)) != len(self.caps):
            raise ValueError("Variants and caps must be distinct")
        if len({publication.publication_id.casefold() for publication in self.publications}) != len(self.publications):
            raise ValueError("A publication must appear exactly once; derivatives are not independent publications")
        if any(set(publication.workflows) != set(self.variants) for publication in self.publications):
            raise ValueError("Each publication must declare exactly the common variant set")
        maximum = len(self.publications) * len(self.variants) * self.repeats * sum(self.caps)
        if maximum > self.max_total_model_calls:
            raise ValueError("The study budget must reserve every planned cell's full model-call cap")
        return self
