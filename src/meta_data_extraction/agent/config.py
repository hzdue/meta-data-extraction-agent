"""Strict offline M00 configuration, separate from inherited live AppConfig."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from meta_data_extraction.config_validation import load_yaml_mapping
from meta_data_extraction.agent.paths import input_file, validate_run_id

SchemaName = Literal[
    "generic.json", "photon_and_neutron.json",
    "generic_no_instructions.json", "photon_and_neutron_no_instructions.json",
]
SOURCE_ROOTS = ("data/raw", "tests/agent/fixtures")


class OfflineRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["offline"] = "offline"
    run_id: str
    sources: tuple[str, ...] = Field(min_length=1)
    schemas: tuple[SchemaName, ...] = ("generic.json", "photon_and_neutron.json")

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        return validate_run_id(value)

    @field_validator("sources", "schemas")
    @classmethod
    def _unique_nonempty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or len(set(values)) != len(values):
            raise ValueError("Expected a nonempty list of distinct inputs")
        return values

    def source_files(self, root: Path | str) -> tuple[Path, ...]:
        paths = tuple(input_file(root, source, allowed_roots=SOURCE_ROOTS) for source in self.sources)
        if len(set(paths)) != len(paths):
            raise ValueError("Source paths resolve to the same file")
        return paths

    def schema_files(self, root: Path | str) -> tuple[Path, ...]:
        return tuple(
            input_file(root, f"schemas/{name}", allowed_roots=("schemas",))
            for name in self.schemas
        )


def load_offline_config(relative: str, root: Path | str) -> OfflineRunConfig:
    """Load only agent configs; all input paths are relative to the checkout root."""
    path = input_file(root, relative, allowed_roots=("configs/agent",))
    config = OfflineRunConfig.model_validate(
        load_yaml_mapping(path.read_text(encoding="utf-8"), source=relative)
    )
    config.source_files(root)
    config.schema_files(root)
    return config
