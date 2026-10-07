from __future__ import annotations

from meta_data_extraction.base.validator import BaseValidator
from meta_data_extraction.data_models import ExtractionResult
from meta_data_extraction.utils import load_merged_schema, top_level_keys


class SimpleValidator(BaseValidator):
    def validate(self, result: ExtractionResult, schemas: list[str]) -> ExtractionResult:
        schema = load_merged_schema(schemas)
        allowed = top_level_keys(schema)
        warnings = list(result.warnings)
        cleaned = {}

        for key, value in result.output.items():
            if key in allowed:
                cleaned[key] = value
            else:
                warnings.append(f"Dropped unknown top-level key: {key!r}")

        result.output = cleaned
        result.warnings = warnings
        return result
