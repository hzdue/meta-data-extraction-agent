from __future__ import annotations

from abc import ABC, abstractmethod

from meta_data_extraction.data_models import ExtractionResult


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, result: ExtractionResult, schemas: list[str]) -> ExtractionResult:
        raise NotImplementedError