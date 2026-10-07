from __future__ import annotations

from abc import ABC, abstractmethod

from meta_data_extraction.data_models import Document, ExtractionResult


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, document: Document) -> ExtractionResult:
        raise NotImplementedError