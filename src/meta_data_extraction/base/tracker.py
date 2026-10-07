from __future__ import annotations

from abc import ABC, abstractmethod

from meta_data_extraction.data_models import ExtractionResult


class BaseTracker(ABC):
    @abstractmethod
    def log_run(self, result: ExtractionResult) -> None:
        raise NotImplementedError