from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from meta_data_extraction.data_models import Document


class BaseParser(ABC):
    """Document parser interface."""

    @abstractmethod
    def parse(self, path: Path) -> Document:
        """Parse one file into a Document.

        Populate id, text, source_format, source_path, and parser_name.
        """
        raise NotImplementedError
