from __future__ import annotations

import re
from pathlib import Path

from meta_data_extraction.base.tracker import BaseTracker
from meta_data_extraction.data_models import ExtractionResult


class LocalTracker(BaseTracker):
    """Write each extraction to <output_dir>/<config_name>/<document_id>.json."""

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_config(config_name: str) -> str:
        return re.sub(r"[^\w\-]", "_", config_name)

    def result_path(self, document_id: str, config_name: str) -> Path:
        """Where log_run() would write for this (document, config)."""
        return self.output_dir / self._safe_config(config_name) / f"{document_id}.json"

    def config_dir(self, config_name: str) -> Path:
        return self.output_dir / self._safe_config(config_name)

    def log_run(self, result: ExtractionResult) -> None:
        config = (result.provenance.config_name or "unknown") if result.provenance else "unknown"
        result_path = self.result_path(result.document_id, config)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
