from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"


def get_schema_refs(schema_files: list[str]) -> dict[str, str | None]:
    """Return schema filename -> declared _meta.schema_version."""
    result: dict[str, str | None] = {}
    for filename in schema_files:
        raw = json.loads((SCHEMAS_DIR / filename).read_text(encoding="utf-8"))
        result[filename] = raw.get("_meta", {}).get("schema_version")
    return result


def load_merged_schema(schema_files: list[str]) -> dict[str, Any]:
    """Load and merge schemas in order, stripping _meta."""
    merged: dict[str, Any] = {}
    for filename in schema_files:
        raw = json.loads((SCHEMAS_DIR / filename).read_text(encoding="utf-8"))
        raw.pop("_meta", None)
        merged.update(raw)
    return merged
