"""Strict output-schema checks shared by validation and held-out evaluation."""
from __future__ import annotations

import re
from typing import Any

from meta_data_extraction.utils import load_merged_schema


DEFAULT_SCHEMAS = ["generic.json", "photon_and_neutron.json"]


def _matches_type(value: Any, name: str) -> bool:
    return {
        "string": isinstance(value, str),
        "boolean": type(value) is bool,
        "integer": type(value) is int,
        "number": type(value) in {int, float} and not isinstance(value, bool),
    }.get(name, True)


def schema_errors(document: dict, schemas: list[str] | None = None) -> list[str]:
    """Validate field structure, types, local IDs and reference endpoints."""
    errors: list[str] = []

    def visit(value: Any, definition: dict, path: str):
        field_type = definition.get("type")
        if not field_type:
            if not isinstance(value, dict):
                errors.append(f"{path}: expected object")
                return
            expected = set(definition)
            for key in sorted(expected - set(value)):
                errors.append(f"{path}.{key}: missing field")
            for key in sorted(set(value) - expected):
                errors.append(f"{path}.{key}: unknown field")
            for key in sorted(expected & set(value)):
                visit(value[key], definition[key], f"{path}.{key}")
            return
        nullable = field_type.endswith("|null")
        base_type = field_type.removesuffix("|null")
        if value is None:
            if not nullable:
                errors.append(f"{path}: null is not allowed")
            return
        if base_type.startswith("list["):
            if not isinstance(value, list):
                errors.append(f"{path}: expected list")
                return
            cardinality = definition.get("cardinality", {})
            if "min" in cardinality and len(value) < cardinality["min"]:
                errors.append(f"{path}: fewer than {cardinality['min']} items")
            if "max" in cardinality and len(value) > cardinality["max"]:
                errors.append(f"{path}: more than {cardinality['max']} items")
            item_type = base_type[5:-1]
            item_schema = definition.get("item_schema")
            for index, item in enumerate(value):
                item_path = f"{path}[{index}]"
                if item_schema is not None:
                    visit(item, item_schema, item_path)
                elif not _matches_type(item, item_type):
                    errors.append(f"{item_path}: expected {item_type}")
            return
        if not _matches_type(value, base_type):
            errors.append(f"{path}: expected {base_type}")
            return
        pattern = definition.get("constraints", {}).get("pattern")
        if pattern and isinstance(value, str) and re.fullmatch(pattern, value) is None:
            errors.append(f"{path}: does not match declared pattern")

    visit(document, load_merged_schema(schemas or DEFAULT_SCHEMAS), "$out")
    ids: dict[str, set[str]] = {}
    for collection, prefix in (("bibliographic.affiliations", "aff"),
                               ("bibliographic.authors", "author"),
                               ("facilities", "facility"), ("samples", "sample"),
                               ("datasets", "dataset")):
        current: Any = document
        for part in collection.split("."):
            current = current.get(part) if isinstance(current, dict) else None
        if not isinstance(current, list):
            continue
        values = [item.get("id") for item in current if isinstance(item, dict)]
        valid = {value for value in values if isinstance(value, str)}
        ids[collection] = valid
        if len(valid) != len(values):
            errors.append(f"$out.{collection}: IDs must be unique strings")
        for value in valid:
            if re.fullmatch(rf"{prefix}_[1-9][0-9]*", value) is None:
                errors.append(f"$out.{collection}: invalid local ID {value!r}")

    def check_refs(items: Any, field: str, targets: set[str], path: str):
        if not isinstance(items, list):
            return
        for index, item in enumerate(items):
            refs = item.get(field) if isinstance(item, dict) else None
            if isinstance(refs, list):
                dangling = [ref for ref in refs if ref not in targets]
                if dangling:
                    errors.append(f"{path}[{index}].{field}: dangling references {dangling!r}")

    bibliographic = document.get("bibliographic") if isinstance(document, dict) else None
    check_refs((bibliographic or {}).get("authors"), "affiliation_refs",
               ids.get("bibliographic.affiliations", set()), "$out.bibliographic.authors")
    check_refs(document.get("datasets"), "sample_refs", ids.get("samples", set()), "$out.datasets")
    check_refs(document.get("datasets"), "facility_refs", ids.get("facilities", set()), "$out.datasets")
    return errors

