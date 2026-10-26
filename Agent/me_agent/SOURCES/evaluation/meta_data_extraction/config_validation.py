"""Strict YAML primitives shared by evaluation configuration readers."""
from typing import Any

import yaml


class UniqueKeyLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
    """Reject duplicate mapping keys instead of silently replacing values."""


def _mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError(f"Invalid YAML mapping key at {key_node.start_mark}") from exc
        if duplicate:
            raise ValueError(f"Duplicate YAML key {key!r} at {key_node.start_mark}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_yaml_mapping(text: str, *, source: str) -> dict[str, Any]:
    try:
        data = yaml.load(text, Loader=UniqueKeyLoader)
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"{source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected a mapping")
    return data


def check_keys(data: dict, allowed: set[str], location: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{location}: unsupported keys {sorted(unknown, key=str)}")
