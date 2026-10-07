from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from meta_data_extraction.app_config import AppConfig

SCHEMAS_DIR = Path(__file__).parents[2] / "schemas"
PROMPT_TEMPLATE = SCHEMAS_DIR / "prompt_template.md"  # zero-shot template
PROMPT_TEMPLATE_FEWSHOT = SCHEMAS_DIR / "prompt_template_fewshot.md"

FEW_SHOT_DIR = Path(__file__).parents[2] / "data" / "few_shot"
_PLACEHOLDER_TOKENS = ("{{SCHEMA_JSON}}", "{{OUTPUT_SKELETON}}", "{{DOCUMENT_TEXT}}", "{{EXAMPLES}}")


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> AppConfig:
    raw = load_yaml(path)
    return AppConfig.model_validate(raw)


def get_prompt_version(template_path: str | Path = PROMPT_TEMPLATE) -> str | None:
    """Read a template's front-matter version; default to the zero-shot template."""
    raw = Path(template_path).read_text(encoding="utf-8")
    if not raw.startswith("---"):
        return None
    _, front_matter, _ = raw.split("---", 2)
    parsed = yaml.safe_load(front_matter)
    return str(parsed["version"]) if isinstance(parsed, dict) and "version" in parsed else None


def get_schema_refs(schema_files: list[str]) -> dict[str, str | None]:
    """Return a mapping of schema filename to its schema_version from _meta."""
    result: dict[str, str | None] = {}
    for filename in schema_files:
        raw = json.loads((SCHEMAS_DIR / filename).read_text(encoding="utf-8"))
        result[filename] = raw.get("_meta", {}).get("schema_version")
    return result


def load_merged_schema(schema_files: list[str]) -> dict[str, Any]:
    """Load and merge one or more schema files in order, stripping _meta."""
    merged: dict[str, Any] = {}
    for filename in schema_files:
        raw = json.loads((SCHEMAS_DIR / filename).read_text(encoding="utf-8"))
        raw.pop("_meta", None)
        merged.update(raw)
    return merged


def top_level_keys(schema: dict[str, Any]) -> set[str]:
    """Return the set of output field keys."""
    return set(schema.keys())


def build_skeleton(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively build an output skeleton from a merged schema."""
    result: dict[str, Any] = {}
    for key, field_def in schema.items():
        if not isinstance(field_def, dict):
            continue
        field_type = field_def.get("type", "")
        if not field_type:
            # namespace container (e.g. bibliographic, document_overview)
            result[key] = build_skeleton(field_def)
        elif field_type.startswith("list["):
            result[key] = []
        elif field_type == "string":
            result[key] = "..."
        else:
            # string|null, boolean|null
            result[key] = None
    return result


def load_few_shot_examples(
    example_ids: list[str], examples_dir: str | Path = FEW_SHOT_DIR
) -> list[dict[str, Any]]:
    """Load frozen document/output exemplars in the configured ID order."""
    directory = Path(examples_dir)

    examples: list[dict[str, Any]] = []
    for arxiv_id in example_ids:
        path = directory / f"{arxiv_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Few-shot exemplar not found: {path}. "
                "Check extractor.few_shot.examples_dir and the released example files."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))

        examples.append(payload)
    return examples


def render_examples(examples: list[dict[str, Any]]) -> str:
    """Render ordered exemplars, preserving Unicode in expected JSON."""
    parts = []
    for i, example in enumerate(examples, 1):
        expected = json.dumps(example["expected_output"], indent=2, ensure_ascii=False)
        parts.append(
            f"### Example {i}\n\n"
            f"#### Document\n\n{example['document_text']}\n\n"
            f"#### Output\n\n{expected}"
        )
    return "\n\n".join(parts)


def _render(
    template_path: Path,
    schemas: list[str],
    document_text: str,
    examples_block: str | None = None,
) -> str:
    """Insert document text last so its placeholder-like strings remain literal."""
    template = template_path.read_text(encoding="utf-8")

    # strip YAML front matter
    if template.startswith("---"):
        _, _, template = template.split("---", 2)

    schema = load_merged_schema(schemas)
    skeleton = build_skeleton(schema)

    rendered = (
        template.strip()
        .replace("{{SCHEMA_JSON}}", json.dumps(schema, indent=2))
        .replace("{{OUTPUT_SKELETON}}", json.dumps(skeleton, indent=2))
    )
    if examples_block is not None:
        rendered = rendered.replace("{{EXAMPLES}}", examples_block)
    return rendered.replace("{{DOCUMENT_TEXT}}", document_text)


def build_prompt(schemas: list[str], document_text: str) -> str:
    """Render the zero-shot template with schema, output skeleton, and document text."""
    return _render(PROMPT_TEMPLATE, schemas, document_text)


def build_few_shot_prompt(
    schemas: list[str], document_text: str, examples: list[dict[str, Any]]
) -> str:
    """Render the few-shot prompt template with in-context exemplars."""
    if not examples:
        raise ValueError("build_few_shot_prompt requires at least one exemplar")
    for example in examples:
        blob = example["document_text"]
        for token in _PLACEHOLDER_TOKENS:
            if token in blob:
                raise ValueError(
                    f"Exemplar {example.get('arxiv_id')} contains the literal template "
                    f"token {token!r}; it would corrupt prompt rendering"
                )
    return _render(
        PROMPT_TEMPLATE_FEWSHOT, schemas, document_text, examples_block=render_examples(examples)
    )
