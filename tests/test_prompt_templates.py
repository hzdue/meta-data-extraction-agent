"""Keep shared instructions identical between zero-shot and few-shot templates."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"
ZERO_SHOT = SCHEMAS_DIR / "prompt_template.md"
FEW_SHOT = SCHEMAS_DIR / "prompt_template_fewshot.md"

# The block the few-shot template adds, and nothing else. Spans from the
# "## Examples" heading up to (not including) the "## Document" heading.
EXAMPLES_BLOCK_RE = re.compile(r"## Examples\n.*?\n\n(?=## Document)", re.DOTALL)


def _split_front_matter(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    assert raw.startswith("---"), f"{path.name} has no YAML front matter"
    _, front_matter, body = raw.split("---", 2)
    return yaml.safe_load(front_matter), body


def test_few_shot_body_is_zero_shot_body_plus_examples_block():
    """Only the examples block may differ between the two prompt bodies."""
    _, zero_body = _split_front_matter(ZERO_SHOT)
    _, few_body = _split_front_matter(FEW_SHOT)

    stripped, n = EXAMPLES_BLOCK_RE.subn("", few_body)
    assert n == 1, f"expected exactly one Examples block, found {n}"
    assert stripped == zero_body, (
        "few-shot template body differs from zero-shot body by more than the "
        "Examples block"
    )


def test_examples_slot_precedes_document_slot():
    """Worked examples precede the target document."""
    _, few_body = _split_front_matter(FEW_SHOT)
    assert few_body.index("{{EXAMPLES}}") < few_body.index("{{DOCUMENT_TEXT}}")


def test_templates_carry_distinct_versions():
    """Recorded prompt versions distinguish zero-shot and few-shot runs."""
    zero_fm, _ = _split_front_matter(ZERO_SHOT)
    few_fm, _ = _split_front_matter(FEW_SHOT)
    assert str(zero_fm["version"]) == "1.1.0"
    assert str(few_fm["version"]) == "1.2.0"
    assert zero_fm["name"] != few_fm["name"]


def test_zero_shot_template_placeholders_unchanged():
    """The zero-shot template has exactly three placeholders."""
    _, zero_body = _split_front_matter(ZERO_SHOT)
    assert set(re.findall(r"\{\{(\w+)\}\}", zero_body)) == {
        "SCHEMA_JSON",
        "OUTPUT_SKELETON",
        "DOCUMENT_TEXT",
    }


def test_few_shot_template_adds_only_the_examples_placeholder():
    _, few_body = _split_front_matter(FEW_SHOT)
    assert set(re.findall(r"\{\{(\w+)\}\}", few_body)) == {
        "SCHEMA_JSON",
        "OUTPUT_SKELETON",
        "DOCUMENT_TEXT",
        "EXAMPLES",
    }
