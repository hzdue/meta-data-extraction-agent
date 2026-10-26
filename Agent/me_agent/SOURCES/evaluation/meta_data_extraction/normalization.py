"""Text canonicalization, tokenization, and scoring normalizers.

These presentation rules are shared with alignment where needed. Reviewed
aliases and identity-specific transformations live in alignment.py.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable


# Shared text canonicalization.

DASH_MAP = str.maketrans(
    {
        "−": "-",
        "–": "-",
        "—": "-",
        "‐": "-",
        "‑": "-",
    }
)

_MISC_MAP = str.maketrans(
    {
        "µ": "μ",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "\u00ad": "",
        "∼": "~",
        "◦": "°",
        "˚": "°",
        "\u00a0": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2009": " ",
    }
)


def text_canon(value: str) -> str:
    """Normalize Unicode, punctuation, and whitespace in one string."""
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(DASH_MAP).translate(_MISC_MAP)
    return " ".join(value.split())


def canonicalize_strings(obj: Any) -> Any:
    """Recursively apply :func:`text_canon` without mutating ``obj``."""
    if isinstance(obj, str):
        return text_canon(obj)
    if isinstance(obj, dict):
        return {key: canonicalize_strings(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [canonicalize_strings(value) for value in obj]
    return obj


# Scoring-plan value normalizers.


def casefold_norm(value: str | None) -> str | None:
    return value.casefold() if value is not None else None


_ORPHANED_AUTHOR_ACUTE_RE = re.compile(r"(?<=[^\W\d_])\s+\u0301(?P<letter>[^\W\d_])")


def normalize_author_name_typography(value: str | None) -> str | None:
    """Repair a spacing acute accent displaced before an author-name letter.

    PDF-derived text can represent ``André`` as ``Andr´e``. NFKC decomposes
    that spacing accent to ``" \N{COMBINING ACUTE ACCENT}"`` before the
    affected letter. Repair only that observed typographic artifact; do not
    strip accents or equate genuinely unaccented and accented names.
    """
    if value is None:
        return None
    text = text_canon(value)

    def attach_acute(match: re.Match[str]) -> str:
        letter = match.group("letter")
        # TeX/PDF extraction commonly emits dotless i before an acute accent.
        if letter == "ı":
            letter = "i"
        return unicodedata.normalize("NFC", f"{letter}\u0301")

    return unicodedata.normalize(
        "NFC", _ORPHANED_AUTHOR_ACUTE_RE.sub(attach_acute, text)
    )


_GRANT_PREFIX_RE = re.compile(
    r"^.*?(?:grant\s+no\.?|award\s+no\.?|contract\s+no\.?"
    r"|discovery\s+grant|advanced\s+grants\s+no\.?|under\s+grant)\s+",
    re.IGNORECASE,
)


def strip_id_prefix(value: str | None) -> str | None:
    if value is None:
        return None
    return _GRANT_PREFIX_RE.sub("", value)


_SAMPLE_HYDRATE_DOTS = str.maketrans({"∙": "·", "⋅": "·", "•": "·"})
_SAMPLE_SPACED_DECIMAL_RE = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")
_SAMPLE_PAREN_VARIABLE_RE = re.compile(
    r"\s*\(\s*([A-Za-z])\s*=\s*([-+]?\d+(?:\.\d+)?)\s*\)"
)
_SAMPLE_VARIABLE_SPACING_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z])\s*=\s*([-+]?\d+(?:\.\d+)?)"
)


def normalize_sample_formula_formatting(value: str | None) -> str | None:
    """Fold presentation-only formula differences for alignment and scoring.

    This deliberately does not expand aliases, case-fold chemical symbols, or
    infer chemical equivalence. Parentheses are removed only when their entire
    content is a simple variable assignment, so chemical groups such as
    ``Fe(OH)3`` remain intact.
    """
    if value is None:
        return None
    text = text_canon(value).translate(_SAMPLE_HYDRATE_DOTS)
    text = _SAMPLE_SPACED_DECIMAL_RE.sub(".", text)
    text = _SAMPLE_PAREN_VARIABLE_RE.sub(r" \1=\2", text)
    text = _SAMPLE_VARIABLE_SPACING_RE.sub(r"\1=\2", text)
    return " ".join(text.split())


# Tokenization shared by scoring and alignment.

_TOKEN_RE = re.compile(r"[\w°Å]+(?:[/.-][\w°Å]+)*")


def tokenize(value: str) -> list[str]:
    """Pinned tokenizer shared by Rouge-L and entity alignment metrics."""
    return _TOKEN_RE.findall(value.casefold())


# Names accepted by the scoring plan's normalizer and key_normalizer settings.
NORMALIZERS: dict[str, Callable[[Any], Any]] = {
    "casefold": casefold_norm,
    "strip_id_prefix": strip_id_prefix,
    "author_name_typography": normalize_author_name_typography,
    "sample_formula_formatting": normalize_sample_formula_formatting,
}
