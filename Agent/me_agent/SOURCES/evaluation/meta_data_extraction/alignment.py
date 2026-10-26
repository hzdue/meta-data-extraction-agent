"""Entity alignment: normalization, identity preparation, then matching.

Entry points: align_similarity, align_funding, align_samples, align_datasets, and
align_facilities. Reviewed forms come from alignment_helper.yaml; attribute scoring
lives in evaluation.py.
"""


from __future__ import annotations

import collections
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from meta_data_extraction.config_validation import load_yaml_mapping
from meta_data_extraction.normalization import (
    DASH_MAP,
    normalize_sample_formula_formatting,
    text_canon,
    tokenize,
)


# Alignment result types.


@dataclass(frozen=True)
class MatchedPair:
    """One accepted prediction-to-gold entity link and its matching stage."""

    pred_index: int
    gold_index: int
    stage: str


@dataclass
class AlignmentResult:
    """Completed correspondence: matched pairs, reference IDs, and diagnostics."""

    pairs: list[MatchedPair]
    id_map: dict[str, str | None] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


# Authors and affiliations.


def make_key_scorer(
    *attrs: str,
    normalizer: Callable[[Any], Any] | None = None,
) -> Callable[[dict, dict], float]:
    """Token F1 of named attributes, optionally normalized for alignment only."""

    def entity_text(entity: dict) -> str:
        text = " ".join(str(entity.get(attr) or "") for attr in attrs)
        if normalizer is not None:
            text = normalizer(text) or ""
        return text

    def key_fn(p: dict, g: dict) -> float:
        return token_overlap_f1(entity_text(p), entity_text(g))

    return key_fn


def align_similarity(
    pred: list[dict] | None,
    gold: list[dict] | None,
    *,
    key_fn: Callable[[dict, dict], float],
    tau: float = 0.5,
) -> AlignmentResult:
    """Match authors or affiliations by maximum-cardinality token similarity."""
    pred, gold = pred or [], gold or []
    sims = [
        (key_fn(p, g), i, j)
        for i, p in enumerate(pred)
        for j, g in enumerate(gold)
    ]
    assigned = _hungarian_assign(sims, tau, len(pred), len(gold))
    pairs = [
        MatchedPair(i, j, stage="similarity")
        for i, j, _similarity in assigned
    ]
    return _alignment_result(pred, gold, pairs)


# Funding.


def normalize_funder(value: str | None) -> str | None:
    """Expand funder aliases and normalize funder wording for alignment."""
    if value is None:
        return None
    aliases = _aliases("bibliographic.funding")
    text = normalize_surface(_expand_aliases(value, aliases))
    return _collapse_repeated_aliases(text, aliases)


_GRANT_LEADING_LABEL_RE = re.compile(
    r"^(?:(?:grant(?:\s+agreement)?|award|contract|project)\s*)?"
    r"(?:no\.?|number|n|nr\.?)\s+",
    re.IGNORECASE,
)

_GRANT_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def grant_identity(value: str | None) -> str:
    """Conservative separator-insensitive identity for a complete grant ID."""
    if not isinstance(value, str) or not value.strip():
        return ""
    normalized = _GRANT_LEADING_LABEL_RE.sub("", text_canon(value).casefold())
    return _GRANT_NON_ALNUM_RE.sub("", normalized)


_GRANT_COMPOUND_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[-/.][A-Za-z0-9]+)+(?![A-Za-z0-9])"
)

_GRANT_DIGIT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z0-9]*\d[A-Za-z0-9]*(?![A-Za-z0-9])"
)

_GRANT_PREFIXED_NUMBER_RE = re.compile(r"\b([A-Za-z]{2,8})\s+(\d{4,})\b")

_GRANT_ANCHOR_PREFIX_STOPWORDS = {"award", "contract", "grant", "no", "number", "project"}


def grant_anchors(value: str | None) -> frozenset[str]:
    """Extract exact, separator-neutral identifier anchors without inference."""
    if not isinstance(value, str) or not value.strip():
        return frozenset()
    text = text_canon(value).casefold()
    anchors: set[str] = set()
    compound_spans: list[tuple[int, int]] = []
    for match in _GRANT_COMPOUND_ID_RE.finditer(text):
        anchor = _GRANT_NON_ALNUM_RE.sub("", match.group(0))
        if len(anchor) >= 5 and any(character.isdigit() for character in anchor):
            anchors.add(anchor)
            compound_spans.append(match.span())
    for match in _GRANT_DIGIT_TOKEN_RE.finditer(text):
        # A compound ID's numeric fragment is not a separate grant anchor.
        if any(start <= match.start() and match.end() <= end for start, end in compound_spans):
            continue
        anchor = _GRANT_NON_ALNUM_RE.sub("", match.group(0))
        if len(anchor) >= 5:
            anchors.add(anchor)
    for prefix, number in _GRANT_PREFIXED_NUMBER_RE.findall(text):
        if prefix not in _GRANT_ANCHOR_PREFIX_STOPWORDS:
            anchors.add(prefix + number)
    return frozenset(anchors)


def complete_funding_identity(entity: dict) -> bool:
    """Whether both schema-required funding alignment keys are present."""
    return all(
        isinstance(entity.get(field), str) and entity[field].strip()
        for field in ("grant_id", "funder")
    )


def _prepare_funding_record(entity: dict, tie_breaker: str) -> dict:
    """Normalize funding evidence once and retain its deterministic ordering key."""
    grant = entity.get("grant_id")
    identity = grant_identity(grant)
    surface = text_canon(grant or "").casefold()
    funder = normalize_funder(entity.get("funder")) or ""
    return {
        "grant": identity,
        "surface": surface,
        "anchors": grant_anchors(grant),
        "funder": funder,
        "key": (identity, funder, surface, tie_breaker),
    }


def _prepare_funding_identities(gold_funding: list[dict]) -> list[dict]:
    """Prepare reconciled gold evidence, using its original index to break ties."""
    return [
        _prepare_funding_record(entity, str(index))
        for index, entity in enumerate(gold_funding, start=1)
    ]


def _match_funding(
    predictions: list[dict],
    entries: list[dict],
    *,
    diagnostics: dict[str, int] | None = None,
) -> dict[int, tuple[int, str]]:
    """Match complete funding records by grant, funder, then grant anchors."""
    prepared = {
        i: _prepare_funding_record(entity, text_canon(entity["funder"]).casefold())
        for i, entity in enumerate(predictions)
        if complete_funding_identity(entity)
    }
    pred_groups: dict[str, list[int]] = collections.defaultdict(list)
    gold_groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, record in prepared.items():
        pred_groups[record["grant"]].append(i)
    for j, entry in enumerate(entries):
        gold_groups[entry["grant"]].append(j)
    matches: dict[int, tuple[int, str]] = {}
    used_p: set[int] = set()
    used_g: set[int] = set()
    for grant in sorted(set(pred_groups) & set(gold_groups)):
        if not grant:
            continue
        ps = sorted(pred_groups[grant], key=lambda index: prepared[index]["key"])
        gs = sorted(gold_groups[grant], key=lambda index: entries[index]["key"])
        # Consume the complete shared-grant group even when it abstains. A
        # weaker grant-anchor fallback must not re-admit a zero-funder edge.
        used_p.update(ps)
        used_g.update(gs)
        if len(ps) == len(gs) == 1:
            matches[ps[0]] = (gs[0], "grant")
            continue
        sims = []
        for li, i in enumerate(ps):
            for lj, j in enumerate(gs):
                funder = token_overlap_f1(prepared[i]["funder"], entries[j]["funder"])
                # Competing records with the same grant require positive funder
                # evidence. Exact grant wording only breaks ties; adding 1 places
                # admissible edges above the assignment threshold.
                score = 0.0
                if funder > 0.0:
                    exact_surface = prepared[i]["surface"] == entries[j]["surface"]
                    score = 1.0 + funder + exact_surface * 1e-9
                sims.append((score, li, lj))
        for li, lj, _score in _hungarian_assign(sims, 1.0, len(ps), len(gs)):
            matches[ps[li]] = (gs[lj], "grant_then_funder")
    ps = sorted(
        (i for i in prepared if i not in used_p),
        key=lambda index: prepared[index]["key"],
    )
    gs = sorted(
        (j for j in range(len(entries)) if j not in used_g),
        key=lambda index: entries[index]["key"],
    )
    sims = []
    for li, i in enumerate(ps):
        p_anchors = prepared[i]["anchors"]
        # A fallback string naming several registry grants cannot select one
        # of them. Check the complete registry, including already assigned
        # grants, so assignment order cannot make a merged record admissible.
        # Co-funders sharing one normalized grant ID remain one grant here.
        anchored_grants = {
            entry["grant"]
            for entry in entries
            if p_anchors & entry["anchors"]
        }
        if len(anchored_grants) > 1:
            if diagnostics is not None:
                diagnostics["ambiguous_grant_anchors"] = (
                    diagnostics.get("ambiguous_grant_anchors", 0) + 1
                )
            continue
        for lj, j in enumerate(gs):
            g_anchors = entries[j]["anchors"]
            shared = p_anchors & g_anchors
            score = 0.0
            if shared:
                specificity = min(max(map(len, shared)), 50) / 50
                coverage = len(shared) / len(p_anchors | g_anchors)
                funder = token_overlap_f1(prepared[i]["funder"], entries[j]["funder"])
                score = 1.0 + specificity * 1e-3 + coverage * 1e-6 + funder * 1e-9
            sims.append((score, li, lj))
    for li, lj, _score in _hungarian_assign(sims, 1.0, len(ps), len(gs)):
        matches[ps[li]] = (gs[lj], "grant_anchor")
    return matches


def align_funding(
    pred: list[dict] | None,
    gold: list[dict] | None,
    *,
    identity_gold: list[dict] | None = None,
) -> AlignmentResult:
    """Match grants first, disambiguate with funders, then use grant anchors.

    For IAA, supply reconciled identity_gold; both annotations link to it
    independently before their corresponding records are joined.
    """
    pred, gold = pred or [], gold or []
    entries = _prepare_funding_identities(gold if identity_gold is None else identity_gold)
    gold_diagnostics: dict[str, int] = {}
    gold_links = _match_funding(gold, entries, diagnostics=gold_diagnostics)
    pred_diagnostics: dict[str, int] = {}
    pred_links = _match_funding(pred, entries, diagnostics=pred_diagnostics)
    return _alignment_result(
        pred, gold, _join_identity_matches(pred_links, gold_links),
        label_fields=("grant_id", "funder"),
        abstentions={
            **{f"pred_{key}": value for key, value in pred_diagnostics.items()},
            **{f"gold_{key}": value for key, value in gold_diagnostics.items()},
        },
    )


# Samples.


def normalize_sample(value: str | None) -> str | None:
    """Normalize sample names and compositions using reviewed aliases and typography.

    Preserve meaningful formula punctuation without inferring chemical equivalence.
    """
    if value is None:
        return None
    expanded = _expand_aliases(value, _aliases("samples"))
    text = normalize_sample_formula_formatting(expanded)
    text = text.casefold()
    # Formula typography: AlO_x -> AlOx. Do not otherwise rewrite symbols.
    text = re.sub(r"(?<=[a-z0-9])_(?=[a-z0-9])", "", text)
    # Isotope typography: [11]B and [11] B -> 11B.
    text = re.sub(r"\[\s*(\d+)\s*\]\s*(?=[a-z])", r"\1", text)
    # A full stop before a letter is a common ASCII rendering of a hydrate dot;
    # decimal points remain untouched.
    text = re.sub(r"(?<=[a-z0-9])\.(?=[a-z])", "Â·", text)
    text = re.sub(r"\s*%\s*", "%", text)
    text = re.sub(r"\s*([/:+=Â·-])\s*", r"\1", text)
    return " ".join(text.split())


_SAMPLE_IDENTITY_TOKEN_RE = re.compile(r"[\wÎ±-Ï‰]+", re.IGNORECASE)


def _sample_identity_tokens(value: str | None) -> set[str]:
    """Return normalized tokens used by sample identity discriminators."""
    return set(_SAMPLE_IDENTITY_TOKEN_RE.findall(value or ""))


_SAMPLE_VARIABLE_RE = re.compile(
    r"(?<![a-z0-9])([a-z])\s*=\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"
    r"(?![a-z0-9_]|\.\d)",
    re.IGNORECASE,
)

_SAMPLE_COUNT_RE = re.compile(
    r"(?<![a-z0-9])n\s*=\s*(\d+)(?![a-z0-9_]|\.\d)", re.IGNORECASE
)

_SAMPLE_UNIT_RE = re.compile(
    r"(?<![a-z0-9_.+-])([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*"
    r"(Â°c|Ã¥|nm|Î¼m|um|at\.?\s*%|atom\s*%)(?![a-z0-9_])",
    re.IGNORECASE,
)


def _sample_variant_identity(*values: str | None) -> dict[str, object]:
    """Extract explicitly printed numeric evidence used to split samples."""
    variables: dict[str, set[str]] = defaultdict(set)
    numeric = {field: set() for field in (
        "counts", "temperatures", "thicknesses", "concentrations"
    )}
    for value in values:
        if not value:
            continue
        normalized = value.casefold().replace("Âµ", "Î¼")
        for name, number in _SAMPLE_VARIABLE_RE.findall(normalized):
            if name.casefold() != "n":
                variables[name.casefold()].add(number)
        numeric["counts"].update(_SAMPLE_COUNT_RE.findall(normalized))
        for number, unit in _SAMPLE_UNIT_RE.findall(normalized):
            unit = re.sub(r"\s+", "", unit.casefold()).replace("at.%", "at%")
            if unit == "Â°c":
                field = "temperatures"
            elif unit in {"Ã¥", "nm", "Î¼m", "um"}:
                field = "thicknesses"
            else:
                field = "concentrations"
            numeric[field].add(f"{number}{unit}")
    result: dict[str, object] = {}
    if variables:
        result["variables"] = {
            name: sorted(numbers) for name, numbers in sorted(variables.items())
        }
    result.update({field: sorted(numbers) for field, numbers in numeric.items() if numbers})
    return result


def _prepare_sample_identities(gold_samples: list[dict], document_id: str) -> list[dict]:
    """Prepare gold identities and the first property that separates each group."""
    curated = load_alignment_helper()["entities"]["samples"]["curated_forms"].get(document_id, {})
    entries = []
    by_composition = defaultdict(list)
    for entity in gold_samples:
        gold_id = str(entity.get("id"))
        reviewed = curated.get(gold_id, {})
        name = normalize_sample(entity.get("sample_name"))
        composition = normalize_sample(entity.get("chemical_composition"))
        names = [name] + [
            normalize_sample(form) for form in reviewed.get("accepted_name_forms", [])
        ]
        compositions = [composition] + [
            normalize_sample(form) for form in reviewed.get("accepted_composition_forms", [])
        ]
        entry = {
            "gold_id": gold_id,
            "normalized_name": name,
            "normalized_composition": composition,
            "accepted_name_forms": list(dict.fromkeys(
                value for value in names if value is not None
            )),
            "accepted_composition_forms": list(dict.fromkeys(
                value for value in compositions if value is not None
            )),
            "composition_is_unique": None if composition is None else True,
            "variant_identity": _sample_variant_identity(
                entity.get("sample_name"), entity.get("chemical_composition")
            ),
            "name_discriminators": [],
            "discriminator": None,
        }
        entries.append(entry)
        if composition is not None:
            by_composition[composition].append(entry)

    shared = sorted(value for value, members in by_composition.items() if len(members) > 1)
    for composition in shared:
        members = by_composition[composition]
        name_tokens = [_sample_identity_tokens(entry["normalized_name"]) for entry in members]
        common_tokens = set.intersection(*name_tokens)
        discriminator = "sample_name"
        for field in ("variables", "counts", "temperatures", "thicknesses", "concentrations"):
            values = [entry["variant_identity"].get(field) for entry in members]
            if all(value is not None for value in values) and len({
                _stable_scalar(value) for value in values
            }) == len(members):
                discriminator = field
                break
        for entry, tokens in zip(members, name_tokens):
            entry["composition_is_unique"] = False
            entry["name_discriminators"] = sorted(tokens - common_tokens)
            entry["discriminator"] = discriminator
    return entries


def _match_samples(
    predictions: list[dict], entries: list[dict],
) -> dict[int, tuple[int, str]]:
    """Collect sample evidence once, then lock matches in the declared stage order."""
    stages = {
        name: defaultdict(list) for name in (
            "exact_name", "unique_composition", "shared_composition_discriminator",
            "unique_name_containment",
        )
    }
    gold_name_tokens = [
        [_sample_identity_tokens(form) for form in entry["accepted_name_forms"]]
        for entry in entries
    ]
    prediction_keys = []
    registry_keys = [
        tuple(_stable_scalar(entry[field]) for field in (
            "normalized_composition", "normalized_name", "gold_id"
        )) for entry in entries
    ]
    for pred_index, prediction in enumerate(predictions):
        name = normalize_sample(prediction.get("sample_name"))
        composition = normalize_sample(prediction.get("chemical_composition"))
        prediction_keys.append(
            (composition or "", name or "", _stable_scalar(prediction.get("id")))
        )
        if not name:
            continue
        tokens = _sample_identity_tokens(name)
        variants = _sample_variant_identity(
            prediction.get("sample_name"), prediction.get("chemical_composition")
        )
        for gold_index, entry in enumerate(entries):
            if name in entry["accepted_name_forms"]:
                stages["exact_name"][pred_index].append((gold_index, 1))

            compositions = entry["accepted_composition_forms"]
            discriminator = entry["discriminator"]
            variant_match = discriminator not in (None, "sample_name") and (
                variants.get(discriminator) == entry["variant_identity"][discriminator]
            )
            if composition is not None and composition in compositions:
                if entry["composition_is_unique"] is True:
                    stages["unique_composition"][pred_index].append((gold_index, 1))
                discriminators = set(entry["name_discriminators"])
                if discriminator == "sample_name":
                    evidence = len(discriminators) if discriminators <= tokens else 0
                else:
                    evidence = 1 if variant_match else 0
                if evidence:
                    stages["shared_composition_discriminator"][pred_index].append(
                        (gold_index, evidence)
                    )

            if tokens and (composition is None or not compositions or composition in compositions):
                overlaps = [
                    len(tokens & gold_tokens) for gold_tokens in gold_name_tokens[gold_index]
                    if tokens <= gold_tokens or gold_tokens <= tokens
                ]
                if overlaps:
                    stages["unique_name_containment"][pred_index].append(
                        (gold_index, max(overlaps))
                    )

    matches = {}
    for stage, candidates in stages.items():
        _lock_stage_candidates(
            matches, candidates, stage, prediction_keys, registry_keys,
            require_mutual_uniqueness=(stage == "unique_composition"),
        )
    return matches


def align_samples(
    pred: list[dict] | None,
    gold: list[dict] | None,
    *,
    document_id: str,
    identity_gold: list[dict] | None = None,
) -> AlignmentResult:
    """Match names, compositions, distinguishing properties, then containment.

    Reviewed forms are paper-local. For IAA, both annotations link to the
    supplied reconciled identity_gold before their corresponding records join.
    """
    pred, gold = pred or [], gold or []
    entries = _prepare_sample_identities(
        gold if identity_gold is None else identity_gold, document_id
    )
    gold_links = _match_samples(gold, entries)
    pred_links = _match_samples(pred, entries)
    return _alignment_result(
        pred, gold, _join_identity_matches(pred_links, gold_links),
        label_fields=("sample_name", "chemical_composition"),
    )


# Datasets.


def _dataset_alignment_text(value: Any) -> str | None:
    """Use nonempty strings for alignment; preserve source values for scoring."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value


_DATASET_PROTECTED_COMPOUNDS = ("de-xas", "nd-ni")

_DATASET_GENERIC_PLURALS = {
    "datasets": "dataset",
    "experiments": "experiment",
    "measurements": "measurement",
    "results": "result",
    "scans": "scan",
    "studies": "study",
}


def normalize_dataset(value: str | None) -> str | None:
    """Normalize dataset names and measurement techniques for alignment."""
    if value is None:
        return None
    aliases = _aliases("datasets")
    text = value
    protected: dict[str, str] = {}
    # These compounds contain alias-like pieces with a different meaning.
    for index, compound in enumerate(_DATASET_PROTECTED_COMPOUNDS):
        marker = f"__blocked_compound_{index}__"
        pattern = re.compile(
            r"(?<![a-z0-9])" + re.escape(compound) + r"(?![a-z0-9])", re.IGNORECASE
        )
        if pattern.search(text):
            protected[marker] = compound
            text = pattern.sub(marker, text)
    text = _expand_aliases(text, aliases)
    for marker, compound in protected.items():
        text = text.replace(marker, compound)
    text = _collapse_repeated_aliases(normalize_surface(text), aliases)
    return " ".join(_DATASET_GENERIC_PLURALS.get(token, token) for token in text.split())


def _dataset_identity_tokens(value: str | None) -> set[str]:
    """Return tokens from an already normalized dataset identity value."""
    return set((value or "").replace("/", " ").split())


def _dataset_generic_tokens(helper: dict) -> set[str]:
    """Normalize generic dataset nouns for identity preparation and matching."""
    return {
        token for term in helper["generic_name_terms"]
        for token in _dataset_identity_tokens(normalize_dataset(term))
    }


def _dataset_minimal_discriminators(forms: list[set[str]]) -> list[list[str]]:
    """Return deterministic subset-minimal non-empty discriminator forms."""
    ordered = sorted(
        (form for form in forms if form), key=lambda item: (len(item), sorted(item))
    )
    minimal: list[set[str]] = []
    for form in ordered:
        if not any(existing <= form for existing in minimal):
            minimal.append(form)
    return [sorted(form) for form in minimal]


def _prepare_dataset_identities(gold_datasets: list[dict], document_id: str) -> list[dict]:
    """Group overlapping gold technique forms, then derive distinctive names."""
    helper = load_alignment_helper()["entities"]["datasets"]
    curated = helper["curated_forms"].get(document_id, {})
    entries = []
    for entity in gold_datasets:
        gold_id = str(entity.get("id"))
        reviewed = curated.get(gold_id, {})
        technique = normalize_dataset(entity.get("measurement_technique"))
        name = normalize_dataset(entity.get("dataset_name"))
        techniques = [technique] + [
            normalize_dataset(form) for form in reviewed.get("accepted_technique_forms", [])
        ]
        names = [name] + [
            normalize_dataset(form) for form in reviewed.get("accepted_name_forms", [])
        ]
        entries.append({
            "gold_id": gold_id,
            "normalized_technique": technique,
            "normalized_name": name,
            "accepted_technique_forms": list(dict.fromkeys(value for value in techniques if value)),
            "accepted_name_forms": list(dict.fromkeys(value for value in names if value)),
            "technique_group": None,
            "name_discriminator_forms": [],
            "name_containment_forms": [],
        })

    # Traverse overlapping forms transitively: A-B and B-C put all three in
    # one group, even if A and C have no technique form in common.
    technique_forms = [set(entry["accepted_technique_forms"]) for entry in entries]
    remaining = set(range(len(entries)))
    components = []
    while remaining:
        pending = [min(remaining)]
        remaining.remove(pending[0])
        component = []
        while pending:
            index = pending.pop()
            component.append(index)
            neighbors = {
                other for other in remaining if technique_forms[index] & technique_forms[other]
            }
            remaining.difference_update(neighbors)
            pending.extend(neighbors)
        if len(component) > 1:
            components.append(sorted(component))

    generic_tokens = _dataset_generic_tokens(helper)
    name_tokens = {
        form: _dataset_identity_tokens(form)
        for entry in entries for form in [entry["normalized_name"], *entry["accepted_name_forms"]]
    }
    technique_tokens = [
        _dataset_identity_tokens(entry["normalized_technique"]) for entry in entries
    ]
    for group_number, component in enumerate(components, start=1):
        common_names = set.intersection(*(
            name_tokens[entries[index]["normalized_name"]] for index in component
        ))
        common_techniques = set.intersection(*(technique_tokens[index] for index in component))
        for index in component:
            entry = entries[index]
            entry["technique_group"] = f"technique_group_{group_number}"
            sibling_names = [
                name_tokens[form] for sibling in component if sibling != index
                for form in entries[sibling]["accepted_name_forms"]
            ]
            discriminators, containment = [], []
            for form in entry["accepted_name_forms"]:
                tokens = name_tokens[form]
                if tokens - generic_tokens - (common_techniques & common_names):
                    containment.append(form)
                alternatives = [tokens - common_techniques - generic_tokens]
                if form == entry["normalized_name"]:
                    alternatives.append(tokens - common_names - generic_tokens)
                discriminators.extend(
                    option for option in alternatives
                    if option and not any(option <= sibling for sibling in sibling_names)
                )
            entry["name_discriminator_forms"] = _dataset_minimal_discriminators(discriminators)
            entry["name_containment_forms"] = sorted(
                set(containment), key=lambda value: (-len(value.split()), value)
            )
    return entries


def _dataset_name_form_contained(name: str, form: str) -> bool:
    """Check whole-phrase containment, excluding an occurrence of 'non <form>'."""
    haystack = f" {name} "
    needle = f" {form} "
    return needle in haystack and f" non {form} " not in haystack


def _match_datasets(
    predictions: list[dict], entries: list[dict], diagnostics: dict[str, int] | None = None,
) -> dict[int, tuple[int, str]]:
    """Match exact names first, then use technique-gated evidence for the rest."""
    generic_tokens = _dataset_generic_tokens(load_alignment_helper()["entities"]["datasets"])
    normalized = [
        (
            normalize_dataset(_dataset_alignment_text(prediction.get("measurement_technique"))),
            normalize_dataset(_dataset_alignment_text(prediction.get("dataset_name"))),
        )
        for prediction in predictions
    ]
    prediction_keys = [
        (technique or "", name or "", _stable_scalar(prediction.get("id")))
        for prediction, (technique, name) in zip(predictions, normalized)
    ]
    registry_keys = [
        tuple(_stable_scalar(entry[field]) for field in (
            "normalized_technique", "normalized_name", "gold_id"
        )) for entry in entries
    ]
    name_tokens = [_dataset_identity_tokens(name) for _, name in normalized]
    compatible = [
        {
            index for index, entry in enumerate(entries)
            if technique and technique in entry["accepted_technique_forms"]
        }
        for technique, _ in normalized
    ]
    matches = {}

    # 1. Both the technique and the complete name agree.
    candidates = {
        index: [
            (gold, 1) for gold in sorted(compatible[index])
            if name in entries[gold]["accepted_name_forms"]
        ]
        for index, (_, name) in enumerate(normalized) if name
    }
    _lock_stage_candidates(
        matches, candidates, "technique_and_name", prediction_keys, registry_keys
    )

    # 2-3. A uniquely owned full name establishes identity even if the
    # technique is wrong. Apply strict equality before removing generic nouns.
    exact_owners, substantive_owners = defaultdict(set), defaultdict(set)
    informative_forms = []
    for gold, entry in enumerate(entries):
        informative = []
        for form in entry["accepted_name_forms"]:
            exact_owners[form].add(gold)
            substantive = frozenset(_dataset_identity_tokens(form) - generic_tokens)
            if substantive:
                substantive_owners[substantive].add(gold)
                informative.append(form)
        informative_forms.append(informative)
    blocked = set()
    for stage, collision_key, owners, keys in (
        (
            "exact_unique_name", "dataset_exact_name_collision",
            exact_owners, [name for _, name in normalized],
        ),
        (
            "generic_normalized_unique_name", "dataset_generic_name_collision",
            substantive_owners, [frozenset(tokens - generic_tokens) for tokens in name_tokens],
        ),
    ):
        candidates = {}
        for index, key in enumerate(keys):
            if index in matches or index in blocked or not key:
                continue
            targets = owners.get(key, set())
            if len(targets) == 1:
                candidates[index] = [(next(iter(targets)), 1)]
        _lock_stage_candidates(matches, candidates, stage, prediction_keys, registry_keys)
        # Duplicate names cannot fall through to a weaker technique match.
        collisions = set(candidates) - matches.keys()
        blocked.update(collisions)
        if diagnostics is not None and collisions:
            diagnostics[collision_key] = len(collisions)

    # 4. Block contradictory names, then disambiguate shared techniques.
    conflicts, candidates = set(), {}
    for index, (technique, name) in enumerate(normalized):
        if index in matches or index in blocked or not technique or not name:
            continue
        groups = {entries[gold]["technique_group"] for gold in compatible[index]} - {None}
        if any(
            _dataset_name_form_contained(name, form)
            for gold, forms in enumerate(informative_forms)
            if gold not in compatible[index] and entries[gold]["technique_group"] not in groups
            for form in forms
        ):
            conflicts.add(index)
            continue
        hits = []
        for gold in sorted(compatible[index]):
            entry = entries[gold]
            if entry["technique_group"] is None:
                continue
            scores = [
                len(form) for form in entry["name_discriminator_forms"]
                if form and set(form) <= name_tokens[index]
            ]
            scores.extend(
                len(form.split()) for form in entry["name_containment_forms"]
                if _dataset_name_form_contained(name, form)
            )
            if scores:
                hits.append((gold, max(scores)))
        if hits:
            candidates[index] = hits
    blocked.update(conflicts)
    if diagnostics is not None and conflicts:
        diagnostics["dataset_name_technique_conflict"] = len(conflicts)
    _lock_stage_candidates(
        matches, candidates, "shared_technique_name_discriminator", prediction_keys, registry_keys
    )

    # 5. Technique alone must identify exactly one remaining gold record,
    # and exactly one remaining prediction must target that record.
    used_gold = {gold for gold, _ in matches.values()}
    sources = defaultdict(list)
    for index in range(len(predictions)):
        if index in matches or index in blocked:
            continue
        targets = [
            gold for gold in compatible[index] - used_gold
            if entries[gold]["technique_group"] is None
        ]
        if len(targets) == 1:
            sources[targets[0]].append(index)
    collisions = sum(len(indices) for indices in sources.values() if len(indices) > 1)
    if diagnostics is not None and collisions:
        diagnostics["dataset_technique_collision"] = collisions
    for gold, indices in sources.items():
        if len(indices) == 1:
            index = indices[0]
            stage = (
                "unique_technique_missing_name" if normalized[index][1] is None else "unique_technique"
            )
            matches[index] = (gold, stage)
    return matches


def align_datasets(
    pred: list[dict] | None,
    gold: list[dict] | None,
    *,
    document_id: str,
    identity_gold: list[dict] | None = None,
) -> AlignmentResult:
    """Match dataset identities using name and technique evidence in stage order.

    Shared techniques require distinguishing names; duplicate and conflicting
    records cannot fall through to weaker matches. For IAA, both annotations
    link independently through the supplied reconciled identity_gold.
    """
    pred, gold = pred or [], gold or []
    entries = _prepare_dataset_identities(
        gold if identity_gold is None else identity_gold, document_id
    )
    gold_diagnostics: dict[str, int] = {}
    gold_links = _match_datasets(gold, entries, gold_diagnostics)
    pred_diagnostics: dict[str, int] = {}
    pred_links = _match_datasets(pred, entries, pred_diagnostics)
    return _alignment_result(
        pred, gold, _join_identity_matches(pred_links, gold_links),
        label_fields=("measurement_technique", "dataset_name"),
        abstentions={
            **{f"gold_{key}": value for key, value in gold_diagnostics.items()},
            **{f"pred_{key}": value for key, value in pred_diagnostics.items()},
        },
    )


# Facilities.


def normalize_facility(value: str | None) -> str | None:
    """Expand whole facility aliases, preserving surrounding instrument wording."""
    if value is None:
        return None
    aliases = {
        item["alias"].casefold(): item["canonical_name"]
        for item in load_alignment_helper()["entities"]["facilities"]["aliases"]
    }
    alternatives = sorted(aliases, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(key) for key in alternatives) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda match: aliases[match.group(0).casefold()], value.casefold())


def _facility_identity_tokens(value: str | None) -> tuple[str, ...]:
    """Represent acronyms and full names with the same length-neutral identity.

    Repeated facility names are absorbed; beamline and instrument words remain.
    """
    if value is None:
        return ()
    forms: list[tuple[str, str]] = []
    canonical_markers: dict[str, str] = {}
    marker_words: dict[str, set[str]] = defaultdict(set)
    for item in load_alignment_helper()["entities"]["facilities"]["aliases"]:
        canonical = text_canon(item["canonical_name"]).casefold()
        marker = canonical_markers.setdefault(
            canonical, f"facilityalias{len(canonical_markers)}"
        )
        marker_words[marker].update(tokenize(canonical))
        forms.extend(((item["canonical_name"], marker), (item["alias"], marker)))
    forms.sort(key=lambda item: len(item[0]), reverse=True)
    normalized = text_canon(value).casefold()
    for form, marker in forms:
        normalized = re.sub(
            r"(?<![\w])" + re.escape(text_canon(form).casefold()) + r"(?![\w])",
            marker, normalized, flags=re.IGNORECASE,
        )
    result: list[str] = []
    seen: set[str] = set()
    for token in tokenize(normalized):
        if token.startswith("facilityalias"):
            if token in seen:
                continue
            seen.add(token)
        result.append(token)
    absorbed_words = {word for marker in seen for word in marker_words[marker]}
    return tuple(
        token for token in result
        if token.startswith("facilityalias") or token not in absorbed_words
    )


def _facility_alignment_values(entity: dict) -> dict:
    """Prepare each facility's text and identity once for the matching stages."""
    raw = text_canon(entity.get("name") or "").casefold()
    identity = _facility_identity_tokens(raw)
    return {
        "raw": raw,
        "identity": identity,
        "aliases": normalize_facility(raw) or "",
        "markers": {token for token in identity if token.startswith("facilityalias")},
        "key": (identity, raw, _stable_scalar(entity.get("id"))),
    }


def _lock_equal_facilities(
    predictions: list[dict],
    gold: list[dict],
    remaining_p: set[int],
    remaining_g: set[int],
    field: str,
    stage: str,
) -> dict[int, tuple[int, str]]:
    matches: dict[int, tuple[int, str]] = {}
    values = sorted(
        {predictions[i][field] for i in remaining_p}
        & {gold[j][field] for j in remaining_g},
        key=str,
    )
    for value in values:
        if not value:
            continue
        ps = sorted(
            (i for i in remaining_p if predictions[i][field] == value),
            key=lambda index: predictions[index]["key"],
        )
        gs = sorted(
            (j for j in remaining_g if gold[j][field] == value),
            key=lambda index: gold[index]["key"],
        )
        for i, j in zip(ps, gs):
            matches[i] = (j, stage)
            remaining_p.remove(i)
            remaining_g.remove(j)
    return matches


def align_facilities(
    pred: list[dict] | None,
    gold: list[dict] | None,
    *,
    tau: float = 1.0,
) -> AlignmentResult:
    """Match exact names, then exact aliases, then unambiguous compatible containment."""
    pred, gold = pred or [], gold or []
    pred_values = [_facility_alignment_values(entity) for entity in pred]
    gold_values = [_facility_alignment_values(entity) for entity in gold]
    remaining_p, remaining_g = set(range(len(pred_values))), set(range(len(gold_values)))
    matches = _lock_equal_facilities(
        pred_values,
        gold_values,
        remaining_p,
        remaining_g,
        "raw",
        "raw_exact",
    )
    matches.update(
        _lock_equal_facilities(
            pred_values,
            gold_values,
            remaining_p,
            remaining_g,
            "identity",
            "alias_exact",
        )
    )
    candidates = defaultdict(list)
    for i in remaining_p:
        for j in remaining_g:
            p, g = pred_values[i], gold_values[j]
            p_raw, g_raw = p["raw"], g["raw"]
            p_identity, g_identity = p["identity"], g["identity"]
            p_markers, g_markers = p["markers"], g["markers"]
            # If gold has known host markers, require at least one shared marker.
            if g_markers and p_markers.isdisjoint(g_markers):
                continue
            p_alias, g_alias = p["aliases"], g["aliases"]
            # Measure containment on shared alias markers so longer facility names do
            # not duplicate alias words.
            containment = token_containment(" ".join(p_identity), " ".join(g_identity))
            if containment < tau:
                continue
            alias_f1 = token_overlap_f1(p_alias, g_alias)
            raw_f1 = token_overlap_f1(p_raw, g_raw)
            unmatched = abs(len(tokenize(p_alias)) - len(tokenize(g_alias)))
            score = (
                1.0
                + alias_f1 * 1e-3
                + raw_f1 * 1e-6
                + (1.0 / (1.0 + unmatched)) * 1e-9
            )
            candidates[i].append((j, score))
    _lock_stage_candidates(
        matches, candidates, "containment",
        [record["key"] for record in pred_values],
        [record["key"] for record in gold_values],
    )
    pairs = [
        MatchedPair(i, j, stage) for i, (j, stage) in matches.items()
    ]
    return _alignment_result(pred, gold, pairs, label_fields=("name",))


# Shared normalization and alias support.


_ALIGNMENT_HELPER_PATH = Path(__file__).resolve().parents[1] / "alignment_helper.yaml"


def load_alignment_helper() -> dict[str, Any]:
    """Read the fixed release helper; its structure is checked in the tests."""
    return load_yaml_mapping(
        _ALIGNMENT_HELPER_PATH.read_text(encoding="utf-8"), source="alignment_helper.yaml"
    )


_SEPARATOR_RE = re.compile(r"[,;:()\[\]{}]+")

_HYPHEN_RE = re.compile(r"\s*-\s*")


def normalize_surface(value: str) -> str:
    """Normalize punctuation and spacing in funder and dataset wording."""
    text = unicodedata.normalize("NFKC", value).translate(DASH_MAP)
    text = text.replace("\u00ad", "").casefold()
    text = _SEPARATOR_RE.sub(" ", text)
    text = _HYPHEN_RE.sub(" ", text)
    return " ".join(text.split())


def _aliases(entity: str) -> dict[str, str]:
    """Read the normalized alias vocabulary for funding, samples, or datasets."""
    return {
        normalize_surface(item["alias"]): normalize_surface(item["canonical_name"])
        for item in load_alignment_helper()["entities"][entity]["aliases"]
    }


def _expand_aliases(value: str, aliases: dict[str, str]) -> str:
    """Replace whole aliases, matching the longest forms first."""
    alternatives = sorted(aliases, key=lambda key: (-len(key), key))
    pattern = re.compile(
        r"(?<![a-z0-9])(?:"
        + "|".join(re.escape(key).replace(r"\ ", r"[-\s]+") for key in alternatives)
        + r")(?![a-z0-9])",
        re.IGNORECASE,
    )
    return pattern.sub(lambda match: aliases[normalize_surface(match.group(0))], value)


def _collapse_repeated_aliases(value: str, aliases: dict[str, str]) -> str:
    """Collapse a full name repeated by expansion of its adjacent acronym."""
    for canonical in sorted(set(aliases.values()), key=len, reverse=True):
        value = re.sub(
            rf"(?:{re.escape(canonical)}\s+){{1,}}{re.escape(canonical)}",
            canonical,
            value,
        )
    return value


def token_overlap_f1(a: str, b: str) -> float:
    """Multiset token F1. Used as the default entity-alignment key score."""
    a_tok = tokenize(a)
    b_tok = tokenize(b)
    if not a_tok or not b_tok:
        return 0.0
    a_c = collections.Counter(a_tok)
    b_c = collections.Counter(b_tok)
    overlap = sum((a_c & b_c).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(a_tok)
    r = overlap / len(b_tok)
    return 2 * p * r / (p + r)


def token_containment(a: str, b: str) -> float:
    """Multiset token overlap divided by the shorter token count.

    Symmetric; returns 1 when the shorter token multiset is contained in the longer,
    such as 'ISIS' within 'WISH beamline, ISIS, United Kingdom'.
    """
    a_tok = tokenize(a)
    b_tok = tokenize(b)
    if not a_tok or not b_tok:
        return 0.0
    a_c = collections.Counter(a_tok)
    b_c = collections.Counter(b_tok)
    overlap = sum((a_c & b_c).values())
    return overlap / min(len(a_tok), len(b_tok))


# Shared one-to-one assignment.


def _hungarian_assign(
    sims: list[tuple[float, int, int]],
    tau: float,
    n_pred: int,
    n_gold: int,
) -> list[tuple[int, int, float]]:
    """Prefer the most matches with score >= tau, then the highest total similarity.

    A large per-match bonus makes cardinality the primary objective.
    """
    if n_pred == 0 or n_gold == 0:
        return []
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    big = 1e6
    sim_m = np.zeros((n_pred, n_gold))
    score_m = np.zeros((n_pred, n_gold))
    for s, i, j in sims:
        sim_m[i, j] = s
        if s >= tau:
            score_m[i, j] = big + s
    rows, cols = linear_sum_assignment(-score_m)
    matches = [
        (int(i), int(j), float(sim_m[i, j]))
        for i, j in zip(rows, cols)
        if sim_m[i, j] >= tau
    ]
    matches.sort(key=lambda t: -t[2])
    return matches


def _lock_stage_candidates(
    matches: dict[int, tuple[int, str]],
    candidates: dict[int, list[tuple[int, int | float]]],
    stage: str,
    prediction_keys: list[tuple],
    registry_keys: list[tuple],
    *,
    require_mutual_uniqueness: bool = False,
) -> None:
    """Assign one stage by maximum cardinality, then total evidence.

    Candidate scores are alignment evidence only. Canonical keys and local IDs
    stabilize tie ordering.
    """
    used_gold = {gold for gold, _ in matches.values()}
    available: dict[int, list[tuple[int, float]]] = {}
    for pred_index, hits in candidates.items():
        if pred_index in matches:
            continue
        filtered = [
            (gold_index, float(score))
            for gold_index, score in hits
            if gold_index not in used_gold
        ]
        if filtered:
            available[pred_index] = filtered

    if require_mutual_uniqueness:
        singletons = {
            pred_index: hits[0]
            for pred_index, hits in available.items()
            if len(hits) == 1
        }
        reverse = collections.Counter(
            gold_index for gold_index, _score in singletons.values()
        )
        available = {
            pred_index: [hit]
            for pred_index, hit in singletons.items()
            if reverse[hit[0]] == 1
        }

    if not available:
        return
    pred_indices = sorted(
        available,
        key=lambda index: (prediction_keys[index], index),
    )
    gold_indices = sorted(
        {gold_index for hits in available.values() for gold_index, _ in hits},
        key=lambda index: (registry_keys[index], index),
    )
    pred_local = {index: local for local, index in enumerate(pred_indices)}
    gold_local = {index: local for local, index in enumerate(gold_indices)}
    sims = [
        (score, pred_local[pred_index], gold_local[gold_index])
        for pred_index, hits in available.items()
        for gold_index, score in hits
    ]
    assigned = _hungarian_assign(sims, 1.0, len(pred_indices), len(gold_indices))
    objective = (len(assigned), sum(score for _i, _j, score in assigned))
    for local_i, local_j, _score in assigned:
        pred_index = pred_indices[local_i]
        # A source with several equally optimal identity choices remains
        # unresolved.  If removing the selected edge lowers cardinality or
        # evidence, the rest of the candidate graph genuinely disambiguates
        # it (for example A->{1,2}, B->{1}).  A source duplicate with only one
        # admissible identity is instead selected by its stable local-ID key.
        if len(available[pred_index]) > 1:
            alternative = _hungarian_assign(
                [
                    item
                    for item in sims
                    if not (item[1] == local_i and item[2] == local_j)
                ],
                1.0,
                len(pred_indices),
                len(gold_indices),
            )
            alternative_objective = (
                len(alternative),
                sum(score for _i, _j, score in alternative),
            )
            if (
                alternative_objective[0] == objective[0]
                and abs(alternative_objective[1] - objective[1]) <= 1e-12
            ):
                continue
        matches[pred_indices[local_i]] = (gold_indices[local_j], stage)


# Shared correspondence and diagnostics.


def _stable_scalar(value: Any) -> str:
    """Return a deterministic string suitable for alignment tie keys."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _entity_label(entity: dict, fields: tuple[str, ...]) -> Any:
    """Use a local ID when present, otherwise the entity's identifying fields."""
    if entity.get("id") is not None:
        return entity["id"]
    return {name: entity.get(name) for name in fields} if fields else None


def _join_identity_matches(
    pred_staged: dict[int, tuple[int, str]], gold_staged: dict[int, tuple[int, str]]
) -> list[MatchedPair]:
    """Join independently aligned sides through their shared gold identities."""
    registry_to_gold = {
        identity: index for index, (identity, _stage) in gold_staged.items()
    }
    return [
        MatchedPair(index, registry_to_gold[identity], stage)
        for index, (identity, stage) in pred_staged.items()
        if identity in registry_to_gold
    ]


def _alignment_result(
    pred: list[dict],
    gold: list[dict],
    pairs: list[MatchedPair],
    *,
    label_fields: tuple[str, ...] = (),
    abstentions: dict[str, int] | None = None,
) -> AlignmentResult:
    """Finalize stable pair order, ID mappings, and report diagnostics."""
    pred_labels = [_entity_label(entity, label_fields) for entity in pred]
    gold_labels = [_entity_label(entity, label_fields) for entity in gold]
    # Pair order also fixes the order of floating-point attribute-score sums.
    pairs = sorted(pairs, key=lambda pair: (
        _stable_scalar(pred_labels[pair.pred_index]),
        _stable_scalar(gold_labels[pair.gold_index]),
        pair.stage,
    ))
    id_map = {entity["id"]: None for entity in pred if entity.get("id") is not None}
    for pair in pairs:
        pred_id = pred[pair.pred_index].get("id")
        if pred_id is not None:
            id_map[pred_id] = gold[pair.gold_index].get("id")

    diagnostics = {
        "match_stages": [
            {
                "pred": pred_labels[pair.pred_index],
                "gold": gold_labels[pair.gold_index],
                "stage": pair.stage,
            }
            for pair in pairs
        ]
    }
    if abstentions:
        diagnostics["alignment_abstentions"] = abstentions
    return AlignmentResult(pairs=pairs, id_map=id_map, diagnostics=diagnostics)

