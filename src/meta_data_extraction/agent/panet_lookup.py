"""Deterministic candidates from an explicitly scoped, frozen PaNET resource.

Scores describe lexical overlap, never publication support or mapping confidence.
Named parent links restrict the search scope; they do not add inferred synonyms,
equivalent concepts, or evidence that a technique was used in a publication.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Literal

from pydantic import Field, model_validator

from meta_data_extraction.agent.evidence import FrozenModel, sha256
from meta_data_extraction.agent.panet_resources import OntologySnapshot


LOOKUP_VERSION = "m04-panet-lexical-v1"
MatchKind = Literal["label", "exact", "alternative", "related", "broad", "narrow"]
_KIND_ORDER = {kind: index for index, kind in enumerate(
    ("label", "exact", "alternative", "related", "broad", "narrow")
)}


def normalize_query(text: str) -> str:
    """Normalize only for matching; do not change source expressions or labels."""
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


class Candidate(FrozenModel):
    concept_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    matched_text: str = Field(min_length=1)
    matched_kind: MatchKind
    match_type: Literal["exact", "token_overlap"]
    score: float = Field(gt=0, le=1, allow_inf_nan=False)
    parents: tuple[str, ...]


def _rank(candidate: Candidate) -> tuple:
    # Every exact text match precedes overlap. Exact labels precede aliases;
    # alternate labels retain their resource-declared kind, not semantic equality.
    exact = candidate.match_type == "exact"
    return (
        0 if exact else 1,
        _KIND_ORDER[candidate.matched_kind] if exact else -candidate.score,
        -candidate.score if exact else _KIND_ORDER[candidate.matched_kind],
        candidate.concept_id,
        candidate.matched_text,
    )


def _cache_key(*, query: str, families: tuple[str, ...], limit: int,
               snapshot_id: str, resource_sha256: str, resource_release: str,
               resource_kind: str) -> str:
    identity = {
        "lookup_version": LOOKUP_VERSION, "query": query, "families": families,
        "limit": limit, "snapshot_id": snapshot_id,
        "resource_sha256": resource_sha256, "resource_release": resource_release,
        "resource_kind": resource_kind,
    }
    return "lookup:" + sha256(json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))


class CandidateSet(FrozenModel):
    lookup_version: Literal["m04-panet-lexical-v1"] = LOOKUP_VERSION
    query: str = Field(min_length=1, max_length=4096, strict=True)
    normalized_query: str = Field(min_length=1)
    families: tuple[str, ...]
    limit: int = Field(ge=1, le=50, strict=True)
    snapshot_id: str = Field(min_length=1)
    resource_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    resource_release: str = Field(min_length=1)
    resource_kind: Literal["authentic", "synthetic"]
    status: Literal["matched", "no_match"]
    candidates: tuple[Candidate, ...]
    total_candidates: int = Field(ge=0, strict=True)
    omitted_candidates: int = Field(ge=0, strict=True)
    cache_key: str

    @model_validator(mode="after")
    def _integrity(self) -> CandidateSet:
        if self.normalized_query != normalize_query(self.query):
            raise ValueError("Query normalization does not match the original expression")
        if not self.families or self.families != tuple(sorted(set(self.families))):
            raise ValueError("Families must be nonempty, distinct, and sorted")
        if (len(self.candidates) != min(self.limit, self.total_candidates)
                or self.omitted_candidates != self.total_candidates - len(self.candidates)):
            raise ValueError("Candidate counts do not match the requested result limit")
        if len({candidate.concept_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("Each concept may occur at most once")
        if self.candidates != tuple(sorted(self.candidates, key=_rank)):
            raise ValueError("Candidates must use the recorded deterministic ranking")
        if self.status != ("matched" if self.candidates else "no_match"):
            raise ValueError("Candidate status does not match its contents")
        expected = _cache_key(
            query=self.query, families=self.families, limit=self.limit,
            snapshot_id=self.snapshot_id, resource_sha256=self.resource_sha256,
            resource_release=self.resource_release, resource_kind=self.resource_kind,
        )
        if self.cache_key != expected:
            raise ValueError("Candidate cache identity does not match its resource/query/scope")
        return self


def _scope(snapshot: OntologySnapshot, families: tuple[str, ...]) -> set[str]:
    concepts = {concept.concept_id: concept for concept in snapshot.concepts}
    if (not families or len(set(families)) != len(families)
            or any(family not in concepts or concepts[family].obsolete for family in families)):
        raise ValueError("Families must be distinct, active concept IDs from the supplied resource")
    selected = set(families)
    # Traverse named edges only, retaining the resource loader's explicit OWL
    # limitations. An obsolete ancestor does not make its labels searchable.
    children: dict[str, list[str]] = {}
    for concept in snapshot.concepts:
        for parent in concept.parents:
            children.setdefault(parent, []).append(concept.concept_id)
    pending = list(families)
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in selected:
                selected.add(child)
                pending.append(child)
    return selected


def lexical_candidates(snapshot: OntologySnapshot, query: str, *,
                       families: tuple[str, ...], limit: int = 5) -> CandidateSet:
    """Search labels and declared aliases in named descendants of selected families.

    No stemming, acronym invention, hierarchy reasoning, publication inspection,
    file discovery, or model call occurs. Ties use declared match kind then ID.
    Exact broad labels outrank overlapping specific labels, but all results remain
    candidates: this function cannot approve a concept assignment.
    """
    snapshot = OntologySnapshot.model_validate(snapshot.model_dump())
    if not isinstance(query, str) or not 1 <= len(query) <= 4096:
        raise ValueError("Query must be a string of 1 to 4096 characters")
    normalized = normalize_query(query)
    if not normalized:
        raise ValueError("Query must contain at least one searchable letter or number")
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("Result limit must be an integer from 1 to 50")
    if not isinstance(families, tuple) or any(not isinstance(f, str) for f in families):
        raise ValueError("Families must be a tuple of explicit resource concept IDs")
    selected = _scope(snapshot, families)
    families = tuple(sorted(families))
    terms = set(normalized.split())
    matches = []
    for concept in snapshot.concepts:
        if concept.obsolete or concept.concept_id not in selected:
            continue
        best = None
        for text, kind in ((concept.label, "label"), *((s.text, s.kind) for s in concept.synonyms)):
            normalized_text = normalize_query(text)
            label_terms = set(normalized_text.split())
            overlap = terms & label_terms
            if not overlap:
                continue
            candidate = Candidate(
                concept_id=concept.concept_id, label=concept.label,
                matched_text=text, matched_kind=kind,
                match_type="exact" if normalized_text == normalized else "token_overlap",
                score=2 * len(overlap) / (len(terms) + len(label_terms)),
                parents=concept.parents,
            )
            if best is None or _rank(candidate) < _rank(best):
                best = candidate
        if best is not None:
            matches.append(best)
    matches.sort(key=_rank)
    resource = snapshot.resource
    return CandidateSet(
        query=query, normalized_query=normalized, families=families, limit=limit,
        snapshot_id=snapshot.snapshot_id, resource_sha256=resource.sha256,
        resource_release=resource.release, resource_kind=resource.kind,
        status="matched" if matches else "no_match", candidates=tuple(matches[:limit]),
        total_candidates=len(matches), omitted_candidates=max(0, len(matches) - limit),
        cache_key=_cache_key(
            query=query, families=families, limit=limit, snapshot_id=snapshot.snapshot_id,
            resource_sha256=resource.sha256, resource_release=resource.release,
            resource_kind=resource.kind,
        ),
    )


def verify_candidates(snapshot: OntologySnapshot, result: CandidateSet) -> CandidateSet:
    """Reject stale or modified replay results by recomputing their entire scope."""
    result = CandidateSet.model_validate(result.model_dump())
    expected = lexical_candidates(snapshot, result.query, families=result.families, limit=result.limit)
    if result != expected:
        raise ValueError("Candidates do not match the supplied frozen resource and lookup version")
    return result
