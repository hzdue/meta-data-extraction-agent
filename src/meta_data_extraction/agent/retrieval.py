"""Deterministic lexical selection over supplied bundles, never a repository index."""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from math import fsum, log
from typing import Literal

from pydantic import Field, model_validator

from meta_data_extraction.agent.evidence import (
    DocumentBundle, EvidenceError, EvidenceRef, FrozenModel, RegionKind, SourceRegion, evidence_ref,
    resolve_evidence,
)


RetrievalStrategy = Literal["lexical_v1", "focused_v2"]

# This fixed, general-language list and window size are part of focused_v2.
# Negation and numbers are retained; there is no domain-specific stopword list.
_STOPWORDS = frozenset("""
a an the and or of to in on at by for from with as is are was were be been being
it its this that these those which what when where who how do does did have has had
can could would should please me us our their they them we you your
""".split())
_FOCUSED_WINDOW_CHARS = 800


def _word_spans(text: str, offset: int) -> tuple[tuple[int, int, str], ...]:
    """Normalize tokens for matching while retaining their original Unicode span.

    Combining marks stay with their base; compatibility characters may normalize
    to multiple words, all of which address the same original contiguous token.
    This mapping never normalizes or reconstructs an evidence quotation.
    """
    spans = []
    begin = None
    for index, char in enumerate(text):
        word_char = bool(re.search(r"\w", unicodedata.normalize("NFKC", char))) or (
            unicodedata.category(char).startswith("M")
        )
        if word_char and begin is None:
            begin = index
        if begin is not None and (not word_char or index == len(text) - 1):
            end = index if not word_char else index + 1
            spans.extend((offset + begin, offset + end, term) for term in sorted(_terms(text[begin:end])))
            begin = None
    return tuple(spans)


def _focused_window(region: SourceRegion, hits: tuple[tuple[int, int, str], ...],
                    weights: dict[str, float], width: int) -> tuple[float, int, int]:
    """Choose one bounded window, scoring only complete matching source tokens."""
    width = min(width, region.end - region.start)
    starts = set()
    for start, end, _ in hits:
        for proposed in (start, end - width, (start + end - width) // 2):
            starts.add(max(region.start, min(region.end - width, proposed)))
    positions = [hit[0] for hit in hits]
    total_weight = fsum(weights.values())
    best = None
    for start in sorted(starts):
        end = start + width
        within = [hit for hit in hits[bisect_left(positions, start):bisect_left(positions, end)]
                  if hit[1] <= end]
        overlap = {hit[2] for hit in within}
        score = min(1.0, fsum(weights[term] for term in sorted(overlap)) / total_weight)
        # Among equal overlaps prefer context balanced around the matched span,
        # then the earliest original offset. Repeated words cannot raise score.
        imbalance = abs(start + end - within[0][0] - within[-1][1]) if within else 0
        candidate = (-score, imbalance, start)
        if best is None or candidate < best[0]:
            best = (candidate, (score, start, end))
    assert best is not None  # callers only pass regions with at least one hit
    return best[1]


class Passage(FrozenModel):
    evidence: EvidenceRef
    kind: RegionKind
    section_path: tuple[str, ...]
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    clipped: bool = False


class Selection(FrozenModel):
    mode: Literal["full_text", "targeted"]
    status: Literal["complete", "truncated", "no_match"]
    text: str
    passages: tuple[Passage, ...]
    source_ids: tuple[str, ...]
    max_chars: int = Field(gt=0, strict=True)
    used_chars: int = Field(ge=0, strict=True)
    available_chars: int | None = Field(default=None, ge=0, strict=True)
    candidate_count: int = Field(ge=0, strict=True)
    omitted_candidates: int = Field(ge=0, strict=True)
    incomplete_sources: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _counts(self) -> Selection:
        if self.used_chars != len(self.text) or self.used_chars > self.max_chars:
            raise ValueError("Selection must respect its recorded character budget")
        if self.omitted_candidates != self.candidate_count - len(self.passages):
            raise ValueError("Selection candidate counts do not match")
        if len(set(self.source_ids)) != len(self.source_ids) or not set(self.incomplete_sources) <= set(self.source_ids):
            raise ValueError("Selection source IDs must describe its scope")
        if any(p.evidence.source_id not in self.source_ids for p in self.passages):
            raise ValueError("Passage is outside the selection scope")
        if self.mode == "targeted":
            if self.text != "\n\n".join(p.evidence.quotation for p in self.passages):
                raise ValueError("Selected text does not match its evidence quotations")
            expected = "no_match" if not self.candidate_count else (
                "truncated" if self.omitted_candidates or any(p.clipped for p in self.passages) else "complete"
            )
        else:
            if len(self.source_ids) != 1 or self.available_chars is None or self.used_chars > self.available_chars:
                raise ValueError("Full-text selection requires one source and its available length")
            if any(self.text[p.evidence.start:p.evidence.end] != p.evidence.quotation for p in self.passages):
                raise ValueError("Full-text evidence does not match the selected prefix")
            expected = "truncated" if self.used_chars < self.available_chars else "complete"
        if self.status != expected:
            raise ValueError("Selection status does not match its omissions")
        return self


def _terms(text: str) -> set[str]:
    # Normalize for matching only. Stored text and quotations are never normalized.
    return set(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _budget(max_chars: int, limit: int = 1) -> None:
    if type(max_chars) is not int or max_chars < 1 or type(limit) is not int or limit < 1:
        raise ValueError("Character budget and result limit must be positive integers")


class SourceCorpus(FrozenModel):
    """The bundles passed here are the entire retrieval scope; no file I/O occurs."""

    bundles: tuple[DocumentBundle, ...]

    @model_validator(mode="after")
    def _scope(self) -> SourceCorpus:
        if not self.bundles or len({b.source_id for b in self.bundles}) != len(self.bundles):
            raise ValueError("Supply a nonempty corpus with distinct source IDs")
        return self

    def _selected(self, source_ids: tuple[str, ...] | None) -> tuple[DocumentBundle, ...]:
        by_id = {b.source_id: b for b in self.bundles}
        if source_ids is None:
            return self.bundles
        if len(set(source_ids)) != len(source_ids) or any(s not in by_id for s in source_ids):
            raise EvidenceError("Source filter must contain distinct IDs from the supplied corpus")
        return tuple(by_id[s] for s in source_ids)

    @staticmethod
    def _incomplete(bundles: tuple[DocumentBundle, ...]) -> tuple[str, ...]:
        return tuple(b.source_id for b in bundles if any(
            w.code in {"truncated", "parser_warning", "images_unavailable"} for w in b.warnings
        ))

    def resolve(self, ref: EvidenceRef) -> str:
        bundle, = self._selected((ref.source_id,))
        return resolve_evidence(bundle, ref)

    def full_text(self, source_id: str, *, max_chars: int) -> Selection:
        """Return an exact prefix, including whitespace, plus refs for its regions."""
        _budget(max_chars)
        bundle, = self._selected((source_id,))
        text = bundle.text[:max_chars]
        passages = tuple(
            Passage(evidence=evidence_ref(bundle, region.region_id, end=min(region.end, len(text))),
                    kind=region.kind, section_path=region.section_path, clipped=region.end > len(text))
            for region in bundle.regions if region.start < len(text)
        )
        return Selection(mode="full_text", status="truncated" if len(text) < len(bundle.text) else "complete",
                         text=text, passages=passages, source_ids=(source_id,),
                         max_chars=max_chars, used_chars=len(text), available_chars=len(bundle.text),
                         candidate_count=len(bundle.regions),
                         omitted_candidates=len(bundle.regions) - len(passages),
                         incomplete_sources=self._incomplete((bundle,)))

    def search(self, query: str, *, max_chars: int, limit: int = 5,
               source_ids: tuple[str, ...] | None = None,
               strategy: RetrievalStrategy = "lexical_v1") -> Selection:
        """Select lexical evidence with an explicitly versioned strategy.

        lexical_v1 retains the historical query-overlap ranking and region-prefix
        clipping. focused_v2 discounts general-language stopwords and uses IDF
        from supplied regions, selecting one context window per matching region.
        Scores are lexical overlap, never confidence or evidence entailment.
        Headings remain searchable; section paths stay with their body passages.
        """
        if strategy not in ("lexical_v1", "focused_v2"):
            raise ValueError("Unknown retrieval strategy")
        if strategy == "focused_v2":
            return self._search_focused(query, max_chars=max_chars, limit=limit, source_ids=source_ids)
        _budget(max_chars, limit)
        bundles = self._selected(source_ids)
        terms = _terms(query)
        if not terms:
            raise ValueError("Query must contain at least one searchable word or number")
        candidates = []
        for bundle in bundles:
            for region in bundle.regions:
                overlap = terms & _terms(bundle.text[region.start:region.end])
                if overlap:
                    score = len(overlap) / len(terms)
                    candidates.append((score, bundle, region))
        candidates.sort(key=lambda item: (-item[0], item[1].source_id, item[2].start))
        passages = []
        pieces = []
        used = 0
        for score, bundle, region in candidates[:limit]:
            separator = 2 if pieces else 0
            remaining = max_chars - used - separator
            if remaining <= 0:
                break
            end = min(region.end, region.start + remaining)
            ref = evidence_ref(bundle, region.region_id, end=end)
            passages.append(Passage(evidence=ref, kind=region.kind, section_path=region.section_path,
                                    score=score, clipped=end < region.end))
            pieces.append(ref.quotation)
            used += separator + len(ref.quotation)
        omitted = len(candidates) - len(passages)
        status = "no_match" if not candidates else (
            "truncated" if omitted or any(p.clipped for p in passages) else "complete"
        )
        return Selection(mode="targeted", status=status, text="\n\n".join(pieces),
                         passages=tuple(passages), source_ids=tuple(b.source_id for b in bundles),
                         max_chars=max_chars, used_chars=used,
                         candidate_count=len(candidates), omitted_candidates=omitted,
                         incomplete_sources=self._incomplete(bundles))

    def _search_focused(self, query: str, *, max_chars: int, limit: int,
                        source_ids: tuple[str, ...] | None) -> Selection:
        _budget(max_chars, limit)
        bundles = self._selected(source_ids)
        terms = _terms(query)
        if not terms:
            raise ValueError("Query must contain at least one searchable word or number")
        # Stopword-only queries keep their explicit terms rather than becoming
        # empty. This fallback still uses focused windows and exact offsets.
        terms = terms - _STOPWORDS or terms
        indexed = [(bundle, region, _word_spans(bundle.text[region.start:region.end], region.start))
                   for bundle in bundles for region in bundle.regions]
        frequencies = {term: 0 for term in terms}
        for _, _, spans in indexed:
            for term in terms & {span[2] for span in spans}:
                frequencies[term] += 1
        weights = {term: 1 + log((1 + len(indexed)) / (1 + frequencies[term])) for term in sorted(terms)}
        candidates = []
        width = min(_FOCUSED_WINDOW_CHARS, max_chars)
        for bundle, region, spans in indexed:
            hits = tuple(span for span in spans if span[2] in terms)
            if hits:
                score, _, _ = _focused_window(region, hits, weights, width)
                candidates.append((score, bundle, region, hits))
        candidates.sort(key=lambda item: (-item[0], item[1].source_id, item[2].start))
        passages = []
        pieces = []
        used = 0
        for _, bundle, region, hits in candidates[:limit]:
            separator = 2 if pieces else 0
            remaining = max_chars - used - separator
            if remaining <= 0:
                break
            score, start, end = _focused_window(region, hits, weights, min(width, remaining))
            ref = evidence_ref(bundle, region.region_id, start=start, end=end)
            passages.append(Passage(evidence=ref, kind=region.kind, section_path=region.section_path,
                                    score=score, clipped=start > region.start or end < region.end))
            pieces.append(ref.quotation)
            used += separator + len(ref.quotation)
        omitted = len(candidates) - len(passages)
        status = "no_match" if not candidates else (
            "truncated" if omitted or any(p.clipped for p in passages) else "complete"
        )
        return Selection(mode="targeted", status=status, text="\n\n".join(pieces),
                         passages=tuple(passages), source_ids=tuple(b.source_id for b in bundles),
                         max_chars=max_chars, used_chars=used,
                         candidate_count=len(candidates), omitted_candidates=omitted,
                         incomplete_sources=self._incomplete(bundles))
