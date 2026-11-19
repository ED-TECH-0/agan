"""Evidence compression: keep most query-relevant sentences within a token budget (S9.5)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agan.retrieval.providers import SearchHit


@dataclass
class Evidence:
    """A single piece of compressed evidence."""

    id: str  # E1, E2, ...
    url: str
    title: str
    text: str
    source: str
    trust: float


@dataclass
class CompressedEvidence:
    items: list[Evidence] = field(default_factory=list)
    total_tokens: int = 0
    budget: int = 0
    truncated: bool = False


def _token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation + whitespace."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _query_overlap(sentence: str, query_terms: set[str]) -> int:
    s = sentence.lower()
    return sum(1 for t in query_terms if t in s)


def compress_evidence(
    hits: list[SearchHit],
    query: str,
    *,
    budget_tokens: int,
    max_per_source: int = 3,
) -> CompressedEvidence:
    """Select and compress the most query-relevant sentences within budget.

    - Scores sentences by query-term overlap.
    - Picks top sentences per source (max_per_source) until budget is exhausted.
    - Assigns E1..En ids.
    """
    query_terms = {t for t in re.split(r"\W+", query.lower()) if len(t) > 2}
    items: list[Evidence] = []
    used = 0
    per_source: dict[str, int] = {}

    # Flatten all sentences across hits with their score
    candidates: list[tuple[int, SearchHit, str]] = []
    for hit in hits:
        for sent in _split_sentences(hit.snippet or hit.title or ""):
            score = _query_overlap(sent, query_terms)
            candidates.append((score, hit, sent))

    # Sort by score desc, then by hit order
    candidates.sort(key=lambda c: (-c[0], c[1].url))

    ev_idx = 1
    for score, hit, sent in candidates:
        if used >= budget_tokens:
            break
        domain = hit.url.split("/")[2] if "://" in hit.url else hit.source
        if per_source.get(domain, 0) >= max_per_source:
            continue
        toks = _token_count(sent)
        if used + toks > budget_tokens and used > 0:
            # Try to fit smaller sentences later; skip this one if it would bust budget
            continue
        items.append(
            Evidence(
                id=f"E{ev_idx}",
                url=hit.url,
                title=hit.title,
                text=sent,
                source=hit.source,
                trust=hit.trust,
            )
        )
        used += toks
        per_source[domain] = per_source.get(domain, 0) + 1
        ev_idx += 1

    total = sum(_token_count(i.text) for i in items)
    return CompressedEvidence(items=items, total_tokens=total, budget=budget_tokens, truncated=used >= budget_tokens and len(candidates) > len(items))
