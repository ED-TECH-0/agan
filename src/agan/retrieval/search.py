"""Ranking: cross-encoder rerank, final score, MMR diversity (S9.5)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from agan.retrieval.providers import SearchHit


@dataclass
class RankedHit:
    """A search hit with its ranking components and final score."""

    hit: SearchHit
    score: float
    components: dict[str, float] = field(default_factory=dict)


class CrossEncoderReranker:
    """Optional cross-encoder reranker (ONNX via sentence-transformers).

    Gracefully degrades to neutral 0.5 scores when no model is available
    (offline/small-profile environments).
    """

    def __init__(self) -> None:
        self._model = None
        self._tried = False

    def _ensure(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:  # noqa: BLE001
            self._model = None

    def rerank(self, query: str, hits: list[SearchHit]) -> list[float]:
        """Return rerank scores in [0, 1] for each hit."""
        self._ensure()
        if self._model is None or not hits:
            return [0.5] * len(hits)
        pairs = [(query, h.snippet or h.title or "") for h in hits]
        try:
            raw = self._model.predict(pairs, show_progress_bar=False)
            return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]
        except Exception:  # noqa: BLE001
            return [0.5] * len(hits)


def _freshness_score(fetched_at: float, ttl_hours: float = 168.0) -> float:
    """Freshness in [0, 1], decaying with age. 1.0 = freshly fetched."""
    if fetched_at <= 0:
        return 0.5
    age_hours = (time.time() - fetched_at) / 3600.0
    return max(0.0, math.exp(-age_hours / ttl_hours))


def _version_match_score(chunk_version: str | None, known_version: str | None) -> float:
    """1.0 if versions match, 0.5 if unknown, 0.0 if mismatched."""
    if known_version is None or chunk_version is None:
        return 0.5
    return 1.0 if chunk_version == known_version else 0.0


def _mmr_select(
    scored: list[tuple[SearchHit, float, dict]],
    query_vec: np.ndarray,
    chunk_vecs: dict[str, np.ndarray],
    lambda_mmr: float = 0.7,
    top_n: int = 10,
) -> list[tuple[SearchHit, float, dict]]:
    """Maximal Marginal Relevance selection for diversity.

    score(d) = lambda * relevance(d) - (1-lambda) * max_sim(d, already_selected)
    """
    if not scored:
        return []

    selected: list[tuple[SearchHit, float, dict]] = []
    remaining = list(scored)

    # Pick the highest relevance first
    remaining.sort(key=lambda x: x[1], reverse=True)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < top_n:
        best_idx = -1
        best_mmr = -1.0
        for i, (hit, rel, comps) in enumerate(remaining):
            # max similarity to already-selected
            max_sim = 0.0
            h_vec = chunk_vecs.get(hit.url)
            if h_vec is not None:
                for sel_hit, _, _ in selected:
                    s_vec = chunk_vecs.get(sel_hit.url)
                    if s_vec is not None:
                        sim = float(np.dot(h_vec, s_vec))  # embeddings are normalized
                        max_sim = max(max_sim, sim)
            mmr = lambda_mmr * rel - (1.0 - lambda_mmr) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))

    return selected


async def rank_results(
    query: str,
    hits: list[SearchHit],
    *,
    top_n: int = 10,
    enable_rerank: bool = True,
    known_version: str | None = None,
    use_mmr: bool = True,
) -> list[RankedHit]:
    """Rank search hits using the S9.5 final score formula.

    score = 0.6 * rerank + 0.2 * trust + 0.1 * freshness + 0.1 * version_match
    Optionally applies MMR (lambda=0.7) for diversity.
    """
    if not hits:
        return []

    reranker = CrossEncoderReranker() if enable_rerank else None
    rerank_scores = reranker.rerank(query, hits) if reranker else [0.5] * len(hits)

    ranked: list[tuple[SearchHit, float, dict]] = []
    for hit, rscore in zip(hits, rerank_scores):
        trust = hit.trust
        freshness = _freshness_score(hit.fetched_at)
        version_match = _version_match_score(getattr(hit, "version_tag", None), known_version)
        final = 0.6 * rscore + 0.2 * trust + 0.1 * freshness + 0.1 * version_match
        components = {
            "rerank": rscore,
            "trust": trust,
            "freshness": freshness,
            "version_match": version_match,
        }
        ranked.append((hit, final, components))

    ranked.sort(key=lambda x: x[1], reverse=True)

    if use_mmr and len(ranked) > 1:
        # Build chunk vectors from snippets for MMR diversity
        from agan.retrieval.index import embed_texts

        texts = [h.snippet or h.title or "" for h, _, _ in ranked]
        vecs = await embed_texts(texts)
        chunk_vecs = {hit.url: v for (hit, _, _), v in zip(ranked, vecs)}
        ranked = _mmr_select(ranked, vecs[0] if vecs else np.zeros(0), chunk_vecs, top_n=top_n)
    else:
        ranked = ranked[:top_n]

    return [RankedHit(hit=h, score=s, components=c) for h, s, c in ranked]
