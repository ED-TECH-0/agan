"""Unified retrieval pipeline: search -> fetch -> extract -> index -> rank -> compress (S9.5)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from agan.retrieval.chunk import chunk_text
from agan.retrieval.compress import compress_evidence
from agan.retrieval.fetch import fetch_and_extract
from agan.retrieval.index import (
    add_chunks,
    embed_texts,
    get_chunks_by_ids,
    hybrid_search,
)
from agan.retrieval.providers import SearchHit, get_search_manager
from agan.retrieval.search import rank_results

# Evidence token budgets by hardware profile (S9.5)
EVIDENCE_BUDGET = {"S": 800, "M": 2500, "L": 4000}

# Domain -> default trust score (higher = more authoritative)
DOMAIN_TRUST = {
    "wikipedia.org": 0.8,
    "en.wikipedia.org": 0.8,
    "stackoverflow.com": 0.7,
    "stackexchange.com": 0.7,
    "docs.python.org": 0.85,
    "developer.mozilla.org": 0.8,
    "arxiv.org": 0.75,
    "export.arxiv.org": 0.75,
    "github.com": 0.6,
}


def _domain_trust(url: str) -> float:
    from urllib.parse import urlparse

    domain = urlparse(url).netloc.lower()
    for d, t in DOMAIN_TRUST.items():
        if domain == d or domain.endswith("." + d):
            return t
    return 0.5


@dataclass
class RetrieveResult:
    query: str
    evidence: list[dict]  # E1..En with text, url, source, trust
    total_tokens: int
    budget: int
    truncated: bool
    n_fetched: int
    n_ranked: int


async def _fetch_candidates(hits: list[SearchHit], max_concurrent: int = 6) -> list[SearchHit]:
    """Fetch + extract + chunk + index candidate pages. Returns hits with fetched_at/trust set."""
    import asyncio

    sem = asyncio.Semaphore(max_concurrent)

    async def _one(hit: SearchHit) -> SearchHit:
        async with sem:
            text, result = await fetch_and_extract(hit.url)
            hit.fetched_at = time.time()
            hit.trust = _domain_trust(hit.url)
            if not result.error and text:
                chunks = chunk_text(
                    text,
                    url=hit.url,
                    fetched_at=hit.fetched_at,
                    trust=hit.trust,
                )
                if chunks:
                    # Set title on chunks from hit title
                    for c in chunks:
                        c.title = hit.title or ""
                    vecs = await embed_texts([c.text for c in chunks])
                    await add_chunks(chunks, vecs)
            return hit

    return await asyncio.gather(*[_one(h) for h in hits])


async def retrieve(
    query: str,
    *,
    profile: str = "M",
    known_version: str | None = None,
    top_n: int = 10,
    providers: list[str] | None = None,
    enable_rerank: bool = True,
    use_mmr: bool = True,
) -> RetrieveResult:
    """End-to-end retrieval for a query.

    1. Hybrid search the local index.
    2. If local coverage is weak/stale, search web providers, fetch, extract, chunk, index.
    3. Rank with cross-encoder + trust + freshness + version_match; MMR for diversity.
    4. Compress to the profile's evidence token budget (E1..En).
    """
    budget = EVIDENCE_BUDGET.get(profile, EVIDENCE_BUDGET["M"])

    # --- Step 1: local hybrid search ---
    query_vecs = await embed_texts([query])
    query_vec = query_vecs[0] if query_vecs else None
    local_ids = await hybrid_search(query, query_vec, limit=top_n * 2)
    chunks = await get_chunks_by_ids([cid for cid, _ in local_ids])

    # --- Step 2: web fallback if local is weak/stale ---
    n_fetched = 0
    hits: list[SearchHit] = []
    if not chunks or max((c.trust for c in chunks), default=0) < 0.4:
        manager = get_search_manager()
        provider_hits = await manager.search(query, max_results=top_n * 2, providers=providers)
        hits = provider_hits
        fetched = await _fetch_candidates(hits)
        n_fetched = len(fetched)
        # Re-run local search to include freshly indexed chunks
        local_ids = await hybrid_search(query, query_vec, limit=top_n * 2)
        chunks = await get_chunks_by_ids([cid for cid, _ in local_ids])

    # Build SearchHit list from indexed chunks for ranking
    if chunks and not hits:
        hits = [
            SearchHit(
                url=c.url,
                title=c.title,
                snippet=c.text[:400],
                source="index",
                trust=c.trust,
                fetched_at=c.fetched_at,
                version_tag=c.version_tag,
            )
            for c in chunks
        ]

    # --- Step 3: rank ---
    ranked = await rank_results(
        query,
        hits,
        top_n=top_n,
        enable_rerank=enable_rerank,
        known_version=known_version,
        use_mmr=use_mmr,
    )
    n_ranked = len(ranked)

    # --- Step 4: compress to evidence budget ---
    compressed = compress_evidence(
        [r.hit for r in ranked],
        query,
        budget_tokens=budget,
    )

    evidence = [
        {
            "id": e.id,
            "text": e.text,
            "url": e.url,
            "title": e.title,
            "source": e.source,
            "trust": e.trust,
        }
        for e in compressed.items
    ]

    return RetrieveResult(
        query=query,
        evidence=evidence,
        total_tokens=compressed.total_tokens,
        budget=compressed.budget,
        truncated=compressed.truncated,
        n_fetched=n_fetched,
        n_ranked=n_ranked,
    )
