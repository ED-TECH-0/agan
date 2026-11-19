"""Retrieval providers for AGAN."""

from __future__ import annotations

import asyncio
from typing import List


async def search_all(query: str, max_results: int = 10) -> List[dict]:
    """Search across all available retrieval providers.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: text, url, source.
    """
    from agan.retrieval import get_search_manager

    manager = get_search_manager()
    hits = await manager.search(query, max_results=max_results)
    # Return as list of dicts for compatibility with reasoner
    return [
        {"text": f"{h.title}: {h.snippet}", "url": h.url, "source": h.source}
        for h in hits
    ]


__all__ = ["search_all"]