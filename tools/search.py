"""Web search tool"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchResult:
    query: str
    results: list[dict]
    bytes_used: int

async def search_web(query: str, max_results: int = 10) -> SearchResult:
    # Placeholder: integrate with SearXNG
    return SearchResult(query=query, results=[], bytes_used=0)
