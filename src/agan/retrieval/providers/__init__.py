"""Retrieval providers for AGAN."""
import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import quote_plus, urlencode

import httpx

from agan.retrieval.fetch import extract_text, fetch_url


@dataclass
class SearchHit:
    """A single search result from a retrieval provider."""

    url: str
    title: str
    snippet: str
    score: float = 0.0
    source: str = ""
    version_tag: str | None = None
    trust: float = 0.5
    fetched_at: float = 0.0


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        pass

    def _rate_limit(self):
        pass


class SearXNGProvider(SearchProvider):
    """SearXNG instance (self-hosted or public)."""
    def __init__(self, base_url: str = "https://searx.be", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "AGAN/0.1"})

    async def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        try:
            resp = await self.client.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json", "categories": "general", "safesearch": "1"},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            hits = []
            for r in data.get("results", [])[:max_results]:
                hits.append(SearchHit(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("content", "")[:300],
                    score=r.get("score", 0.0),
                    source="searxng",
                ))
            return hits
        except Exception:
            return []

    async def close(self):
        await self.client.aclose()


class WikipediaProvider(SearchProvider):
    """Wikipedia API search."""
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "AGAN/0.1"})

    async def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        try:
            resp = await self.client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": max_results,
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            hits = []
            for r in data.get("query", {}).get("search", []):
                title = r["title"]
                url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
                hits.append(SearchHit(
                    url=url,
                    title=title,
                    snippet=r.get("snippet", "")[:300],
                    source="wikipedia",
                ))
            return hits
        except Exception:
            return []

    async def close(self):
        await self.client.aclose()


class StackExchangeProvider(SearchProvider):
    """Stack Exchange API (Stack Overflow, etc.)."""
    def __init__(self, site: str = "stackoverflow", timeout: float = 10.0, key: str | None = None):
        self.site = site
        self.timeout = timeout
        self.key = key
        self.client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "AGAN/0.1"})

    async def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        try:
            params = {
                "order": "desc",
                "sort": "relevance",
                "intitle": query,
                "site": self.site,
                "pagesize": max_results,
            }
            if self.key:
                params["key"] = self.key
            resp = await self.client.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params=params,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            hits = []
            for r in data.get("items", []):
                hits.append(SearchHit(
                    url=r.get("link", ""),
                    title=r.get("title", ""),
                    snippet=r.get("excerpt", "")[:300],
                    source=f"stackexchange:{self.site}",
                ))
            return hits
        except Exception:
            return []

    async def close(self):
        await self.client.aclose()


class GitHubProvider(SearchProvider):
    """GitHub code/documentation search (no auth, limited)."""
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "AGAN/0.1"})

    async def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        try:
            resp = await self.client.get(
                "https://api.github.com/search/code",
                params={"q": query, "per_page": max_results},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            hits = []
            for r in data.get("items", []):
                hits.append(SearchHit(
                    url=r.get("html_url", ""),
                    title=r.get("name", ""),
                    snippet=r.get("path", "")[:300],
                    source="github",
                ))
            return hits
        except Exception:
            return []

    async def close(self):
        await self.client.aclose()


class ArxivProvider(SearchProvider):
    """arXiv API search."""
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "AGAN/0.1"})

    async def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        try:
            resp = await self.client.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            if resp.status_code != 200:
                return []
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            hits = []
            for entry in root.findall("atom:entry", ns):
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                id_elem = entry.find("atom:id", ns)
                title = title_elem.text.strip() if title_elem is not None else ""
                summary = summary_elem.text.strip() if summary_elem is not None else ""
                url = id_elem.text.strip() if id_elem is not None else ""
                hits.append(SearchHit(
                    url=url,
                    title=title,
                    snippet=summary[:300],
                    source="arxiv",
                ))
            return hits
        except Exception:
            return []

    async def close(self):
        await self.client.aclose()


class DocSiteProvider(SearchProvider):
    """Documentation site search via site:domain queries to SearXNG/DDG."""
    def __init__(self, searxng: SearXNGProvider):
        self.searxng = searxng

    async def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        doc_domains = [
            "docs.python.org", "docs.djangoproject.com", "fastapi.tiangolo.com",
            "pytorch.org/docs", "tensorflow.org/api_docs", "numpy.org/doc",
            "developer.mozilla.org", "docs.aws.amazon.com", "cloud.google.com/docs",
            "kubernetes.io/docs", "docs.docker.com", "terraform.io/docs",
            "docs.ansible.com", "grafana.com/docs", "prometheus.io/docs",
            "react.dev", "vuejs.org", "svelte.dev", "nextjs.org/docs",
            "rust-lang.github.io", "doc.rust-lang.org", "golang.org/doc",
            "nodejs.org/docs", "docs.deno.com", "bun.sh/docs",
        ]
        all_hits = []
        for domain in doc_domains[:5]:
            site_query = f"site:{domain} {query}"
            hits = await self.searxng.search(site_query, max_results=3)
            for h in hits:
                h.source = f"docsite:{domain}"
            all_hits.extend(hits)
        all_hits.sort(key=lambda h: h.score, reverse=True)
        return all_hits[:max_results]


class SearchManager:
    """Manages multiple search providers and routes queries."""
    def __init__(self):
        self.providers: dict[str, SearchProvider] = {}
        self._init_default_providers()

    def _init_default_providers(self):
        self.providers["searxng"] = SearXNGProvider()
        self.providers["wikipedia"] = WikipediaProvider()
        self.providers["stackoverflow"] = StackExchangeProvider("stackoverflow")
        self.providers["github"] = GitHubProvider()
        self.providers["arxiv"] = ArxivProvider()
        self.providers["docsite"] = DocSiteProvider(self.providers["searxng"])

    async def search(
        self,
        query: str,
        max_results: int = 10,
        providers: list[str] | None = None,
    ) -> list[SearchHit]:
        """Search across multiple providers."""
        if providers is None:
            providers = ["searxng", "wikipedia", "stackoverflow", "github", "arxiv", "docsite"]

        tasks = []
        for name in providers:
            if name in self.providers:
                provider = self.providers[name]
                tasks.append(provider.search(query, max_results))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_hits = []
        for hits in results:
            if isinstance(hits, list):
                all_hits.extend(hits)

        seen = set()
        unique_hits = []
        for hit in all_hits:
            if hit.url not in seen:
                seen.add(hit.url)
                unique_hits.append(hit)

        unique_hits.sort(key=lambda h: h.score, reverse=True)
        return unique_hits[:max_results]

    async def close(self):
        for provider in self.providers.values():
            if hasattr(provider, "close"):
                await provider.close()


# Global search manager instance
_search_manager: SearchManager | None = None


def get_search_manager() -> SearchManager:
    global _search_manager
    if _search_manager is None:
        _search_manager = SearchManager()
    return _search_manager


async def close_search_manager():
    global _search_manager
    if _search_manager:
        await _search_manager.close()
        _search_manager = None


async def search_all(query: str, max_results: int = 10) -> list[dict]:
    """Search across all available retrieval providers.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: text, url, source.
    """
    manager = get_search_manager()
    hits = await manager.search(query, max_results=max_results)
    return [
        {"text": f"{h.title}: {h.snippet}", "url": h.url, "source": h.source}
        for h in hits
    ]


__all__ = [
    "ArxivProvider",
    "DocSiteProvider",
    "GitHubProvider",
    "SearXNGProvider",
    "SearchHit",
    "SearchManager",
    "SearchProvider",
    "StackExchangeProvider",
    "WikipediaProvider",
    "close_search_manager",
    "get_search_manager",
    "search_all",
]
