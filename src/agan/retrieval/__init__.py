"""AGAN retrieval package: providers, fetch, extract, index, search, compress."""

from .compress import compress_evidence
from .fetch import FetchResult, extract_text, fetch_and_extract, fetch_url
from .index import (
    add_chunks,
    embed_texts,
    get_chunks_by_ids,
    get_page_info,
    hybrid_search,
    keyword_search,
    prune_old_cache,
    update_page_trust,
    vector_search,
)
from .pipeline import EVIDENCE_BUDGET, RetrieveResult, retrieve
from .providers import (
    ArxivProvider,
    DocSiteProvider,
    GitHubProvider,
    SearchHit,
    SearchManager,
    SearchProvider,
    SearXNGProvider,
    StackExchangeProvider,
    WikipediaProvider,
    close_search_manager,
    get_search_manager,
    search_all,
)
from .search import CrossEncoderReranker, RankedHit, rank_results

__all__ = [
    "EVIDENCE_BUDGET",
    "ArxivProvider",
    "CrossEncoderReranker",
    "DocSiteProvider",
    "FetchResult",
    "GitHubProvider",
    "RankedHit",
    "RetrieveResult",
    "SearXNGProvider",
    "SearchHit",
    "SearchManager",
    "SearchProvider",
    "StackExchangeProvider",
    "WikipediaProvider",
    "add_chunks",
    "close_search_manager",
    "compress_evidence",
    "embed_texts",
    "extract_text",
    "fetch_and_extract",
    "fetch_url",
    "get_chunks_by_ids",
    "get_page_info",
    "get_search_manager",
    "hybrid_search",
    "keyword_search",
    "prune_old_cache",
    "rank_results",
    "retrieve",
    "search_all",
    "update_page_trust",
    "vector_search",
]
