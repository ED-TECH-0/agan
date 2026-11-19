"""Tests for M2 Retrieval (S9.5): rank, compress, pipeline, SSRF, robots."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agan.retrieval.compress import compress_evidence
from agan.retrieval.fetch import _check_robots, _check_ssrf, _get_robots_rules, _is_private_ip
from agan.retrieval.pipeline import EVIDENCE_BUDGET, RetrieveResult, retrieve
from agan.retrieval.providers import SearchHit
from agan.retrieval.search import CrossEncoderReranker, rank_results
from agan.schemas import Chunk

# ---------------------------------------------------------------- compress_evidence

def _mk_hit(url: str, title: str, text: str, trust: float = 0.5) -> SearchHit:
    return SearchHit(url=url, title=title, snippet=text[:200], source="web", trust=trust, fetched_at=time.time())


def test_compress_budgets_s_m_l():
    hits = [
        _mk_hit("https://example.com/1", "One", "alpha beta gamma delta epsilon " * 40, trust=0.9),
        _mk_hit("https://example.com/2", "Two", "zeta eta theta iota kappa lambda " * 40, trust=0.6),
        _mk_hit("https://example.com/3", "Three", "mu nu xi omicron pi rho sigma " * 40, trust=0.3),
    ]
    for profile, budget in (("S", 800), ("M", 2500), ("L", 4000)):
        result = compress_evidence(hits, "alpha beta gamma", budget_tokens=budget)
        assert result.budget == budget
        assert result.total_tokens <= budget + 50  # small slack for sentence boundaries
        assert [e.id for e in result.items] == [f"E{i+1}" for i in range(len(result.items))]
        assert all(e.text for e in result.items)
        assert isinstance(result.truncated, bool)


def test_compress_keeps_query_relevant_sentences():
    hits = [
        _mk_hit("https://example.com/a", "A", "The quick brown fox jumps over the lazy dog. Unrelated sentence about cooking pasta. " * 5),
    ]
    result = compress_evidence(hits, "quick brown fox", budget_tokens=200)
    assert result.items
    # Should contain the query-relevant sentence
    joined = " ".join(e.text for e in result.items)
    assert "quick brown fox" in joined


# ---------------------------------------------------------------- rank_results

def _mk_chunk(url: str, trust: float = 0.5, fetched_at: float | None = None, version_tag: str | None = None) -> Chunk:
    return Chunk(
        id=url,
        url=url,
        title=url.split("/")[-1],
        text="sample text about retrieval systems",
        fetched_at=fetched_at if fetched_at is not None else time.time(),
        trust=trust,
        version_tag=version_tag,
        tainted=True,
    )


@pytest.mark.asyncio
async def test_rank_scoring_formula():
    hits = [
        SearchHit(url="https://example.com/high", title="High", snippet="retrieval", trust=1.0, fetched_at=time.time()),
        SearchHit(url="https://example.com/low", title="Low", snippet="retrieval", trust=0.0, fetched_at=time.time()),
    ]
    ranked = await rank_results("retrieval", hits, top_n=5, enable_rerank=False, use_mmr=False)
    assert len(ranked) == 2
    # Higher trust should rank higher (same rerank=0.5, same freshness)
    assert ranked[0].score >= ranked[1].score
    assert ranked[0].hit.url == "https://example.com/high"
    # Components present
    assert set(ranked[0].components) == {"rerank", "trust", "freshness", "version_match"}


@pytest.mark.asyncio
async def test_rank_version_match():
    now = time.time()
    hits = [
        SearchHit(url="https://example.com/v1", title="V1", snippet="retrieval", trust=0.5, fetched_at=now, version_tag="3.0"),
        SearchHit(url="https://example.com/v2", title="V2", snippet="retrieval", trust=0.5, fetched_at=now, version_tag="3.1"),
    ]
    ranked = await rank_results("retrieval", hits, top_n=5, enable_rerank=False, known_version="3.1", use_mmr=False)
    assert ranked[0].hit.url == "https://example.com/v2"
    assert ranked[0].components["version_match"] == 1.0


@pytest.mark.asyncio
async def test_rank_empty():
    ranked = await rank_results("retrieval", [], top_n=5, enable_rerank=False, use_mmr=False)
    assert ranked == []


# ---------------------------------------------------------------- CrossEncoderReranker degradation

@pytest.mark.asyncio
async def test_reranker_degrades_when_model_unavailable():
    reranker = CrossEncoderReranker()
    hits = [SearchHit(url="https://example.com/1", title="T", snippet="s")]

    def _fail_ensure(self) -> None:
        self._tried = True
        self._model = None

    with patch.object(CrossEncoderReranker, "_ensure", _fail_ensure):
        scores = reranker.rerank("query", hits)
    assert scores == [0.5]


# ---------------------------------------------------------------- retrieve() pipeline

class _FakeManager:
    def __init__(self, hits):
        self.hits = hits

    async def search(self, query, max_results=10, providers=None):
        return list(self.hits)


@pytest.mark.asyncio
async def test_retrieve_web_fallback_path():
    """When local index is empty, pipeline should search providers, fetch, index, rank, compress."""
    provider_hits = [
        SearchHit(url="https://example.com/page1", title="Page 1", snippet="retrieval systems overview", source="web", trust=0.7, fetched_at=time.time()),
    ]

    fake_chunk = Chunk(
        id="chunk-1",
        url="https://example.com/page1",
        title="Page 1",
        text="retrieval systems overview with detailed content about hybrid search and ranking.",
        fetched_at=time.time(),
        trust=0.7,
        version_tag=None,
        tainted=True,
    )

    with patch("agan.retrieval.pipeline.embed_texts", new=AsyncMock(return_value=[])), \
         patch("agan.retrieval.pipeline.hybrid_search", new=AsyncMock(return_value=[])), \
         patch("agan.retrieval.pipeline.get_chunks_by_ids", new=AsyncMock(return_value=[])), \
         patch("agan.retrieval.pipeline.get_search_manager", return_value=_FakeManager(provider_hits)), \
         patch("agan.retrieval.pipeline.fetch_and_extract", new=AsyncMock(return_value=("retrieval systems overview with detailed content about hybrid search and ranking.", MagicMock(error="")))), \
         patch("agan.retrieval.pipeline.chunk_text", return_value=[fake_chunk]), \
         patch("agan.retrieval.pipeline.add_chunks", new=AsyncMock()):
        result = await retrieve("retrieval systems", profile="M", top_n=5, enable_rerank=False, use_mmr=False)

    assert isinstance(result, RetrieveResult)
    assert result.query == "retrieval systems"
    assert result.budget == 2500  # M profile
    assert result.n_fetched == 1
    assert result.n_ranked >= 1
    assert result.evidence
    assert result.evidence[0]["id"] == "E1"
    assert result.total_tokens <= result.budget + 50


@pytest.mark.asyncio
async def test_retrieve_local_index_path():
    """When local index has good chunks, pipeline should not fetch from web."""
    now = time.time()
    chunk = Chunk(
        id="chunk-local",
        url="https://example.com/local",
        title="Local",
        text="retrieval systems content",
        fetched_at=now,
        trust=0.9,
        version_tag=None,
        tainted=True,
    )

    with patch("agan.retrieval.pipeline.embed_texts", new=AsyncMock(return_value=[])), \
         patch("agan.retrieval.pipeline.hybrid_search", new=AsyncMock(return_value=[("chunk-local", 1.0)])), \
         patch("agan.retrieval.pipeline.get_chunks_by_ids", new=AsyncMock(return_value=[chunk])), \
         patch("agan.retrieval.pipeline.get_search_manager") as mgr_mock:
        result = await retrieve("retrieval systems", profile="S", top_n=5, enable_rerank=False, use_mmr=False)

    mgr_mock.assert_not_called()  # no web fallback
    assert result.budget == 800  # S profile
    assert result.n_fetched == 0
    assert result.n_ranked >= 1


# ---------------------------------------------------------------- SSRF / robots

def test_ssrf_blocks_private_ips():
    assert _check_ssrf("http://127.0.0.1/") is not None
    assert _check_ssrf("http://10.0.0.1/") is not None
    assert _check_ssrf("http://192.168.1.1/") is not None
    assert _check_ssrf("http://169.254.169.254/") is not None  # AWS metadata
    assert _check_ssrf("https://example.com/") is None
    assert _check_ssrf("http://example.com/") is None


def test_is_private_ip():
    assert _is_private_ip("127.0.0.1")
    assert _is_private_ip("10.0.0.1")
    assert _is_private_ip("192.168.1.1")
    assert _is_private_ip("169.254.169.254")
    assert not _is_private_ip("93.184.216.34")  # example.com
    assert not _is_private_ip("1.1.1.1")


@pytest.mark.asyncio
async def test_robots_disallowed_blocked():
    rules = {"allow": ["/"], "disallow": ["/private"]}
    assert _check_robots("https://example.com/public/page", rules) is True
    assert _check_robots("https://example.com/private/data", rules) is False


@pytest.mark.asyncio
async def test_get_robots_rules_returns_dict():
    rules = await _get_robots_rules("example.com")
    assert isinstance(rules, dict)
    assert "allow" in rules and "disallow" in rules


# ---------------------------------------------------------------- budgets constant

def test_evidence_budgets_constant():
    assert EVIDENCE_BUDGET == {"S": 800, "M": 2500, "L": 4000}
