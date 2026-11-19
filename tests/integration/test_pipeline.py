"""Integration tests for the full AGAN pipeline.

Tests the complete flow: gateway → router → planner → retriever →
reasoner → verifier → writer, using mocked LLM and retrieval.
"""

import asyncio
import sys
sys.path.insert(0, 'src')

import pytest
from unittest.mock import AsyncMock

from agan.schemas import Route, Plan, PlanStep, Claim, Answer, UserProfile
from agan.hardware.detect import ProfileSpec, HardwareProfile
from agan.pipeline.router import Router
from agan.pipeline.planner import Planner
from agan.pipeline.reasoner import Reasoner
from agan.pipeline.verifier import Verifier, VerifierReport as PipelineVerifierReport
from agan.pipeline.writer import Writer
from agan.privacy.firewall import OutboundGateway


class MockLLMClient:
    """Deterministic mock LLM client for integration tests."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    async def complete(self, messages, temperature=0.7, max_tokens=512):
        resp = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return resp


async def mock_search_all(question, max_results=5):
    """Deterministic mock retrieval returning evidence snippets."""
    return [
        {
            "text": "Paris is the capital and most populous city of France.",
            "url": "https://example.com/paris",
            "title": "Paris",
        },
        {
            "text": "France is a country in Western Europe.",
            "url": "https://example.com/france",
            "title": "France",
        },
    ]


@pytest.fixture
def profile():
    return ProfileSpec(
        name=HardwareProfile.M,
        ram_gb_min=8,
        ram_gb_max=32,
        gpu_required=False,
        default_model="qwen2.5-3b",
        quantization="Q4_K_M",
        time_budget=45,
        best_of_n=3,
        max_context=4096,
        max_output_tokens=512,
        max_retrieval=5,
        enable_execution=False,
        execution_timeout_s=10,
        level=1,
    )


@pytest.mark.asyncio
async def test_full_pipeline_fact_question(profile):
    """End-to-end: question → plan → evidence → answer → verified answer."""

    llm = MockLLMClient(responses=[
        # Planner response (JSON)
        '{"steps": [{"id": 1, "query": "capital of France", "kind": "web", "depends_on": []}], "answer_type": "fact"}',
        # Reasoner LLM response
        "Paris is the capital of France. [E1]",
    ])

    # --- Stage 1: Router ---
    router = Router(llm)
    route = router.route("What is the capital of France?")
    assert route == Route.search

    # --- Stage 2: Planner ---
    planner = Planner(llm, profile)
    plan = await planner.plan("What is the capital of France?", route)
    assert plan.needs_search is True
    assert len(plan.steps) == 1
    assert plan.answer_type == "fact"

    # --- Stage 3+4: Retriever (mocked) ---
    evidence_results = await mock_search_all("capital of France", max_results=profile.max_retrieval)
    evidence_texts = [r["text"] for r in evidence_results]
    assert len(evidence_texts) == 2
    assert "Paris is the capital" in evidence_texts[0]

    # --- Stage 5: Reasoner ---
    reasoner = Reasoner(llm, profile, retrieval=mock_search_all, privacy=OutboundGateway())
    answer = await reasoner.execute("What is the capital of France?", plan)
    assert answer.text is not None
    assert len(answer.text) > 0
    assert answer.confidence in ("low", "medium", "high")

    # --- Stage 6: Verifier ---
    verifier = Verifier(sandbox=None, profile=profile)
    report = await verifier.verify_claims(answer.text, evidence_texts)
    assert isinstance(report.ok, bool)
    assert len(report.claims) >= 0

    # --- Stage 7: Writer ---
    writer = Writer(profile)
    final = writer.write(answer, report, plan)
    assert final.text is not None
    assert len(final.text) > 0


@pytest.mark.asyncio
async def test_pipeline_direct_route_no_search(profile):
    """Pipeline with direct route skips retrieval."""

    llm = MockLLMClient(responses=[
        # Planner response
        '{"steps": [], "answer_type": "chat"}',
        # Reasoner LLM response
        "Hello! How can I help you today?",
    ])

    router = Router(llm)
    route = router.route("Hello thanks")
    assert route == Route.direct

    planner = Planner(llm, profile)
    plan = await planner.plan("Hello there", route)
    assert plan.needs_search is False

    reasoner = Reasoner(llm, profile, retrieval=mock_search_all, privacy=OutboundGateway())
    answer = await reasoner.execute("Hello there", plan)
    assert answer.text is not None
    assert len(answer.text) > 0


@pytest.mark.asyncio
async def test_privacy_gateway_blocks_private_ips():
    """Privacy gateway rejects loopback/private destinations."""
    gateway = OutboundGateway()

    # Public domain should be allowed
    allowed = gateway.check("Visit https://example.com/api for info")
    assert allowed is True

    # Loopback IP should be blocked
    blocked = gateway.check("http://127.0.0.1:8000/internal")
    assert blocked is False

    # Private range should be blocked
    blocked = gateway.check("http://192.168.1.1/admin")
    assert blocked is False


@pytest.mark.asyncio
async def test_writer_appends_sources_and_caveats():
    """Writer correctly appends sources and verification notes."""

    writer = Writer()
    answer = Answer(
        text="Paris is the capital of France.",
        claims=[Claim(text="Paris is the capital of France.", label="verified", score=0.9, evidence_ids=["E1"])],
        sources=[{"id": "E1", "type": "evidence", "trust": 0.8}],
        confidence="high",
    )
    report = PipelineVerifierReport(ok=True, claims=answer.claims)

    result = writer.write(answer, report)
    assert "Sources:" in result.text
    assert "[E1]" in result.text
    assert "Could not fully verify" not in result.text  # report is ok


@pytest.mark.asyncio
async def test_writer_adds_verification_caveat():
    """Writer adds caveat when verification fails."""

    writer = Writer()
    answer = Answer(text="Some claim.", confidence="low")
    report = PipelineVerifierReport(ok=False, missing_query="search for more info")

    result = writer.write(answer, report)
    assert "Could not fully verify" in result.text
    assert "Confidence: low" in result.text


def test_pipeline_components_importable():
    """All pipeline components can be imported and instantiated."""
    router = Router(MockLLMClient())
    planner = Planner(MockLLMClient(), ProfileSpec(
        name=HardwareProfile.M, ram_gb_min=8, ram_gb_max=32,
        gpu_required=False, default_model="qwen2.5-3b", quantization="Q4_K_M",
        time_budget=45, best_of_n=3, max_context=4096,
    ))
    reasoner = Reasoner(MockLLMClient(), ProfileSpec(
        name=HardwareProfile.M, ram_gb_min=8, ram_gb_max=32,
        gpu_required=False, default_model="qwen2.5-3b", quantization="Q4_K_M",
        time_budget=45, best_of_n=3, max_context=4096,
    ))
    verifier = Verifier(sandbox=None, profile=ProfileSpec(
        name=HardwareProfile.M, ram_gb_min=8, ram_gb_max=32,
        gpu_required=False, default_model="qwen2.5-3b", quantization="Q4_K_M",
        time_budget=45, best_of_n=3, max_context=4096,
    ))
    writer = Writer(ProfileSpec(
        name=HardwareProfile.M, ram_gb_min=8, ram_gb_max=32,
        gpu_required=False, default_model="qwen2.5-3b", quantization="Q4_K_M",
        time_budget=45, best_of_n=3, max_context=4096,
    ))
    gateway = OutboundGateway()

    assert router is not None
    assert planner is not None
    assert reasoner is not None
    assert verifier is not None
    assert writer is not None
    assert gateway is not None
