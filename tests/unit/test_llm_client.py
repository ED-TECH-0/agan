"""Tests for the Ollama LLM client (S9.1).

Deterministic httpx.MockTransport tests run without a live Ollama server, so
they are safe in CI. A separate integration test exercises a real server when
one is reachable.
"""

import json
from collections.abc import Callable

import httpx
import pytest

from agan.llm.client import LLMClient
from agan.schemas import Plan


def _plan_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": content}}
            ]
        },
    )


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> LLMClient:
    transport = httpx.MockTransport(handler)
    client = LLMClient(base_url="http://localhost:11434/v1", model="qwen2.5-3b:latest")
    client._client = httpx.AsyncClient(  # type: ignore[assignment]
        transport=transport, base_url="http://localhost:11434/v1"
    )
    return client


PLAN_JSON = json.dumps(
    {
        "needs_search": False,
        "steps": [],
        "answer_type": "fact",
        "skill_ids": [],
        "lesson_ids": [],
    }
)


@pytest.mark.asyncio
async def test_chat_json_plan_schema_validates_fifty_times():
    """S9.1 acceptance: 50 chat_json calls on the Plan schema all validate."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _plan_response(PLAN_JSON)

    client = _make_client(handler)
    try:
        for _ in range(50):
            plan = await client.chat_json(
                [{"role": "user", "content": "make a plan"}], Plan
            )
            assert isinstance(plan, Plan)
            assert plan.answer_type == "fact"
            assert isinstance(plan.steps, list)
        assert calls["n"] == 50
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_json_parse_and_repair():
    """Invalid first output is repaired on the second attempt."""
    state = {"attempt": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempt"] += 1
        if state["attempt"] == 1:
            return _plan_response("this is definitely not json {{{")
        return _plan_response(PLAN_JSON)

    client = _make_client(handler)
    try:
        plan = await client.chat_json(
            [{"role": "user", "content": "make a plan"}], Plan, max_repair=2
        )
        assert isinstance(plan, Plan)
        assert state["attempt"] == 2  # repaired once
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_json_recovers_when_server_comes_back():
    """Killed server mid-call: connection retries then success (no crash)."""
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            raise httpx.ConnectError("server killed mid-call")
        return _plan_response(PLAN_JSON)

    client = _make_client(handler)
    try:
        plan = await client.chat_json(
            [{"role": "user", "content": "make a plan"}], Plan, connect_retries=3
        )
        assert isinstance(plan, Plan)
        assert state["calls"] == 2  # recovered on retry
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_json_fallback_when_server_stays_down():
    """Permanently unreachable server: deterministic fallback, no crash."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("server gone")

    client = _make_client(handler)
    try:
        plan = await client.chat_json(
            [{"role": "user", "content": "make a plan"}], Plan, connect_retries=2
        )
        assert isinstance(plan, Plan)
        assert plan.answer_type == "fact"
        assert plan.steps == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_complete_backward_compat():
    """legacy complete() still returns raw text."""
    def handler(request: httpx.Request) -> httpx.Response:
        return _plan_response("hello world")

    client = _make_client(handler)
    try:
        text = await client.complete([{"role": "user", "content": "hi"}])
        assert text == "hello world"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_json_real_ollama():
    """End-to-end against a live Ollama server (skipped when none is up)."""
    import socket

    try:
        probe = socket.create_connection(("localhost", 11434), timeout=2)
        probe.close()
    except OSError:
        pytest.skip("no Ollama server on localhost:11434")

    client = LLMClient(model="qwen2.5-3b:latest")
    try:
        plan = await client.chat_json(
            [{"role": "user", "content": "Plan how to summarize a research paper."}],
            Plan,
        )
        assert isinstance(plan, Plan)
        assert plan.answer_type in {
            "fact", "howto", "code", "compare", "design", "chat",
        }
        assert isinstance(plan.steps, list)
        assert isinstance(plan.needs_search, bool)
    finally:
        await client.close()
