from typing import Optional
import httpx
from agan.schemas import Answer, Plan, Route


class LLMClient:
    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "qwen2.5:3b-instruct-q4_k_m", api_key: str = "ollama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0, headers={"Authorization": f"Bearer {api_key}"})

    async def complete(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 2048, stream: bool = False) -> str:
        resp = await self.client.post("/chat/completions", json={
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        })
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def close(self):
        await self.client.aclose()


async def route_prompt(client: LLMClient, question: str) -> Route:
    prompt = f"""Classify this question into one of three routes:
- direct: Simple chat, greetings, opinions, creative writing
- search: Factual questions needing current info, how-to, definitions
- deep: Complex multi-step reasoning, coding tasks, analysis, comparison

Question: {question}

Respond with ONLY: direct, search, or deep"""
    resp = await client.complete([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10)
    text = resp.strip().lower()
    if "deep" in text:
        return Route.deep
    elif "search" in text:
        return Route.search
    return Route.direct


class Router:
    def __init__(self, client: LLMClient):
        self.client = client

    async def route(self, question: str) -> Route:
        return await route_prompt(self.client, question)