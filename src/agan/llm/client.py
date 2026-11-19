"""LLM client for the local-first Ollama backend.

Talks to Ollama's OpenAI-compatible endpoint at ``http://localhost:11434/v1``.
Ollama's OpenAI compatibility layer does not reliably honor ``response_format``,
so structured output is produced with a strict *parse-and-repair* loop: ask for
JSON, try to validate it against a pydantic schema, and on failure ask the
model to re-emit *only* valid JSON matching the schema. If repair fails too, a
deterministic fallback value is returned so the pipeline never blocks on a
misbehaving model.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Type, TypeVar  # noqa: UP035

import httpx
from pydantic import BaseModel, ValidationError

from agan.schemas import Route

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "qwen2.5-3b:latest"


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = DEFAULT_MODEL,
        api_key: str = "ollama",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        stream: bool = False,
        schema: dict | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if schema is not None:
            payload["schema"] = schema
        if stop is not None:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed

        resp = await self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str:
        return await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

    async def chat_json(
        self,
        messages: list[dict],
        model_cls: Type[T],  # noqa: UP006
        max_repair: int = 2,
        connect_retries: int = 3,
        **kw: Any,
    ) -> T:
        """Ask for JSON validated against ``model_cls`` with parse-and-repair.

        Returns a validated pydantic instance. On any parse/validation failure
        the model is asked once more to emit *only* valid JSON matching the
        schema. If the Ollama server is unreachable (e.g. it was killed
        mid-call) the client retries the connection a few times and then
        returns a deterministic fallback so the pipeline never crashes.
        """
        schema = model_cls.model_json_schema()
        schema_hint = json.dumps(schema, sort_keys=True)
        last_error: Exception | None = None

        for attempt in range(max_repair + 1):
            content = await self._chat_resilient(messages, connect_retries, **kw)
            if content is None:
                # Server unreachable even after connect retries: recover with
                # a deterministic fallback instead of crashing.
                return self._fallback(model_cls, last_error)
            try:
                obj = self._extract_json(content)
                return model_cls.model_validate(obj)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < max_repair:
                    repair_user = (
                        "The previous output was invalid. Reply with ONLY valid "
                        f"JSON matching this schema: {schema_hint}"
                    )
                    messages = list(messages) + [
                        {"role": "assistant", "content": content or ""},
                        {"role": "user", "content": repair_user},
                    ]

        return self._fallback(model_cls, last_error)

    async def _chat_resilient(
        self,
        messages: list[dict],
        connect_retries: int,
        **kw: Any,
    ) -> str | None:
        """Run ``chat``, retrying on connection errors; return ``None`` if the
        server stays unreachable after all retries."""
        delay = 0.5
        for i in range(connect_retries):
            try:
                return await self.chat(messages, **kw)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if i < connect_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
        return None

    @staticmethod
    def _extract_json(content: str) -> Any:
        text = content.strip()
        # Strip markdown fences if present.
        if text.startswith("```"):
            text = text.split("```", 2)
            text = text[1] if len(text) > 1 else ""
            if text.startswith("json"):
                text = text.removeprefix("json")
        text = text.strip()

        # Try the whole string first.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fall back to the first balanced {...} block.
        start = text.find("{")
        if start == -1:
            raise ValueError("no JSON object found in model output")
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
        raise ValueError("no balanced JSON object found in model output")

    @staticmethod
    def _fallback(
        model_cls: Type[T],  # noqa: UP006
        last_error: Exception | None,
    ) -> T:
        """Deterministic fallback so the pipeline never blocks."""
        try:
            return model_cls()
        except Exception:  # noqa: BLE001  # last resort: swallow any constructor failure
            # Last resort: build with empty/None defaults for every field.
            kwargs: dict[str, Any] = {}
            for name, field in model_cls.model_fields.items():
                if field.is_required():
                    annotation = field.annotation
                    if annotation is str or (getattr(annotation, "__origin__", None) is str):
                        kwargs[name] = ""
                    elif annotation is int or getattr(annotation, "__origin__", None) is int:
                        kwargs[name] = 0
                    elif annotation is bool or getattr(annotation, "__origin__", None) is bool:
                        kwargs[name] = False
                    elif annotation is float or getattr(annotation, "__origin__", None) is float:
                        kwargs[name] = 0.0
                    elif annotation is list or getattr(annotation, "__origin__", None) is list:
                        kwargs[name] = []
                    elif annotation is dict or getattr(annotation, "__origin__", None) is dict:
                        kwargs[name] = {}
                    else:
                        kwargs[name] = None
            return model_cls(**kwargs)


async def route_prompt(client: LLMClient, question: str) -> Route:
    prompt = f"""Classify this question into one of three routes:
- direct: Simple chat, greetings, opinions, creative writing
- search: Factual questions needing current info, how-to, definitions
- deep: Complex multi-step reasoning, coding tasks, analysis, comparison

Question: {question}

Respond with ONLY: direct, search, or deep"""
    resp = await client.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10)
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
