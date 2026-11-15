"""
LLM Client — supports Ollama native API and llama.cpp OpenAI-compatible API
"""
import asyncio
import httpx
from dataclasses import dataclass
from typing import Optional, TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:11434"  # Ollama default
    model: str = "qwen2.5:7b-instruct-q4_k_m"
    timeout: float = 180.0
    temperature: float = 0.3
    max_tokens: int = 2048
    backend: str = "ollama"  # "ollama" or "llamacpp"

class LLMClient:
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.client = httpx.AsyncClient(base_url=self.config.base_url, timeout=self.config.timeout)
    
    async def chat(self, messages: list[dict], **kwargs) -> str:
        """Raw chat completion"""
        if self.config.backend == "ollama":
            return await self._chat_ollama(messages, **kwargs)
        else:
            return await self._chat_llamacpp(messages, **kwargs)
    
    async def _chat_ollama(self, messages: list[dict], **kwargs) -> str:
        body = {
            "model": self.config.model,
            "messages": messages,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
            "stream": False,
        }
        if "stop" in kwargs:
            body["options"]["stop"] = kwargs["stop"]
        
        response = await self.client.post("/api/chat", json=body)
        response.raise_for_status()
        return response.json()["message"]["content"]
    
    async def _chat_llamacpp(self, messages: list[dict], **kwargs) -> str:
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False,
        }
        if "stop" in kwargs:
            body["stop"] = kwargs["stop"]
        
        response = await self.client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    async def chat_json(self, messages: list[dict], model_cls: Type[T], **kwargs) -> T:
        """Chat with structured JSON output (uses prompt-based JSON for compatibility)"""
        # Add JSON schema instruction to the last user message
        schema = model_cls.model_json_schema()
        schema_prompt = f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema}"
        
        if messages and messages[-1]["role"] == "user":
            messages = messages.copy()
            messages[-1] = messages[-1].copy()
            messages[-1]["content"] += schema_prompt
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                content = await self.chat(messages, temperature=0.1, **kwargs)
                # Try to parse JSON from response
                import json
                # Extract JSON if wrapped in markdown
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                return model_cls.model_validate_json(content)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                # Retry with error feedback
                messages = messages.copy()
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Invalid JSON. Error: {e}. Please output valid JSON only."})
        raise ValueError("Failed to get valid JSON after retries")
    
    async def close(self):
        await self.client.aclose()

class ModelManager:
    """Manages model lifecycle and configuration"""
    def __init__(self):
        self.clients: dict[str, LLMClient] = {}
    
    def get_client(self, profile: str = "standard") -> LLMClient:
        if profile in self.clients:
            return self.clients[profile]
        
        model_map = {
            "nano": "qwen2.5:1.5b-instruct-q4_k_m",
            "lite": "qwen2.5:3b-instruct-q4_k_m",
            "standard": "qwen2.5:7b-instruct-q4_k_m",
            "pro": "qwen2.5:14b-instruct-q4_k_m",
            "max": "qwen2.5:32b-instruct-q4_k_m",
        }
        
        config = LLMConfig(model=model_map.get(profile, model_map["standard"]), backend="ollama")
        client = LLMClient(config)
        self.clients[profile] = client
        return client
    
    async def close_all(self):
        for client in self.clients.values():
            await client.close()
        self.clients.clear()

if __name__ == "__main__":
    async def test():
        mgr = ModelManager()
        client = mgr.get_client("lite")
        try:
            resp = await client.chat([{"role": "user", "content": "Say hello in one sentence"}])
            print(f"Response: {resp}")
        finally:
            await mgr.close_all()
    
    asyncio.run(test())