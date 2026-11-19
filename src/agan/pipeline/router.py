from typing import Optional
from agan.schemas import Route
from agan.llm.client import LLMClient


class Router:
    """Router decides the processing path: direct, search, or deep research."""
    
    DIRECT_KEYWORDS = {
        "greetings": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you"],
        "simple_rewrite": ["rewrite", "rephrase", "summarize", "explain"],
        "simple_arithmetic": ["what is", "calculate", "compute", "+", "-", "*", "/"],
    }
    
    DEEP_KEYWORDS = {
        "freshness": ["latest", "today", "current", "new", "2024", "2025", "price", "news", "stock", "version"],
        "comparisons": ["compare", "vs", "versus", "difference between", "better than"],
        "multi_part": ["how do I", "steps for", "guide to", "tutorial"],
    }
    
    def __init__(self, client: LLMClient):
        self.client = client
    
    def route(self, question: str) -> Route:
        """Route a question to the appropriate processing path.
        
        Returns:
            Route.direct, Route.search, or Route.deep
        """
        q_lower = question.lower()
        
        # Check for deep research signals first
        deep_indicators = 0
        for keyword in self.DEEP_KEYWORDS["freshness"]:
            if keyword in q_lower:
                deep_indicators += 1
        for keyword in self.DEEP_KEYWORDS["comparisons"]:
            if keyword in q_lower:
                deep_indicators += 1
        if any(kw in q_lower for kw in self.DEEP_KEYWORDS["multi_part"]):
            deep_indicators += 1
        
        if deep_indicators >= 2:
            return Route.deep
        
        # Check for direct answer signals
        direct_indicators = 0
        for keyword in self.DIRECT_KEYWORDS["greetings"]:
            if keyword in q_lower:
                direct_indicators += 1
        for keyword in self.DIRECT_KEYWORDS["simple_rewrite"]:
            if keyword in q_lower:
                direct_indicators += 1
        for keyword in self.DIRECT_KEYWORDS["simple_arithmetic"]:
            if keyword in q_lower:
                direct_indicators += 1
        
        if direct_indicators >= 2:
            return Route.direct
        
        # Default to search for uncertain cases
        return Route.search
    
    async def route_with_llm(self, question: str) -> Route:
        """Use LLM to assist with routing when keyword-based routing is ambiguous."""
        prompt = f"""Given the user question, decide the routing path:

Question: "{question}"

Choose one:
- "direct": Simple greeting, rewriting, summarizing, or simple arithmetic
- "search": Needs up-to-date information, research, or facts not in the model's training
- "deep": Multi-step research, comparisons, or complex design questions

Return only: direct, search, or deep"""
        
        try:
            response = await self.client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            response = response.strip().lower()
            if response in ["direct", "search", "deep"]:
                return Route(response)
        except Exception:
            pass
        
        # Fall back to keyword-based routing
        return self.route(question)