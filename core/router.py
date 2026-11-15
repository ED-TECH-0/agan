"""
Router — decides: direct | search | deep | action
"""
from enum import Enum
from dataclasses import dataclass
import re

class RouteType(str, Enum):
    DIRECT = "direct"
    SEARCH = "search"
    DEEP = "deep"
    ACTION = "action"

@dataclass
class RouteDecision:
    route: RouteType
    confidence: float
    reason: str

# Keywords that suggest need for retrieval
SEARCH_KEYWORDS = [
    "latest", "current", "recent", "version", "release", "changelog",
    "how to", "tutorial", "guide", "documentation", "docs", "api",
    "error", "exception", "bug", "issue", "fix", "solve",
    "compare", "vs", "versus", "difference", "better", "best",
    "price", "cost", "benchmark", "performance", "specs",
    "2024", "2025", "2026", "new", "updated", "changed"
]

DEEP_KEYWORDS = [
    "design", "architecture", "plan", "roadmap", "strategy",
    "implement", "build", "create", "develop", "write a",
    "analyze", "research", "investigate", "comprehensive",
    "end-to-end", "full", "complete", "system", "pipeline"
]

ACTION_KEYWORDS = [
    "run", "execute", "install", "create file", "write file",
    "delete", "modify", "edit", "refactor", "rename",
    "commit", "push", "pull", "build", "compile",
    "render", "simulate", "lint", "format"
]

# Keywords that need word-boundary matching (single words)
ACTION_KEYWORDS_WB = ["run", "execute", "install", "delete", "modify", "edit", "refactor", "rename",
    "commit", "push", "pull", "build", "compile", "render", "simulate", "lint", "format"]

DIRECT_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no)\.?$",
    r"^\d+\s*[\+\-\*/]\s*\d+$",  # simple arithmetic
]

def classify_route(message: str, profile: str = "standard") -> RouteDecision:
    msg_lower = message.lower().strip()
    
    # Direct: chit-chat, simple arithmetic, rewriting
    for pat in DIRECT_PATTERNS:
        if re.match(pat, msg_lower, re.IGNORECASE):
            return RouteDecision(RouteType.DIRECT, 0.9, "Simple direct interaction")
    
    # Action: system/app operations - use word boundaries for single words
    for kw in ACTION_KEYWORDS:
        if " " in kw:
            # Multi-word phrase: exact substring match
            if kw in msg_lower:
                return RouteDecision(RouteType.ACTION, 0.8, f"Action keyword: {kw}")
        else:
            # Single word: word boundary match
            if re.search(rf"\b{re.escape(kw)}\b", msg_lower):
                return RouteDecision(RouteType.ACTION, 0.8, f"Action keyword: {kw}")
    
    # Deep: multi-part research/design
    deep_score = sum(1 for kw in DEEP_KEYWORDS if kw in msg_lower)
    if deep_score >= 2 or (deep_score >= 1 and len(msg_lower.split()) > 30):
        return RouteDecision(RouteType.DEEP, 0.75, f"Deep task indicators: {deep_score}")
    
    # Search: factual, version, how-to
    search_score = sum(1 for kw in SEARCH_KEYWORDS if kw in msg_lower)
    if search_score >= 1:
        return RouteDecision(RouteType.SEARCH, 0.7, f"Search keywords: {search_score}")
    
    # Default to direct for short queries, search for longer
    if len(msg_lower.split()) < 10:
        return RouteDecision(RouteType.DIRECT, 0.6, "Short query, no keywords")
    return RouteDecision(RouteType.SEARCH, 0.5, "Default to search")

if __name__ == "__main__":
    test_msgs = [
        "hi",
        "2 + 2",
        "latest Blender version",
        "how to fix AttributeError in bpy",
        "design a RAG system for code generation",
        "run the test suite",
        "compare PyTorch vs JAX performance",
    ]
    for msg in test_msgs:
        decision = classify_route(msg)
        print(f"'{msg}' -> {decision.route.value} ({decision.confidence:.2f}) - {decision.reason}")
