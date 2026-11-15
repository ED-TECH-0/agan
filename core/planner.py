"""
Planner — decomposes questions into search steps with dependencies.
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import json

class SourceType(str, Enum):
    WEB = "web"
    WIKI = "wiki"
    DOCS = "docs"
    LOCAL = "local"
    PROJECT = "project"

class AnswerType(str, Enum):
    FACT = "fact"
    COMPARISON = "comparison"
    HOWTO = "howto"
    CODE = "code"
    DESIGN = "design"
    REPORT = "report"

@dataclass
class PlanStep:
    id: int
    query: str
    source_type: SourceType = SourceType.WEB
    depends_on: list[int] = field(default_factory=list)
    tool: Optional[str] = None

@dataclass
class Plan:
    needs_search: bool
    steps: list[PlanStep]
    expected_answer_type: AnswerType
    tools: list[str]
    skillpack: Optional[str] = None

STEP_CAPS = {
    "nano": 2,
    "lite": 3,
    "standard": 4,
    "pro": 6,
    "max": 8,
}

SKILLPACK_KEYWORDS = {
    "code": ["python", "javascript", "rust", "c++", "go", "debug", "refactor", "git", "api", "library"],
    "blender": ["blender", "bpy", "geometry nodes", "shader nodes", "rigging", "animation", "render", "3d"],
    "unreal": ["unreal", "ue4", "ue5", "blueprint", "gameplay", "c++", "editor", "packaging"],
    "sysarch": ["architecture", "system design", "microservices", "database", "scaling", "capacity", "adr"],
    "hwarch": ["verilog", "systemverilog", "vhdl", "fpga", "rtl", "testbench", "kicad", "pcb", "asic"],
}

def detect_skillpack(message: str) -> Optional[str]:
    msg_lower = message.lower()
    scores = {}
    for pack, keywords in SKILLPACK_KEYWORDS.items():
        scores[pack] = sum(1 for kw in keywords if kw in msg_lower)
    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
    return None

def make_plan(message: str, profile: str = "standard") -> Plan:
    skillpack = detect_skillpack(message)
    max_steps = STEP_CAPS.get(profile, 4)
    
    # Simple heuristic planner (replace with model-based in Phase 2)
    words = message.lower().split()
    
    if any(w in words for w in ["compare", "vs", "versus"]):
        # Comparison: one search per item
        items = [w for w in words if w not in ["compare", "vs", "versus", "the", "a", "an", "and", "or"]]
        steps = [
            PlanStep(id=i+1, query=f"{item} specifications features", source_type=SourceType.DOCS)
            for i, item in enumerate(items[:max_steps])
        ]
        return Plan(
            needs_search=True,
            steps=steps,
            expected_answer_type=AnswerType.COMPARISON,
            tools=["search", "fetch"],
            skillpack=skillpack,
        )
    
    if any(w in words for w in ["how to", "tutorial", "guide"]):
        steps = [
            PlanStep(id=1, query=message, source_type=SourceType.WEB),
            PlanStep(id=2, query=f"{message} best practices", source_type=SourceType.DOCS, depends_on=[1]),
        ]
        return Plan(
            needs_search=True,
            steps=steps[:max_steps],
            expected_answer_type=AnswerType.HOWTO,
            tools=["search", "fetch"],
            skillpack=skillpack,
        )
    
    if any(w in words for w in ["design", "architecture", "plan", "implement", "build"]):
        steps = [
            PlanStep(id=1, query=f"{message} requirements", source_type=SourceType.WEB),
            PlanStep(id=2, query=f"{message} architecture patterns", source_type=SourceType.DOCS, depends_on=[1]),
            PlanStep(id=3, query=f"{message} implementation approach", source_type=SourceType.WEB, depends_on=[1, 2]),
        ]
        return Plan(
            needs_search=True,
            steps=steps[:max_steps],
            expected_answer_type=AnswerType.DESIGN,
            tools=["search", "fetch", "python"],
            skillpack=skillpack,
        )
    
    # Default: single search
    steps = [PlanStep(id=1, query=message, source_type=SourceType.WEB)]
    return Plan(
        needs_search=True,
        steps=steps,
        expected_answer_type=AnswerType.FACT,
        tools=["search", "fetch"],
        skillpack=skillpack,
    )

if __name__ == "__main__":
    test_msgs = [
        "latest Blender version",
        "compare PyTorch vs JAX",
        "how to fix bpy AttributeError",
        "design a RAG system for code generation",
        "Verilog testbench for FIFO",
    ]
    for msg in test_msgs:
        plan = make_plan(msg, "standard")
        print(f"\n'{msg}' -> skillpack: {plan.skillpack}, type: {plan.expected_answer_type.value}")
        for step in plan.steps:
            print(f"  Step {step.id}: {step.query} (depends_on: {step.depends_on})")
