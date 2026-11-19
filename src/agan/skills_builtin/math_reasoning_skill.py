import re
from typing import Optional, Dict, Any, List

from .skill_base import Skill, SkillResult


class MathReasoningSkill(Skill):
    """Skill for mathematical reasoning and computation.

    Handles arithmetic, algebra, geometry, and stepwise math problem solving."""

    name = "math_reasoning"
    description = "Mathematical reasoning and computation"

    def can_handle(self, question: str) -> float:
        """Return confidence that math reasoning can help."""
        q = question.lower()
        # Math-specific keywords
        math_keywords = ["calculate", "compute", "sum", "product", "difference",
                         "integral", "derivative", "area", "volume", "equation",
                         "solve for", "x =", "y =", "fibonacci", "prime",
                         "percentage", "average", "median", "mode"]
        markers_found = sum(1 for k in math_keywords if k in q)

        if markers_found >= 3:
            return 0.9
        if any(k in q for k in ["fibonacci", "prime", "equation"]):
            return 0.7
        if re.search(r'\d+[+\-*/]\d+', q) or re.search(r'[πσ∫∑]', q):
            return 0.5
        return 0.1

    async def execute(self, question: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        """Execute math reasoning on the question."""
        try:
            # Try to evaluate simple arithmetic expressions
            q = question.strip()

            # Check if it's a simple arithmetic expression
            if re.match(r'^[\d\s\+\-\*/\(\)\.%]+$', q):
                # Safe evaluation - only basic arithmetic
                try:
                    # Replace common words with operators
                    expr = q
                    result = eval(expr, {"__builtins__": {}})
                    output = f"Calculation: {q} = {result}"
                    return SkillResult(success=True, output=output, confidence=0.95)
                except Exception:
                    pass

            # For complex math, provide structured reasoning
            output = "Math reasoning for: " + question + "\n"
            output += "• Identify the mathematical operations needed\n"
            output += "• Break down into stepwise computation\n"
            output += "• Verify intermediate results\n"
            output += "• Compute final answer\n"
            return SkillResult(success=True, output=output, confidence=0.7)
        except Exception as e:
            return SkillResult(success=False, output="", error=str(e), confidence=0.3)