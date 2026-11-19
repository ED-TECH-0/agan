from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class SkillResult:
    """Result from running a skill."""
    success: bool
    output: str = ""
    error: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    skill_name: str = ""


class Skill:
    """Abstract base class for all AGAN built-in skills."""

    name: str = "base_skill"
    description: str = "Base skill class"
    reputation: float = 1.0  # Trust/reputation score 0.0 - 1.0

    def can_handle(self, question: str) -> float:
        """Return confidence [0,1] that this skill can handle the question.
        Subclasses override to implement domain-specific matching."""
        return 0.0

    async def execute(self, question: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        """Execute the skill on the given question with optional context.
        Subclasses must override."""
        return SkillResult(success=False, output="", error="Not implemented")

    def get_description(self) -> str:
        return self.description