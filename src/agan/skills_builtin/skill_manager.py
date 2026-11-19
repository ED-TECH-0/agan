from typing import Optional, Dict, Any, List
from collections import defaultdict

from .skill_base import Skill, SkillResult
from .web_search_skill import WebSearchSkill
from .math_reasoning_skill import MathReasoningSkill
from .python_debug_skill import PythonDebugSkill
from .code_exec_skill import CodeExecutionSkill
from .summarization_skill import SummarizationSkill


class SkillManager:
    """Manages the registry of built-in skills with reputation tracking
    and intelligent selection based on question type and skill history."""

    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.reputation: Dict[str, float] = defaultdict(lambda: 1.0)
        self.skill_usage: Dict[str, int] = defaultdict(int)
        self._init_skills()

    def _init_skills(self):
        """Register all built-in skills."""
        self.register(WebSearchSkill())
        self.register(MathReasoningSkill())
        self.register(PythonDebugSkill())
        self.register(CodeExecutionSkill())
        self.register(SummarizationSkill())

    def register(self, skill: Skill):
        """Register a skill with the manager."""
        self.skills[skill.name] = skill
        # Ensure reputation entry exists
        if skill.name not in self.reputation:
            self.reputation[skill.name] = 1.0

    def select_skills(self, question: str, min_reputation: float = 0.3) -> List[tuple[Skill, float]]:
        """Select and rank skills that can handle the given question.

        Returns list of (skill, confidence) tuples sorted by effective score
        (reputation * capability confidence)."""
        candidates = []
        for name, skill in self.skills.items():
            capability = skill.can_handle(question)
            reputation = self.reputation.get(name, 1.0)
            effective = capability * reputation
            if effective >= min_reputation:
                candidates.append((skill, effective))
        # Sort by effective score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    async def execute_best(self, question: str, context: Optional[Dict[str, Any]] = None
                          ) -> Optional[SkillResult]:
        """Execute the best-matching skill for the question.

        Updates reputation based on success/failure."""
        candidates = self.select_skills(question)
        if not candidates:
            return None

        # Try top candidate first
        skill, score = candidates[0]
        result = await skill.execute(question, context)

        # Update reputation based on outcome
        if result and result.success:
            self.reputation[skill.name] = min(1.0, self.reputation.get(skill.name, 1.0) + 0.05)
        elif result and not result.success:
            self.reputation[skill.name] = max(0.0, self.reputation.get(skill.name, 1.0) - 0.1)

        # If top skill failed reasonably low, try next candidate
        if result and not result.success and len(candidates) > 1:
            for alt_skill, alt_score in candidates[1:3]:
                result = await alt_skill.execute(question, context)
                if result and result.success:
                    self.reputation[alt_skill.name] = min(1.0, self.reputation.get(alt_skill.name, 1.0) + 0.05)
                    break

        return result

    def get_skill_descriptions(self) -> Dict[str, str]:
        """Get descriptions for all registered skills."""
        return {name: skill.get_description() for name, skill in self.skills.items()}

    def get_reputation_report(self) -> Dict[str, float]:
        """Get current reputation scores for all skills."""
        return dict(self.reputation)