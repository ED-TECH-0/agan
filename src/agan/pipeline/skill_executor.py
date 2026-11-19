"""SkillExecutor: runs skills by name through a SkillManager."""

from typing import Dict, Any
from agan.skills_builtin import SkillManager, SkillResult


class SkillExecutor:
    """Executes named skills via a SkillManager, returning per-skill results."""

    def __init__(self, skill_manager: SkillManager):
        self.skill_manager = skill_manager

    async def execute_all_skills(
        self, skill_names: list[str], question: str, context: Dict[str, Any] | None = None
    ) -> Dict[str, SkillResult]:
        """Execute each named skill for the given question.

        Returns a dict mapping skill name -> SkillResult.
        Skills that don't exist return a failure result.
        """
        results: Dict[str, SkillResult] = {}
        for name in skill_names:
            skill = self.skill_manager.skills.get(name)
            if skill is None:
                results[name] = SkillResult(
                    success=False,
                    output=f"Skill '{name}' not found.",
                    error="skill_not_found",
                )
                continue
            try:
                result = await skill.execute(question, context)
                if result is None:
                    results[name] = SkillResult(
                        success=False,
                        output=f"Skill '{name}' returned no result.",
                        error="no_result",
                    )
                else:
                    results[name] = result
            except Exception as e:  # noqa: BLE001
                results[name] = SkillResult(
                    success=False,
                    output=f"Skill '{name}' raised an exception.",
                    error=str(e),
                )
        return results

    async def execute_best(self, question: str, context: Dict[str, Any] | None = None):
        """Execute the single best-matching skill for the question."""
        return await self.skill_manager.execute_best(question, context)
