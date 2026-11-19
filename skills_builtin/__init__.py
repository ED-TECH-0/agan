# AGAN Skill System — built-in skills for the local AI
# SkillManager provides intelligent skill selection based on question type
# Individual skills can be imported directly from their modules

from agan.skills_builtin.skill_manager import SkillManager
from agan.skills_builtin.skill_base import Skill, SkillResult

__all__ = ["SkillManager", "Skill", "SkillResult"]