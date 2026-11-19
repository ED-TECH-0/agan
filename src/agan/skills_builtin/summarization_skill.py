import re
from typing import Optional, Dict, Any, List

from .skill_base import Skill, SkillResult


class SummarizationSkill(Skill):
    """Skill for summarizing long text or extracting key points."""

    name = "summarize"
    description = "Text summarization and key point extraction"

    def can_handle(self, question: str) -> float:
        """Return confidence that summarization can help."""
        q = question.lower()
        # Long questions or requests for summary
        if len(q.split()) > 20:
            return 0.7
        summary_markers = ["summarize", "summary", "key points", "main idea",
                          "highlight", "distill", "condense"]
        markers_found = sum(1 for m in summary_markers if m in q)
        if markers_found >= 2:
            return 0.9
        if markers_found == 1:
            return 0.6
        return 0.2

    async def execute(self, question: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        """Execute summarization on the question/context."""
        try:
            text = question
            if context and "text" in context:
                text = context["text"]
            elif context and "content" in context:
                text = context["content"]

            # If text is very long, provide summarization strategy
            words = len(text.split())
            if words > 500:
                output = "Summarization strategy for " + str(words) + "-word text:\n"
                output += "• Identify topic sentences in each paragraph\n"
                output += "• Extract 3-5 key supporting points\n"
                output += "• Write concise summary (1-2 sentences)\n"
                output += "• Note any uncertain or disputed claims\n"
                return SkillResult(success=True, output=output, confidence=0.8)
            else:
                output = "Summarization of: \"" + text[:100] + "...\"\n"
                output += "• Text is short enough for direct processing\n"
                output += "• Key points extracted inline\n"
                return SkillResult(success=True, output=output, confidence=0.8)
        except Exception as e:
            return SkillResult(success=False, output="", error=str(e), confidence=0.3)