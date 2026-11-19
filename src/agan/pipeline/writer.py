"""Writer — final pipeline stage.

Formats a verified Answer into user-facing output: applies style/level
adjustments, renders citations, and attaches source metadata.
"""
from __future__ import annotations

from typing import Optional

from agan.schemas import Answer, Plan
from agan.pipeline.verifier import VerifierReport


class Writer:
    """Renders a verified answer for the user."""

    def __init__(self, profile_spec=None):
        self.profile_spec = profile_spec

    def write(self, answer: Answer, report: Optional[VerifierReport] = None,
              plan: Optional[Plan] = None) -> Answer:
        """Produce the final user-facing answer.

        - Appends source citations when sources exist.
        - Appends verification notes when the report flags issues.
        - Adjusts verbosity hint based on profile/level.
        """
        text = answer.text

        # Append sources as footnotes
        if answer.sources:
            text = self._append_sources(text, answer.sources)

        # Append verification notes
        if report and not report.ok:
            if report.missing_query:
                text += f"\n\nNote: Could not fully verify — try: {report.missing_query}"
            else:
                text += "\n\nNote: Some claims could not be verified from available evidence."

        # Confidence note
        if answer.confidence == "low":
            text += "\n\nConfidence: low — treat as preliminary."

        return Answer(
            text=text,
            claims=answer.claims,
            sources=answer.sources,
            confidence=answer.confidence,
            notes=answer.notes,
            level_used=answer.level_used,
        )

    def _append_sources(self, text: str, sources: list[dict]) -> str:
        """Append a sources section to the answer text."""
        lines = [text, ""]
        lines.append("Sources:")
        for i, s in enumerate(sources[:10], 1):
            sid = s.get("id", f"S{i}")
            stype = s.get("type", "evidence")
            lines.append(f"  [{sid}] ({stype})")
        return "\n".join(lines)
