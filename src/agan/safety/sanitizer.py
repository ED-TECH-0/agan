"""Input sanitizer for AGAN safety pipeline."""

from __future__ import annotations


def sanitize_input(text: str) -> str:
    """Sanitize user input by removing potentially harmful content.

    Args:
        text: The text to sanitize.

    Returns:
        The sanitized text.
    """
    # Strip common AI prompt injection markers
    import re
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\b(?:ignore|override|system|prompt)\b', '', text, flags=re.IGNORECASE)
    return text.strip()
