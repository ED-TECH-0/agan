"""Outbound privacy gateway for AGAN."""

from __future__ import annotations


class OutboundGateway:
    """Gatekeeper for outbound data flows respecting privacy boundaries."""

    def check(self, question: str) -> bool:
        """Return True if the question passes privacy checks.

        Subclasses or middleware may override this to enforce
        more sophisticated rules (redaction, consent, etc.).
        """
        return True

    def redact_sensitive(self, text: str) -> str:
        """Remove or mask sensitive references from ``text``."""
        return text
