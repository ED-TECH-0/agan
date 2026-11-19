"""Tool registry for AGAN skill execution."""

from __future__ import annotations


class ToolRegistry:
    """Registry of available tools/skills for the Reasoner.

    Maps skill names to callable implementations.
    """

    def __init__(self) -> None:
        self._tools: dict[str, callable] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default tool implementations."""
        # Placeholder - concrete implementations register themselves
        pass

    def register(self, name: str, func: callable) -> None:
        """Register a tool by name.

        Args:
            name: The tool/skill name.
            func: Async callable that takes a question string and returns output.
        """
        self._tools[name] = func

    def get(self, name: str) -> callable | None:
        """Get a registered tool by name.

        Args:
            name: The tool/skill name.

        Returns:
            The callable if registered, else None.
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: The tool/skill name.

        Returns:
            True if the tool is registered.
        """
        return name in self._tools
