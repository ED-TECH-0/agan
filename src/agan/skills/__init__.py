"""Built-in skills for AGAN."""

from __future__ import annotations


class SkillResult:
    """Result of a skill execution."""

    def __init__(self, success: bool, output: str | None = None, error: str | None = None) -> None:
        self.success = success
        self.output = output
        self.error = error


class SkillManager:
    """Manager for built-in skills used by the Reasoner."""

    def __init__(self) -> None:
        self.skills: dict[str, dict] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default skill implementations."""
        # Core reasoning skills
        self.skills["web_search"] = {
            "name": "web_search",
            "description": "Search the web for information",
        }
        self.skills["math_reasoning"] = {
            "name": "math_reasoning",
            "description": "Perform mathematical reasoning",
        }
        self.skills["code_exec"] = {
            "name": "code_exec",
            "description": "Execute Python code snippets",
        }
        self.skills["summarize"] = {
            "name": "summarize",
            "description": "Summarize text or information",
        }
        self.skills["python_debug"] = {
            "name": "python_debug",
            "description": "Debug Python code errors",
        }

    def select_skills(self, query: str) -> list[tuple[dict, float]]:
        """Select relevant skills for a given query.

        Args:
            query: The user's question.

        Returns:
            List of (skill_info, confidence) tuples sorted by relevance.
        """
        query_lower = query.lower()
        results: list[tuple[dict, float]] = []

        # Simple keyword-based skill selection
        if any(
            kw in query_lower
            for kw in ["calculate", "compute", "math", "add", "subtract", "multiply", "divide", "fibonacci", "power"]
        ):
            results.append((self.skills["math_reasoning"], 0.8))

        if any(kw in query_lower for kw in ["search", "find", "lookup", "google", "bing"]):
            results.append((self.skills["web_search"], 0.8))

        if any(kw in query_lower for kw in ["fix", "error", "debug", "trace", "NameError", "TypeError"]):
            results.append((self.skills["python_debug"], 0.8))

        if any(kw in query_lower for kw in ["summarize", "summary", "reduce", "brief"]):
            results.append((self.skills["summarize"], 0.8))

        if any(kw in query_lower for kw in ["python", "code", "program", "script"]):
            results.append((self.skills["code_exec"], 0.7))

        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def execute(self, name: str, query: str) -> SkillResult:
        """Execute a registered skill by name.

        Args:
            name: The skill name.
            query: The question or input for the skill.

        Returns:
            A SkillResult instance.
        """
        if name not in self.skills:
            return SkillResult(success=False, error=f"Unknown skill: {name}")

        skill_info = self.skills[name]

        # Dispatch to appropriate handler
        if name == "math_reasoning":
            return self._math_reasoning(query)
        elif name == "web_search":
            return self._web_search(query)
        elif name == "code_exec":
            return self._code_exec(query)
        elif name == "summarize":
            return self._summarize(query)
        elif name == "python_debug":
            return self._python_debug(query)
        else:
            return SkillResult(success=False, error=f"Skill '{name}' not yet implemented")

    # --- Individual skill handlers ---
    def _math_reasoning(self, query: str) -> SkillResult:
        """Handle math reasoning requests."""
        import re

        # Try to extract a calculation expression
        expr_match = re.search(r"(\d+)\s*\*\s*(\d+)", query)
        if expr_match:
            a, b = int(expr_match.group(1)), int(expr_match.group(2))
            result = a * b
            return SkillResult(success=True, output=f"{a} * {b} = {result}")

        # Look for "What is X * Y" patterns
        match = re.search(r"(\d+)\s*\*\s*(\d+)\s*\?", query)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            return SkillResult(success=True, output=f"{a} * {b} = {a * b}")

        return SkillResult(success=False, error="Could not parse mathematical expression")

    def _web_search(self, query: str) -> SkillResult:
        """Handle web search requests (placeholder)."""
        # In a full implementation, this would use a search API
        return SkillResult(
            success=True,
            output=f"Web search for: {query}\n(Note: web search is a placeholder in this demo)",
        )

    def _code_exec(self, query: str) -> SkillResult:
        """Handle code execution requests (placeholder)."""
        # In a full implementation, this would execute Python code safely
        return SkillResult(
            success=True,
            output=f"Code execution placeholder for: {query}",
        )

    def _summarize(self, query: str) -> SkillResult:
        """Handle summarize requests (placeholder)."""
        return SkillResult(
            success=True,
            output=f"Summary of: {query}\n(Note: summarize is a placeholder in this demo)",
        )

    def _python_debug(self, query: str) -> SkillResult:
        """Handle Python debug requests (placeholder)."""
        # Try to extract and report on the error
        if "NameError" in query:
            return SkillResult(
                success=True,
                output="NameError: Check variable names and ensure they're defined before use",
            )
        return SkillResult(success=True, output="Debug placeholder - review the code for issues")
