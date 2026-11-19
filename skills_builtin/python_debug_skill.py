import re
from typing import Optional, Dict, Any, List

from .skill_base import Skill, SkillResult


class PythonDebugSkill(Skill):
    """Skill for Python code debugging — identifies errors, suggests fixes."""

    name = "python_debug"
    description = "Python code debugging and error resolution"

    def can_handle(self, question: str) -> float:
        """Return confidence that Python debugging can help."""
        q = question.lower()
        debug_markers = ["error", "exception", "traceback", "fix", "debug",
                         "bug", "doesn't work", "not running", "python file",
                         "syntax error", "name error", "type error"]
        markers_found = sum(1 for m in debug_markers if m in q)

        if markers_found >= 3:
            return 0.95
        if any(k in q for k in ["python script", "python code", "py file"]):
            return 0.8
        if re.search(r'```python|def |\nimport ', q):
            return 0.7
        if "python" in q and ("error" in q or "fix" in q):
            return 0.6
        return 0.1

    async def execute(self, question: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        """Execute Python debugging on the question."""
        try:
            code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', question, re.DOTALL)
            if code_blocks:
                code = code_blocks[0]
                # Simple syntax check
                import ast
                try:
                    ast.parse(code)
                    syntax_ok = True
                except SyntaxError as e:
                    syntax_ok = False
                    error_msg = str(e)

                if syntax_ok:
                    output = "Python code looks syntactically valid.\n```python\n" + code + "\n```"
                    return SkillResult(success=True, output=output, confidence=0.9)
                else:
                    output = "Syntax error found:\n```\n" + error_msg + "\n```\n\nSuggested fix:\n• Check indentation\n• Verify all parentheses/brackets are closed\n• Ensure all variables are defined\n• Check for typos in function names"
                    return SkillResult(success=True, output=output, confidence=0.8)
            else:
                # No code block, but question mentions Python errors
                output = "Python debugging analysis for: " + question + "\n"
                output += "• Common Python errors: NameError, TypeError, IndentationError, KeyError\n"
                output += "• Check: undefined variables, incorrect indentation, missing imports\n"
                output += "• Verify function signatures and argument types\n"
                return SkillResult(success=True, output=output, confidence=0.7)
        except Exception as e:
            return SkillResult(success=False, output="", error=str(e), confidence=0.3)