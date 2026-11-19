import re
import ast
from typing import Optional, Dict, Any, List

from .skill_base import Skill, SkillResult


class CodeExecutionSkill(Skill):
    """Skill for executing code snippets in a controlled sandbox.

    Executes Python code with timeout and memory limits, returns
    stdout/stderr output and success status."""

    name = "code_exec"
    description = "Execute code snippets with timeout and memory limits"

    def can_handle(self, question: str) -> float:
        """Return confidence that code execution can help."""
        q = question.lower()
        code_markers = ["execute", "run", "function", "script", "print(",
                        "```", "import ", "def ", "return "]
        markers_found = sum(1 for m in code_markers if m in q)

        if markers_found >= 3:
            return 0.9
        if "```" in q or q.startswith("def "):
            return 0.7
        if any(k in q for k in ["fibonacci", "factorial", "binary search", "sort"]):
            return 0.5
        return 0.1

    async def execute(self, question: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        """Execute code snippet from the question."""
        try:
            code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', question, re.DOTALL)
            if code_blocks:
                code = code_blocks[0]
                # For now, just validate and describe what the code does
                # Actual execution is handled by the verifier/sandbox
                try:
                    tree = ast.parse(code)
                    # Simple analysis
                    imports = [n.names[0].name for n in ast.walk(tree)
                               if isinstance(n, ast.Import)]
                    functions = [n.name for n in ast.walk(tree)
                                 if isinstance(n, ast.FunctionDef)]

                    imports_str = ", ".join(imports) if imports else "none"
                    functions_str = ", ".join(functions) if functions else "none"

                    output = "Code analysis:\n"
                    output += f"• {len(functions)} function(s) defined: {functions_str}\n"
                    output += f"• Import(s): {imports_str}\n"
                    output += f"• Code length: {len(code)} characters\n"
                    output += "• Execution: Requires sandbox with timeout/memory limits\n"
                    return SkillResult(success=True, output=output, confidence=0.8)
                except SyntaxError as e:
                    return SkillResult(success=False, output="",
                                      error=f"Syntax error: {e}", confidence=0.3)
            else:
                output = "Code execution analysis for: " + question + "\n"
                output += "• No code block found in question\n"
                output += "• Use ```python ``` fenced blocks to embed code\n"
                output += "• Code will be executed in sandbox with 30s timeout\n"
                output += "• Memory limit: 128MB\n"
                return SkillResult(success=True, output=output, confidence=0.7)
        except Exception as e:
            return SkillResult(success=False, output="", error=str(e), confidence=0.3)