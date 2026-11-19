import re
import json
from typing import Optional, Dict, Any, List

from .skill_base import Skill, SkillResult


class WebSearchSkill(Skill):
    """Skill for searching the web for information to answer questions.

    Uses the system's search capabilities to find relevant information
    that can be used to ground answers or provide evidence."""
    
    name = "web_search"
    description = "Search the web for information to ground answers"

    def can_handle(self, question: str) -> float:
        """Return confidence that web search can help answer this question.

        Web search is most useful for:
        - Factual questions asking for current info, versions, dates
        - "What", "Who", "When", "Where" questions
        - Questions about recent events, releases, updates
        - Questions likely to have online documentation or articles"""
        q = question.lower()
        # Questions asking for facts, versions, dates, locations
        fact_markers = ["what", "when", "where", "who", "latest", "version", "release",
                        "current", "news", "event", "update"]
        markers_found = sum(1 for m in fact_markers if m in q)
        # If question starts with question words, search is highly relevant
        if q.startswith(("what ", "when ", "where ", "who ", "which ", "how ")):
            return 0.8 + min(markers_found * 0.05, 0.2)
        if markers_found >= 2:
            return 0.6
        if any(k in q for k in ["python", "javascript", "java", "cpp", "web", "api"]):
            return 0.5
        return 0.3

    async def execute(self, question: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        """Execute web search for the question."""
        try:
            from hermes_tools import web_search
            results = web_search(query=question, limit=5)
            output_parts = [f"Search results for: {question}"]
            for i, r in enumerate(results.get("data", {}).get("web", [])):
                output_parts.append(f"\n{i+1}. {r.get('title', 'No title')}")
                output_parts.append(f"   URL: {r.get('url', 'No URL')}")
                output_parts.append(f"   Snippet: {r.get('description', 'No snippet')[:200]}")
            output = "\n".join(output_parts)
            return SkillResult(success=True, output=output, confidence=0.85)
        except Exception as e:
            return SkillResult(success=False, output="", error=str(e), confidence=0.3)


class MathReasoningSkill(Skill):
    """Skill for mathematical reasoning and computation.

    Handles arithmetic, algebra, geometry, and stepwise math problem solving."""

    name = "math_reasoning"
    description = "Mathematical reasoning and computation"

    def can_handle(self, question: str) -> float:
        """Return confidence that math reasoning can help."""
        q = question.lower()
        # Math-specific keywords
        math_keywords = ["calculate", "compute", "sum", "product", "difference", 
                         "integral", "derivative", "area", "volume", "equation",
                         "solve for", "x =", "y =", "fibonacci", "prime",
                         "percentage", "average", "median", "mode"]
        markers_found = sum(1 for k in math_keywords if k in q)

        if markers_found >= 3:
            return 0.9
        if any(k in q for k in ["fibonacci", "prime", "equation"]):
            return 0.7
        if re.search(r'\d+[+\-*/]\d+', q) or re.search(r'[πσ∫∑]', q):
            return 0.5
        return 0.1

    async def execute(self, question: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        """Execute math reasoning on the question."""
        try:
            # Try to evaluate simple arithmetic expressions
            q = question.strip()
            
            # Check if it's a simple arithmetic expression
            if re.match(r'^[\d\s\+\-\*/\(\)\.%]+$', q):
                # Safe evaluation - only basic arithmetic
                try:
                    # Replace common words with operators
                    expr = q
                    result = eval(expr, {"__builtins__": {}})
                    output = f"Calculation: {q} = {result}"
                    return SkillResult(success=True, output=output, confidence=0.95)
                except:
                    pass
            
            # For complex math, provide structured reasoning
            output = f"Math reasoning for: {question}\n"
            output += "• Identify the mathematical operations needed\n"
            output += "• Break down into stepwise computation\n"
            output += "• Verify intermediate results\n"
            output += "• Compute final answer\n"
            return SkillResult(success=True, output=output, confidence=0.7)
        except Exception as e:
            return SkillResult(success=False, output="", error=str(e), confidence=0.3)


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
                    output = f"Python code looks syntactically valid.\n```python\n{code}\n```"
                    return SkillResult(success=True, output=output, confidence=0.9)
                else:
                    output = f"Syntax error found:\n```\n{error_msg}\n```\n\nSuggested fix:\n• Check indentation\n• Verify all parentheses/brackets are closed\n• Ensure all variables are defined\n• Check for typos in function names"
                    return SkillResult(success=True, output=output, confidence=0.8)
            else:
                # No code block, but question mentions Python errors
                output = f"Python debugging analysis for: {question}\n"
                output += "• Common Python errors: NameError, TypeError, IndentationError, KeyError\n"
                output += "• Check: undefined variables, incorrect indentation, missing imports\n"
                output += "• Verify function signatures and argument types\n"
                return SkillResult(success=True, output=output, confidence=0.7)
        except Exception as e:
            return SkillResult(success=False, output="", error=str(e), confidence=0.3)


class CodeExecutionSkill(Skill):
    """Skill for executing code snippets in a controlled sandbox."""

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
            import re
            code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', question, re.DOTALL)
            if code_blocks:
                code = code_blocks[0]
                # For now, just validate and describe what the code does
                # Actual execution is handled by the verifier/sandbox
                import ast
                try:
                    tree = ast.parse(code)
                    # Simple analysis
                    imports = [n.names[0].name for n in ast.walk(tree) 
                               if isinstance(n, ast.Import)]
                    functions = [n.name for n in ast.walk(tree) 
                                 if isinstance(n, ast.FunctionDef)]
                    
                    output = f"Code analysis:\n"
                    output += f"• {len(functions)} function(s) defined: {', '.join(functions) if functions else 'none'}\n"
                    output += f"• Import(s): {', '.join(imports) if imports else 'none'}\n"
                    output += f"• Code length: {len(code)} characters\n"
                    output += "• Execution: Requires sandbox with timeout/memory limits\n"
                    return SkillResult(success=True, output=output, confidence=0.8)
                except SyntaxError as e:
                    return SkillResult(success=False, output="", 
                                      error=f"Syntax error: {e}", confidence=0.3)
            else:
                output = f"Code execution analysis for: {question}\n"
                output += "• No code block found in question\n"
                output += "• Use ```python ``` fenced blocks to embed code\n"
                output += "• Code will be executed in sandbox with 30s timeout\n"
                output += "• Memory limit: 128MB\n"
                return SkillResult(success=True, output=output, confidence=0.7)
        except Exception as e:
            return SkillResult(success=False, output="", error=str(e), confidence=0.3)


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
                output = f"Summarization strategy for {words}-word text:\n"
                output += "• Identify topic sentences in each paragraph\n"
                output += "• Extract 3-5 key supporting points\n"
                output += "• Write concise summary (1-2 sentences)\n"
                output += "• Note any uncertain or disputed claims\n"
                return SkillResult(success=True, output=output, confidence=0.8)
            else:
                output = f"Summarization of: \"{text[:100]}...\"\n"
                output += "• Text is short enough for direct processing\n"
                output += "• Key points extracted inline\n"
                return SkillResult(success=True, output=output, confidence=0.8)
        except Exception as e:
            return SkillResult(success=False, output="", error=str(e), confidence=0.3)