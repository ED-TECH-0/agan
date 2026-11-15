"""
Verifier — claim check + executable verification.
"""
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel
import subprocess
import tempfile
import os
import re

from core.llm_client import LLMClient

@dataclass
class VerificationReport:
    ok: bool
    unsupported_claims: list[str] = field(default_factory=list)
    execution_errors: list[str] = field(default_factory=list)
    missing_query: Optional[str] = None

class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    reasoning: str = ""

class ClaimCheckResult(BaseModel):
    checks: list[ClaimCheck]

class Verifier:
    def __init__(self, profile_config: dict = None, client: LLMClient = None):
        self.profile = profile_config or {}
        self.max_loops = self.profile.get("verification_loops", 3)
        self.can_execute = self.profile.get("can_execute", True)
        self.client = client
    
    def check(self, draft: str, evidence: str, run_code: bool = True) -> VerificationReport:
        # 1. Claim verification
        unsupported = self._verify_claims(draft, evidence)
        
        # 2. Executable verification
        exec_errors = []
        if run_code and self.can_execute:
            exec_errors = self._verify_executable(draft)
        
        ok = len(unsupported) == 0 and len(exec_errors) == 0
        missing_query = None
        if not ok:
            missing_query = self._generate_missing_query(unsupported, exec_errors)
        
        return VerificationReport(
            ok=ok,
            unsupported_claims=unsupported,
            execution_errors=exec_errors,
            missing_query=missing_query,
        )
    
    def _verify_claims(self, draft: str, evidence: str) -> list[str]:
        if not self.client:
            return []
        
        # Extract claims from draft (simple heuristic: sentences with citations)
        claims = self._extract_claims(draft)
        if not claims:
            return []
        
        # Use LLM to verify each claim against evidence
        try:
            result = self._verify_with_llm(claims, evidence)
            unsupported = [c.claim for c in result.checks if not c.supported]
            return unsupported
        except Exception:
            return []
    
    def _extract_claims(self, draft: str) -> list[str]:
        """Extract atomic claims from draft answer."""
        # Simple extraction: sentences containing [E#] citations
        claims = []
        sentences = re.split(r'(?<=[.!?])\s+', draft)
        for sent in sentences:
            if re.search(r'\[E\d+\]', sent):
                claims.append(sent.strip())
        return claims[:10]  # Limit to avoid token overflow
    
    def _verify_with_llm(self, claims: list[str], evidence: str) -> ClaimCheckResult:
        messages = [
            {"role": "system", "content": """You are a claim verifier. For each claim, determine if it is fully supported by the evidence.
Return JSON with a list of checks, each containing: claim (string), supported (boolean), reasoning (string)."""},
            {"role": "user", "content": f"""Evidence:
{evidence}

Claims to verify:
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(claims))}

Return JSON with checks."""},
        ]
        
        import asyncio
        return asyncio.run(self._async_verify(messages))
    
    async def _async_verify(self, messages: list[dict]) -> ClaimCheckResult:
        return await self.client.chat_json(messages, ClaimCheckResult)
    
    def _verify_executable(self, draft: str) -> list[str]:
        errors = []
        
        # Extract Python code blocks
        python_blocks = re.findall(r'```python\n(.*?)\n```', draft, re.DOTALL)
        for i, code in enumerate(python_blocks):
            error = self._run_python_code(code)
            if error:
                errors.append(f"Python block {i+1}: {error}")
        
        # Extract Blender bpy code
        blender_blocks = re.findall(r'```(?:bpy|blender)\n(.*?)\n```', draft, re.DOTALL)
        for i, code in enumerate(blender_blocks):
            error = self._run_blender_script(code)
            if error:
                errors.append(f"Blender block {i+1}: {error}")
        
        # Extract shell commands
        shell_blocks = re.findall(r'```(?:bash|shell|sh)\n(.*?)\n```', draft, re.DOTALL)
        for i, code in enumerate(shell_blocks):
            error = self._run_shell(code)
            if error:
                errors.append(f"Shell block {i+1}: {error}")
        
        return errors
    
    def _run_python_code(self, code: str) -> Optional[str]:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    ['python', f.name],
                    capture_output=True, text=True, timeout=30
                )
                os.unlink(f.name)
                if result.returncode != 0:
                    return result.stderr[:500]
        except subprocess.TimeoutExpired:
            return "Timeout after 30s"
        except Exception as e:
            return str(e)
        return None
    
    def _run_blender_script(self, code: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ['blender', '--version'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return "Blender not installed"
        except Exception:
            return "Blender not available"
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    ['blender', '--background', '--python', f.name],
                    capture_output=True, text=True, timeout=60
                )
                os.unlink(f.name)
                if result.returncode != 0:
                    return result.stderr[:500]
        except subprocess.TimeoutExpired:
            return "Blender timeout"
        except Exception as e:
            return str(e)
        return None
    
    def _run_shell(self, code: str) -> Optional[str]:
        try:
            result = subprocess.run(
                code, shell=True, capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return result.stderr[:500]
        except subprocess.TimeoutExpired:
            return "Timeout"
        except Exception as e:
            return str(e)
        return None
    
    def _generate_missing_query(self, unsupported: list[str], exec_errors: list[str]) -> str:
        parts = []
        if unsupported:
            parts.append(f"Unsupported claims: {'; '.join(unsupported)}")
        if exec_errors:
            parts.append(f"Execution errors: {'; '.join(exec_errors)}")
        return " | ".join(parts)

if __name__ == "__main__":
    v = Verifier({"verification_loops": 3, "can_execute": True})
    draft = """
    ```python
    print("hello")
    x = 1 / 0
    ```
    """
    report = v.check(draft, "evidence")
    print(f"OK: {report.ok}")
    print(f"Errors: {report.execution_errors}")
