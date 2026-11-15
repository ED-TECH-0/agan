"""
Verifier — claim check + executable verification.
"""
from dataclasses import dataclass, field
from typing import Optional
import subprocess
import tempfile
import os

@dataclass
class VerificationReport:
    ok: bool
    unsupported_claims: list[str] = field(default_factory=list)
    execution_errors: list[str] = field(default_factory=list)
    missing_query: Optional[str] = None

class Verifier:
    def __init__(self, profile_config: dict = None):
        self.profile = profile_config or {}
        self.max_loops = self.profile.get("verification_loops", 3)
        self.can_execute = self.profile.get("can_execute", True)
    
    def check(self, draft: str, evidence: str, run_code: bool = True) -> VerificationReport:
        # 1. Claim verification (placeholder)
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
        # Placeholder: extract atomic claims and check against evidence
        # In practice: use NLI model or LLM to verify each claim
        return []
    
    def _verify_executable(self, draft: str) -> list[str]:
        errors = []
        
        # Extract Python code blocks
        import re
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
        # Check if blender is available
        try:
            result = subprocess.run(
                ['blender', '--version'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return "Blender not installed"
        except Exception:
            return "Blender not available"
        
        # Run headless (placeholder)
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
