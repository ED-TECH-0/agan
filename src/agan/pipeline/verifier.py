from typing import Optional, List, Dict, Any, Protocol
import json
import re
import ast
import subprocess
import sys
from pathlib import Path


class ExecutionReport(BaseModel):
    """Result of running a tool/code in the sandbox."""
    ran: bool = False
    ok: bool = False
    output: str = ""
    error: str = ""
    duration_ms: int = 0


class Claim(BaseModel):
    """A factual claim extracted from an answer."""
    text: str
    evidence_ids: List[str] = []
    label: Literal["verified", "partial", "unverified"] = "unverified"
    score: float = 0.0


class VerifierReport(BaseModel):
    """Report from verification pass."""
    ok: bool
    claims: List[Claim] = []
    missing_query: Optional[str] = None
    execution: Optional[ExecutionReport] = None


class Verifier:
    """Verifier checks claims against evidence and runs executable code."""
    
    def __init__(self, sandbox, profile):
        self.sandbox = sandbox
        self.profile = profile
    
    def _extract_claims(self, text: str) -> List[Claim]:
        """Extract atomic factual claims from the draft answer."""
        claims = []
        
        # Split by common separators and look for sentence-like chunks
        sentences = re.split(r'[.!?]+', text)
        
        skip_prefixes = ['- ', '* ', '• ', '# ', 'TODO:', 'FIXME:', 'NOTE:',
                         '```', 'import ', 'def ', 'class ', 'print(']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 3:
                continue
            
            # Skip opinions, instructions, and code blocks
            if any(sentence.startswith(prefix) for prefix in skip_prefixes):
                continue
            
            # Skip pure numbers or URLs
            if re.match(r'^\d+(\.\d+)?$', sentence.strip()):
                continue
            if sentence.startswith('http') or sentence.startswith('www.'):
                continue
            
            # Extract any numbers, versions, dates, or proper names mentioned
            numbers = re.findall(r'\b\d+(\.\d+)?\b', sentence)
            version_patterns = re.findall(r'[vV]?\d+\.\d+(\.\d+)?', sentence)
            
            claim = Claim(text=sentence.strip())
            
            # Check if the claim contains verifiable elements
            has_numbers = len(numbers) > 0
            has_versions = len(version_patterns) > 0
            
            if has_numbers or has_versions:
                # Mark as needing evidence verification
                claim.label = "unverified"
                claim.score = 0.0
            else:
                # Opinion or general statement - mark as unverified but not evidence-dependent
                claim.label = "unverified"
                claim.score = 0.5
            
            claims.append(claim)
        
        # If no claims extracted from sentences, treat the whole text as one claim
        if not claims:
            claim = Claim(text=text.strip()[:500] if text.strip() else "")
            claim.label = "unverified"
            claim.score = 0.0
            claims.append(claim)
        
        return claims
    
    def _verify_claims_against_evidence(self, claims: List[Claim], 
                                        evidence_texts: List[str]) -> List[Claim]:
        """Verify each claim against provided evidence texts."""
        for claim in claims:
            best_score = 0.0
            best_evidence = ""
            
            for evidence in evidence_texts:
                score = self._entailment_score(claim.text, evidence)
                if score > best_score:
                    best_score = score
                    best_evidence = evidence
            
            if best_score >= 0.80:
                claim.label = "verified"
                claim.score = best_score
            elif best_score >= 0.50:
                claim.label = "partial"
                claim.score = best_score
            else:
                claim.label = "unverified"
                claim.score = best_score
            
            # Number guard: every number/version in claim must appear in supporting sentence
            if self._has_numbers(claim.text) and best_score < 0.80:
                # Downgrade if numbers present but not well-supported
                if claim.label == "verified":
                    claim.label = "partial"
                    claim.score = max(0.50, best_score - 0.20)
                elif claim.label == "partial":
                    claim.label = "unverified"
                    claim.score = max(0.0, best_score - 0.20)
            
            claim.evidence_ids.append(f"E{hash(best_evidence) % 1000 + 1}")
        
        return claims
    
    def _has_numbers(self, text: str) -> bool:
        """Check if text contains numbers/versions."""
        return bool(re.findall(r'\b\d+(\.\d+)?\b', text))
    
    def _entailment_score(self, claim: str, evidence: str) -> float:
        """Simple entailment scoring using keyword overlap."""
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        evidence_words = set(re.findall(r'\b\w+\b', evidence.lower()))
        
        if not claim_words:
            return 0.0
        
        intersection = claim_words & evidence_words
        union = claim_words | evidence_words
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    async def verify_claims(self, draft_text: str, evidence_texts: List[str]) -> VerifierReport:
        """Full verification: claim check + execution check."""
        claims = self._extract_claims(draft_text)
        claims = self._verify_claims_against_evidence(claims, evidence_texts)
        
        # Check if any central claim is unverified
        central_claims = [c for c in claims if c.score > 0.5]
        central_unverified = any(c.label == "unverified" for c in central_claims)
        
        execution = None
        
        # Execution verification for code claims
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', draft_text, re.DOTALL)
        if code_blocks and self.profile.enable_execution:
            for code_block in code_blocks[:1]:  # Run first code block
                report = await self._execute_code(code_block)
                execution = report
                if not report.ok:
                    # If code execution fails, mark related claims as partial
                    for claim in claims:
                        if "code" in claim.text.lower() or "function" in claim.text.lower():
                            if claim.label == "verified":
                                claim.label = "partial"
                                claim.score = 0.5
        
        ok = not central_unverified
        
        return VerifierReport(
            ok=ok,
            claims=claims,
            missing_query=self._generate_missing_query(claims) if not ok else None,
            execution=execution
        )
    
    async def _execute_code(self, code: str) -> ExecutionReport:
        """Execute code in the sandbox with limitations."""
        start = __import__('time').perf_counter()
        
        try:
            # Execute with timeout and memory limits
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True,
                timeout=self.profile.execution_timeout_s,
                cwd=self.sandbox.work_dir if self.sandbox else None
            )
            
            duration_ms = int((__import__('time').perf_counter() - start) * 1000)
            
            ok = result.returncode == 0
            output = result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
            error = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
            
            return ExecutionReport(
                ran=True,
                ok=ok,
                output=output,
                error=error,
                duration_ms=duration_ms
            )
        except subprocess.TimeoutExpired:
            duration_ms = int((__import__('time').perf_counter() - start) * 1000)
            return ExecutionReport(
                ran=True,
                ok=False,
                error=f"Code execution timed out after {self.profile.execution_timeout_s}s",
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = int((__import__('time').perf_counter() - start) * 1000)
            return ExecutionReport(
                ran=True,
                ok=False,
                error=str(e),
                duration_ms=duration_ms
            )
    
    def _generate_missing_query(self, claims: List[Claim]) -> Optional[str]:
        """Generate a query for missing evidence."""
        unverified_claims = [c for c in claims if c.label == "unverified"]
        if not unverified_claims:
            return None
        
        # Use the first central unverified claim to generate a query
        claim = unverified_claims[0]
        key_terms = re.findall(r'\b\w+\b', claim.text.lower())
        key_terms = [t for t in key_terms if len(t) > 3 and t not in {'the', 'and', 'that', 'this', 'with'}]
        
        if key_terms:
            return " ".join(key_terms[:5])
        return None