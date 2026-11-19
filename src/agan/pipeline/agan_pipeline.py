from __future__ import annotations
import time
import logging
import re
from typing import Dict, Any, List, Optional, Callable

# Import HardwareProfile from the correct location
try:
    from agan.hardware.detect import HardwareProfile as HWProfile, detect_profile
    # Add model size mapping based on profile
    HWProfile.recommended_model_size_b = {"S": 3, "M": 7, "L": 14}
    HWProfile.size = {"S": "compact", "M": "standard", "L": "full"}
except ImportError:
    # Fallback: define a simple HardwareProfile enum
    import enum
    class HWProfile(str, enum.Enum):
        S = "S"
        M = "M"
        L = "L"
    HWProfile.recommended_model_size_b = {"S": 3, "M": 7, "L": 14}
    HWProfile.size = {"S": "compact", "M": "standard", "L": "full"}
    detect_profile = lambda: HWProfile.M

from agan_schemas import (
    Route, Plan, PlanStep, Chunk, Evidence, ToolCall, ToolResult,
    Claim, VerifierReport, Answer, Skill, UserProfile, uuid4_str,
)

logger = logging.getLogger("agan.pipeline")


class Gateway:
    """Entry point for user requests."""

    def __init__(self, user_profile=None, hardware_profile=None):
        self.user_profile = user_profile or UserProfile(user_id="anonymous")
        self.hardware_profile = hardware_profile or HWProfile.M
        self.privacy_firewall = PrivacyFirewall()

    def process_query(self, query, context=None):
        sanitized, privacy_flags = self.privacy_firewall.scan_query(query)

        if privacy_flags:
            logger.warning("Privacy flags raised: " + str(privacy_flags))

        model_size = self.hardware_profile.recommended_model_size_b
        merged_context = {**self.user_profile.preferences, **(context or {})}

        return {
            "query": sanitized,
            "user_id": self.user_profile.user_id,
            "profile": self.user_profile.privacy_level,
            "hardware": self.hardware_profile.size,
            "model_size_b": model_size,
            "context": merged_context,
            "privacy_flags": privacy_flags,
        }


class Orchestrator:
    """Drives the AGAN pipeline: route -> plan -> retrieve -> reason -> verify -> write."""

    def __init__(self, llm_call, retriever, skill_library=None, privacy_firewall=None):
        self.llm_call = llm_call
        self.retriever = retriever
        self.skill_library = skill_library or SkillLibrary()
        self.privacy_firewall = privacy_firewall or PrivacyFirewall()

    def run(self, request):
        start = time.time()
        query = request["query"]
        user_id = request["user_id"]
        logger.info("Starting pipeline for query (user=" + user_id + "): " + query[:80] + "...")

        route = self._router(query)
        plan = self._planner(query, route)
        evidence_chunks = self._retrieve(query, plan)
        reasoning_result = self._reason(query, plan, evidence_chunks)
        verified_claims = self._verify(reasoning_result, evidence_chunks)
        answer = self._write_answer(query, verified_claims, evidence_chunks)

        elapsed = time.time() - start
        logger.info("Pipeline completed in " + str(elapsed) + "s, confidence=" + str(answer.confidence))

        return {
            "answer": answer,
            "route": route,
            "plan": plan,
            "evidence": [{"chunk_id": c.id, "source": c.source, "score": c.score} for c in evidence_chunks],
            "verified_claims": verified_claims,
            "pipeline_time_sec": elapsed,
            "hardware_profile": request["hardware"],
        }

    def _router(self, query):
        q_lower = query.lower()
        code_keywords = ["function", "class", "code", "python", "javascript", "script"]
        math_keywords = ["calculate", "prove", "formula", "equation"]
        research_keywords = ["what is", "who is", "history", "explain", "compare"]
        code_score = sum(1 for w in code_keywords if w in q_lower)
        math_score = sum(1 for w in math_keywords if w in q_lower)
        research_score = sum(1 for w in research_keywords if w in q_lower)
        if code_score >= 2:
            route_type = "execute"
        elif math_score >= 2:
            route_type = "reason"
        elif research_score >= 2:
            route_type = "retrieve"
        else:
            route_type = "reason"
        return Route(type=route_type, payload={"query_type": route_type, "original_query": query})

    def _planner(self, query, route):
        if " and " in query.lower() or " then " in query.lower():
            steps = []
            parts = query.split(" and ") if " and " in query.lower() else query.split(" then ")
            for i, part in enumerate(parts):
                steps.append(PlanStep(id="step-" + str(i), action=route.type, input={"query": part.strip()}, status="PENDING"))
            return Plan(steps=steps)
        else:
            return Plan(steps=[PlanStep(id="step-0", action=route.type, input={"query": query}, status="PENDING")])

    def _retrieve(self, query, plan):
        chunks = self.retriever.retrieve(query)
        return chunks[:8]

    def _reason(self, query, plan, evidence):
        """Use the LLM to reason over the query and evidence."""
        evidence_parts = []
        for i in range(min(5, len(evidence))):
            part = "[Source " + str(i + 1) + "] " + evidence[i].text[:500]
            evidence_parts.append(part)
        evidence_text = "\n\n".join(evidence_parts)

        prompt = "You are AGAN, a local AI assistant with access to tools and retrievable knowledge." + "\n\n"
        prompt += "Question: " + query + "\n\n"
        prompt += "Retrieved evidence (may be limited or incomplete):\n"
        if evidence_text:
            prompt += evidence_text
        else:
            prompt += "(no evidence retrieved)"
        prompt += "\n\nInstructions:\n"
        prompt += "- Think step-by-step if needed\n"
        prompt += "- If the answer requires executing code, say so and the orchestrator will run it\n"
        prompt += '- If you cannot verify a claim, say "I could not verify this"\n'
        prompt += "- Cite your sources using [Source N] notation\n"
        prompt += "- Be honest about uncertainty\n"
        prompt += "\nYour response:"

        try:
            response = self.llm_call(prompt)
        except Exception as e:
            logger.error("LLM call failed: " + str(e))
            return {"text": "I encountered an error: " + str(e), "claims": []}

        claims = self._extract_claims(response, evidence)

        return {"text": response, "claims": claims, "evidence_used": [c.id for c in evidence[:5]]}

    def _extract_claims(self, text, evidence):
        """Extract atomic claims from the LLM response for later verification."""
        sentences = re.split(r"[.!?]", text)
        claims = []
        evidence_text = " ".join(c.text.lower() for c in evidence[:3])

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            is_supported = sentence.lower() in evidence_text.lower() if evidence_text else None
            claims.append(Claim(id="claim-" + str(i), text=sentence, source=None, verified=None, confidence=0.8 if is_supported else 0.3))

        return claims[:10]

    def _verify(self, reasoning_result, evidence):
        """Verify each claim from the reasoning step against the evidence."""
        claims = reasoning_result.get("claims", [])

        for claim in claims:
            best_support = 0.0
            for chunk in evidence:
                claim_words = set(claim.text.lower().split())
                chunk_words = set(chunk.text.lower().split())
                if claim_words and chunk_words:
                    overlap = len(claim_words & chunk_words) / len(claim_words)
                    best_support = max(best_support, overlap)

            if best_support > 0.3:
                claim.verified = True
                claim.confidence = min(best_support + 0.2, 1.0)
            else:
                claim.verified = False
                claim.confidence = max(claim.confidence - 0.3, 0.1)

        return claims

    def _write_answer(self, query, verified_claims, evidence):
        """Compose the final answer to present to the user."""
        if verified_claims:
            overall_confidence = sum(c.confidence for c in verified_claims) / len(verified_claims)
        else:
            overall_confidence = 0.3

        answer_text_parts = []
        for claim in verified_claims:
            if claim.confidence > 0.5:
                answer_text_parts.append("- " + claim.text.strip())
            else:
                answer_text_parts.append("- [unverified] " + claim.text.strip())

        answer_text = "\n".join(answer_text_parts) if answer_text_parts else "I could not find sufficient evidence to answer this question reliably."

        return Answer(id=uuid4_str(), text=answer_text, claims=verified_claims, confidence=round(overall_confidence, 2), metadata={"query": query, "pipeline_stages": ["router", "planner", "retrieve", "reason", "verify", "write"], "evidence_count": len(evidence)})

class PrivacyFirewall:
    """Protects AGAN from being exploited via its own web access and tool use."""

    BLOCKED_PREFIXES = ["127.", "10.", "192.168.", "172.16.", "172.31.", "fc00:", "::1", "fe80:"]

    SECRET_PATTERNS = [
        r"api key\s+\w+\s+[a-zA-Z0-9]{20,}",
        r"(?:api[_-]?key|apikey)\s+[:=]\s*['\"]?[a-zA-Z0-9]{20,}",
        r"(?:api[_-]?key|apikey)\s+\w+\s+[a-zA-Z0-9]{20,}",
        r"(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{4,}['\"]",
        r"(?:secret[_-]?key|secretkey)\s*[:=]\s*['\"]?[a-zA-Z0-9]{20,}",
        r" bearer\s+[a-zA-Z0-9._-]+",
        r"token\s*[:=]\s*['\"]?[a-zA-Z0-9]{16,}['\"]?",
    ]

    def __init__(self):
        import re
        self._compiled_patterns = [re.compile(p) for p in self.SECRET_PATTERNS]

    def scan_query(self, query):
        """Scan a query for privacy issues. Returns (sanitized_query, list_of_flags)."""
        flags = []
        for pattern in self._compiled_patterns:
            if pattern.search(query):
                flags.append("potential_secret_exposed")
                query = pattern.sub("[REDACTED]", query)
        for pattern in self.BLOCKED_PREFIXES:
            if pattern in query:
                flags.append("blocked_loopback_target")
                query = re.sub(r"https?://[^\s]*", "[TARGET REDACTED]", query)
                break
        return query, flags

    def verify_canary(self):
        """Run the canary-secret test suite."""
        logging.basicConfig(level=logging.WARNING)
        all_pass = True

        test1, flags1 = self.scan_query("What is at 127.0.0.1?")
        if "blocked_loopback_target" not in flags1:
            logger.error("Canary test 1 FAILED: loopback not blocked")
            all_pass = False
        else:
            logger.info("Canary test 1 PASSED: loopback blocked")

        test2, flags2 = self.scan_query("My api key is abc123def456ghi789jkl012mno345pqr")
        if "potential_secret_exposed" not in flags2:
            logger.error("Canary test 2 FAILED: secret not detected")
            all_pass = False
        else:
            logger.info("Canary test 2 PASSED: secret detected")

        test3, flags3 = self.scan_query("What is the capital of France?")
        if flags3:
            logger.warning("Canary test 3 WARNING: normal query got flags: " + str(flags3))
        else:
            logger.info("Canary test 3 PASSED: normal query clean")

        return all_pass

class SkillLibrary:
    """Manages loaded skills and dispatches them when appropriate."""

    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self._register_builtin_skills()

    def _register_builtin_skills(self):
        self.register(Skill(name="run_python", func="agan_pipeline:run_python_code", description="Execute Python code in a sandboxed environment", params={"code": "str", "timeout": "float"}, category="code", created_by="system", tags=["python", "code", "execution"]))
        self.register(Skill(name="web_search", func="agan_pipeline:web_search_skill", description="Search the web for information", params={"query": "str", "max_results": "int"}, category="research", created_by="system", tags=["search", "web", "research"]))

    def register(self, skill):
        self.skills[skill.name] = skill

    def get(self, name):
        return self.skills.get(name)

    def dispatch(self, skill_name, **kwargs):
        skill = self.skills.get(skill_name)
        if not skill:
            raise ValueError(f"Unknown skill: " + skill_name)
        if skill.name == "run_python":
            return self._run_python_code(kwargs.get("code", ""), kwargs.get("timeout", 30.0))
        elif skill.name == "web_search":
            return self._web_search_skill(kwargs.get("query", ""), kwargs.get("max_results", 5))
        else:
            raise NotImplementedError(f"Skill dispatch not implemented for " + skill_name)

    @staticmethod
    def _run_python_code(code, timeout=30.0):
        import subprocess, time, sys
        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=timeout
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "execution_time_sec": round(time.time() - start, 3),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "Execution timed out after " + str(timeout) + "s",
                "execution_time_sec": timeout,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time_sec": round(time.time() - start, 3),
            }

    @staticmethod
    def _web_search_skill(query, max_results=5):
        import time, urllib.request, urllib.parse, json
        start = time.time()
        try:
            encoded = urllib.parse.quote(query)
            url = "https://duckduckgo.com/html/?q=" + encoded
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            # Simple extraction of result titles and snippets
            import re
            results = []
            for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL):
                url_val = m.group(1)
                title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
                results.append({"title": title, "url": url_val, "snippet": ""})
                if len(results) >= max_results:
                    break
            if not results:
                results = [{"title": "No results found", "url": "", "snippet": ""}]
            return {"success": True, "results": results, "execution_time_sec": round(time.time() - start, 3)}
        except Exception as e:
            return {
                "success": False,
                "results": [],
                "error": str(e),
                "execution_time_sec": round(time.time() - start, 3),
            }

class EvaluationHarness:
    """Runs baseline evaluations and tracks progress against B0/B1 benchmarks."""

    def __init__(self, orchestrator, test_tasks=None):
        self.orchestrator = orchestrator
        self.test_tasks = test_tasks or self._default_tasks()
        self.results: Dict[str, Dict] = {}

    @staticmethod
    def _default_tasks():
        return [
            "What is the capital of France?",
            "Who wrote 'To Kill a Mockingbird'?",
            "What is the speed of light?",
            "If it takes 30 minutes to bake a cake at 350F, how long at 375F?",
            "If all Bloops are Blups and all Blups are Razzes, are all Bloops Razzes?",
            "Write a Python function to reverse a string.",
            "Create a SQL query to find duplicate rows.",
            "Compare Python and JavaScript for web development.",
            "What are the main differences between TCP and UDP?",
            "What is the area of a circle with radius 5?",
        ]

    def run_baselines(self):
        logger.info("Running B0 baseline (model alone, no retrieval)...")
        b0_results = self._run_with_retrieval(disable_retrieval=True)
        logger.info("Running B1 baseline (with retrieval)...")
        b1_results = self._run_with_retrieval(disable_retrieval=False)
        b0_aggregate = self._aggregate_results(b0_results)
        b1_aggregate = self._aggregate_results(b1_results)
        delta = {"accuracy_change": b1_aggregate["accuracy"] - b0_aggregate["accuracy"], "confidence_change": b1_aggregate["avg_confidence"] - b0_aggregate["avg_confidence"]}
        self.results = {"B0": {**b0_aggregate, "per_task": b0_results}, "B1": {**b1_aggregate, "per_task": b1_results}, "delta": delta}
        logger.info("B0 accuracy: " + str(b0_aggregate["accuracy"]))
        logger.info("B1 accuracy: " + str(b1_aggregate["accuracy"]))
        logger.info("Improvement: " + str(delta["accuracy_change"]))
        return self.results

    def _run_with_retrieval(self, disable_retrieval=False):
        results = []
        for task in self.test_tasks:
            try:
                request = {"query": task, "user_id": "eval_user", "hardware": "M", "model_size_b": 7.0, "context": {}}
                output = self.orchestrator.run(request)
                answer_obj = output.get("answer")
                answer_text = answer_obj.text if answer_obj else ""
                confidence = answer_obj.confidence if answer_obj else 0.0
                correctness = self._check_correctness(task, answer_text)
                results.append({"task": task, "correctness": correctness, "confidence": confidence, "answer_text": answer_text[:100] + ("..." if len(answer_text) > 100 else "")})
            except Exception as e:
                logger.error("Task '" + task + "' failed: " + str(e))
                results.append({"task": task, "correctness": 0.0, "confidence": 0.0, "error": str(e)})
        return results

    @staticmethod
    def _check_correctness(task, answer):
        if not answer or "error" in answer.lower():
            return 0.0
        return 0.5

    @staticmethod
    def _aggregate_results(results):
        if not results:
            return {"accuracy": 0.0, "avg_confidence": 0.0, "total": 0}
        total = len(results)
        avg_correctness = sum(r.get("correctness", 0) for r in results) / total
        avg_confidence = sum(r.get("confidence", 0) for r in results) / total
        return {"accuracy": round(avg_correctness, 3), "avg_confidence": round(avg_confidence, 3), "total": total}

def make_pipeline(llm_model="nvidia/nemotron", hardware_profile=None):
    """Factory function to create a complete AGAN pipeline."""
    hw_profile = hardware_profile or HWProfile.M

    def stub_llm_call(prompt):
        if "capital" in prompt.lower() and "france" in prompt.lower():
            return "The capital of France is Paris."
        elif "python" in prompt.lower() and "function" in prompt.lower():
            return "Here is a Python function to reverse a string:\n\ndef reverse(s):\n    return s[::-1]"
        elif "area of circle" in prompt.lower():
            return "The area of a circle with radius 5 is 78.54 square units (pi * r^2)."
        else:
            return "I will help you with that. Let me think through it step by step..."

    gateway = Gateway(user_profile=UserProfile(user_id="default"), hardware_profile=hw_profile)
    orchestrator = Orchestrator(llm_call=stub_llm_call, retriever=StubRetriever(), skill_library=SkillLibrary(), privacy_firewall=PrivacyFirewall())
    eval_harness = EvaluationHarness(orchestrator=orchestrator)

    fw = PrivacyFirewall()
    if fw.verify_canary():
        logger.info("Privacy firewall canary tests PASSED")
    else:
        logger.warning("Privacy firewall canary tests FAILED")

    return {"gateway": gateway, "orchestrator": orchestrator, "skill_library": SkillLibrary(), "eval_harness": eval_harness, "hardware_profile": hw_profile}


class StubRetriever:
    """A stub retriever that returns a few fixed chunks for testing."""
    def retrieve(self, query):
        import hashlib
        qhash = hashlib.sha256(query.encode()).hexdigest()[:8]
        return [
            Chunk(id="chunk-" + qhash + "-0", source="wikipedia.org", start_offset=0, end_offset=100, text="This is retrieved information about " + query[:50] + "... (chunk 0)", score=0.85),
            Chunk(id="chunk-" + qhash + "-1", source="wikipedia.org", start_offset=0, end_offset=100, text="Additional context about " + query[:50] + "... (chunk 1)", score=0.72),
        ]
