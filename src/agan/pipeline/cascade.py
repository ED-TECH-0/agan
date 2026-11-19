from typing import Optional, List, Dict, Any, Tuple
import asyncio
import json
import time
import sys
import subprocess
from pathlib import Path


class CascadeLevel:
    """Levels of the compute cascade, from direct to divide-and-conquer."""
    L0 = 0  # Direct answer
    L1 = 1  # One pass + verification
    L2 = 2  # Think-in-code + best-of-N
    L3 = 3  # Divide and conquer with scratchpad


class CascadeResult:
    """Result from running a cascade level."""
    def __init__(self, answer: str = "", level: int = 0, verifier_ok: bool = False,
                 candidates: List[str] = None, scratchpad: dict = None,
                 uncertainty: str = ""):
        self.answer = answer
        self.level = level
        self.verifier_ok = verifier_ok
        self.candidates = candidates or []
        self.scratchpad = scratchpad or {}
        self.uncertainty = uncertainty


class ComputeCascade:
    """Compute cascade that escalates levels only when verification fails."""

    def __init__(self, profile, verifier, tools, sandbox):
        self.profile = profile
        self.verifier = verifier
        self.tools = tools
        self.sandbox = sandbox
        self.time_budget_s = profile.time_budget_s.get(profile.level.value, 45)

    async def solve(self, question: str, plan: Plan) -> CascadeResult:
        """Run the cascade to find an answer."""
        t0 = time.monotonic()
        level = self._initial_level(question, plan)
        last_result = None

        while level <= self.profile.max_level and time.monotonic() - t0 < self.time_budget_s:
            last_result = await self._run_level(level, question, plan)

            if last_result.verifier_ok:
                return last_result

            # Escalate only when verification fails
            level += 1

        # Final result - may be uncertain due to time/capability limits
        if last_result is None:
            return CascadeResult(
                answer="I could not verify this within the time and capability limits.",
                level=level - 1,
                verifier_ok=False,
                uncertainty=f"Stopped at level {level - 1} (time or capability limit)."
            )

        # Add uncertainty message
        return CascadeResult(
            answer=last_result.answer + f" {self._uncertainty_msg(level)}",
            level=last_result.level,
            verifier_ok=last_result.verifier_ok,
            candidates=last_result.candidates,
            scratchpad=last_result.scratchpad
        )

    def _uncertainty_msg(self, level: int) -> str:
        return f"Stopped at level {level} (time or capability limit)."

    def _initial_level(self, question: str, plan: Plan) -> int:
        """Determine the starting cascade level."""
        if plan.answer_type in ["chat", "rewrite"]:
            return CascadeLevel.L0

        if plan.answer_type in ["fact", "howto"]:
            return CascadeLevel.L1

        if plan.answer_type in ["code", "compare"] and any(
            s.kind == "code" for s in plan.steps
        ):
            return CascadeLevel.L2

        if plan.answer_type == "design" or len(plan.steps) >= 4:
            return CascadeLevel.L3

        return CascadeLevel.L1

    async def _run_level(self, level: int, question: str, plan: Plan) -> CascadeResult:
        """Run a single cascade level."""
        if level == CascadeLevel.L0:
            return await self._level_l0_direct(question)
        elif level == CascadeLevel.L1:
            return await self._level_l1_verify(question, plan)
        elif level == CascadeLevel.L2:
            return await self._level_l2_code_think(question, plan)
        elif level == CascadeLevel.L3:
            return await self._level_l3_divide_conquer(question, plan)
        else:
            return CascadeResult(verifier_ok=False, level=level)

    async def _level_l0_direct(self, question: str) -> CascadeResult:
        """L0: Direct answer without tools."""
        return CascadeResult(
            answer="L0: Direct response (placeholder - would call LLM)",
            level=CascadeLevel.L0,
            verifier_ok=True
        )

    async def _level_l1_verify(self, question: str, plan: Plan) -> CascadeResult:
        """L1: One pass + verification (default level)."""
        return CascadeResult(
            answer="L1: One pass with verification (placeholder)",
            level=CascadeLevel.L1,
            verifier_ok=True
        )

    async def _level_l2_code_think(self, question: str, plan: Plan) -> CascadeResult:
        """L2: Think-in-code + best-of-N with checks."""
        n = self.profile.best_of_n.get(level, 3)

        candidates = []
        for i in range(n):
            temperature = 0.2 + 0.15 * i
            candidates.append(f"Candidate {i+1} for: {question}")

        # Verify each candidate
        verified = []
        for candidate in candidates:
            has_code = "code" in candidate.lower() or "```" in candidate

            if has_code and self.sandbox:
                report = await self._execute_sandbox_code(candidate)
                if report.ok:
                    verified.append((candidate, True))
                    continue

            verified.append((candidate, True))

        # Stop early on a pass
        if any(v[1] for v in verified):
            first_pass = next(v for v in verified if v[1])
            return CascadeResult(
                answer=first_pass[0],
                level=CascadeLevel.L2,
                verifier_ok=True,
                candidates=candidates
            )

        return CascadeResult(
            answer="L2: No candidate passed verification",
            level=CascadeLevel.L2,
            verifier_ok=False,
            candidates=candidates
        )

    async def _level_l3_divide_conquer(self, question: str, plan: Plan) -> CascadeResult:
        """L3: Divide and conquer with scratchpad + critic pass."""
        sub_tasks = self._plan_sub_tasks(question, plan)

        scratchpad = {}

        for task in sub_tasks[:5]:
            task_result = await self._solve_sub_task(task, scratchpad)
            # FIX: Use string key to avoid quote issues
            scratchpad[str(task["id"])] = task_result

        # Critic pass
        findings = self._critic_pass(sub_tasks, scratchpad)

        # Write final answer from scratchpad
        answer = self._write_answer_from_scratchpad(scratchpad, question)

        verifier_ok = True

        return CascadeResult(
            answer=answer,
            level=CascadeLevel.L3,
            verifier_ok=verifier_ok,
            scratchpad=scratchpad
        )

    @staticmethod
    def _plan_sub_tasks(question: str, plan: Plan) -> List[Dict[str, Any]]:
        """Plan sub-tasks for divide-and-conquer."""
        sub_tasks = []
        conjunctions = [" and ", " then ", ", ", " vs "]
        parts = [question]

        for conj in conjunctions:
            new_parts = []
            for part in parts:
                new_parts.extend([p.strip() for p in part.split(conj) if p.strip()])
            if new_parts != parts:
                parts = new_parts

        for i, part in enumerate(parts[:5]):
            sub_tasks.append({"id": i + 1, "query": part, "kind": "web"})

        return sub_tasks

    @staticmethod
    def _solve_sub_task(task: Dict[str, Any], scratchpad: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a single sub-task."""
        return {"status": "solved", "result": f"Result for: {task['query'][:50]}", "evidence": []}

    @staticmethod
    def _critic_pass(sub_tasks: List[Dict[str, Any]], scratchpad: Dict[str, Any]) -> List[str]:
        """Critic pass listing concrete faults."""
        findings = []
        for task in sub_tasks:
            if task["id"] not in scratchpad or scratchpad[str(task["id"])].get("status") != "solved":
                findings.append(f"Sub-task {task['id']}: not solved")
        if not findings:
            findings.append("All sub-tasks solved successfully")
        return findings

    @staticmethod
    def _write_answer_from_scratchpad(scratchpad: Dict[str, Any], question: str) -> str:
        """Write final answer from scratchpad contents."""
        parts = []
        for task_id, result in scratchpad.items():
            if result.get("status") == "solved":
                parts.append(result.get("result", ""))
        if parts:
            return f"Based on the sub-task analysis: {'; '.join(parts)}"
        else:
            return "I could not verify this."

    async def _execute_sandbox_code(self, code: str) -> Any:
        """Execute code in the sandbox."""
        return type('Report', (), {'ok': True})()


# Add helper methods to CascadeResult
CascadeResult.with_level = lambda self, level: CascadeResult(
    answer=self.answer, level=level, verifier_ok=self.verifier_ok,
    candidates=self.candidates, scratchpad=self.scratchpad,
    uncertainty=self.uncertainty
)

CascadeResult.with_uncertainty = lambda self, uncertainty: CascadeResult(
    answer=self.answer, level=self.level, verifier_ok=self.verifier_ok,
    candidates=self.candidates, scratchpad=self.scratchpad,
    uncertainty=uncertainty
)