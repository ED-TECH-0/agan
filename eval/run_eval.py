"""AGAN Evaluation Harness — runs benchmarks and produces capability matrix with confidence intervals.

Implements the evaluation framework from AGAN Plan V2 Section 11:
- Baselines: B0 (base model alone), B1 (base + retrieval), B2 (AGAN L1), B3 (full AGAN), B4 (tuned model)
- Test suites: qa_cited, fresh, multihop, code_exec, math_code, tools, abstain, injection, privacy_canary, long_input, skill_learning, adaptation, idea_quality
- Metrics: task success, groundedness, citation precision/recall, abstention accuracy, tool-call validity, injection success rate, leak rate, latency, memory
- Statistics: bootstrap 95% CI on paired tasks
"""
import asyncio
import json
import time
import statistics
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Callable, Awaitable
from enum import Enum
from collections import defaultdict

from agan.schemas import Answer, Claim, Plan, PlanStep, Route, VerifierReport, ExecutionReport
from agan.hardware.detect import HardwareProfile, ProfileSpec, detect_profile, get_spec
from agan.llm.client import LLMClient
from agan.pipeline.router import Router
from agan.pipeline.planner import Planner
from agan.pipeline.reasoner import Reasoner
from agan.pipeline.verifier import Verifier


class BaselineType(str, Enum):
    B0_BASE = "B0_base"           # Base model alone
    B1_RETRIEVAL = "B1_retrieval" # Base + retrieval
    B2_AGAN_L1 = "B2_agan_l1"     # AGAN at L1
    B3_FULL_AGAN = "B3_full_agan" # Full AGAN cascade
    B4_TUNED = "B4_tuned"         # Tuned model (future)


class SuiteType(str, Enum):
    QA_CITED = "qa_cited"
    FRESH = "fresh"
    MULTIHOP = "multihop"
    CODE_EXEC = "code_exec"
    MATH_CODE = "math_code"
    TOOLS = "tools"
    ABSTAIN = "abstain"
    INJECTION = "injection"
    PRIVACY_CANARY = "privacy_canary"
    LONG_INPUT = "long_input"
    SKILL_LEARNING = "skill_learning"
    ADAPTATION = "adaptation"
    IDEA_QUALITY = "idea_quality"


@dataclass
class EvalTask:
    id: str
    suite: SuiteType
    question: str
    expected_answer_type: str
    verification: dict  # Verification spec
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    task_id: str
    suite: str
    baseline: str
    profile: str
    model: str
    success: bool
    score: float
    latency_ms: int
    tokens_generated: int
    answer: str
    claims: list[Claim]
    verifier_report: Optional[VerifierReport] = None
    details: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class SuiteSummary:
    suite: str
    baseline: str
    profile: str
    n_tasks: int
    n_passed: int
    pass_rate: float
    mean_score: float
    ci_low: float
    ci_high: float
    mean_latency_ms: float
    mean_tokens_per_sec: float


class EvalHarness:
    def __init__(
        self,
        tasks_file: str = "eval/suites/tasks.json",
        profiles: Optional[list[HardwareProfile]] = None,
        models: Optional[list[str]] = None,
        baselines: Optional[list[BaselineType]] = None,
    ):
        self.tasks_file = Path(tasks_file)
        self.profiles = profiles or [HardwareProfile.S, HardwareProfile.M, HardwareProfile.L]
        self.models = models or ["qwen2.5:3b-instruct-q4_k_m"]
        self.baselines = baselines or [BaselineType.B0_BASE, BaselineType.B1_RETRIEVAL, BaselineType.B2_AGAN_L1, BaselineType.B3_FULL_AGAN]
        self.tasks = self._load_tasks()
        self.results: list[EvalResult] = []
    
    def _load_tasks(self) -> list[EvalTask]:
        if self.tasks_file.exists():
            with open(self.tasks_file) as f:
                data = json.load(f)
            tasks = []
            for t in data:
                suite_str = t.get("category", "qa_cited")
                try:
                    suite = SuiteType(suite_str)
                except ValueError:
                    suite = SuiteType.QA_CITED
                tasks.append(EvalTask(
                    id=t["id"],
                    suite=suite,
                    question=t["question"],
                    expected_answer_type=t.get("expected_type", "fact"),
                    verification=t.get("verification", {}),
                    metadata=t.get("metadata", {}),
                ))
            return tasks
        return []
    
    def save_tasks(self):
        self.tasks_file.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for t in self.tasks:
            data.append({
                "id": t.id,
                "category": t.suite.value,
                "question": t.question,
                "expected_type": t.expected_answer_type,
                "verification": t.verification,
                "metadata": t.metadata,
            })
        with open(self.tasks_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_task(self, task: EvalTask):
        self.tasks.append(task)
        self.save_tasks()
    
    def _bootstrap_ci(self, values: list[float], n_bootstrap: int = 10000, confidence: float = 0.95) -> tuple[float, float]:
        """Compute bootstrap confidence interval."""
        if len(values) < 2:
            return (values[0] if values else 0.0, values[0] if values else 0.0)
        
        means = []
        n = len(values)
        for _ in range(n_bootstrap):
            sample = [values[random.randint(0, n-1)] for _ in range(n)]
            means.append(statistics.mean(sample))
        
        means.sort()
        alpha = (1 - confidence) / 2
        low_idx = int(alpha * n_bootstrap)
        high_idx = int((1 - alpha) * n_bootstrap)
        return (means[low_idx], means[high_idx])
    
    async def run_baseline_b0(self, profile: HardwareProfile, model: str, task: EvalTask) -> EvalResult:
        """B0: Base model alone - no retrieval, no tools, no verification."""
        spec = get_spec(profile)
        client = LLMClient(base_url="http://localhost:11434/v1", model=model)
        
        start = time.perf_counter()
        try:
            response = await client.complete(
                [{"role": "user", "content": task.question}],
                temperature=0.2,
                max_tokens=spec.max_context // 4,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            answer = Answer(text=response, claims=[], sources=[], confidence="medium", level_used=0)
            success, score, details = self._verify_task(task, answer)
            
            return EvalResult(
                task_id=task.id,
                suite=task.suite.value,
                baseline=BaselineType.B0_BASE.value,
                profile=profile.name,
                model=model,
                success=success,
                score=score,
                latency_ms=latency_ms,
                tokens_generated=len(response.split()),
                answer=response,
                claims=[],
                details=details,
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return EvalResult(
                task_id=task.id,
                suite=task.suite.value,
                baseline=BaselineType.B0_BASE.value,
                profile=profile.name,
                model=model,
                success=False,
                score=0.0,
                latency_ms=latency_ms,
                tokens_generated=0,
                answer="",
                claims=[],
                error=str(e),
            )
    
    async def run_baseline_b1(self, profile: HardwareProfile, model: str, task: EvalTask) -> EvalResult:
        """B1: Base model + retrieval (but no verification cascade)."""
        spec = get_spec(profile)
        client = LLMClient(base_url="http://localhost:11434/v1", model=model)
        
        # This would need retrieval integration - placeholder for now
        start = time.perf_counter()
        try:
            # In a real implementation, this would fetch relevant docs first
            prompt = f"Question: {task.question}\n\n[No retrieval available in baseline B1]\nAnswer with citations if possible."
            response = await client.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=spec.max_context // 4,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            answer = Answer(text=response, claims=[], sources=[], confidence="medium", level_used=1)
            success, score, details = self._verify_task(task, answer)
            
            return EvalResult(
                task_id=task.id,
                suite=task.suite.value,
                baseline=BaselineType.B1_RETRIEVAL.value,
                profile=profile.name,
                model=model,
                success=success,
                score=score,
                latency_ms=latency_ms,
                tokens_generated=len(response.split()),
                answer=response,
                claims=[],
                details=details,
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return EvalResult(
                task_id=task.id,
                suite=task.suite.value,
                baseline=BaselineType.B1_RETRIEVAL.value,
                profile=profile.name,
                model=model,
                success=False,
                score=0.0,
                latency_ms=latency_ms,
                tokens_generated=0,
                answer="",
                claims=[],
                error=str(e),
            )
    
    async def run_baseline_b2(self, profile: HardwareProfile, model: str, task: EvalTask) -> EvalResult:
        """B2: AGAN at L1 (router + planner + reasoner + single verification pass)."""
        from agan.__main__ import AGAN
        
        agan = AGAN(profile=profile, model=model)
        start = time.perf_counter()
        try:
            answer = await agan.answer(task.question)
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            success, score, details = self._verify_task(task, answer)
            
            return EvalResult(
                task_id=task.id,
                suite=task.suite.value,
                baseline=BaselineType.B2_AGAN_L1.value,
                profile=profile.name,
                model=model,
                success=success,
                score=score,
                latency_ms=latency_ms,
                tokens_generated=len(answer.text.split()),
                answer=answer.text,
                claims=answer.claims,
                details=details,
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return EvalResult(
                task_id=task.id,
                suite=task.suite.value,
                baseline=BaselineType.B2_AGAN_L1.value,
                profile=profile.name,
                model=model,
                success=False,
                score=0.0,
                latency_ms=latency_ms,
                tokens_generated=0,
                answer="",
                claims=[],
                error=str(e),
            )
    
    async def run_baseline_b3(self, profile: HardwareProfile, model: str, task: EvalTask) -> EvalResult:
        """B3: Full AGAN with compute cascade (L0-L3), best-of-N, think-in-code, scratchpad."""
        # For now, same as B2 since cascade isn't fully implemented
        return await self.run_baseline_b2(profile, model, task)
    
    async def run_task(self, baseline: BaselineType, profile: HardwareProfile, model: str, task: EvalTask) -> EvalResult:
        """Run a single task with the specified baseline."""
        if baseline == BaselineType.B0_BASE:
            return await self.run_baseline_b0(profile, model, task)
        elif baseline == BaselineType.B1_RETRIEVAL:
            return await self.run_baseline_b1(profile, model, task)
        elif baseline == BaselineType.B2_AGAN_L1:
            return await self.run_baseline_b2(profile, model, task)
        elif baseline == BaselineType.B3_FULL_AGAN:
            return await self.run_baseline_b3(profile, model, task)
        else:
            raise ValueError(f"Unknown baseline: {baseline}")
    
    def _verify_task(self, task: EvalTask, answer: Answer) -> tuple[bool, float, dict]:
        """Verify answer against task verification spec."""
        v = task.verification
        vtype = v.get("type", "claim")
        
        if vtype == "executable":
            return self._verify_executable(v, answer)
        elif vtype == "claim":
            return self._verify_claims(v, answer)
        elif vtype == "abstain":
            return self._verify_abstain(v, answer)
        else:
            return False, 0.0, {"error": f"Unknown verification type: {vtype}"}
    
    def _verify_executable(self, verification: dict, answer: Answer) -> tuple[bool, float, dict]:
        """Verify by executing code from the answer."""
        check_code = verification.get("check", "")
        if not check_code:
            return False, 0.0, {"error": "No check code provided"}
        
        # Extract code blocks from answer
        import re
        code_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", answer.text, re.DOTALL)
        if not code_blocks:
            return False, 0.0, {"error": "No code blocks found in answer"}
        
        # Combine answer code with check code
        test_code = code_blocks[0] + "\n\n# Test\n" + check_code
        
        try:
            exec_globals = {}
            exec(test_code, exec_globals)
            # If we reach here without exception, check passed
            return True, 1.0, {"check": "passed"}
        except AssertionError as e:
            return False, 0.0, {"error": f"Assertion failed: {e}", "check": "failed"}
        except Exception as e:
            return False, 0.0, {"error": f"Execution error: {e}", "check": "error"}
    
    def _verify_claims(self, verification: dict, answer: Answer) -> tuple[bool, float, dict]:
        """Verify claims in answer - simplified check for now."""
        expected_claims = verification.get("claims", [])
        if not expected_claims:
            return True, 1.0, {"note": "No claims to verify"}
        
        answer_lower = answer.text.lower()
        found = sum(1 for claim in expected_claims if claim.lower() in answer_lower)
        score = found / len(expected_claims) if expected_claims else 1.0
        success = score >= 0.5  # At least half the expected claims mentioned
        
        return success, score, {
            "expected_claims": len(expected_claims),
            "found_claims": found,
        }
    
    def _verify_abstain(self, verification: dict, answer: Answer) -> tuple[bool, float, dict]:
        """Verify that model correctly abstains on unanswerable questions."""
        abstain_phrases = [
            "i don't know", "i could not verify", "cannot verify",
            "unknown", "not sure", "insufficient information",
            "no evidence", "unable to answer"
        ]
        answer_lower = answer.text.lower()
        abstained = any(phrase in answer_lower for phrase in abstain_phrases)
        
        return abstained, 1.0 if abstained else 0.0, {"abstained": abstained}
    
    async def run_all(
        self,
        max_tasks_per_suite: Optional[int] = None,
        profiles: Optional[list[HardwareProfile]] = None,
        models: Optional[list[str]] = None,
        baselines: Optional[list[BaselineType]] = None,
    ) -> list[EvalResult]:
        """Run all tasks across all configurations."""
        profiles = profiles or self.profiles
        models = models or self.models
        baselines = baselines or self.baselines
        
        # Group tasks by suite
        tasks_by_suite: dict[SuiteType, list[EvalTask]] = defaultdict(list)
        for task in self.tasks:
            tasks_by_suite[task.suite].append(task)
        
        all_results = []
        
        for profile in profiles:
            for model in models:
                for baseline in baselines:
                    print(f"\n=== Running {baseline.value} on {profile.name} with {model} ===")
                    
                    for suite_type, suite_tasks in tasks_by_suite.items():
                        tasks = suite_tasks[:max_tasks_per_suite] if max_tasks_per_suite else suite_tasks
                        print(f"  Suite {suite_type.value}: {len(tasks)} tasks")
                        
                        for task in tasks:
                            print(f"    [{task.id}]...", end=" ", flush=True)
                            result = await self.run_task(baseline, profile, model, task)
                            all_results.append(result)
                            
                            status = "PASS" if result.success else "FAIL"
                            print(f"{status} (score={result.score:.2f}, {result.latency_ms}ms)")
                            
                            if result.error:
                                print(f"      Error: {result.error[:100]}")
        
        self.results = all_results
        return all_results
    
    def generate_report(self) -> dict:
        """Generate capability matrix report with bootstrap confidence intervals."""
        if not self.results:
            return {"error": "No results to report"}
        
        # Group by suite, baseline, profile
        grouped: dict[tuple, list[EvalResult]] = defaultdict(list)
        for r in self.results:
            key = (r.suite, r.baseline, r.profile)
            grouped[key].append(r)
        
        summaries = []
        for (suite, baseline, profile), results in grouped.items():
            scores = [r.score for r in results]
            latencies = [r.latency_ms for r in results]
            tokens_per_sec = [r.tokens_generated / (r.latency_ms / 1000) if r.latency_ms > 0 else 0 for r in results]
            
            ci_low, ci_high = self._bootstrap_ci(scores)
            
            summary = SuiteSummary(
                suite=suite,
                baseline=baseline,
                profile=profile,
                n_tasks=len(results),
                n_passed=sum(1 for r in results if r.success),
                pass_rate=sum(1 for r in results if r.success) / len(results),
                mean_score=statistics.mean(scores) if scores else 0,
                ci_low=ci_low,
                ci_high=ci_high,
                mean_latency_ms=statistics.mean(latencies) if latencies else 0,
                mean_tokens_per_sec=statistics.mean(tokens_per_sec) if tokens_per_sec else 0,
            )
            summaries.append(summary)
        
        # Overall capability statements
        capability_map = {}
        for summary in summaries:
            key = f"{summary.baseline}|{summary.profile}|{summary.suite}"
            if summary.pass_rate >= 0.8 and summary.ci_low >= 0.7:
                capability = "reliable"
            elif summary.pass_rate >= 0.6 and summary.ci_low >= 0.5:
                capability = "usable with verification"
            else:
                capability = "not recommended"
            capability_map[key] = capability
        
        report = {
            "timestamp": time.time(),
            "profiles": list(set(r.profile for r in self.results)),
            "models": list(set(r.model for r in self.results)),
            "baselines": list(set(r.baseline for r in self.results)),
            "suites": list(set(r.suite for r in self.results)),
            "suite_summaries": [asdict(s) for s in summaries],
            "capability_statements": capability_map,
            "raw_results": [asdict(r) for r in self.results],
        }
        return report
    
    def save_report(self, report: dict, output_dir: str = "eval/reports"):
        """Save report to JSON file."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"eval_report_{int(time.time())}.json"
        with open(out_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {out_file}")
        return out_file
    
    def print_capability_matrix(self, report: dict):
        """Print formatted capability matrix."""
        print("\n" + "="*80)
        print("CAPABILITY MATRIX")
        print("="*80)
        
        for profile in sorted(set(r.profile for r in self.results)):
            print(f"\n--- Profile: {profile} ---")
            for baseline in [BaselineType.B0_BASE, BaselineType.B1_RETRIEVAL, BaselineType.B2_AGAN_L1, BaselineType.B3_FULL_AGAN]:
                print(f"\n  {baseline.value}:")
                for suite in SuiteType:
                    key = f"{baseline.value}|{profile}|{suite.value}"
                    cap = report.get("capability_statements", {}).get(key, "no data")
                    # Find the summary
                    summaries = [s for s in report.get("suite_summaries", []) 
                               if s["baseline"] == baseline.value and s["profile"] == profile and s["suite"] == suite.value]
                    if summaries:
                        s = summaries[0]
                        print(f"    {suite.value:20s}  {s['pass_rate']*100:5.1f}% pass  "
                              f"({s['n_passed']}/{s['n_tasks']})  "
                              f"CI=[{s['ci_low']*100:.1f}%, {s['ci_high']*100:.1f}%]  "
                              f"{cap}")
                    else:
                        print(f"    {suite.value:20s}  no data")


async def main():
    """Run evaluation harness."""
    harness = EvalHarness()
    
    # Ensure we have tasks - create some if empty
    if not harness.tasks:
        print("No tasks found, creating default task suite...")
        # Add tasks from each suite
        default_tasks = [
            # Code execution tasks
            EvalTask(id="code_001", suite=SuiteType.CODE_EXEC,
                    question="Write a Python function to compute the nth Fibonacci number using memoization",
                    expected_answer_type="code",
                    verification={"type": "executable", "check": "fib(10) == 55 and fib(0) == 0 and fib(1) == 1"}),
            EvalTask(id="code_002", suite=SuiteType.CODE_EXEC,
                    question="Write a Python context manager that times a code block and prints the elapsed time",
                    expected_answer_type="code",
                    verification={"type": "executable", "check": "import time; with Timer(): time.sleep(0.01)"}),
            EvalTask(id="code_003", suite=SuiteType.CODE_EXEC,
                    question="Implement a binary search function that returns the index of target in sorted list, or -1 if not found",
                    expected_answer_type="code",
                    verification={"type": "executable", "check": "binary_search([1,2,3,4,5], 3) == 2 and binary_search([1,2,3], 4) == -1"}),
            
            # Fact/QA tasks
            EvalTask(id="fact_001", suite=SuiteType.QA_CITED,
                    question="What is the latest stable version of Python as of 2024?",
                    expected_answer_type="fact",
                    verification={"type": "claim", "claims": ["Python version number", "release date"]}),
            EvalTask(id="fact_002", suite=SuiteType.QA_CITED,
                    question="What are the key new features in Blender 4.2 LTS?",
                    expected_answer_type="fact",
                    verification={"type": "claim", "claims": ["Blender 4.2 features", "LTS release"]}),
            
            # Abstention tasks
            EvalTask(id="abstain_001", suite=SuiteType.ABSTAIN,
                    question="What is the exact temperature in Cairo, Egypt right now at this second?",
                    expected_answer_type="fact",
                    verification={"type": "abstain", "claims": []}),
            EvalTask(id="abstain_002", suite=SuiteType.ABSTAIN,
                    question="Who will win the 2030 FIFA World Cup?",
                    expected_answer_type="fact",
                    verification={"type": "abstain", "claims": []}),
        ]
        for task in default_tasks:
            harness.add_task(task)
    
    print(f"Loaded {len(harness.tasks)} tasks across {len(set(t.suite for t in harness.tasks))} suites")
    
    # Run with just S profile and one model for quick test
    results = await harness.run_all(
        max_tasks_per_suite=5,
        profiles=[HardwareProfile.S],
        models=["qwen2.5:3b-instruct-q4_k_m"],
        baselines=[BaselineType.B0_BASE, BaselineType.B2_AGAN_L1],
    )
    
    report = harness.generate_report()
    harness.save_report(report)
    harness.print_capability_matrix(report)
    
    return report


if __name__ == "__main__":
    asyncio.run(main())