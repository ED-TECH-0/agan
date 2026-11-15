"""
Evaluation harness — runs benchmarks and produces capability matrix.
"""
import json
import asyncio
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

from core.pipeline import Pipeline, PipelineConfig
from core.governor import ProfileName

@dataclass
class EvalTask:
    id: str
    category: str
    question: str
    expected_type: str
    verification: dict  # {"type": "executable", "check": "..."} or {"type": "claim", "claims": [...]}

@dataclass
class EvalResult:
    task_id: str
    profile: str
    answer: str
    passed: bool
    score: float
    details: dict

class EvalHarness:
    def __init__(self, tasks_file: str = "eval/suites/tasks.json"):
        self.tasks_file = Path(tasks_file)
        self.tasks = self._load_tasks()
    
    def _load_tasks(self) -> list[EvalTask]:
        if self.tasks_file.exists():
            with open(self.tasks_file) as f:
                data = json.load(f)
            return [EvalTask(**t) for t in data]
        return []
    
    def save_tasks(self):
        self.tasks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tasks_file, 'w') as f:
            json.dump([asdict(t) for t in self.tasks], f, indent=2)
    
    def add_task(self, task: EvalTask):
        self.tasks.append(task)
        self.save_tasks()
    
    async def run(self, profiles: list[ProfileName] = None, max_tasks: int = None) -> list[EvalResult]:
        profiles = profiles or [ProfileName.STANDARD]
        tasks = self.tasks[:max_tasks] if max_tasks else self.tasks
        results = []
        
        for profile in profiles:
            pipeline = Pipeline(PipelineConfig(profile=profile))
            for task in tasks:
                answer = await pipeline.answer(task.question)
                passed, score, details = self._verify(task, answer)
                results.append(EvalResult(
                    task_id=task.id,
                    profile=profile.value,
                    answer=answer,
                    passed=passed,
                    score=score,
                    details=details,
                ))
        return results
    
    def _verify(self, task: EvalTask, answer: str) -> tuple[bool, float, dict]:
        v = task.verification
        if v.get("type") == "executable":
            # Run code and check
            return self._verify_executable(v, answer)
        elif v.get("type") == "claim":
            # Check claims in answer
            return self._verify_claims(v, answer)
        return False, 0.0, {"error": "Unknown verification type"}
    
    def _verify_executable(self, verification: dict, answer: str) -> tuple[bool, float, dict]:
        # Placeholder
        return True, 1.0, {}
    
    def _verify_claims(self, verification: dict, answer: str) -> tuple[bool, float, dict]:
        # Placeholder
        return True, 1.0, {}

async def main():
    harness = EvalHarness()
    
    # Add some initial tasks if empty
    if not harness.tasks:
        harness.add_task(EvalTask(
            id="code_001",
            category="code",
            question="Write a Python function to compute fibonacci numbers",
            expected_type="code",
            verification={"type": "executable", "check": "fibonacci(10) == 55"}
        ))
        harness.add_task(EvalTask(
            id="blender_001",
            category="blender",
            question="Create a Blender bpy script to add a cube at origin",
            expected_type="code",
            verification={"type": "executable", "check": "bpy.ops.mesh.primitive_cube_add"}
        ))
        harness.add_task(EvalTask(
            id="fact_001",
            category="fact",
            question="What is the latest stable version of Python?",
            expected_type="fact",
            verification={"type": "claim", "claims": ["Python version"]}
        ))
    
    results = await harness.run([ProfileName.STANDARD], max_tasks=10)
    
    # Save results
    out_file = Path("eval/reports") / f"eval_{int(asyncio.get_event_loop().time())}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"Completed {len(results)} evaluations")
    for r in results:
        print(f"  {r.task_id} ({r.profile}): {'PASS' if r.passed else 'FAIL'} score={r.score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
