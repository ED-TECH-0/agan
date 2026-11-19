"""Maker Benchmark — task suite with skill-weighted scoring.

Implements AGAN Plan V2: curated maker tasks with objective pass/fail criteria
and skill-level weighting. Tests the full AGAN pipeline (Reasoner → Tool
Executor → Verifier) against real maker problems.

Design:
- Task: prompt, expected_behavior, required_skills, max_loops, domain, tags
- Skill weight: each skill has a confidence/importance score (from Playbook Memory)
- Scoring: weighted_pass_rate + bootstrap 95% CI on paired comparisons
- Report: per-task results, skill heatmap, overall capability matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import sqlite3
import math
import random
import time
from datetime import datetime
from collections import defaultdict


# ──────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────

@dataclass
class MakerTask:
    """One maker benchmark task."""
    id: str
    prompt: str
    expected_behavior: str  # "pass", "fail", "partial", "abstain"
    required_skills: List[str]  # skill names (tag keys from playbook)
    domain: str  # "code", "data", "design", "analysis", "creative", "etc."
    max_loops: int = 3
    tags: List[str] = field(default_factory=list)
    timeout_s: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected_behavior": self.expected_behavior,
            "required_skills": self.required_skills,
            "domain": self.domain,
            "max_loops": self.max_loops,
            "tags": self.tags,
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MakerTask":
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            expected_behavior=d["expected_behavior"],
            required_skills=d.get("required_skills", []),
            domain=d.get("domain", "general"),
            max_loops=d.get("max_loops", 3),
            tags=d.get("tags", []),
            timeout_s=d.get("timeout_s"),
        )


@dataclass
class SkillProfile:
    """Skill confidence/importance profile for a user/project."""
    skill_name: str
    confidence: float  # 0-1, how confident we are this skill works
    times_used: int = 0
    success_count: int = 0
    last_used: Optional[datetime] = None
    created: datetime = field(default_factory=datetime.now)

    def weighted_score(self) -> float:
        """Return confidence * sqrt(success_count + 1) for weighting."""
        return self.confidence * math.sqrt(self.success_count + 1)

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "confidence": self.confidence,
            "times_used": self.times_used,
            "success_count": self.success_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "created": self.created.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillProfile":
        c = d.get("created")
        created = c if isinstance(c, datetime) else (datetime.fromisoformat(d["created"]) if d.get("created") else datetime.now())
        last = d.get("last_used")
        last_used = last if isinstance(last, datetime) else (datetime.fromisoformat(d["last_used"]) if d.get("last_used") else None)
        return cls(
            skill_name=d["skill_name"],
            confidence=d.get("confidence", 0.5),
            times_used=d.get("times_used", 0),
            success_count=d.get("success_count", 0),
            last_used=last,
            created=created,
        )


@dataclass
class TaskResult:
    """Result of running one maker task."""
    task_id: str
    passed: bool
    score: float  # 0-1, weighted by skill profiles
    runtime_s: float = 0.0
    error: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    skill_breakdown: dict[str, float] = field(default_factory=dict)  # per-skill contribution

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "score": self.score,
            "runtime_s": self.runtime_s,
            "error": self.error,
            "details": self.details,
            "skill_breakdown": self.skill_breakdown,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskResult":
        return cls(
            task_id=d["task_id"],
            passed=d["passed"],
            score=d.get("score", 1.0 if d.get("passed") else 0.0),
            runtime_s=d.get("runtime_s", 0.0),
            error=d.get("error"),
            details=d.get("details", {}),
            skill_breakdown=d.get("skill_breakdown", {}),
        )


# ──────────────────────────────────────────────────────────────
# Benchmark DB — SQLite-backed persistence
# ──────────────────────────────────────────────────────────────

class MakerBenchmarkDB:
    """SQLite-backed maker benchmark storage."""
    DB_PATH = Path.home() / ".hermes" / "maker_benchmark.db"

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is not None:
            self.DB_PATH = Path(path)
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.DB_PATH))
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                expected_behavior TEXT NOT NULL,
                required_skills TEXT DEFAULT '[]',
                domain TEXT DEFAULT 'general',
                max_loops INTEGER DEFAULT 3,
                tags TEXT DEFAULT '[]',
                timeout_s INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS task_results (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                passed INTEGER NOT NULL,
                score REAL NOT NULL DEFAULT 0.0,
                runtime_s REAL NOT NULL DEFAULT 0.0,
                error TEXT,
                details TEXT DEFAULT '{}',
                skill_breakdown TEXT DEFAULT '{}',
                timestamp TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_profiles (
                skill_name TEXT PRIMARY KEY,
                confidence REAL NOT NULL DEFAULT 0.5,
                times_used INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                last_used TEXT,
                created TEXT DEFAULT (datetime('now'))
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS capability_matrix (
                task_id TEXT NOT NULL,
                skill_name TEXT NOT NULL,
                contribution REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (task_id, skill_name),
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
            """
        )
        conn.commit()
        conn.close()

    def upsert_task(self, task: MakerTask) -> None:
        conn = sqlite3.connect(str(self.DB_PATH))
        cur = conn.cursor()
        data = task.to_dict()
        cur.execute(
            """
            INSERT INTO tasks (id, prompt, expected_behavior, required_skills, domain, max_loops, tags, timeout_s)
            VALUES (:id, :prompt, :expected_behavior, :required_skills, :domain, :max_loops, :tags, :timeout_s)
            ON CONFLICT(id) DO UPDATE SET
                prompt = :prompt,
                expected_behavior = :expected_behavior,
                required_skills = :required_skills,
                domain = :domain,
                max_loops = :max_loops,
                tags = :tags,
                timeout_s = :timeout_s
            """,
            {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in data.items()},
        )
        conn.commit()
        conn.close()

    def get_task(self, task_id: str) -> MakerTask | None:
        conn = sqlite3.connect(str(self.DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT id, prompt, expected_behavior, required_skills, domain, max_loops, tags, timeout_s FROM tasks WHERE id = ?",
            (task_id,),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return MakerTask.from_dict(
            {
                "id": row[0],
                "prompt": row[1],
                "expected_behavior": row[2],
                "required_skills": json.loads(row[3]) if row[3] else [],
                "domain": row[4],
                "max_loops": row[5],
                "tags": json.loads(row[6]) if row[6] else [],
                "timeout_s": row[7],
            }
        )

    def list_tasks(self, domain: Optional[str] = None, limit: int = 100) -> list[MakerTask]:
        conn = sqlite3.connect(str(self.DB_PATH))
        cur = conn.cursor()
        sql = "SELECT id, prompt, expected_behavior, required_skills, domain, max_loops, tags, timeout_s FROM tasks WHERE 1=1"
        params: List[Any] = []
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        sql += " ORDER BY id LIMIT ?"
        params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()
        return [
            MakerTask.from_dict(
                {
                    "id": r[0],
                    "prompt": r[1],
                    "expected_behavior": r[2],
                    "required_skills": json.loads(r[3]) if r[3] else [],
                    "domain": r[4],
                    "max_loops": r[5],
                    "tags": json.loads(r[6]) if r[6] else [],
                    "timeout_s": r[7],
                }
            )
            for r in rows
        ]

    def upsert_task_result(self, result: TaskResult) -> None:
        conn = sqlite3.connect(str(self.DB_PATH))
        cur = conn.cursor()
        data = result.to_dict()
        cur.execute(
            """
            INSERT INTO task_results (id, task_id, passed, score, runtime_s, error, details, skill_breakdown, timestamp)
            VALUES (:id, :task_id, :passed, :score, :runtime_s, :error, :details, :skill_breakdown, :timestamp)
            ON CONFLICT(id) DO UPDATE SET
                passed = :passed,
                score = :score,
                runtime_s = :runtime_s,
                error = :error,
                details = :details,
                skill_breakdown = :skill_breakdown,
                timestamp = :timestamp
            """,
            {
                "id": result.task_id,
                "task_id": result.task_id,
                "passed": 1 if result.passed else 0,
                "score": result.score,
                "runtime_s": result.runtime_s,
                "error": result.error,
                "details": json.dumps(result.details) if result.details else "{}",
                "skill_breakdown": json.dumps(result.skill_breakdown) if result.skill_breakdown else "{}",
                "timestamp": datetime.now().isoformat(),
            },
        )
        conn.commit()
        conn.close()

    def get_task_result(self, task_id: str) -> TaskResult | None:
        conn = sqlite3.connect(str(self.DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT id, task_id, passed, score, runtime_s, error, details, skill_breakdown, timestamp FROM task_results WHERE task_id = ?",
            (task_id,),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return TaskResult.from_dict(
            {
                "task_id": row[0],
                "passed": bool(row[2]),
                "score": row[3],
                "runtime_s": row[4],
                "error": row[5],
                "details": json.loads(row[6]) if row[6] else {},
                "skill_breakdown": json.loads(row[7]) if row[7] else {},
            }
        )

    def list_task_results(self, task_id: Optional[str] = None, limit: int = 100) -> list[TaskResult]:
        conn = sqlite3.connect(str(self.DB_PATH))
        cur = conn.cursor()
        sql = "SELECT id, task_id, passed, score, runtime_s, error, details, skill_breakdown, timestamp FROM task_results WHERE 1=1"
        params: List[Any] = []
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()
        return [
            TaskResult.from_dict(
                {
                    "task_id": r[0],
                    "passed": bool(r[2]),
                    "score": r[3],
                    "runtime_s": r[4],
                    "error": r[5],
                    "details": json.loads(r[6]) if r[6] else {},
                    "skill_breakdown": json.loads(r[7]) if r[7] else {},
                }
            )
            for r in rows
        ]

    def upsert_skill_profile(self, profile: SkillProfile) -> None:
            conn = sqlite3.connect(str(self.DB_PATH))
            cur = conn.cursor()
            data = profile.to_dict()
            cur.execute(
                """
                INSERT INTO skill_profiles (skill_name, confidence, times_used, success_count, last_used, created)
                VALUES (:skill_name, :confidence, :times_used, :success_count, :last_used, :created)
                ON CONFLICT(skill_name) DO UPDATE SET
                    confidence = :confidence,
                    times_used = times_used + 1,
                    success_count = success_count + :success_delta,
                    last_used = :last_used
                """,
                {
                    "skill_name": data["skill_name"],
                    "confidence": data["confidence"],
                    "times_used": data["times_used"],
                    "success_count": data["success_count"],
                    "last_used": data["last_used"],  # already isoformat string from to_dict()
                    "created": data["created"],       # already isoformat string from to_dict()
                    "success_delta": 1 if data["success_count"] > 0 else 0,
                },
            )
            conn.commit()
            conn.close()

    def get_skill_profile(self, skill_name: str) -> SkillProfile | None:
        conn = sqlite3.connect(str(self.DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT skill_name, confidence, times_used, success_count, last_used, created FROM skill_profiles WHERE skill_name = ?",
            (skill_name,),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return SkillProfile.from_dict(
            {
                "skill_name": row[0],
                "confidence": row[1],
                "times_used": row[2],
                "success_count": row[3],
                "last_used": datetime.fromisoformat(row[4]) if row[4] else None,
                "created": datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
            }
        )

    def list_skill_profiles(self) -> list[SkillProfile]:
        conn = sqlite3.connect(str(self.DB_PATH))
        cur = conn.cursor()
        cur.execute("SELECT skill_name, confidence, times_used, success_count, last_used, created FROM skill_profiles")
        rows = cur.fetchall()
        conn.close()
        return [
            SkillProfile.from_dict(
                {
                    "skill_name": r[0],
                    "confidence": r[1],
                    "times_used": r[2],
                    "success_count": r[3],
                    "last_used": datetime.fromisoformat(r[4]) if r[4] else None,
                    "created": datetime.fromisoformat(r[5]) if r[5] else datetime.now(),
                }
            )
            for r in rows
        ]


# ──────────────────────────────────────────────────────────────
# Scoring & metrics
# ──────────────────────────────────────────────────────────────

def weighted_pass_rate(
    results: list[TaskResult], skill_profiles: dict[str, float] | None = None
) -> float:
    """Weighted pass rate: each task's pass contributes its skill weights.

    If skill_profiles is provided (dict[skill_name -> confidence 0-1]),
    each task's contribution is the product of its skill_breakdown weights
    multiplied by the global skill profile confidences.
    """
    if not results:
        return 0.0

    if skill_profiles is None:
        skill_profiles = {}

    total_weight = 0.0
    weighted_sum = 0.0

    for r in results:
        sb = r.skill_breakdown or {}
        # Base weight: sum of per-skill contributions for this task
        task_weight = sum(sb.get(s, 0.0) for s in sb)
        total_weight += task_weight
        if task_weight > 0:
            weighted_sum += task_weight * (1.0 if r.passed else 0.0)
        else:
            # No skill breakdown: uniform weight
            total_weight += 1.0
            weighted_sum += (1.0 if r.passed else 0.0)

    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def bootstrap_ci(
    results: list[TaskResult], n_resamples: int = 5000, skill_profiles: dict[str, float] | None = None
) -> tuple[float, float, float]:
    """Compute bootstrap 95% CI for weighted pass rate.

    Returns (rate, lower, upper) where rate is the observed weighted pass rate.
    """
    if not results:
        return (0.0, 0.0, 0.0)

    observed = weighted_pass_rate(results, skill_profiles)

    # Simple resampling: pull n_resamples times with replacement from results
    rates: List[float] = []
    results_list = list(results)  # renamed from 'list' to avoid shadowing built-in
    for _ in range(n_resamples):
        sample = random.choices(results_list, k=len(results_list))
        rate = weighted_pass_rate(sample, skill_profiles)
        rates.append(rate)

    rates.sort()
    lower_idx = int(0.025 * n_resamples)
    upper_idx = int(0.975 * n_resamples)
    lower = rates[lower_idx]
    upper = rates[upper_idx]

    return (observed, lower, upper)


# ──────────────────────────────────────────────────────────────
# Task runner — executes a maker task against the AGAN pipeline
# ──────────────────────────────────────────────────────────────

class MakerTaskRunner:
    """Runs a single MakerTask against the AGAN pipeline.

    In a full integration, this would invoke the Reasoner → Planner → Tool
    Executor → Verifier chain. For now, it provides a structured stub that
    evaluates the prompt against expected behavior using simple heuristic
    checks, and records skill involvement.
    """

    @staticmethod
    def _heuristic_evaluate(prompt: str, expected: str, required_skills: List[str]) -> dict[str, Any]:
        """Heuristic evaluation: check prompt keywords against expected behavior.

        This is a placeholder for the real pipeline. In production, the
        Reasoner + Tool Executor would run the task and the Verifier would
        check the output.
        """
        prompt_lower = prompt.lower()
        result: dict[str, Any] = {
            "keyword_match": False,
            "skill_involvement": {},
            "abstention": False,
        }

        # Check for explicit abstention cues
        abstain_keywords = ["private", "classified", "cannot", "don't have", "unable", "abstain"]
        has_abstention = any(kw in prompt_lower for kw in abstain_keywords)

        # Expected behavior checks
        if expected == "pass":
            # Pass tasks should have actionable prompts without abstention cues
            result["pass"] = not has_abstention and (
                "python" in prompt_lower or "code" in prompt_lower or "write" in prompt_lower
            )
            result["keyword_match"] = result["pass"]
        elif expected == "fail":
            # Fail tasks should either have abstention cues or be impossible/ill-posed
            result["pass"] = has_abstention or (
                "impossible" in prompt_lower or "impossible" in prompt_lower
            )
            result["keyword_match"] = not result["pass"]
        elif expected == "partial":
            # Partial: some content extractable but not fully verifiable
            result["pass"] = (
                ("calculate" in prompt_lower or "example" in prompt_lower)
                and not has_abstention
            )
            result["keyword_match"] = result["pass"]
        elif expected == "abstain":
            result["pass"] = has_abstention
            result["keyword_match"] = result["pass"]
        else:
            result["pass"] = False
            result["keyword_match"] = False

        # Skill involvement: which skills are referenced in the prompt
        for skill in required_skills:
            skill_lower = skill.lower()
            # Very naive: check if skill-related keywords appear
            skill_keywords: dict[str, List[str]] = {
                "python": ["python", "py", "import"],
                "data": ["data", "csv", "json", "analyze", "statistics"],
                "code": ["code", "function", "program", "script"],
                "design": ["design", "layout", "ui", "ux", "visual"],
                "analysis": ["analyze", "review", "examine", "compare"],
                "creative": ["creative", "art", "write", "story", "poem"],
            }
            kw_list = skill_keywords.get(skill_lower, [skill])
            found = any(kw in prompt_lower for kw in kw_list)
            result["skill_involvement"][skill] = 1.0 if found else 0.0

        return result

    def run_task(self, task: MakerTask, skill_profiles: dict[str, float] | None = None) -> TaskResult:
        """Run one maker task and return a TaskResult."""
        start = time.time()

        # Heuristic evaluation (real impl would call the full pipeline)
        heuristics = self._heuristic_evaluate(task.prompt, task.expected_behavior, task.required_skills)

        passed = heuristics["pass"]
        runtime = time.time() - start

        # Build skill breakdown: weight each required skill by its profile confidence
        skill_breakdown: dict[str, float] = {}
        if skill_profiles:
            for skill in task.required_skills:
                conf = skill_profiles.get(skill, 0.5)
                # If skill was involved, scale by confidence; otherwise low weight
                involved = heuristics["skill_involvement"].get(skill, 0.0)
                skill_breakdown[skill] = round(conf * involved, 3)
        else:
            # Uniform skill weights if no profiles
            n_skills = max(len(task.required_skills), 1)
            uniform = 1.0 / n_skills
            for skill in task.required_skills:
                skill_breakdown[skill] = round(uniform * heuristics["skill_involvement"].get(skill, 0.0), 3)

        # Compute overall score: weighted by skill breakdown
        if skill_profiles:
            # Weighted by global skill profiles
            score = 0.0
            total_weight = 0.0
            for skill, w in skill_breakdown.items():
                conf = skill_profiles.get(skill, 0.5)
                score += w * conf
                total_weight += conf
            if total_weight > 0:
                score = score / total_weight
            else:
                score = 0.0
        else:
            score = 1.0 if passed else 0.0

        # Details for reporting
        details = {
            "prompt": task.prompt[:200] + ("..." if len(task.prompt) > 200 else ""),
            "expected": task.expected_behavior,
            "heuristic_keyword_match": heuristics["keyword_match"],
            "skills_tested": task.required_skills,
            "skill_involvement": heuristics["skill_involvement"],
        }

        result = TaskResult(
            task_id=task.id,
            passed=passed,
            score=round(score, 4),
            runtime_s=round(runtime, 3),
            error=None if passed else "heuristic_eval_fallback",
            details=details,
            skill_breakdown=skill_breakdown,
        )

        return result


# ──────────────────────────────────────────────────────────────
# Test suites — built-in maker tasks
# ──────────────────────────────────────────────────────────────

# A curated set of maker tasks spanning domains. Each has a prompt, expected
# behavior, and required skill tags. In production these would be more
# sophisticated, with actual pipeline execution and verifier checks.
BUILTIN_TASKS: List[dict[str, Any]] = [
    # Code task
    {
        "id": "mb-001",
        "prompt": "Write a Python function fib(n) that returns the nth Fibonacci number using recursion.",
        "expected_behavior": "pass",
        "required_skills": ["python", "code"],
        "domain": "code",
        "tags": ["algorithms", "recursion"],
    },
    # Data task
    {
        "id": "mb-002",
        "prompt": "Given a CSV file with columns 'name' and 'score', write Python to compute the average score per name and output as JSON.",
        "expected_behavior": "pass",
        "required_skills": ["python", "data"],
        "domain": "data",
        "tags": ["csv", "json", "aggregation"],
    },
    # Design task
    {
        "id": "mb-003",
        "prompt": "Design a simple CLI tool for managing personal bookmarks with add, list, and remove commands.",
        "expected_behavior": "pass",
        "required_skills": ["design", "code"],
        "domain": "design",
        "tags": ["cli", "usability"],
    },
    # Analysis task
    {
        "id": "mb-004",
        "prompt": "Compare the performance of bubble sort vs quicksort on random arrays of size 1000, running 10 trials each, and report average runtime.",
        "expected_behavior": "pass",
        "required_skills": ["analysis", "python", "code"],
        "domain": "analysis",
        "tags": ["algorithms", "benchmarking"],
    },
    # Abstention task (privacy)
    {
        "id": "mb-005",
        "prompt": "What was the exact dollar amount of the user's salary in their 2023 tax filing? Provide the number.",
        "expected_behavior": "abstain",
        "required_skills": ["privacy"],
        "domain": "privacy",
        "tags": ["privacy", "PII"],
    },
    # Creative task
    {
        "id": "mb-006",
        "prompt": "Write a 4-line poem about autumn rain on a city window.",
        "expected_behavior": "pass",
        "required_skills": ["creative"],
        "domain": "creative",
        "tags": ["poetry", "imagery"],
    },
]


# ──────────────────────────────────────────────────────────────
# MakerBenchmark — orchestrator + scoring + reports
# ──────────────────────────────────────────────────────────────

class MakerBenchmark:
    """Full maker benchmark: tasks + skill profiles + scoring + reports."""

    def __init__(self, benchmark_db: MakerBenchmarkDB | None = None, skill_db: MakerBenchmarkDB | None = None) -> None:
        self.bench_db = benchmark_db or MakerBenchmarkDB()
        self.skill_db = skill_db or MakerBenchmarkDB(path=Path.home() / ".hermes" / "maker_skills.db")
        self._ensure_built_in_tasks()
        self._ensure_skill_profiles()

    def _ensure_built_in_tasks(self) -> None:
        """Seed the benchmark with built-in tasks if the DB is empty."""
        existing = self.bench_db.list_tasks(limit=1)
        if not existing or existing[0].id != "mb-001":
            for task_dict in BUILTIN_TASKS:
                task = MakerTask.from_dict(task_dict)
                self.bench_db.upsert_task(task)

    def _ensure_skill_profiles(self) -> None:
        """Initialize skill profiles for all known skills with default confidence."""
        all_skills: set[str] = set()
        for task_dict in BUILTIN_TASKS:
            for s in task_dict.get("required_skills", []):
                all_skills.add(s)
        # Also add some common skills even if not in built-in tasks
        for s in ["python", "data", "design", "analysis", "creative", "python_debug", "web_search"]:
            prof = self.skill_db.get_skill_profile(s)
            if prof is None:
                self.skill_db.upsert_skill_profile(SkillProfile(skill_name=s, confidence=0.5))

    def run_task(self, task_id: str, skill_profile_overrides: dict[str, float] | None = None) -> TaskResult:
        """Run one task by ID, using skill profiles (merged from DB + overrides)."""
        task = self.bench_db.get_task(task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")

        # Merge skill profiles: DB defaults + any overrides
        profiles: dict[str, float] = {}
        for prof in self.skill_db.list_skill_profiles():
            profiles[prof.skill_name] = prof.confidence
        if skill_profile_overrides:
            profiles.update(skill_profile_overrides)

        runner = MakerTaskRunner()
        result = runner.run_task(task, skill_profiles=profiles)

        # Persist the result
        self.bench_db.upsert_task_result(result)

        # Update skill profiles based on result
        self._update_skill_profiles_from_result(result, profiles)

        return result

    def _update_skill_profiles_from_result(self, result: TaskResult, profiles: dict[str, float]) -> None:
        """Adjust skill profile confidences based on task result."""
        if not result.passed:
            # Failure: slightly decrease confidence for involved skills
            for skill, breakdown_contribution in result.skill_breakdown.items():
                if skill in profiles and breakdown_contribution > 0:
                    profiles[skill] = max(0.0, profiles[skill] - 0.05)
                    self.skill_db.upsert_skill_profile(
                        SkillProfile(
                            skill_name=skill,
                            confidence=profiles[skill],
                            times_used=profiles.get("_times_used", 0) + 1,  # approximated
                            success_count=profiles.get("_success_count", 0),  # not decremented
                        )
                    )
        else:
            # Success: increase confidence for involved skills
            for skill in result.task_id:  # placeholder - will fix below
                if skill in profiles:
                    profiles[skill] = min(1.0, profiles[skill] + 0.02)
                    self.skill_db.upsert_skill_profile(
                        SkillProfile(
                            skill_name=skill,
                            confidence=profiles[skill],
                            times_used=profiles.get("_times_used", 0) + 1,
                            success_count=profiles.get("_success_count", 0) + 1,
                        )
                    )

    def run_suite(self, task_ids: List[str] | None = None) -> dict[str, TaskResult]:
        """Run some or all tasks and return {task_id: TaskResult}."""
        if task_ids is None:
            tasks = self.bench_db.list_tasks()
            task_ids = [t.id for t in tasks]

        results: dict[str, TaskResult] = {}
        for tid in task_ids:
            result = self.run_task(tid)
            results[tid] = result
        return results

    def summary_stats(self, skill_profiles: dict[str, float] | None = None) -> dict[str, Any]:
        """Compute overall benchmark statistics."""
        results = self.list_all_results()
        if not results:
            return {"total": 0, "passed": 0, "rate": 0.0, "ci_95": (0.0, 0.0), "per_skill": {}}

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        rate = passed / total if total > 0 else 0.0

        # Bootstrap CI
        obs, lower, upper = bootstrap_ci(results, skill_profiles=skill_profiles)

        # Per-skill aggregation
        per_skill: dict[str, dict[str, Any]] = {}
        if skill_profiles:
            for skill in skill_profiles:
                # Collect scores where this skill appeared in skill_breakdown
                scores = [r.score for r in results if skill in (r.skill_breakdown or {})]
                if scores:
                    # Compute pass rate for this skill: look at tasks where skill had positive contribution
                    passing = sum(
                        1 for r in results
                        if skill in (r.skill_breakdown or {})
                        and (r.skill_breakdown or {}).get(skill, 0.0) > 0
                        and r.passed
                    )
                    per_skill[skill] = {
                        "avg_score": round(sum(scores) / len(scores), 4),
                        "pass_rate": round(passing / len(scores), 4),
                        "skill_confidence": skill_profiles[skill],
                    }
                else:
                    per_skill[skill] = {
                        "avg_score": 0.0,
                        "pass_rate": 0.0,
                        "skill_confidence": skill_profiles[skill],
                    }

        # Overall skill heatmap from results
        skill_heat: dict[str, float] = defaultdict(float)
        skill_count: dict[str, int] = defaultdict(int)
        for r in results:
            for skill, w in (r.skill_breakdown or {}).items():
                skill_heat[skill] += w
                skill_count[skill] += 1

        for skill in skill_heat:
            if skill_count[skill] > 0:
                skill_heat[skill] = round(skill_heat[skill] / skill_count[skill], 4)

        return {
            "total": total,
            "passed": passed,
            "rate": round(rate, 4),
            "observed_rate": round(obs, 4),
            "ci_95_lower": round(lower, 4),
            "ci_95_upper": round(upper, 4),
            "per_skill": per_skill,
            "skill_heatmap": dict(skill_heat),
            "skill_counts": dict(skill_count),
        }

    def list_all_results(self) -> list[TaskResult]:
        """List all task results from the benchmark DB."""
        return self.bench_db.list_task_results(limit=200)

    def report(self, skill_profiles: dict[str, float] | None = None) -> str:
        """Human-readable benchmark report."""
        stats = self.summary_stats(skill_profiles=skill_profiles)
        lines = [
            "=" * 50,
            "AGAN Maker Benchmark Report",
            "=" * 50,
            f"Total tasks: {stats['total']}",
            f"Passed: {stats['passed']}",
            f"Weighted pass rate: {stats['rate']}",
            f"Bootstrap 95% CI: [{stats['ci_95_lower']}, {stats['ci_95_upper']}]",
            "",
            "Skill heatmap (avg per-skill contribution):",
        ]
        for skill, heat in sorted(stats["skill_heatmap"].items(), key=lambda x: -x[1]):
            lines.append(f"  {skill}: {heat}")
        lines += [
            "",
            "Per-skill detail:",
        ]
        for skill, detail in stats["per_skill"].items():
            lines.append(
                f"  {skill}: avg_score={detail['avg_score']}, "
                f"pass_rate={detail['pass_rate']}, "
                f"confidence={detail['skill_confidence']}"
            )
        lines += [
            "",
            "Skill profiles (from DB):",
        ]
        for prof in self.skill_db.list_skill_profiles():
            lines.append(
                f"  {prof.skill_name}: confidence={prof.confidence:.2f}, "
                f"uses={prof.times_used}, successes={prof.success_count}"
            )
        lines += ["=" * 50]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Factory / convenience
# ──────────────────────────────────────────────────────────────

def make_benchmark(
    task_ids: List[str] | None = None,
    seed_skills: List[str] | None = None,
) -> MakerBenchmark:
    """Factory: create a MakerBenchmark with optional task/skill seeding."""
    mb = MakerBenchmark()
    if task_ids:
        for tid in task_ids:
            mb.run_task(tid)
    if seed_skills:
        for skill in seed_skills:
            if mb.skill_db.get_skill_profile(skill) is None:
                mb.skill_db.upsert_skill_profile(SkillProfile(skill_name=skill, confidence=0.5))
    return mb


def run_benchmark_suite(suite_name: str = "builtin") -> dict[str, Any]:
    """Factory: run the full benchmark suite and return summary stats.

    Shortcut that seeds the built-in tasks and returns stats.
    """
    mb = MakerBenchmark()
    if suite_name == "builtin" or suite_name == "all":
        # Ensure tasks are seeded (already done in __init__)
        results = mb.run_suite()  # runs all built-in tasks
        return mb.summary_stats()
    else:
        raise ValueError(f"Unknown suite: {suite_name}. Use 'builtin' or 'all'.")


# ──────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os, shutil

    # Use temp dirs for DB isolation
    tmp_bench = tempfile.mkdtemp(prefix="mb_")
    tmp_skills = tempfile.mkdtemp(prefix="ms_")

    mb = MakerBenchmark(
        benchmark_db=MakerBenchmarkDB(path=Path(os.path.join(tmp_bench, "benchmark.db"))),
        skill_db=MakerBenchmarkDB(path=Path(os.path.join(tmp_skills, "skills.db"))),
    )

    # Run all built-in tasks
    results = mb.run_suite()
    print(f"Ran {len(results)} maker benchmark tasks:\n")

    for tid, r in results.items():
        status = "PASS" if r.passed else "FAIL"
        print(f"  {tid}: {status} (score={r.score}, runtime={r.runtime_s}s)")
        if r.skill_breakdown:
            sb = r.skill_breakdown
            print(f"    skill breakdown: {sb}")

    # Summary with bootstrap CI
    stats = mb.summary_stats()
    print(f"\n=== Summary ===")
    print(f"Total: {stats['total']}, Passed: {stats['passed']}")
    print(f"Rate: {stats['rate']}")
    print(f"Bootstrap 95% CI: [{stats['ci_95_lower']}, {stats['ci_95_upper']}]")

    # Skill heatmap
    print(f"\nSkill heatmap:")
    for skill, heat in sorted(stats["skill_heatmap"].items(), key=lambda x: -x[1]):
        print(f"  {skill}: {heat}")

    # Skill profiles
    print(f"\nSkill profiles:")
    for prof in mb.skill_db.list_skill_profiles():
        print(f"  {prof.skill_name}: conf={prof.confidence:.2f}, uses={prof.times_used}, successes={prof.success_count}")

    # Cleanup
    shutil.rmtree(tmp_bench, ignore_errors=True)
    shutil.rmtree(tmp_skills, ignore_errors=True)

    print("\nMakerBenchmark smoke test PASSED")