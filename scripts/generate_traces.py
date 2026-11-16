"""Trace generation using Qwen2.5 teacher model via Ollama.

Generates synthetic skill traces:
1. Seed question → Planner (router + planner modules)
2. Retriever fetches real evidence (SearXNG / mock)
3. Teacher (Qwen2.5:3B via Ollama) produces plan + tool calls + cited answer
4. Verifier filters: evidence-supported, self-consistent, execution passes
5. Output: SFT traces and DPO preference pairs
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.pipeline import Pipeline, PipelineConfig
from core.governor import ProfileName
from core.privacy import PrivacyFirewall

DATA_DIR = Path("data")
TRACES_DIR = DATA_DIR / "traces"
DPO_DIR = DATA_DIR / "dpo"
DB_PATH = DATA_DIR / "traces.db"


@dataclass
class SFTTrace:
    id: str
    question: str
    plan: dict
    evidence: list[dict]
    answer: str
    citations: list[dict]
    skillpack: Optional[str]
    created_at: str
    license: str = "synthetic-open-license"


@dataclass
class DPOTrace:
    id: str
    question: str
    chosen: str          # grounded, cited answer
    rejected: str        # hallucinated or uncited answer
    skillpack: Optional[str]
    created_at: str


class TraceStore:
    """SQLite-backed trace storage with dedup by content hash."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    question TEXT,
                    plan TEXT,
                    evidence TEXT,
                    answer TEXT,
                    citations TEXT,
                    skillpack TEXT,
                    created_at TEXT,
                    license TEXT,
                    content_hash TEXT UNIQUE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dpo_pairs (
                    id TEXT PRIMARY KEY,
                    question TEXT,
                    chosen TEXT,
                    rejected TEXT,
                    skillpack TEXT,
                    created_at TEXT
                )
            """)

    def _content_hash(self, question: str, answer: str) -> str:
        return hashlib.sha256(f"{question}:{answer}".encode()).hexdigest()[:16]

    def add_sft(self, trace: SFTTrace) -> bool:
        """Returns True if inserted, False if duplicate."""
        chash = self._content_hash(trace.question, trace.answer)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO traces (id, question, plan, evidence, answer, citations, skillpack, created_at, license, content_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        trace.id, trace.question, json.dumps(trace.plan),
                        json.dumps(trace.evidence), trace.answer,
                        json.dumps(trace.citations), trace.skillpack,
                        trace.created_at, trace.license, chash,
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def add_dpo(self, trace: DPOTrace) -> bool:
        chash = hashlib.sha256(f"{trace.question}:{trace.chosen}:{trace.rejected}".encode()).hexdigest()[:16]
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO dpo_pairs (id, question, chosen, rejected, skillpack, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (trace.id, trace.question, trace.chosen, trace.rejected,
                     trace.skillpack, trace.created_at, chash),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def counts(self) -> dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            sft = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            dpo = conn.execute("SELECT COUNT(*) FROM dpo_pairs").fetchone()[0]
        return {"sft": sft, "dpo": dpo}


class TraceGenerator:
    def __init__(self, profile: ProfileName = ProfileName.LITE):
        self.profile = profile
        self.pipeline = Pipeline(PipelineConfig(profile=profile))
        self.firewall = PrivacyFirewall()
        self.store = TraceStore()

    async def generate_sft_trace(self, question: str, skillpack: Optional[str] = None) -> Optional[SFTTrace]:
        """Generate one SFT trace: plan → retrieve → sanitize → reason → verify."""
        # Plan
        plan = self.pipeline.planner(question, self.profile.value)

        # Retrieve + sanitize
        all_chunks = []
        for step in plan.steps:
            docs = await self.pipeline.retriever.search(step.query, step.source_type)
            for doc in docs:
                safe = self.pipeline.safety.sanitize(doc.content, doc.url)
                if not safe.safe:
                    continue
                chunks = self.pipeline.retriever.chunk_document(doc)
                all_chunks.extend(chunks)

        if not all_chunks:
            return None

        # Rerank + compress
        ranked = self.pipeline.reranker.rank(question, all_chunks, top_k=self.pipeline.governor.current_profile().max_chunks)
        evidence = self.pipeline.compressor.compress_chunks(question, ranked)

        # Reason
        draft = await self.pipeline.reasoner.synthesize(question, evidence, skillpack)

        # Verify
        report = self.pipeline.verifier.check(draft.answer, evidence)
        if not report.ok:
            return None  # Skip unverified traces

        trace = SFTTrace(
            id=hashlib.sha256(f"{question}:{datetime.now(timezone.utc)}".encode()).hexdigest()[:16],
            question=question,
            plan=asdict(plan),
            evidence=[{"id": r.chunk_id, "text": r.text, "score": r.score, "source_id": r.source_id} for r in ranked],
            answer=draft.answer,
            citations=[{"claim": c.text, "source_id": c.source_id, "passage": c.passage} for c in draft.citations],
            skillpack=skillpack,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return trace

    async def generate_batch(self, questions: list[str], skillpack: Optional[str] = None, max_traces: int = 100) -> int:
        """Generate traces for a batch of questions. Returns count of new traces stored."""
        count = 0
        for q in questions:
            trace = await self.generate_sft_trace(q, skillpack)
            if trace and self.store.add_sft(trace):
                count += 1
            if count >= max_traces:
                break
        return count

    async def close(self) -> None:
        await self.pipeline.close()


async def main() -> None:
    gen = TraceGenerator()
    try:
        questions = [
            "How do I create a cube in Blender using bpy?",
            "Write a Python function to compute fibonacci numbers",
            "Compare PyTorch vs JAX for machine learning",
        ]
        n = await gen.generate_batch(questions, skillpack="blender", max_traces=3)
        print(f"Generated {n} new traces")
        print(f"Store counts: {gen.store.counts()}")
    finally:
        await gen.close()


if __name__ == "__main__":
    asyncio.run(main())
