"""DPO (Direct Preference Optimization) pair generator.

Generates preference pairs from SFT traces:
- chosen: grounded, cited answer (from verified trace)
- rejected: hallucinated or uncited answer (synthesized by removing citations/evidence)

Output: DPO pairs stored in data/traces.db for DPO training.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/traces.db")


@dataclass
class DPOTrace:
    id: str
    question: str
    chosen: str          # grounded, cited answer
    rejected: str        # hallucinated or uncited answer
    skillpack: Optional[str]
    created_at: str


def make_rejected_answer(answer: str) -> str:
    """Create a rejected version by removing citations and weakening the answer."""
    # Remove citation markers [E1], [E2], etc.
    rejected = re.sub(r"\[E\d+\]", "", answer)
    # Remove "Based on the retrieved evidence" type framing
    rejected = re.sub(r"Based on the retrieved evidence[,]?\s*", "", rejected, flags=re.IGNORECASE)
    rejected = re.sub(r"According to the sources[,]?\s*", "", rejected, flags=re.IGNORECASE)
    # Add uncertainty markers
    if not rejected.strip():
        rejected = "I'm not sure about this."
    rejected = rejected.strip()
    if not rejected.endswith((".", "!", "?")):
        rejected += "."
    return rejected


def generate_dpo_from_sft(traces: list[dict]) -> list[DPOTrace]:
    """Convert SFT traces to DPO pairs."""
    pairs = []
    for trace in traces:
        chosen = trace["answer"]
        rejected = make_rejected_answer(chosen)
        
        if chosen == rejected:
            continue  # Skip if rejection didn't change anything
        
        trace_id = hashlib.sha256(
            f"{trace['question']}:{chosen}:{rejected}".encode()
        ).hexdigest()[:16]
        
        pairs.append(DPOTrace(
            id=trace_id,
            question=trace["question"],
            chosen=chosen,
            rejected=rejected,
            skillpack=trace.get("skillpack"),
            created_at=datetime.now(timezone.utc).isoformat(),
        ))
    return pairs


class DPOStore:
    """SQLite storage for DPO pairs."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
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

    def add(self, trace: DPOTrace) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO dpo_pairs (id, question, chosen, rejected, skillpack, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (trace.id, trace.question, trace.chosen, trace.rejected,
                     trace.skillpack, trace.created_at),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def counts(self) -> dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            sft = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            dpo = conn.execute("SELECT COUNT(*) FROM dpo_pairs").fetchone()[0]
        return {"sft": sft, "dpo": dpo}


async def generate_dpo_from_traces(max_pairs: int = 100) -> int:
    """Generate DPO pairs from existing SFT traces."""
    store = DPOStore()
    
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT question, plan, evidence, answer, citations, skillpack FROM traces "
            "ORDER BY created_at DESC LIMIT ?",
            (max_pairs * 2,),
        ).fetchall()
    
    traces = []
    for row in rows:
        traces.append({
            "question": row[0],
            "plan": json.loads(row[1]),
            "evidence": json.loads(row[2]),
            "answer": row[3],
            "citations": json.loads(row[4]),
            "skillpack": row[5],
        })
    
    pairs = generate_dpo_from_sft(traces)
    count = 0
    for pair in pairs:
        if store.add(pair):
            count += 1
        if count >= max_pairs:
            break
    
    return count, store.counts()


async def main() -> None:
    count, counts = await generate_dpo_from_traces(max_pairs=50)
    print(f"Generated {count} new DPO pairs")
    print(f"Store counts: {counts}")


if __name__ == "__main__":
    asyncio.run(main())
