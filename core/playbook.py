"""Playbook Memory — per-folder index of user's project patterns & workflows.

Tracks reusable skill sequences, successful prompt patterns, and
project-level preferences so the Reasoner can shortcut to proven
workflows instead of re-inventing steps.

Design:
- SQLite-backed per-project store under ~/.hermes/playbooks/
- Each entry: pattern_name, trigger_keywords, skill_sequence, success_score, last_used
- Skill sequences are serialized PlanStep lists compatible with the Planner
- Supports "pattern match → auto-suggest steps" and "record successful flow"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import sqlite3
import json
import hashlib
from datetime import datetime


PLAYBOOK_ROOT = Path.home() / ".hermes" / "playbooks"


@dataclass
class PlaybookEntry:
    pattern_name: str
    trigger_keywords: list[str]
    skill_sequence: list[dict]  # PlanStep dicts
    success_score: float = 0.0
    times_used: int = 0
    last_used: Optional[datetime] = None
    created: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name,
            "trigger_keywords": self.trigger_keywords,
            "skill_sequence": [s.model_dump() if hasattr(s, "model_dump") else s for s in self.skill_sequence],
            "success_score": self.success_score,
            "times_used": self.times_used,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "created": self.created.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlaybookEntry":
        return cls(
            pattern_name=d["pattern_name"],
            trigger_keywords=d["trigger_keywords"],
            skill_sequence=d["skill_sequence"],
            success_score=d.get("success_score", 0.0),
            times_used=d.get("times_used", 0),
            last_used=datetime.fromisoformat(d["last_used"]) if d.get("last_used") else None,
            created=datetime.fromisoformat(d["created"]) if d.get("created") else datetime.now(),
        )


class PlaybookDB:
    """SQLite-backed playbook storage."""

    def __init__(self, root: Path = PLAYBOOK_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "playbooks.db"
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS playbooks (
                pattern_name TEXT PRIMARY KEY,
                trigger_keywords TEXT,
                skill_sequence TEXT,
                success_score REAL DEFAULT 0,
                times_used INTEGER DEFAULT 0,
                last_used TEXT,
                created TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
        conn.close()

    def upsert(self, entry: PlaybookEntry) -> None:
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        data = entry.to_dict()
        cur.execute(
            """INSERT INTO playbooks
               (pattern_name, trigger_keywords, skill_sequence, success_score, times_used, last_used, created)
               VALUES (:pattern_name, :trigger_keywords, :skill_sequence, :success_score, :times_used, :last_used, :created)
               ON CONFLICT(pattern_name) DO UPDATE SET
                 trigger_keywords = :trigger_keywords,
                 skill_sequence = :skill_sequence,
                 success_score = :success_score,
                 times_used = times_used + 1,
                 last_used = :last_used""",
            {
                "pattern_name": data["pattern_name"],
                "trigger_keywords": json.dumps(data["trigger_keywords"]),
                "skill_sequence": json.dumps(data["skill_sequence"]),
                "success_score": data["success_score"],
                "times_used": data["times_used"],
                "last_used": data["last_used"],
                "created": data["created"],
            },
        )
        conn.commit()
        conn.close()

    def find_by_keywords(self, keywords: list[str]) -> list[PlaybookEntry]:
        """Return entries whose trigger_keywords intersect with the given keywords."""
        matches: list[PlaybookEntry] = []
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        # Store keywords as JSON for pattern matching
        kw_json = json.dumps(sorted(keywords))
        cur.execute(
            """
            SELECT pattern_name, trigger_keywords, skill_sequence, success_score, times_used, last_used
            FROM playbooks
            WHERE trigger_keywords LIKE ?
            ORDER BY success_score DESC, times_used DESC
            """,
            (f"%{','.join(sorted(keywords))}%",),
        )
        for row in cur.fetchall():
            entry = PlaybookEntry.from_dict(
                {
                    "pattern_name": row[0],
                    "trigger_keywords": json.loads(row[1]),
                    "skill_sequence": json.loads(row[2]),
                    "success_score": row[3],
                    "times_used": row[4],
                    "last_used": row[5],
                }
            )
            matches.append(entry)
        conn.close()
        return matches

    def get(self, pattern_name: str) -> PlaybookEntry | None:
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT pattern_name, trigger_keywords, skill_sequence, success_score, times_used, last_used FROM playbooks WHERE pattern_name = ?",
            (pattern_name,),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return PlaybookEntry.from_dict(
            {
                "pattern_name": row[0],
                "trigger_keywords": json.loads(row[1]),
                "skill_sequence": json.loads(row[2]),
                "success_score": row[3],
                "times_used": row[4],
                "last_used": row[5],
            }
        )


__all__ = ["PlaybookEntry", "PlaybookDB", "PLAYBOOK_ROOT"]
