"""
Local Project Memory — per-folder index of user's projects with decision log.
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import sqlite3
import json
import hashlib
import fnmatch
from datetime import datetime

@dataclass
class ProjectMemoryConfig:
    root_path: Path
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "*.env", "*.key", "*.pem", "*.p12", "*.pfx",
        "id_rsa*", "id_ed25519*", "*.kdbx",
        ".git/", "__pycache__/", "node_modules/",
        "*.pyc", "*.pyo", "*.so", "*.dll",
        "*.log", "*.tmp", "*.cache",
    ])
    chunk_size: int = 500
    chunk_overlap: int = 50

@dataclass
class ProjectChunk:
    chunk_id: str
    project_id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    embedding: Optional[list[float]] = None

@dataclass
class DecisionLogEntry:
    entry_id: str
    project_id: str
    timestamp: str
    decision: str
    rationale: str
    context: str
    tags: list[str]

class ProjectMemory:
    def __init__(self, config: ProjectMemoryConfig):
        self.config = config
        self.db_path = config.root_path / ".agan" / "project_memory.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    file_path TEXT,
                    content TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    embedding BLOB,
                    indexed_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    entry_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    timestamp TEXT,
                    decision TEXT,
                    rationale TEXT,
                    context TEXT,
                    tags TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project_id)")
    
    def _project_id(self) -> str:
        return hashlib.sha256(str(self.config.root_path.resolve()).encode()).hexdigest()[:16]
    
    def _should_exclude(self, path: Path) -> bool:
        rel_path = str(path.relative_to(self.config.root_path))
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(path.name, pattern):
                return True
        return False
    
    def index_project(self) -> int:
        project_id = self._project_id()
        chunks_indexed = 0
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE project_id = ?", (project_id,))
            
            for file_path in self.config.root_path.rglob("*"):
                if file_path.is_file() and not self._should_exclude(file_path):
                    try:
                        chunks = self._chunk_file(file_path, project_id)
                        for chunk in chunks:
                            conn.execute("""
                                INSERT INTO chunks (chunk_id, project_id, file_path, content, start_line, end_line, indexed_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (chunk.chunk_id, chunk.project_id, chunk.file_path, chunk.content,
                                  chunk.start_line, chunk.end_line, datetime.utcnow().isoformat()))
                            chunks_indexed += 1
                    except Exception as e:
                        print(f"Failed to index {file_path}: {e}")
        
        return chunks_indexed
    
    def _chunk_file(self, file_path: Path, project_id: str) -> list[ProjectChunk]:
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return []
        
        lines = content.splitlines()
        chunks = []
        start = 0
        idx = 0
        
        while start < len(lines):
            end = min(start + self.config.chunk_size, len(lines))
            chunk_lines = lines[start:end]
            chunk_content = "\n".join(chunk_lines)
            
            chunk_id = hashlib.sha256(f"{project_id}:{file_path}:{idx}".encode()).hexdigest()[:16]
            chunks.append(ProjectChunk(
                chunk_id=chunk_id,
                project_id=project_id,
                file_path=str(file_path.relative_to(self.config.root_path)),
                content=chunk_content,
                start_line=start + 1,
                end_line=end,
            ))
            start += self.config.chunk_size - self.config.chunk_overlap
            idx += 1
        
        return chunks
    
    def search(self, query: str, max_results: int = 10) -> list[ProjectChunk]:
        project_id = self._project_id()
        query_terms = query.lower().split()
        
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT chunk_id, file_path, content, start_line, end_line
                FROM chunks WHERE project_id = ?
            """, (project_id,)).fetchall()
        
        results = []
        for row in rows:
            content_lower = row[2].lower()
            score = sum(1 for term in query_terms if term in content_lower)
            if score > 0:
                results.append((score, ProjectChunk(
                    chunk_id=row[0], project_id=project_id, file_path=row[1],
                    content=row[2], start_line=row[3], end_line=row[4]
                )))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:max_results]]
    
    def add_decision(self, decision: str, rationale: str, context: str = "", tags: list[str] = None) -> str:
        entry_id = hashlib.sha256(f"{self._project_id()}:{decision}:{datetime.utcnow()}".encode()).hexdigest()[:16]
        entry = DecisionLogEntry(
            entry_id=entry_id,
            project_id=self._project_id(),
            timestamp=datetime.utcnow().isoformat(),
            decision=decision,
            rationale=rationale,
            context=context,
            tags=tags or [],
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO decisions (entry_id, project_id, timestamp, decision, rationale, context, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (entry.entry_id, entry.project_id, entry.timestamp, entry.decision,
                  entry.rationale, entry.context, json.dumps(entry.tags)))
        return entry_id
    
    def get_decisions(self, tags: list[str] = None) -> list[DecisionLogEntry]:
        with sqlite3.connect(self.db_path) as conn:
            if tags:
                rows = conn.execute(
                    "SELECT * FROM decisions WHERE project_id = ? AND tags LIKE ?",
                    (self._project_id(), f"%{tags[0]}%")
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM decisions WHERE project_id = ?",
                    (self._project_id(),)
                ).fetchall()
        
        return [DecisionLogEntry(
            entry_id=r[0], project_id=r[1], timestamp=r[2], decision=r[3],
            rationale=r[4], context=r[5], tags=json.loads(r[6])
        ) for r in rows]
    
    def get_stats(self) -> dict:
        project_id = self._project_id()
        with sqlite3.connect(self.db_path) as conn:
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE project_id = ?", (project_id,)).fetchone()[0]
            decision_count = conn.execute("SELECT COUNT(*) FROM decisions WHERE project_id = ?", (project_id,)).fetchone()[0]
        return {
            "project_id": project_id,
            "root_path": str(self.config.root_path),
            "chunks_indexed": chunk_count,
            "decisions_logged": decision_count,
        }

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "test.py").write_text("def hello():\n    print('hello')\n\n# Decision: use print for simplicity\n")
        config = ProjectMemoryConfig(tmp)
        pm = ProjectMemory(config)
        count = pm.index_project()
        print(f"Indexed {count} chunks")
        results = pm.search("hello")
        print(f"Search results: {len(results)}")
        pm.add_decision("Use print for output", "Simple and works everywhere", tags=["output"])
        decisions = pm.get_decisions()
        print(f"Decisions: {len(decisions)}")
        print(pm.get_stats())