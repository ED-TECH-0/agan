"""
Retriever — search → fetch → clean → chunk → embed → hybrid rank
"""
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import hashlib
import sqlite3
from pathlib import Path
import json

class SourceType(str, Enum):
    WEB = "web"
    WIKI = "wiki"
    DOCS = "docs"
    LOCAL = "local"
    PROJECT = "project"

@dataclass
class Document:
    id: str
    url: str
    title: str
    content: str
    source_type: SourceType
    timestamp: str
    domain: str
    trust_score: float = 0.5

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    embedding: Optional[list[float]] = None

class Retriever:
    def __init__(self, cache_dir: Path = Path("cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "retrieval.db"
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE,
                    title TEXT,
                    content TEXT,
                    source_type TEXT,
                    timestamp TEXT,
                    domain TEXT,
                    trust_score REAL,
                    embeddings BLOB
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    text TEXT,
                    start_char INTEGER,
                    end_char INTEGER,
                    embedding BLOB,
                    FOREIGN KEY(doc_id) REFERENCES documents(id)
                )
            """)
    
    def _doc_id(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def _chunk_id(self, doc_id: str, index: int) -> str:
        return f"{doc_id}_{index:04d}"
    
    async def search(self, query: str, source_type: SourceType = SourceType.WEB, max_results: int = 10, cache: bool = True) -> list[Document]:
        # Placeholder: integrate with SearXNG, Wikipedia API, etc.
        # Returns mock results for now
        return [
            Document(
                id=self._doc_id(f"https://example.com/{i}"),
                url=f"https://example.com/{i}",
                title=f"Result {i} for: {query}",
                content=f"This is mock content for query '{query}' from source {i}. " * 50,
                source_type=source_type,
                timestamp="2024-01-01T00:00:00Z",
                domain="example.com",
                trust_score=0.7,
            )
            for i in range(min(max_results, 5))
        ]
    
    async def fetch(self, url: str) -> Optional[Document]:
        # Check cache first
        doc_id = self._doc_id(url)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if row:
                return Document(
                    id=row[0], url=row[1], title=row[2], content=row[3],
                    source_type=SourceType(row[4]), timestamp=row[5],
                    domain=row[6], trust_score=row[7]
                )
        # Placeholder: actual HTTP fetch
        return None
    
    def chunk_document(self, doc: Document, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]:
        text = doc.content
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(Chunk(
                doc_id=doc.id,
                chunk_id=self._chunk_id(doc.id, idx),
                text=chunk_text,
                start_char=start,
                end_char=end,
            ))
            start += chunk_size - overlap
            idx += 1
        return chunks
    
    def store_document(self, doc: Document, chunks: list[Chunk]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO documents (id, url, title, content, source_type, timestamp, domain, trust_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (doc.id, doc.url, doc.title, doc.content, doc.source_type.value, doc.timestamp, doc.domain, doc.trust_score))
            for chunk in chunks:
                conn.execute("""
                    INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, start_char, end_char)
                    VALUES (?, ?, ?, ?, ?)
                """, (chunk.chunk_id, chunk.doc_id, chunk.text, chunk.start_char, chunk.end_char))
    
    def get_chunks(self, doc_ids: list[str]) -> list[Chunk]:
        if not doc_ids:
            return []
        placeholders = ",".join("?" * len(doc_ids))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(f"SELECT * FROM chunks WHERE doc_id IN ({placeholders})", doc_ids).fetchall()
            return [Chunk(doc_id=r[1], chunk_id=r[0], text=r[2], start_char=r[3], end_char=r[4]) for r in rows]

# Domain trust scoring
TRUSTED_DOMAINS = {
    "python.org": 1.0, "docs.python.org": 1.0,
    "blender.org": 1.0, "docs.blender.org": 1.0,
    "unrealengine.com": 1.0, "docs.unrealengine.com": 1.0,
    "github.com": 0.9, "gitlab.com": 0.9,
    "stackoverflow.com": 0.8, "stackexchange.com": 0.8,
    "wikipedia.org": 0.9,
    "arxiv.org": 0.95,
    "ieee.org": 0.95, "acm.org": 0.95,
    "nvidia.com": 0.9, "amd.com": 0.9, "intel.com": 0.9,
}

def score_domain(domain: str) -> float:
    domain = domain.lower()
    for trusted, score in TRUSTED_DOMAINS.items():
        if trusted in domain:
            return score
    return 0.5

if __name__ == "__main__":
    import asyncio
    async def test():
        r = Retriever()
        docs = await r.search("Blender bpy tutorial")
        for d in docs:
            print(f"{d.title} | {d.domain} | trust: {d.trust_score}")
            chunks = r.chunk_document(d)
            print(f"  -> {len(chunks)} chunks")
    asyncio.run(test())
