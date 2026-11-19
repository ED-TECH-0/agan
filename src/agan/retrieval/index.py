"""SQLite FTS5 + vector search index for retrieval."""
import asyncio
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from agan.schemas import Chunk

DB_PATH = Path.home() / ".agan" / "retrieval.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Vector dimension for sentence-transformers all-MiniLM-L6-v2
VECTOR_DIM = 384


def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=NORMAL;
    PRAGMA cache_size=-32768;

    CREATE TABLE IF NOT EXISTS pages (
        url TEXT PRIMARY KEY,
        title TEXT,
        fetched_at REAL,
        etag TEXT,
        last_modified TEXT,
        kind TEXT,
        ttl_hours INTEGER,
        trust REAL DEFAULT 0.5,
        version_tag TEXT
    );

    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        url TEXT,
        ord INTEGER,
        heading_path TEXT,
        text TEXT,
        tokens INTEGER,
        emb BLOB,
        FOREIGN KEY(url) REFERENCES pages(url)
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text, heading_path, content='chunks', content_rowid='rowid'
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_url ON chunks(url);
    CREATE INDEX IF NOT EXISTS idx_pages_fetched ON pages(fetched_at);
    """)


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        _init_db(conn)
        yield conn
    finally:
        conn.close()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


async def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Embed texts using sentence-transformers. Runs in thread pool to avoid blocking."""
    from sentence_transformers import SentenceTransformer

    # Load model once (cached globally)
    if not hasattr(embed_texts, "_model"):
        embed_texts._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _encode():
        return embed_texts._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, _encode)
    return [emb for emb in embeddings]


async def add_chunks(chunks: list[Chunk], embeddings: list[np.ndarray]):
    """Add chunks and their embeddings to the index."""
    with _get_conn() as conn:
        for chunk, emb in zip(chunks, embeddings):
            # Insert/update page
            conn.execute("""
                INSERT OR REPLACE INTO pages (url, title, fetched_at, etag, last_modified, kind, ttl_hours, trust, version_tag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (chunk.url, chunk.title, chunk.fetched_at, None, None, "web", 48, chunk.trust, chunk.version_tag))

            # Insert chunk
            conn.execute("""
                INSERT OR REPLACE INTO chunks (id, url, ord, heading_path, text, tokens, emb)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (chunk.id, chunk.url, 0, "", chunk.text, 0, _vec_to_blob(emb)))

            # FTS5 will be updated via trigger or manually
            conn.execute("INSERT INTO chunks_fts(rowid, text, heading_path) VALUES (last_insert_rowid(), ?, ?)", (chunk.text, ""))

        conn.commit()


async def keyword_search(query: str, limit: int = 50) -> list[tuple[str, float]]:
    """BM25 keyword search via FTS5. Returns (chunk_id, score)."""
    with _get_conn() as conn:
        # FTS5 rank function: bm25()
        rows = conn.execute("""
            SELECT c.id, bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
        """, (query, limit)).fetchall()
        return [(row["id"], row["score"]) for row in rows]


async def vector_search(query_vec: np.ndarray, limit: int = 50, min_score: float = 0.0) -> list[tuple[str, float]]:
    """Exact cosine similarity search over all vectors. Returns (chunk_id, score)."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT id, emb FROM chunks WHERE emb IS NOT NULL").fetchall()
        results = []
        for row in rows:
            vec = _blob_to_vec(row["emb"])
            score = _cosine_sim(query_vec, vec)
            if score >= min_score:
                results.append((row["id"], score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


async def hybrid_search(query: str, query_vec: np.ndarray, limit: int = 20, k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of keyword and vector search."""
    kw_results = await keyword_search(query, limit=limit * 2)
    vec_results = await vector_search(query_vec, limit=limit * 2)

    # RRF: score(d) = sum(1 / (k + rank_i(d)))
    scores: dict[str, float] = {}
    for rank, (cid, _) in enumerate(kw_results):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(vec_results):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:limit]


async def get_chunks_by_ids(ids: list[str]) -> list[Chunk]:
    """Retrieve full Chunk objects by IDs."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT c.id, c.url, p.title, c.text, c.heading_path, c.tokens, p.fetched_at, p.trust, p.version_tag
            FROM chunks c
            JOIN pages p ON p.url = c.url
            WHERE c.id IN ({placeholders})
        """, ids).fetchall()
        return [Chunk(
            id=row["id"],
            url=row["url"],
            title=row["title"] or "",
            text=row["text"],
            fetched_at=row["fetched_at"],
            trust=row["trust"],
            version_tag=row["version_tag"],
            tainted=True,
        ) for row in rows]


async def get_page_info(url: str) -> dict | None:
    """Get cached page metadata."""
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM pages WHERE url = ?", (url,)).fetchone()
        return dict(row) if row else None


async def update_page_trust(url: str, trust: float):
    """Update trust score for a domain/page."""
    with _get_conn() as conn:
        conn.execute("UPDATE pages SET trust = ? WHERE url = ?", (trust, url))
        conn.commit()


async def prune_old_cache(max_age_days: int = 30):
    """Remove pages older than max_age_days."""
    cutoff = time.time() - max_age_days * 86400
    with _get_conn() as conn:
        conn.execute("DELETE FROM chunks WHERE url IN (SELECT url FROM pages WHERE fetched_at < ?)", (cutoff,))
        conn.execute("DELETE FROM pages WHERE fetched_at < ?", (cutoff,))
        conn.commit()


__all__ = [
    "DB_PATH",
    "VECTOR_DIM",
    "add_chunks",
    "embed_texts",
    "get_chunks_by_ids",
    "get_page_info",
    "hybrid_search",
    "keyword_search",
    "prune_old_cache",
    "update_page_trust",
    "vector_search",
]