"""
Reranker + Compressor — cross-encode and extract relevant sentences.
"""
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class RankedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source_id: str

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = False
    
    def rank(self, query: str, chunks: list, top_k: int = 5) -> list[RankedChunk]:
        self._load_model()
        if self._model is False:
            # Fallback: simple keyword overlap scoring
            return self._fallback_rank(query, chunks, top_k)
        
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [
            RankedChunk(
                chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text,
                score=float(s), source_id=c.doc_id
            )
            for c, s in ranked[:top_k]
        ]
    
    def _fallback_rank(self, query: str, chunks: list, top_k: int) -> list[RankedChunk]:
        query_terms = set(query.lower().split())
        scored = []
        for c in chunks:
            text_terms = set(c.text.lower().split())
            overlap = len(query_terms & text_terms)
            score = overlap / max(len(query_terms), 1)
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RankedChunk(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text, score=s, source_id=c.doc_id)
            for c, s in scored[:top_k]
        ]

class Compressor:
    @staticmethod
    def extract_relevant_sentences(query: str, text: str, max_sentences: int = 3) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        query_terms = set(query.lower().split())
        scored = []
        for s in sentences:
            s_terms = set(s.lower().split())
            overlap = len(query_terms & s_terms)
            score = overlap / max(len(query_terms), 1)
            scored.append((s, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in scored[:max_sentences]]
        return " ".join(selected)
    
    @staticmethod
    def compress_chunks(query: str, chunks: list[RankedChunk], max_total_chars: int = 2000) -> str:
        parts = []
        total = 0
        for rc in chunks:
            compressed = Compressor.extract_relevant_sentences(query, rc.text)
            if total + len(compressed) > max_total_chars:
                break
            parts.append(f"[Source {rc.source_id}] {compressed}")
            total += len(compressed)
        return "\n\n".join(parts)

if __name__ == "__main__":
    from core.retriever import Chunk
    chunks = [
        Chunk("d1", "c1", "Blender bpy tutorial shows how to create a cube. Use bpy.ops.mesh.primitive_cube_add().", 0, 100),
        Chunk("d2", "c2", "Blender geometry nodes are powerful. Create procedural assets with node trees.", 0, 100),
        Chunk("d3", "c3", "Python scripting in Blender uses the bpy module. Import bpy first.", 0, 100),
    ]
    reranker = Reranker()
    ranked = reranker.rank("Blender bpy create cube", chunks, top_k=2)
    for r in ranked:
        print(f"Score: {r.score:.3f} | {r.text[:80]}")
    compressed = Compressor.compress_chunks("Blender bpy create cube", ranked)
    print(f"\nCompressed:\n{compressed}")
