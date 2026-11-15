"""
Writer — formats final answer with citations and uncertainty.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class WriterOutput:
    answer: str
    sources: list[dict]
    uncertainties: list[str]

class Writer:
    def __init__(self):
        pass
    
    def finalize(self, draft: str, evidence: str, citations: list[dict] = None) -> WriterOutput:
        # Clean up draft, ensure citations are formatted
        answer = draft.strip()
        
        # Format sources
        sources = []
        if citations:
            for c in citations:
                sources.append({
                    "id": c.get("source_id", "unknown"),
                    "claim": c.get("claim", ""),
                    "passage": c.get("passage", "")[:200],
                })
        
        return WriterOutput(answer=answer, sources=sources, uncertainties=[])
    
    def finalize_with_uncertainty(self, draft: str, evidence: str, unsupported: list[str] = None, exec_errors: list[str] = None) -> WriterOutput:
        answer = draft.strip()
        
        if unsupported:
            answer += "\n\n**Unverified claims:**\n" + "\n".join(f"- {u}" for u in unsupported)
        if exec_errors:
            answer += "\n\n**Execution issues:**\n" + "\n".join(f"- {e}" for e in exec_errors)
        
        return WriterOutput(
            answer=answer,
            sources=[],
            uncertainties=unsupported or [] + exec_errors or []
        )

if __name__ == "__main__":
    w = Writer()
    out = w.finalize("Answer with [source_1] citation.", "evidence", [{"source_id": "source_1", "claim": "test", "passage": "passage"}])
    print(out.answer)
    print(out.sources)
