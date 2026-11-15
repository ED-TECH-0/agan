"""
Reasoner — synthesizes answer from evidence with citations.
"""
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class ReasonerOutput:
    answer: str
    citations: list[dict]  # [{"claim": str, "source_id": str, "passage": str}]
    reasoning_trace: str

class Reasoner:
    def __init__(self, model=None, adapters: list = None):
        self.model = model
        self.adapters = adapters or []
    
    def synthesize(self, question: str, evidence: str, skillpack: Optional[str] = None) -> ReasonerOutput:
        # Placeholder: actual model inference here
        # For now, return a template response
        
        prompt = self._build_prompt(question, evidence, skillpack)
        
        # Mock response
        answer = f"Based on the retrieved evidence, here is what I found regarding: {question}\n\n"
        answer += "The evidence shows key information from multiple sources. "
        answer += "Specific details would be synthesized here with citations."
        
        # Mock citations
        citations = [
            {"claim": "Example claim from evidence", "source_id": "source_1", "passage": "Relevant passage..."},
        ]
        
        reasoning = "1. Analyzed question type\n2. Retrieved relevant evidence\n3. Synthesized answer with citations"
        
        return ReasonerOutput(answer=answer, citations=citations, reasoning_trace=reasoning)
    
    def _build_prompt(self, question: str, evidence: str, skillpack: Optional[str]) -> str:
        skill_instruction = ""
        if skillpack:
            skill_instruction = f"\nYou are an expert in {skillpack}. Apply domain-specific knowledge."
        
        return f"""You are a retrieval-augmented assistant. Answer the question using ONLY the provided evidence.
        {skill_instruction}
        
        Rules:
        - Factual claims MUST come from evidence; cite sources inline like [source_1]
        - Logic, math, code structure can come from reasoning
        - If evidence is insufficient, say "I could not verify X"
        - Never invent sources or facts
        
        Question: {question}
        
        Evidence:
        {evidence}
        
        Answer with citations:"""
    
    def repair_prompt(self, question: str, errors: list[str]) -> str:
        return f"""Previous attempt had errors:
        {chr(10).join(errors)}
        
        Original question: {question}
        
        Revise your approach and try again."""

if __name__ == "__main__":
    r = Reasoner()
    out = r.synthesize("How to create a cube in Blender bpy?", "Evidence: bpy.ops.mesh.primitive_cube_add() creates a cube. [source_1]")
    print(out.answer)
    print("\nCitations:", out.citations)
