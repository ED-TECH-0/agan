"""
Reasoner — synthesizes answer from evidence with citations using real LLM.
"""
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field
from core.llm_client import LLMClient, ModelManager

class Claim(BaseModel):
    text: str
    source_id: str
    passage: str = ""

class ReasonerOutput(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    reasoning_trace: str = ""

class Reasoner:
    def __init__(self, client: LLMClient = None, skillpack: str = None):
        self.client = client
        self.skillpack = skillpack
    
    def _build_prompt(self, question: str, evidence: str, skillpack: Optional[str] = None) -> list[dict]:
        skill_instruction = ""
        if skillpack:
            skill_instruction = f"\nYou are an expert in {skillpack}. Apply domain-specific knowledge."
        
        system_prompt = f"""You are a retrieval-augmented assistant. Answer the question using ONLY the provided evidence.
{skill_instruction}

Rules:
- Factual claims MUST come from evidence; cite sources inline like [E1], [E2]
- Logic, math, code structure can come from reasoning
- If evidence is insufficient, say "I could not verify X"
- Never invent sources or facts
- Be concise and practical
- For code, give runnable code and state how to run it"""
        
        user_prompt = f"""Question: {question}

Evidence:
{evidence}

Answer with citations:"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    
    async def synthesize(self, question: str, evidence: str, skillpack: Optional[str] = None) -> ReasonerOutput:
        if not self.client:
            # Fallback mock
            return ReasonerOutput(
                answer=f"[Mock] Based on evidence: {question[:100]}...",
                claims=[Claim(text="Mock claim", source_id="E1", passage="Mock passage")],
                reasoning_trace="Mock reasoning"
            )
        
        sp = skillpack or self.skillpack
        messages = self._build_prompt(question, evidence, sp)
        
        try:
            # Use structured output for claims
            class ClaimsOutput(BaseModel):
                answer: str
                claims: list[Claim]
            
            result = await self.client.chat_json(messages, ClaimsOutput)
            return ReasonerOutput(
                answer=result.answer,
                claims=result.claims,
                reasoning_trace="LLM synthesis with structured claims"
            )
        except Exception as e:
            # Fallback to raw chat
            content = await self.client.chat(messages)
            return ReasonerOutput(
                answer=content,
                claims=[],
                reasoning_trace=f"Raw chat (structured failed: {e})"
            )
    
    def repair_prompt(self, question: str, errors: list[str]) -> list[dict]:
        error_text = "\n".join(errors)
        return [
            {"role": "system", "content": "You are fixing a previous answer that had errors."},
            {"role": "user", "content": f"""Previous attempt had errors:
{error_text}

Original question: {question}

Revise your approach and try again."""},
        ]

if __name__ == "__main__":
    import asyncio
    from core.llm_client import ModelManager
    
    async def test():
        mgr = ModelManager()
        client = mgr.get_client("lite")
        try:
            reasoner = Reasoner(client)
            out = await reasoner.synthesize(
                "How to create a cube in Blender bpy?",
                "Evidence [E1]: bpy.ops.mesh.primitive_cube_add() creates a cube at origin.\nEvidence [E2]: Use size parameter for dimensions."
            )
            print(f"Answer: {out.answer[:300]}")
            print(f"Claims: {len(out.claims)}")
            for c in out.claims:
                print(f"  - {c.text} [{c.source_id}]")
        finally:
            await mgr.close_all()
    
    asyncio.run(test())
