"""
Core Pipeline — orchestrates the full retrieval-augmented loop.
"""
from dataclasses import dataclass
from typing import Optional
import asyncio

from core.governor import ResourceGovernor, ProfileName
from core.router import classify_route, RouteType
from core.planner import make_plan, Plan
from core.retriever import Retriever, SourceType
from core.reranker import Reranker, Compressor
from core.reasoner import Reasoner
from core.verifier import Verifier
from core.writer import Writer
from core.safety import SafetySanitizer
from core.llm_client import ModelManager, LLMClient

@dataclass
class PipelineConfig:
    profile: ProfileName = ProfileName.STANDARD
    max_loops: int = 3

class Pipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.model_mgr = ModelManager()
        self.governor = ResourceGovernor()
        self.router = classify_route
        self.planner = make_plan
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.compressor = Compressor()
        self.writer = Writer()
        self.safety = SafetySanitizer()
        
        # Initialize LLM-backed components
        self._init_llm_components()
    
    def _init_llm_components(self):
        profile_name = self.config.profile.value
        client = self.model_mgr.get_client(profile_name)
        self.reasoner = Reasoner(client=client)
        self.verifier = Verifier(
            profile_config=self.governor.current_profile().__dict__,
            client=client
        )
    
    async def answer(self, question: str) -> str:
        profile = self.governor.current_profile()
        profile_name = profile.name if hasattr(profile, 'name') else self.config.profile
        
        # Route
        route_decision = self.router(question, profile_name.value)
        if route_decision.route == RouteType.DIRECT:
            result = await self.reasoner.synthesize(question, "")
            return result.answer
        
        # Plan
        plan = self.planner(question, profile_name.value)
        
        # Retrieve
        all_chunks = []
        for step in plan.steps:
            docs = await self.retriever.search(step.query, step.source_type)
            for doc in docs:
                # Safety sanitize
                safe_result = self.safety.sanitize(doc.content, doc.url)
                if not safe_result.safe:
                    continue
                # Chunk
                chunks = self.retriever.chunk_document(doc)
                all_chunks.extend(chunks)
        
        if not all_chunks:
            return "I could not find relevant information to answer your question."
        
        # Rerank + compress
        ranked = self.reranker.rank(question, all_chunks, top_k=profile.max_chunks)
        evidence = self.compressor.compress_chunks(question, ranked)
        
        # Reason + verify loop
        for attempt in range(self.config.max_loops):
            draft_output = await self.reasoner.synthesize(question, evidence, plan.skillpack)
            report = self.verifier.check(draft_output.answer, evidence)
            
            if report.ok:
                return self.writer.finalize(draft_output.answer, evidence, [
                    {"claim": c.text, "source_id": c.source_id, "passage": c.passage}
                    for c in draft_output.claims
                ]).answer
            
            # Repair: search for missing info
            if report.missing_query:
                new_docs = await self.retriever.search(report.missing_query, SourceType.WEB)
                for doc in new_docs:
                    chunks = self.retriever.chunk_document(doc)
                    all_chunks.extend(chunks)
                ranked = self.reranker.rank(question, all_chunks, top_k=profile.max_chunks)
                evidence = self.compressor.compress_chunks(question, ranked)
        
        # Max loops reached, return with uncertainty
        return self.writer.finalize_with_uncertainty(
            draft_output.answer, evidence,
            report.unsupported_claims, report.execution_errors
        ).answer
    
    async def close(self):
        await self.model_mgr.close_all()

async def main():
    pipeline = Pipeline()
    try:
        question = "How do I create a cube in Blender using bpy?"
        answer = await pipeline.answer(question)
        print(answer)
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())
