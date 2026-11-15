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

@dataclass
class PipelineConfig:
    profile: ProfileName = ProfileName.STANDARD
    max_loops: int = 3

class Pipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.governor = ResourceGovernor()
        self.router = classify_route
        self.planner = make_plan
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.compressor = Compressor()
        self.reasoner = Reasoner()
        self.verifier = Verifier(self.governor.current_profile().__dict__ if hasattr(self.governor, 'current_profile') else {})
        self.writer = Writer()
        self.safety = SafetySanitizer()
    
    async def answer(self, question: str) -> str:
        profile = self.governor.current_profile()
        profile_name = profile.name if hasattr(profile, 'name') else self.config.profile
        
        # Route
        route_decision = self.router(question, profile_name.value)
        if route_decision.route == RouteType.DIRECT:
            return self.reasoner.synthesize(question, "").answer
        
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
            draft_output = self.reasoner.synthesize(question, evidence, plan.skillpack)
            report = self.verifier.check(draft_output.answer, evidence)
            
            if report.ok:
                return self.writer.finalize(draft_output.answer, evidence, draft_output.citations).answer
            
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

async def main():
    pipeline = Pipeline()
    question = "How do I create a cube in Blender using bpy?"
    answer = await pipeline.answer(question)
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
