"""AGAN pipeline orchestrator — wires router→planner→reasoner→verifier→writer."""

from __future__ import annotations

import time
import logging
import uuid
from typing import Any, Dict, List, Optional

from agan.schemas import Plan, Answer, Claim, Route, UserProfile
from agan.llm.client import LLMClient
from agan.pipeline.router import Router
from agan.pipeline.planner import Planner
from agan.pipeline.reasoner import Reasoner
from agan.pipeline.verifier import Verifier, VerifierReport
from agan.pipeline.writer import Writer
from agan.hardware.detect import detect_profile, ProfileSpec, SPECS, HardwareProfile
from agan.privacy.firewall import OutboundGateway

logger = logging.getLogger("agan.pipeline")


class Gateway:
    """Entry point for user requests — sanitises and assembles a pipeline request."""

    def __init__(self, user_profile: Optional[UserProfile] = None, hardware_profile=None):
        self.user_profile = user_profile or UserProfile()
        self.hardware_profile = hardware_profile or detect_profile()
        self.privacy_firewall = OutboundGateway()

    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Sanitise a user query and return a request dict for the orchestrator."""
        if not self.privacy_firewall.check(query):
            return {
                "query": query,
                "user_id": str(uuid.uuid4()),
                "blocked": True,
                "reason": "privacy firewall blocked the query",
            }

        merged_context = {**self.user_profile.stated, **(context or {})}

        return {
            "query": query,
            "user_id": str(uuid.uuid4()),
            "blocked": False,
            "profile": self.hardware_profile.value,
            "hardware": self.hardware_profile.value,
            "context": merged_context,
        }


class Orchestrator:
    """Drives the AGAN pipeline: route -> plan -> reason -> verify -> write.

    The Reasoner internally handles retrieval + verification; the orchestrator
    adds a second verification pass and final writing stage for polish.
    """

    def __init__(
        self,
        client: LLMClient,
        profile: Optional[ProfileSpec] = None,
        verifier=None,
        sandbox=None,
    ):
        self.client = client
        hw = profile.name if profile else detect_profile()
        self.profile = profile or SPECS[hw]
        self.router = Router(client)
        self.planner = Planner(client, profile=self.profile)
        self.reasoner = Reasoner(client, profile=self.profile)
        self.verifier = verifier or Verifier(sandbox=sandbox, profile=self.profile)
        self.writer = Writer(profile_spec=self.profile)
        self.privacy_firewall = OutboundGateway()

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the full M3 pipeline for a single query."""
        start = time.time()

        # Privacy check
        if not self.privacy_firewall.check(query):
            return {
                "query": query,
                "blocked": True,
                "reason": "privacy firewall blocked the query",
                "pipeline_time_sec": time.time() - start,
            }

        # 1. Route
        route = self.router.route(query)
        logger.info("Route: %s", route)

        # 2. Plan
        plan = await self.planner.plan(query, route)

        # 3. Reason (internally retrieves evidence + verifies + produces Answer)
        answer = await self.reasoner.execute(query, plan)

        # 4. Verify again against evidence (separate pass for the report)
        evidence_texts: List[str] = [c.text for c in answer.claims] if answer.claims else []
        report = await self.verifier.verify_claims(answer.text, evidence_texts)

        # 5. Write final answer
        final_answer = self.writer.write(answer, report, plan)

        elapsed = time.time() - start
        logger.info("Pipeline completed in %.2fs, confidence=%s", elapsed, final_answer.confidence)

        return {
            "answer": final_answer,
            "route": route,
            "plan": plan,
            "verification": report,
            "pipeline_time_sec": elapsed,
            "hardware_profile": self.profile.name.value,
        }


def make_pipeline(
    llm_model: str = "qwen2.5-3b:latest",
    hardware_profile: Optional[HardwareProfile] = None,
) -> Dict[str, Any]:
    """Factory function to create a complete AGAN pipeline."""
    hw = hardware_profile or detect_profile()
    spec = SPECS[hw]
    client = LLMClient(model=llm_model)

    gateway = Gateway(user_profile=UserProfile(), hardware_profile=hw)
    orchestrator = Orchestrator(
        client=client,
        profile=spec,
    )

    return {
        "gateway": gateway,
        "orchestrator": orchestrator,
        "client": client,
        "hardware_profile": hw,
    }


__all__ = [
    "Gateway",
    "Orchestrator",
    "make_pipeline",
    "OutboundGateway",
    "ProfileSpec",
    "HardwareProfile",
    "UserProfile",
    "Answer",
    "Plan",
    "Claim",
    "Route",
]
