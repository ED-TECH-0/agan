from typing import Optional, List, Dict, Any
import asyncio
import re
import time

from agan.schemas import Answer, Plan, Claim, Evidence, Route
from agan.retrieval.providers import search_all
from agan.privacy.firewall import OutboundGateway
from agan.tools.registry import ToolRegistry
from agan.safety.sanitizer import sanitize_input
from agan.skills_builtin import SkillManager, SkillResult


class Reasoner:
    """Reasoner answers questions using evidence, tools, and verification."""

    def __init__(self, client, profile, retrieval=None, privacy=None, tools=None, safety=None):
        self.client = client
        self.profile = profile
        self.retrieval = retrieval or search_all
        self.privacy = privacy or OutboundGateway()
        self.tools = tools or ToolRegistry()
        self.safety = safety or sanitize_input
        self.skill_executor = SkillExecutor(SkillManager())

    async def execute(self, question: str, plan: Plan) -> Answer:
        """Execute the reasoning pipeline for a question."""
        # Step 1: Gather evidence if needed
        context = ""
        evidence_texts = []

        if plan.needs_search:
            # Check privacy firewall before searching
            if not self._check_privacy(question):
                return Answer(
                    text="I cannot search for this due to privacy restrictions.",
                    confidence="low",
                    level_used=self.profile.level.value
                )

            evidence_results = await self.retrieval(question, max_results=self.profile.max_retrieval)
            evidence_texts = [r.get("text", str(r)) for r in evidence_results]
            context = self._format_evidence(evidence_texts)

        # Step 2: Execute skill steps from the plan
        skill_results_context = ""
        if plan.steps or plan.skill_ids:
            skill_results = await self._execute_plan_skills(question, plan)
            if skill_results:
                skill_results_context = self._format_skill_results(skill_results)

        # Step 3: Build the reasoning prompt
        prompt = self._build_prompt(question, context, plan, skill_results_context)

        # Step 4: Get answer from model
        text = await self.client.complete(
            [{"role": "user", "content": prompt}],
            temperature=self._get_temperature(plan.answer_type),
            max_tokens=self.profile.max_output_tokens
        )

        # Step 5: Sanitize and extract
        text = self.safety.sanitize_text(text)

        # Step 6: Verify claims
        claims = await self._verify_draft(text, evidence_texts)

        # Step 7: Determine confidence
        confidence = self._determine_confidence(claims)

        # Step 8: Extract sources
        sources = self._extract_sources(evidence_texts, claims)

        return Answer(
            text=text,
            claims=claims,
            sources=sources,
            confidence=confidence,
            level_used=self.profile.level.value
        )

    def _check_privacy(self, question: str) -> bool:
        """Check if the question violates privacy firewall rules."""
        # Simple check - in full implementation, this would use the OutboundGateway
        private_indicators = ["password", "api_key", "token", "secret", "home path"]
        q_lower = question.lower()
        return not any(ind in q_lower for ind in private_indicators)

    def _format_evidence(self, evidence_texts: List[str]) -> str:
        """Format evidence texts for the prompt."""
        if not evidence_texts:
            return ""

        parts = []
        for i, evidence in enumerate(evidence_texts[:5], 1):
            # Truncate long evidence
            text = evidence[:500] if len(evidence) > 500 else evidence
            parts.append(f"[E{i}] {text}")

        return "\n".join(parts)

    def _get_temperature(self, answer_type: str) -> float:
        """Get appropriate temperature based on answer type."""
        temps = {
            "fact": 0.1,
            "howto": 0.2,
            "code": 0.0,
            "compare": 0.1,
            "design": 0.3,
            "chat": 0.4
        }
        return temps.get(answer_type, 0.3)

    def _build_prompt(self, question: str, context: str, plan: Plan, skill_results_context: str = "") -> str:
        """Build the reasoning prompt with evidence and plan."""
        # Include skills and lessons if available
        skills_section = ""
        if plan.skill_ids:
            skills_section = f"\nSkills available: {', '.join(plan.skill_ids)}"

        lessons_section = ""
        if plan.lesson_ids:
            lessons_section = f"\nLessons to avoid: {', '.join(plan.lesson_ids)}"

        # Format plan steps
        steps_section = ""
        if plan.steps:
            steps_section = "\nPlanned steps:\n"
            for step in plan.steps:
                steps_section += f"- Step {step.id}: {step.query} ({step.kind})\n"

        prompt = f"""Question: {question}

Context (evidence):
{context if context else "No external evidence needed."}{skills_section}{lessons_section}{steps_section}{skill_results_context}

Instructions:
1. Answer using the evidence above and your reasoning.
2. For facts about the world, cite your evidence sources using [E1], [E2], etc.
3. For exact calculations, write and run code - never compute mentally.
4. If the evidence does not support a claim, say "I could not verify this."
5. Match the user's level ({self.profile.level.value}) and style (balanced).
6. Include code blocks for any calculations or scripts.
7. If uncertain, say "I could not verify this."

Answer:"""

        return prompt

    async def _verify_draft(self, text: str, evidence_texts: List[str]) -> List[Claim]:
        """Verify claims in the draft answer against evidence."""
        # Extract claims from the draft
        claims = self._extract_claims_from_text(text)

        # Verify against evidence if we have any
        if evidence_texts and claims:
            from agan.pipeline.verifier import Verifier
            verifier = Verifier(sandbox=None, profile=self.profile)
            report = await verifier.verify_claims(text, evidence_texts)
            return report.claims

        # If no evidence or no claims, mark everything as unverified
        result_claims = []
        for claim in claims:
            claim.label = "unverified"
            claim.score = 0.0
            result_claims.append(claim)

        if not result_claims:
            claim = Claim(text=text[:500] if text else "")
            claim.label = "unverified"
            claim.score = 0.0
            result_claims.append(claim)

        return result_claims

    def _extract_claims_from_text(self, text: str) -> List[Claim]:
        """Extract atomic factual claims from text."""

        claims = []

        # Split by sentence endings
        sentences = re.split(r'[.!?]+', text)

        skip_prefixes = ['- ', '* ', '• ', '# ', 'TODO:', 'FIXME:', 'NOTE:',
                         '```', 'import ', 'def ', 'class ', 'print(']

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 3:
                continue

            if any(sentence.startswith(prefix) for prefix in skip_prefixes):
                continue

            # Skip pure numbers
            if re.match(r'^\d+(\.\d+)?$', sentence):
                continue

            claim = Claim(text=sentence)
            claims.append(claim)

        # If no claims extracted, treat whole text as one claim
        if not claims:
            claim = Claim(text=text[:500] if text else "")
            claims.append(claim)

        return claims

    def _determine_confidence(self, claims: List[Claim]) -> str:
        """Determine overall confidence based on claim verification."""
        if not claims:
            return "low"

        verified = sum(1 for c in claims if c.label == "verified")
        partial = sum(1 for c in claims if c.label == "partial")
        total = len(claims)

        if total == 0:
            return "low"

        verified_ratio = verified / total
        partial_ratio = partial / total

        if verified_ratio >= 0.80:
            return "high"
        elif verified_ratio >= 0.50 or partial_ratio > 0:
            return "medium"
        else:
            return "low"

    def _extract_sources(self, evidence_texts: List[str], claims: List[Claim]) -> List[dict]:
        """Extract source information from evidence and claims."""
        sources = []

        # Add evidence sources
        for i, evidence in enumerate(evidence_texts[:5], 1):
            sources.append({
                "id": f"E{i}",
                "type": "evidence",
                "trust": 0.7
            })

        # Add claim evidence IDs
        for claim in claims:
            for eid in claim.evidence_ids:
                if eid not in [s["id"] for s in sources]:
                    sources.append({
                        "id": eid,
                        "type": "claim_evidence",
                        "trust": 0.5
                    })

        return sources

    async def _execute_plan_skills(self, question: str, plan: Plan) -> Dict[str, str]:
        """Execute skill steps from the plan and return results."""
        skill_results = {}

        # Collect all skill names referenced in plan steps
        skill_names = []
        for step in plan.steps:
            if step.kind == "skill" and step.tool:
                skill_names.append(step.tool)

        # Also check skill_ids from the plan
        for skill_id in plan.skill_ids:
            if skill_id and skill_id not in skill_names:
                skill_names.append(skill_id)

        if not skill_names:
            return {}

        # Execute each skill
        executor = self.skill_executor
        results = await executor.execute_all_skills(skill_names, question)

        for name, result in results.items():
            if result and result.success:
                skill_results[name] = result.output
            else:
                skill_results[name] = f"Skill '{name}' did not produce output."

        return results

    def _format_skill_results(self, skill_results: Dict[str, str]) -> str:
        """Format skill execution results for the prompt."""
        if not skill_results:
            return ""

        parts = ["Skill execution results:"]
        for name, output in skill_results.items():
            parts.append(f"  [{name}]: {output[:300]}")
        return "\n".join(parts)