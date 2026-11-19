from typing import Optional, List, Dict, Any
import json
from agan.schemas import Plan, PlanStep, Route
from agan.llm.client import LLMClient


class Planner:
    """Planner creates a structured plan (steps + answer type) from a question."""
    
    def __init__(self, client: LLMClient, profile=None):
        self.client = client
        self.profile = profile
    
    def _build_prompt(self, question: str, route: Route) -> str:
        base = f"""Create a plan for: {question}

Return JSON ONLY with this exact schema:
{{
  "steps": [
    {{
      "id": 1,
      "query": "...",
      "kind": "web" | "code" | "project" | "skill",
      "depends_on": []
    }}
  ],
  "answer_type": "fact" | "howto" | "code" | "compare" | "design" | "chat"
}}

Rules:
- Queries must be 3-10 keywords, no personal data, secrets, or long code
- If a skill or lesson is provided, reference it by ID
- answer_type must be one of the listed values
- For search: only "web" kind steps
- For deep: can include code, project, or skill steps
"""
        if route == Route.search:
            return base + "\n- answer_type should be 'fact' or 'howto'"
        else:
            return base + "\n- answer_type can be 'code', 'compare', 'design', or 'chat'"
    
    async def plan(self, question: str, route: Route) -> Plan:
        """Create a structured plan from a question."""
        prompt = self._build_prompt(question, route)
        
        resp = await self.client.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512
        )
        
        try:
            data = json.loads(resp.strip())
        except json.JSONDecodeError:
            # Repair: extract JSON from possible markdown formatting
            start = resp.find("{")
            end = resp.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(resp[start:end])
                except json.JSONDecodeError:
                    data = {}
        
        steps_data = data.get("steps", [])
        steps = []
        for sd in steps_data:
            step_kind = sd.get("kind", "web")
            step = PlanStep(
                id=sd.get("id", 1),
                query=sd.get("query", question),
                kind=step_kind,
                depends_on=sd.get("depends_on", [])
            )
            steps.append(step)
        
        answer_type = data.get("answer_type", "fact")
        if answer_type not in ["fact", "howto", "code", "compare", "design", "chat"]:
            answer_type = "fact"
        
        needs_search = route == Route.search or any(s.kind == "web" for s in steps)
        
        return Plan(
            needs_search=needs_search,
            steps=steps,
            answer_type=answer_type,
            skill_ids=data.get("skill_ids", []),
            lesson_ids=data.get("lesson_ids", [])
        )