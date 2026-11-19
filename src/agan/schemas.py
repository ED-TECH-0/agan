from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Route(str, Enum):
    direct = "direct"
    search = "search"
    deep = "deep"


class PlanStep(BaseModel):
    id: int
    query: str
    kind: Literal["web", "docs", "code", "project", "skill"] = "web"
    depends_on: list[int] = Field(default_factory=list)
    tool: Optional[str] = None


class Plan(BaseModel):
    needs_search: bool
    steps: list[PlanStep] = Field(default_factory=list)
    answer_type: Literal["fact", "howto", "code", "compare", "design", "chat"] = "fact"
    skill_ids: list[str] = Field(default_factory=list)
    lesson_ids: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str
    url: str
    title: str = ""
    text: str
    fetched_at: float
    trust: float = Field(default=0.5, ge=0.0, le=1.0)
    version_tag: Optional[str] = None
    tainted: bool = True


class Evidence(BaseModel):
    eid: str
    chunk: Chunk
    score: float
    extract: str


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict
    origin_tainted: bool = False


class ToolResult(BaseModel):
    call_id: str
    ok: bool
    output: str = ""
    error: str = ""
    duration_ms: int = 0


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    label: Literal["verified", "partial", "unverified"] = "unverified"
    score: float = 0.0


class ExecutionReport(BaseModel):
    ran: bool
    ok: bool = False
    runner: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


class VerifierReport(BaseModel):
    ok: bool
    claims: list[Claim] = Field(default_factory=list)
    missing_query: Optional[str] = None
    execution: Optional[ExecutionReport] = None


class Answer(BaseModel):
    text: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    notes: list[str] = Field(default_factory=list)
    level_used: int = 1


class SearchHit(BaseModel):
    """A single search result from a retrieval provider."""

    url: str
    title: str = ""
    snippet: str = ""
    source: str = ""  # provider name, e.g. "wikipedia", "searxng"
    fetched_at: float = 0.0
    trust: float = Field(default=0.5, ge=0.0, le=1.0)


class Skill(BaseModel):
    id: str
    name: str
    description: str
    instructions: str
    checker: Optional[str] = None
    tools: list[str] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    status: Literal["candidate", "trusted", "stale", "archived"] = "candidate"
    origin: Literal["learned", "user", "imported", "builtin"] = "learned"
    successes: int = 0
    failures: int = 0
    created_at: float = 0.0
    last_used: float = 0.0


class Lesson(BaseModel):
    id: str
    situation: str
    wrong: str
    right: str
    evidence: list[str] = Field(default_factory=list)
    created_at: float = 0.0
    hits: int = 0


class UserProfile(BaseModel):
    stated: dict[str, str] = Field(default_factory=dict)
    level: dict[str, Literal["beginner", "intermediate", "advanced"]] = Field(default_factory=dict)
    style: Literal["concise", "balanced", "detailed", "step_by_step"] = "balanced"
    tools: dict[str, str] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    observed: list[dict] = Field(default_factory=list)