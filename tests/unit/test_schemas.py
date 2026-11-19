"""Tests for schemas.py - JSON round-trip validation."""
import json
import pytest
from agan.schemas import (
    Route, PlanStep, Plan, Chunk, Evidence,
    ToolCall, ToolResult, Claim, ExecutionReport,
    VerifierReport, Answer, Skill, Lesson, UserProfile,
)


class TestRoute:
    def test_route_values(self):
        assert Route.direct == "direct"
        assert Route.search == "search"
        assert Route.deep == "deep"

    def test_route_from_string(self):
        assert Route("direct") == Route.direct
        assert Route("search") == Route.search
        assert Route("deep") == Route.deep


class TestPlanStep:
    def test_minimal(self):
        step = PlanStep(id=1, query="test query")
        assert step.id == 1
        assert step.query == "test query"
        assert step.kind == "web"
        assert step.depends_on == []
        assert step.tool is None

    def test_full(self):
        step = PlanStep(
            id=2,
            query="another query",
            kind="code",
            depends_on=[1],
            tool="run_python",
        )
        assert step.kind == "code"
        assert step.depends_on == [1]
        assert step.tool == "run_python"

    def test_json_roundtrip(self):
        step = PlanStep(id=1, query="test", kind="docs", depends_on=[], tool="search")
        data = step.model_dump()
        restored = PlanStep.model_validate(data)
        assert restored == step


class TestPlan:
    def test_minimal(self):
        plan = Plan(needs_search=True)
        assert plan.needs_search is True
        assert plan.steps == []
        assert plan.answer_type == "fact"
        assert plan.skill_ids == []
        assert plan.lesson_ids == []

    def test_with_steps(self):
        plan = Plan(
            needs_search=True,
            steps=[PlanStep(id=1, query="q1"), PlanStep(id=2, query="q2", depends_on=[1])],
            answer_type="code",
            skill_ids=["skill1"],
            lesson_ids=["lesson1"],
        )
        assert len(plan.steps) == 2
        assert plan.answer_type == "code"

    def test_json_roundtrip(self):
        plan = Plan(
            needs_search=False,
            steps=[PlanStep(id=1, query="q")],
            answer_type="howto",
        )
        data = plan.model_dump()
        restored = Plan.model_validate(data)
        assert restored == plan


class TestChunk:
    def test_minimal(self):
        chunk = Chunk(id="c1", url="http://example.com", text="content", fetched_at=1234567890.0)
        assert chunk.id == "c1"
        assert chunk.url == "http://example.com"
        assert chunk.title == ""
        assert chunk.trust == 0.5
        assert chunk.version_tag is None
        assert chunk.tainted is True

    def test_full(self):
        chunk = Chunk(
            id="c2",
            url="http://example.com/page",
            title="Page Title",
            text="Content here",
            fetched_at=1234567890.0,
            trust=0.9,
            version_tag="v1.0",
            tainted=False,
        )
        assert chunk.trust == 0.9
        assert chunk.version_tag == "v1.0"
        assert chunk.tainted is False

    def test_json_roundtrip(self):
        chunk = Chunk(id="c1", url="http://example.com", text="content", fetched_at=1234567890.0)
        data = chunk.model_dump()
        restored = Chunk.model_validate(data)
        assert restored == chunk


class TestEvidence:
    def test_basic(self):
        chunk = Chunk(id="c1", url="http://example.com", text="content", fetched_at=1234567890.0)
        evidence = Evidence(eid="E1", chunk=chunk, score=0.85, extract="relevant extract")
        assert evidence.eid == "E1"
        assert evidence.score == 0.85
        assert evidence.extract == "relevant extract"

    def test_json_roundtrip(self):
        chunk = Chunk(id="c1", url="http://example.com", text="content", fetched_at=1234567890.0)
        evidence = Evidence(eid="E1", chunk=chunk, score=0.85, extract="extract")
        data = evidence.model_dump()
        restored = Evidence.model_validate(data)
        assert restored == evidence


class TestToolCall:
    def test_basic(self):
        call = ToolCall(id="tc1", name="search", arguments={"query": "test"}, origin_tainted=False)
        assert call.name == "search"
        assert call.arguments == {"query": "test"}
        assert call.origin_tainted is False

    def test_json_roundtrip(self):
        call = ToolCall(id="tc1", name="run_python", arguments={"code": "1+1"}, origin_tainted=True)
        data = call.model_dump()
        restored = ToolCall.model_validate(data)
        assert restored == call


class TestToolResult:
    def test_success(self):
        result = ToolResult(call_id="tc1", ok=True, output="2", duration_ms=50)
        assert result.ok is True
        assert result.output == "2"
        assert result.error == ""
        assert result.duration_ms == 50

    def test_failure(self):
        result = ToolResult(call_id="tc1", ok=False, error="timeout", duration_ms=5000)
        assert result.ok is False
        assert result.error == "timeout"

    def test_json_roundtrip(self):
        result = ToolResult(call_id="tc1", ok=True, output="out", duration_ms=100)
        data = result.model_dump()
        restored = ToolResult.model_validate(data)
        assert restored == result


class TestClaim:
    def test_minimal(self):
        claim = Claim(text="The sky is blue")
        assert claim.text == "The sky is blue"
        assert claim.evidence_ids == []
        assert claim.label == "unverified"
        assert claim.score == 0.0

    def test_with_evidence(self):
        claim = Claim(
            text="Python 3.11 released in 2022",
            evidence_ids=["E1", "E2"],
            label="verified",
            score=0.95,
        )
        assert claim.label == "verified"
        assert claim.score == 0.95

    def test_json_roundtrip(self):
        claim = Claim(text="test", evidence_ids=["E1"], label="partial", score=0.6)
        data = claim.model_dump()
        restored = Claim.model_validate(data)
        assert restored == claim


class TestExecutionReport:
    def test_not_ran(self):
        report = ExecutionReport(ran=False)
        assert report.ran is False
        assert report.ok is False
        assert report.runner == ""

    def test_ran_success(self):
        report = ExecutionReport(ran=True, ok=True, runner="python", stdout_tail="hello", stderr_tail="")
        assert report.ran is True
        assert report.ok is True

    def test_ran_failure(self):
        report = ExecutionReport(ran=True, ok=False, runner="python", stdout_tail="", stderr_tail="Error")
        assert report.ok is False

    def test_json_roundtrip(self):
        report = ExecutionReport(ran=True, ok=True, runner="python", stdout_tail="out", stderr_tail="err")
        data = report.model_dump()
        restored = ExecutionReport.model_validate(data)
        assert restored == report


class TestVerifierReport:
    def test_ok(self):
        claim = Claim(text="test", label="verified", score=0.9)
        report = VerifierReport(ok=True, claims=[claim])
        assert report.ok is True
        assert len(report.claims) == 1
        assert report.missing_query is None
        assert report.execution is None

    def test_not_ok_with_missing_query(self):
        claim = Claim(text="unverified claim", label="unverified", score=0.2)
        report = VerifierReport(ok=False, claims=[claim], missing_query="search for more")
        assert report.ok is False
        assert report.missing_query == "search for more"

    def test_with_execution(self):
        exec_report = ExecutionReport(ran=True, ok=True, runner="python")
        claim = Claim(text="2+2=4", label="verified", score=1.0)
        report = VerifierReport(ok=True, claims=[claim], execution=exec_report)
        assert report.execution is not None
        assert report.execution.ok is True

    def test_json_roundtrip(self):
        claim = Claim(text="test", label="verified", score=0.9)
        report = VerifierReport(ok=True, claims=[claim])
        data = report.model_dump()
        restored = VerifierReport.model_validate(data)
        assert restored == report


class TestAnswer:
    def test_minimal(self):
        answer = Answer(text="Hello world")
        assert answer.text == "Hello world"
        assert answer.claims == []
        assert answer.sources == []
        assert answer.confidence == "medium"
        assert answer.notes == []
        assert answer.level_used == 1

    def test_full(self):
        claim = Claim(text="fact", label="verified", score=0.9)
        answer = Answer(
            text="Answer with citation [E1]",
            claims=[claim],
            sources=[{"eid": "E1", "url": "http://example.com", "title": "Source"}],
            confidence="high",
            notes=["Verified against source"],
            level_used=2,
        )
        assert answer.confidence == "high"
        assert answer.level_used == 2

    def test_json_roundtrip(self):
        answer = Answer(text="test", confidence="low", level_used=3)
        data = answer.model_dump()
        restored = Answer.model_validate(data)
        assert restored == answer


class TestSkill:
    def test_minimal(self):
        skill = Skill(id="s1", name="Test Skill", description="Does something", instructions="Steps...")
        assert skill.id == "s1"
        assert skill.name == "Test Skill"
        assert skill.checker is None
        assert skill.tools == []
        assert skill.status == "candidate"
        assert skill.origin == "learned"
        assert skill.successes == 0
        assert skill.failures == 0

    def test_full(self):
        skill = Skill(
            id="s2",
            name="Debug Python",
            description="Fix Python errors from tracebacks",
            instructions="...",
            checker="python_tests",
            tools=["run_python", "search"],
            tool_versions={"python": "3.11"},
            status="trusted",
            origin="builtin",
            successes=5,
            failures=0,
            created_at=1234567890.0,
            last_used=1234567890.0,
        )
        assert skill.status == "trusted"
        assert skill.checker == "python_tests"

    def test_json_roundtrip(self):
        skill = Skill(id="s1", name="Test", description="Desc", instructions="Instr")
        data = skill.model_dump()
        restored = Skill.model_validate(data)
        assert restored == skill


class TestLesson:
    def test_minimal(self):
        lesson = Lesson(id="l1", situation="when user asks X", wrong="old answer", right="new answer")
        assert lesson.situation == "when user asks X"
        assert lesson.wrong == "old answer"
        assert lesson.right == "new answer"
        assert lesson.evidence == []
        assert lesson.hits == 0

    def test_with_evidence(self):
        lesson = Lesson(
            id="l2",
            situation="context",
            wrong="bad",
            right="good",
            evidence=["E1"],
            hits=3,
        )
        assert lesson.evidence == ["E1"]
        assert lesson.hits == 3

    def test_json_roundtrip(self):
        lesson = Lesson(id="l1", situation="s", wrong="w", right="r", evidence=["E1"], hits=1)
        data = lesson.model_dump()
        restored = Lesson.model_validate(data)
        assert restored == lesson


class TestUserProfile:
    def test_defaults(self):
        profile = UserProfile()
        assert profile.stated == {}
        assert profile.level == {}
        assert profile.style == "balanced"
        assert profile.tools == {}
        assert profile.constraints == []
        assert profile.rules == []
        assert profile.observed == []

    def test_full(self):
        profile = UserProfile(
            stated={"topic": "electronics"},
            level={"python": "advanced", "electronics": "beginner"},
            style="step_by_step",
            tools={"python": "3.11", "gcc": "12"},
            constraints=["no paid services"],
            rules=["always give Linux commands"],
            observed=[{"pattern": "user prefers concise", "confidence": 0.8}],
        )
        assert profile.style == "step_by_step"
        assert profile.constraints == ["no paid services"]
        assert len(profile.observed) == 1

    def test_json_roundtrip(self):
        profile = UserProfile(style="concise", constraints=["no cloud"])
        data = profile.model_dump()
        restored = UserProfile.model_validate(data)
        assert restored == profile


class TestInvalidInputRejected:
    """Test that invalid input is properly rejected."""

    def test_plan_invalid_answer_type(self):
        with pytest.raises(Exception):
            Plan(needs_search=True, answer_type="invalid_type")

    def test_chunk_invalid_trust(self):
        with pytest.raises(Exception):
            Chunk(id="c1", url="http://example.com", text="content", fetched_at=0.0, trust=1.5)

    def test_claim_invalid_label(self):
        with pytest.raises(Exception):
            Claim(text="test", label="invalid_label")

    def test_answer_invalid_confidence(self):
        with pytest.raises(Exception):
            Answer(text="test", confidence="invalid")

    def test_skill_invalid_status(self):
        with pytest.raises(Exception):
            Skill(id="s1", name="Test", description="Desc", instructions="Instr", status="invalid")

    def test_user_profile_invalid_style(self):
        with pytest.raises(Exception):
            UserProfile(style="invalid_style")

    def test_user_profile_invalid_level(self):
        with pytest.raises(Exception):
            UserProfile(level={"python": "expert"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])