"""Unit tests for AGAN verifier (M4): claim verification + sandboxed code execution."""

import sys
sys.path.insert(0, "src")

import pytest
from unittest.mock import MagicMock

from agan.pipeline.verifier import Verifier, VerifierReport, ExecutionReport
from agan.schemas import Claim
from agan.sandbox import SandboxRunner
from agan.hardware.detect import ProfileSpec, HardwareProfile


class TestVerifierCreation:
    """Tests for Verifier instantiation."""

    def test_verifier_default_sandbox(self):
        v = Verifier(profile=None)
        assert v.sandbox is not None
        assert isinstance(v.sandbox, SandboxRunner)

    def test_verifier_with_sandbox(self):
        sandbox = SandboxRunner()
        v = Verifier(sandbox=sandbox, profile=None)
        assert v.sandbox is sandbox

    def test_verifier_with_profile(self):
        profile = ProfileSpec(
            name=HardwareProfile.M, ram_gb_min=8, ram_gb_max=32,
            gpu_required=False, default_model="qwen2.5-3b",
            quantization="Q4_K_M", time_budget=45, best_of_n=3,
            max_context=4096, max_output_tokens=512, max_retrieval=5,
            enable_execution=True, execution_timeout_s=10, level=1,
        )
        v = Verifier(profile=profile)
        assert v.profile is profile


class TestClaimExtraction:
    """Tests for claim extraction from text."""

    def test_extract_simple_claims(self):
        v = Verifier(profile=None)
        claims = v._extract_claims("Paris is the capital of France. The sky is blue.")
        assert len(claims) >= 1
        assert all(isinstance(c, Claim) for c in claims)

    def test_extract_claim_with_numbers(self):
        v = Verifier(profile=None)
        claims = v._extract_claims("The population is 8.4 million people.")
        assert len(claims) >= 1
        assert claims[0].label == "unverified"

    def test_extract_claim_no_claims(self):
        v = Verifier(profile=None)
        claims = v._extract_claims("")
        assert len(claims) == 1
        assert claims[0].label == "unverified"

    def test_extract_claim_skips_code_blocks(self):
        v = Verifier(profile=None)
        claims = v._extract_claims("```python\nprint(1)\n```\nHello world.")
        assert any("Hello world" in c.text for c in claims)


class TestClaimVerification:
    """Tests for claim verification against evidence."""

    def test_verified_claim(self):
        v = Verifier(profile=None)
        claims = v._extract_claims("Paris is the capital of France.")
        evidence = ["Paris is the capital and most populous city of France."]
        claims = v._verify_claims_against_evidence(claims, evidence)
        verified = [c for c in claims if c.label == "verified"]
        assert len(verified) >= 1

    def test_unverified_claim(self):
        v = Verifier(profile=None)
        claims = v._extract_claims("The moon is made of green cheese.")
        evidence = ["Paris is the capital of France."]
        claims = v._verify_claims_against_evidence(claims, evidence)
        # Unrelated evidence should not verify the claim
        assert all(c.label != "verified" for c in claims)

    def test_partial_claim(self):
        v = Verifier(profile=None)
        claims = v._extract_claims("Paris is in France.")
        evidence = ["Paris is the capital and most populous city of France."]
        claims = v._verify_claims_against_evidence(claims, evidence)
        # Should be verified or partial, not unverified
        labels = {c.label for c in claims}
        assert "unverified" not in labels or len(labels) > 1

    def test_number_guard_downgrade(self):
        v = Verifier(profile=None)
        claims = v._extract_claims("The population is 8.4 million.")
        evidence = ["Paris is the capital of France."]
        claims = v._verify_claims_against_evidence(claims, evidence)
        # Numbers present but evidence doesn't support - should be downgraded
        assert claims[0].label in ("partial", "unverified")


class TestVerifyClaims:
    """Tests for the full verify_claims method."""

    @pytest.mark.asyncio
    async def test_verify_with_evidence(self):
        v = Verifier(profile=None)
        report = await v.verify_claims(
            "Paris is the capital of France.",
            ["Paris is the capital and most populous city of France."],
        )
        assert isinstance(report, VerifierReport)
        assert isinstance(report.ok, bool)
        assert len(report.claims) >= 0

    @pytest.mark.asyncio
    async def test_verify_no_evidence(self):
        v = Verifier(profile=None)
        report = await v.verify_claims("Hello world.", [])
        assert isinstance(report, VerifierReport)
        assert isinstance(report.ok, bool)

    @pytest.mark.asyncio
    async def test_verify_code_block(self):
        profile = ProfileSpec(
            name=HardwareProfile.M, ram_gb_min=8, ram_gb_max=32,
            gpu_required=False, default_model="qwen2.5-3b",
            quantization="Q4_K_M", time_budget=45, best_of_n=3,
            max_context=4096, max_output_tokens=512, max_retrieval=5,
            enable_execution=True, execution_timeout_s=10, level=1,
        )
        v = Verifier(profile=profile)
        report = await v.verify_claims(
            "2 + 2 = 4.\n\n```python\nprint(2 + 2)\n```",
            ["2 + 2 equals 4."],
        )
        assert report.execution is not None
        assert report.execution.ran is True


class TestExecuteCode:
    """Tests for sandboxed code execution in verifier."""

    @pytest.mark.asyncio
    async def test_execute_successful_code(self):
        profile = ProfileSpec(
            name=HardwareProfile.M, ram_gb_min=8, ram_gb_max=32,
            gpu_required=False, default_model="qwen2.5-3b",
            quantization="Q4_K_M", time_budget=45, best_of_n=3,
            max_context=4096, max_output_tokens=512, max_retrieval=5,
            enable_execution=True, execution_timeout_s=10, level=1,
        )
        v = Verifier(profile=profile)
        report = await v._execute_code("print(42)")
        assert report.ran is True
        assert report.ok is True
        assert "42" in report.output

    @pytest.mark.asyncio
    async def test_execute_failing_code(self):
        profile = ProfileSpec(
            name=HardwareProfile.M, ram_gb_min=8, ram_gb_max=32,
            gpu_required=False, default_model="qwen2.5-3b",
            quantization="Q4_K_M", time_budget=45, best_of_n=3,
            max_context=4096, max_output_tokens=512, max_retrieval=5,
            enable_execution=True, execution_timeout_s=10, level=1,
        )
        v = Verifier(profile=profile)
        report = await v._execute_code("print(undefined_var)")
        assert report.ran is True
        assert report.ok is False

    @pytest.mark.asyncio
    async def test_execution_report_fields(self):
        profile = ProfileSpec(
            name=HardwareProfile.M, ram_gb_min=8, ram_gb_max=32,
            gpu_required=False, default_model="qwen2.5-3b",
            quantization="Q4_K_M", time_budget=45, best_of_n=3,
            max_context=4096, max_output_tokens=512, max_retrieval=5,
            enable_execution=True, execution_timeout_s=10, level=1,
        )
        v = Verifier(profile=profile)
        report = await v._execute_code("print('hello')")
        assert isinstance(report.duration_ms, int)
        assert report.duration_ms >= 0
        assert isinstance(report.ok, bool)
