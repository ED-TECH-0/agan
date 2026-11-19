"""Tests for eval harness (src/agan/eval/harness.py)."""
import pytest
from pathlib import Path
from agan.eval.harness import EvalHarness, BaselineResult


class TestEvalHarness:
    def test_record_rate_pass(self):
        """Rate metrics: higher is better (pass when value >= target)."""
        harness = EvalHarness(Path("eval/reports"))
        harness.record("test_precision", 0.8, "rate", 0.7)
        assert len(harness.results) == 1
        assert harness.results[0].passed is True
        assert harness.results[0].value == 0.8
    
    def test_record_rate_fail(self):
        """Rate metrics: higher is better (fail when value < target)."""
        harness = EvalHarness(Path("eval/reports"))
        harness.record("test_precision", 0.5, "rate", 0.7)
        assert len(harness.results) == 1
        assert harness.results[0].passed is False
    
    def test_record_latency_pass(self):
        """Latency metrics: lower is better (pass when value <= target)."""
        harness = EvalHarness(Path("eval/reports"))
        harness.record("test_latency", 100, "ms", 200)
        assert len(harness.results) == 1
        assert harness.results[0].passed is True
    
    def test_record_latency_fail(self):
        """Latency metrics: lower is better (fail when value > target)."""
        harness = EvalHarness(Path("eval/reports"))
        harness.record("test_latency", 300, "ms", 200)
        assert len(harness.results) == 1
        assert harness.results[0].passed is False
    
    def test_extract_claims_with_citations(self):
        harness = EvalHarness(Path("eval/reports"))
        text = "- First claim [source]\n- Second claim\n- Third claim [ref]"
        claims = harness._extract_claims(text)
        assert len(claims) == 3
        assert claims[0]["citation"] is True
        assert claims[1]["citation"] is False
        assert claims[2]["citation"] is True
    
    def test_extract_claims_with_code(self):
        harness = EvalHarness(Path("eval/reports"))
        text = "- Code claim with ```python block```"
        claims = harness._extract_claims(text)
        assert len(claims) == 1
        assert claims[0]["executed"] is True
    
    def test_extract_claims_empty(self):
        harness = EvalHarness(Path("eval/reports"))
        text = "Just a plain answer without bullet points."
        claims = harness._extract_claims(text)
        assert len(claims) == 1
        assert claims[0]["text"] == text[:200]
    
    def test_save_results(self, tmp_path):
        harness = EvalHarness(tmp_path)
        harness.record("metric1", 0.8, "rate", 0.7)
        harness.record("metric2", 150, "ms", 200)
        harness.save()
        
        import json
        saved_files = list(tmp_path.glob("eval_*.json"))
        assert len(saved_files) == 1
        data = json.loads(saved_files[0].read_text())
        assert len(data) == 2
        assert data[0]["name"] == "metric1"
        assert data[1]["name"] == "metric2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])