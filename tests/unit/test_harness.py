import pytest
from agan.eval.harness import EvalHarness
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.mark.asyncio
async def test_harness_b0_runs():
    harness = EvalHarness(Path("eval/reports/test"))
    async def answer_fn(prompt, tools=False, retrieval=False):
        return "I don't know."
    
    with patch("agan.eval.harness.load_dataset") as mock_load:
        mock_split = MagicMock()
        mock_split.select.return_value = [
            {"question": "Q1", "correct_answers": ["A1"], "incorrect_answers": ["B1"], "best_answer": "A1"},
            {"question": "Q2", "correct_answers": ["A2"], "incorrect_answers": ["B2"], "best_answer": "A2"},
        ]
        mock_load.return_value = {"validation": mock_split}
        
        await harness.run_b0_truthful_qa(answer_fn)
    
    assert len(harness.results) == 1
    assert harness.results[0].name == "B0_hallucination_rate"
    assert 0.0 <= harness.results[0].value <= 1.0


@pytest.mark.asyncio
async def test_harness_b1_runs():
    harness = EvalHarness(Path("eval/reports/test"))
    async def answer_fn(prompt, tools=False, retrieval=False):
        return "- Claim [source]\n- Claim without citation"
    
    with patch("agan.eval.harness.load_dataset") as mock_load:
        mock_split = MagicMock()
        mock_split.select.return_value = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        mock_load.return_value = {"validation": mock_split}
        
        await harness.run_b1_search_only(answer_fn)
    
    assert len(harness.results) == 2
    names = {r.name for r in harness.results}
    assert "B1_citation_precision" in names
    assert "B1_unsupported_claim_rate" in names


@pytest.mark.asyncio
async def test_harness_b2_runs():
    harness = EvalHarness(Path("eval/reports/test"))
    async def answer_fn(prompt, tools=False, retrieval=False):
        return "- Claim [source]\n- Calculated: 42 ```python\nprint(42)\n```"
    
    with patch("agan.eval.harness.load_dataset") as mock_load:
        mock_split = MagicMock()
        mock_split.select.return_value = [
            {"question": "Q1", "answer": "42"},
            {"question": "Q2", "answer": "24"},
        ]
        mock_load.return_value = {"test": mock_split}
        
        await harness.run_b2_verification(answer_fn)
    
    names = {r.name for r in harness.results}
    assert "B2_unsupported_claim_rate" in names
    assert "B2_execution_success" in names