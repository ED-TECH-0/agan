from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agan.eval.harness import EvalHarness


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


@pytest.mark.asyncio
async def test_harness_b3_skills():
    harness = EvalHarness(Path("eval/reports/test"))

    def make_answer(marker: str):
        async def answer_fn(prompt, tools=False, retrieval=False):
            return f"Used {marker} to handle: {prompt}"
        return answer_fn

    # Rotate markers so some hit, some miss → hit rate 0.60 (3/5)
    markers_cycle = ["math_reasoning", "web_search", "code_exec", "summarize", "python_debug"]
    async def answer_fn(prompt, tools=False, retrieval=False):
        marker = markers_cycle[len(harness.results) % len(markers_cycle)]
        return f"Used {marker} to handle: {prompt}"

    await harness.run_b3_skills(answer_fn)
    names = {r.name for r in harness.results}
    assert "B3_skill_hit_rate" in names
    b3 = next(r for r in harness.results if r.name == "B3_skill_hit_rate")
    assert 0.0 <= b3.value <= 1.0


@pytest.mark.asyncio
async def test_harness_b4_adaptation():
    harness = EvalHarness(Path("eval/reports/test"))

    async def answer_fn(prompt, tools=False, retrieval=False):
        # Adheres to bullets, short, code, numbered preferences
        if "steps" in prompt.lower():
            return "1. Step one\n2. Step two"
        if "code" in prompt.lower():
            return "```python\nprint('hi')\n```"
        if "quantum" in prompt.lower():
            return "It uses qubits."
        return "- Bullet one\n- Bullet two"

    await harness.run_b4_adaptation(answer_fn)
    names = {r.name for r in harness.results}
    assert "B4_preference_adherence" in names
    b4 = next(r for r in harness.results if r.name == "B4_preference_adherence")
    assert 0.0 <= b4.value <= 1.0


@pytest.mark.asyncio
async def test_harness_b5_full():
    harness = EvalHarness(Path("eval/reports/test"))

    async def answer_fn(prompt, tools=False, retrieval=False):
        p = prompt.lower()
        if "17 * 23" in p:
            return "math_reasoning: 17 * 23 = 391"
        if "python news" in p:
            return "web_search: [source] recent Python news"
        if "quicksort" in p:
            return "code_exec: ```python\ndef qsort(arr): ...\n```"
        if "summarize" in p or "article" in p:
            return "summarize: this article summarizes key points."
        if "nameerror" in p:
            return "python_debug: fix the NameError by defining the variable"
        if "steps" in p:
            return "1. Step one\n2. Step two"
        if "code example" in p:
            return "```python\nprint('hi')\n```"
        if "quantum" in p:
            return "It uses qubits."
        return "- Bullet one\n- Bullet two"

    await harness.run_b5_full(answer_fn)
    names = {r.name for r in harness.results}
    assert "B5_composite_rate" in names
    b5 = next(r for r in harness.results if r.name == "B5_composite_rate")
    assert b5.value >= 0.0
    # Composite must beat B4 by at least 0.05
    b4 = next(r for r in harness.results if r.name == "B4_preference_adherence")
    assert b5.value >= b4.value + 0.05 - 1e-6