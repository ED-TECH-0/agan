"""B1 demo: 20 questions through the AGAN M3 pipeline (router→planner→reason→verify→write).

Reproduces the B1 baseline (base + retrieval, with citations) through AGAN.
Each question is answered end-to-end; results are printed and saved to
`eval/reports/b1_demo.json`.

Run:  .venv/bin/python scripts/b1_demo.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from agan.llm.client import LLMClient
from agan.pipeline.agan_pipeline import make_pipeline


# 20 questions spanning fact lookup, how-to, comparison, and current-events-style
# queries that benefit from retrieval (B1 = base model + search).
QUESTIONS = [
    "What is the capital of Australia?",
    "Who developed the first successful polio vaccine?",
    "What is the chemical symbol for gold?",
    "Which ocean borders California?",
    "What is the boiling point of water at sea level?",
    "Who wrote the novel 1984?",
    "What is the largest planet in our solar system?",
    "Which element has the chemical symbol O?",
    "What is the speed of light in a vacuum?",
    "Who was the first president of the United States?",
    "What is the currency of the United Kingdom?",
    "Which planet is known as the Red Planet?",
    "What is the tallest mountain on Earth?",
    "Who painted the Mona Lisa?",
    "What is the smallest prime number?",
    "Which year did World War II end?",
    "What is the capital of Japan?",
    "Who invented the telephone?",
    "What is the largest ocean on Earth?",
    "Which country launched the first satellite?",
]


def _score(answer_text: str, expected_keywords: list[str]) -> float:
    """Simple keyword-hit score for the demo (0..1)."""
    lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower)
    return hits / len(expected_keywords) if expected_keywords else 0.0


# Expected keywords per question (for a rough correctness signal).
EXPECTED = [
    ["canberra"],
    ["jonas salk"],
    ["au"],
    ["pacific"],
    ["100", "celsius", "212"],
    ["george orwell", "orwell"],
    ["jupiter"],
    ["oxygen"],
    ["299792458", "300000"],
    ["george washington", "washington"],
    ["pound", "gbp", "british pound"],
    ["mars"],
    ["mount everest", "everest"],
    ["leonardo da vinci", "da vinci"],
    ["2"],
    ["1945"],
    ["tokyo"],
    ["alexander graham bell", "bell"],
    ["pacific"],
    ["soviet union", "ussr", "russia"],
]


async def main() -> None:
    pipeline = make_pipeline(llm_model="qwen2.5-3b:latest")
    orchestrator = pipeline["orchestrator"]

    out_dir = Path("eval/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_start = time.time()

    print(f"B1 Demo — {len(QUESTIONS)} questions through AGAN M3 pipeline\n")
    print(f"{'#':>3}  {'Route':<9} {'Conf':<7} {'Score':<6}  Question")
    print("-" * 80)

    for i, (question, keywords) in enumerate(zip(QUESTIONS, EXPECTED), 1):
        t0 = time.time()
        result = await orchestrator.run(question)
        elapsed = time.time() - t0

        answer = result["answer"]
        route = result["route"]
        confidence = answer.confidence
        score = _score(answer.text, keywords)
        pipeline_time = result["pipeline_time_sec"]

        results.append({
            "n": i,
            "question": question,
            "route": str(route),
            "confidence": confidence,
            "score": round(score, 3),
            "answer": answer.text,
            "claims": len(answer.claims),
            "sources": len(answer.sources),
            "pipeline_time_sec": round(pipeline_time, 3),
            "level_used": answer.level_used,
        })

        print(
            f"{i:>3}  {str(route):<9} {confidence:<7} {score:<6.2f}  {question}"
        )

    total_elapsed = time.time() - total_start
    avg_score = sum(r["score"] for r in results) / len(results)
    high_conf = sum(1 for r in results if r["confidence"] == "high")
    with_sources = sum(1 for r in results if r["sources"] > 0)

    print("-" * 80)
    print(f"Completed {len(QUESTIONS)} questions in {total_elapsed:.1f}s")
    print(f"  Avg keyword score: {avg_score:.3f}")
    print(f"  High confidence: {high_conf}/{len(QUESTIONS)}")
    print(f"  Answers with sources: {with_sources}/{len(QUESTIONS)}")

    report = {
        "demo": "b1_20_question",
        "total_questions": len(QUESTIONS),
        "total_time_sec": round(total_elapsed, 3),
        "avg_score": round(avg_score, 3),
        "high_confidence": high_conf,
        "with_sources": with_sources,
        "results": results,
    }

    report_path = out_dir / "b1_demo.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
