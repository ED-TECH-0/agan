#!/usr/bin/env python3
"""
Baseline benchmark runner using Ollama.
Tests the 34 eval tasks against available models and measures performance.
"""
import asyncio
import json
import time
import httpx
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

OLLAMA_URL = "http://127.0.0.1:11434"

@dataclass
class BenchmarkResult:
    task_id: str
    category: str
    model: str
    success: bool
    latency_ms: int
    tokens_generated: int
    tok_per_sec: float
    answer_preview: str
    error: str = ""

async def run_task(client: httpx.AsyncClient, model: str, task, max_tokens: int = 256) -> BenchmarkResult:
    """Run a single task against a model."""
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task["question"]}],
        "options": {"temperature": 0.2, "num_predict": max_tokens},
        "stream": False,
    }
    
    start = time.perf_counter()
    try:
        response = await client.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        elapsed = time.perf_counter() - start
        
        content = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        
        return BenchmarkResult(
            task_id=task["id"],
            category=task["category"],
            model=model,
            success=True,
            latency_ms=int(elapsed * 1000),
            tokens_generated=eval_count,
            tok_per_sec=eval_count / elapsed if elapsed > 0 else 0,
            answer_preview=content[:200],
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return BenchmarkResult(
            task_id=task["id"],
            category=task["category"],
            model=model,
            success=False,
            latency_ms=int(elapsed * 1000),
            tokens_generated=0,
            tok_per_sec=0,
            answer_preview="",
            error=str(e),
        )

async def benchmark_model(model: str, tasks: list, max_tokens: int = 256) -> list[BenchmarkResult]:
    """Run all tasks against a model."""
    async with httpx.AsyncClient(base_url=OLLAMA_URL, timeout=180.0) as client:
        # Warmup
        await client.post(f"{OLLAMA_URL}/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "options": {"num_predict": 10},
            "stream": False,
        })
        
        results = []
        for task in tasks:
            print(f"  [{model}] {task['id']} ({task['category']})...", end=" ")
            result = await run_task(client, model, task, max_tokens)
            results.append(result)
            if result.success:
                print(f"OK ({result.tok_per_sec:.1f} tok/s, {result.latency_ms}ms)")
            else:
                print(f"FAIL: {result.error[:80]}")
        return results

async def main():
    # Load tasks
    tasks_file = Path("eval/suites/tasks.json")
    with open(tasks_file) as f:
        tasks = json.load(f)
    
    # Models to test
    models = ["qwen2.5:3b-instruct-q4_k_m"]
    
    all_results = {}
    
    for model in models:
        print(f"\n=== Benchmarking {model} ===")
        results = await benchmark_model(model, tasks)
        all_results[model] = [asdict(r) for r in results]
        
        # Summary
        successful = [r for r in results if r.success]
        if successful:
            avg_tps = sum(r.tok_per_sec for r in successful) / len(successful)
            avg_latency = sum(r.latency_ms for r in successful) / len(successful)
            print(f"  Summary: {len(successful)}/{len(results)} passed, "
                  f"avg {avg_tps:.1f} tok/s, avg {avg_latency:.0f}ms latency")
    
    # Save results
    output = {
        "timestamp": time.time(),
        "models": models,
        "results": all_results,
    }
    
    out_dir = Path("eval/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"baseline_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {out_file}")
    
    # Print capability matrix
    print("\n=== CAPABILITY MATRIX ===")
    for model in models:
        results = all_results[model]
        by_cat = {}
        for r in results:
            if r["category"] not in by_cat:
                by_cat[r["category"]] = {"pass": 0, "total": 0}
            by_cat[r["category"]]["total"] += 1
            if r["success"]:
                by_cat[r["category"]]["pass"] += 1
        
        print(f"\n{model}:")
        for cat, stats in sorted(by_cat.items()):
            pct = stats["pass"] / stats["total"] * 100 if stats["total"] else 0
            print(f"  {cat}: {stats['pass']}/{stats['total']} ({pct:.0f}%)")

if __name__ == "__main__":
    asyncio.run(main())