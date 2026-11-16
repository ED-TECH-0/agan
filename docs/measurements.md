# Measurements (Phase 1 baseline)

All numbers measured on the owner's laptop (Windows 11, 12 threads / 8 cores, 15.7 GB RAM, **no GPU detected by `nvidia-smi`**). Every row is a real measurement, not an estimate. Re-measure after the CUDA build lands.

## Hardware fingerprint

| Item | Value |
|---|---|
| CPU | 8 cores / 12 threads (x86-64) |
| RAM | 15.7 GB total, ~6.5 GB free during runs |
| GPU | Not detected (no `nvidia-smi` on PATH) |
| Disk free | ~50 GB |
| OS | Windows 11 (10.0.26200) |
| Recommended profile (by current detection) | `lite` |
| Note | RTX 5050 was expected by the plan; detection sees no CUDA device. Investigate before Phase 5 (training). |

## Inference speed (Ollama, CPU-only fallback)

| Model | Quant | Backend | Warm eval speed | Cold load | Notes |
|---|---|---|---|---|---|
| `qwen2.5:3b-instruct-q4_k_m` | Q4_K_M | Ollama (CPU) | **~39 tok/s** total / **52.2 tok/s** eval-only | ~3.3 s per 128 tokens | Usable interactively |
| `qwen2.5:1.5b-instruct-q4_k_m` | Q4_K_M | Ollama (CPU) | not yet measured | — | Pulled, pending run |
| `nandhank/qwen3.5-abliterated-truthful:9b` | Q4 | Ollama (CPU) | **5.1 tok/s** | ~30 s | Too slow for interactive use |

Raw evidence for the 3B number: `eval_count=100`, `eval_duration=1.917 s` → 52.2 tok/s. For the 9B: `eval_count=100`, `eval_duration=19.511 s` → 5.1 tok/s.

## Model size vs profile fit (measured by file size)

| Model | On-disk size | Fits profile |
|---|---|---|
| 1.5B Q4_K_M | 1.1 GB | Nano, Lite |
| 3B Q4_K_M | 2.1 GB | Lite, Standard |
| 9B Q4 | 6.6 GB | Standard (tight), Pro |

## Eval suite

- `eval/suites/tasks.json` — 34 tasks across 8 categories (code 10, blender 10, fact 5, howto 3, design 2, comparison 2, sysarch 1, hwarch 1).
- `eval/run_baseline.py` — runs every task through Ollama, records latency, tokens generated, tok/s, and an answer preview; writes `eval/reports/baseline_<ts>.json` and prints a per-category capability matrix.

## Known gaps (must close before Phase 5)

1. **No CUDA build.** `llama.cpp` CUDA build is blocked on a missing Windows SDK resource compiler (`rc.exe`). Ollama is the working backend meanwhile. Without GPU, only Lite/Nano profiles are realistic.
2. **`mmap`/KV-cache tuning unmeasured.** Token rates above are default settings.
3. **Reranker + embedding models not yet wired in.** Files fetched (`nomic-embed-text-v1.5.f16.gguf`, `ms-marco-MiniLM-L-6-v2.f16.gguf`) but the pipeline still uses the keyword-overlap fallback.
4. **9B is out of budget on CPU.** Keep it out of the default profile set until a GPU build exists.

## How to reproduce

```bash
# 1. start the backend
ollama serve
ollama pull qwen2.5:3b-instruct-q4_k_m

# 2. run the suite
python eval/run_baseline.py
```

Update this file after each phase with new measured values; never replace a measured number with an estimate.
