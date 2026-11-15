# Phase 1: Model Candidates (Permissive Licenses Only)
# Updated monthly — verify at build time

## Nano (0.6B–1.7B)

| Model | License | Params | Context | Notes |
|---|---|---|---|---|
| SmolLM2-1.7B | Apache-2.0 | 1.7B | 8K | Strong for size, good reasoning |
| Qwen2.5-1.5B | Apache-2.0 | 1.5B | 32K | Multilingual, tool-calling |
| Gemma-2-2B | Gemma License* | 2B | 8K | *Check current terms |
| Phi-3.5-mini | MIT | 3.8B | 128K | Slightly over budget, very strong |
| TinyLlama-1.1B | Apache-2.0 | 1.1B | 2K | Older, baseline only |

## Lite (1.7B–4B)

| Model | License | Params | Context | Notes |
|---|---|---|---|---|
| Qwen2.5-3B | Apache-2.0 | 3B | 32K | Strong, tool-calling, multilingual |
| SmolLM2-3.6B | Apache-2.0 | 3.6B | 8K | Strong reasoning for size |
| Gemma-2-2B | Gemma License* | 2B | 8K | *Verify terms |
| Phi-3.5-mini | MIT | 3.8B | 128K | Excellent reasoning |
| StableLM-2-1.6B | Apache-2.0 | 1.6B | 4K | Lower end of range |

## Standard (4B–8B) — Primary Target (RTX 5050 8GB)

| Model | License | Params | Context | Quantization (4-bit) | Notes |
|---|---|---|---|---|---|
| Qwen2.5-7B | Apache-2.0 | 7B | 32K | ~4.2 GB | Top choice: strong, tool-calling, Apache-2.0 |
| Llama-3.2-3B | Llama 3.2 License* | 3B | 128K | ~2 GB | *Check terms; not Apache-2.0 |
| Gemma-2-9B | Gemma License* | 9B | 8K | ~5.5 GB | *Slightly over, verify terms |
| Phi-3.5-medium | MIT | 14B | 128K | ~8 GB | *Over budget for 4-bit |
| Nemotron-3-8B | NVIDIA License* | 8B | 4K | ~4.8 GB | *Check terms |
| StableLM-2-7B | Apache-2.0 | 7B | 4K | ~4.2 GB | Good alternative |
| Mistral-7B-v0.3 | Apache-2.0 | 7B | 32K | ~4.2 GB | No native tool-calling |

## Pro (8B–14B)

| Model | License | Params | Context | Notes |
|---|---|---|---|---|
| Qwen2.5-14B | Apache-2.0 | 14B | 32K | Strong, tool-calling |
| Llama-3.1-8B | Llama 3.1 License* | 8B | 128K | *Check terms |
| Nemotron-3-14B | NVIDIA License* | 14B | 4K | *Check terms |
| Phi-4 | MIT | 14B | 16K | New, strong reasoning |

## Max (14B–32B+)

| Model | License | Params | Context | Notes |
|---|---|---|---|---|
| Qwen2.5-32B | Apache-2.0 | 32B | 32K | Top Apache-2.0 choice |
| Llama-3.3-70B | Llama 3.3 License* | 70B | 128K | *Check terms, needs quantization |
| Nemotron-3-Ultra | NVIDIA License* | 53B | 4K | *Check terms |

## Embedding Models

| Model | License | Dim | Size | Notes |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | Apache-2.0 | 384 | 80 MB | Fast, good baseline |
| bge-small-en-v1.5 | MIT | 384 | 130 MB | Stronger |
| nomic-embed-text-v1.5 | Apache-2.0 | 768 | 270 MB | Long context |
| jina-embeddings-v2-small-en | MIT | 512 | 130 MB | 8K context |

## Reranker Models

| Model | License | Size | Notes |
|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | Apache-2.0 | 80 MB | Cross-encoder, fast |
| bge-reranker-base | MIT | 270 MB | Stronger |
| jina-reranker-v2-base-multilingual | MIT | 270 MB | Multilingual |

## Selection Criteria (in order)
1. **License**: Must be Apache-2.0, MIT, BSD, or similarly permissive
2. **Native tool-calling**: Prefer models with function-calling training
3. **Reasoning benchmarks**: MMLU, GSM8K, HumanEval, BBH
4. **Size efficiency**: Score per GB (quantized)
5. **Context length**: Longer is better for RAG
6. **Community adoption**: Active maintenance, GGUF availability

## Immediate Next Steps
1. [ ] Download Qwen2.5-7B-Instruct-GGUF (Q4_K_M) for Standard profile testing
2. [ ] Download Qwen2.5-3B-Instruct-GGUF for Lite profile testing
3. [ ] Download SmolLM2-1.7B-Instruct-GGUF for Nano profile testing
4. [ ] Run baseline benchmarks on each (lm-eval-harness)
5. [ ] Measure tokens/sec and VRAM/RAM on RTX 5050
6. [ ] Record results in MODELS.md with checksums