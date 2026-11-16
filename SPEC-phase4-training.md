# Spec: Phase 4 — Training Pipeline — V1

## Objective
Train small retrieval-augmented models (Lite 3B, Standard 7B) via QLoRA on the owner's RTX 5050 laptop, using open-license skill data only, then export GGUF weights and re-run the 34-task benchmark to measure gains over the untrained baseline (~88 tok/s CPU, 100% pass rate on formatted outputs, quality TBD).

Success = trained adapters that pass the 34-task suite with ≥ same pass rate as baseline + measurable quality improvement on maker tasks (Blender bpy scripts, code with runnable checks).

## Tech Stack
- Python 3.11+, PyTorch 2.x, Transformers, PEFT, TRL, bitsandbytes
- Unsloth (preferred for 8GB VRAM QLoRA) or Axolotl fallback
- llama.cpp GGUF export via llama-quantize (CUDA build pending Windows SDK)
- Ollama for inference; GGUF models placed in `models/`

## Commands
- Generate traces: `python scripts/generate_traces.py --skill code --count 5000`
- SFT train: `python scripts/train_sft.py --model qwen2.5-3b --data data/traces/ --output adapters/code/`
- DPO train: `python scripts/train_dpo.py --model qwen2.5-3b-adapter --data data/dpo/ --output adapters/code-dpo/`
- GRPO train: `python scripts/train_grpo.py --model qwen2.5-3b-dpo --tasks eval/suites/tasks.json --output adapters/code-grpo/`
- Export GGUF: `llama-quantize adapters/code-grpo/final.gguf q4_k_m models/`
- Benchmark: `python eval/run_baseline.py`

## Project Structure
```
data/
  traces/           SFT traces (skill data, license-clean)
  dpo/              Preference pairs (grounded vs hallucinated)
  rl/               RL tasks with verifiable rewards
  datasheet.md      Sources, licenses, filters, known gaps
scripts/
  generate_traces.py
  train_sft.py
  train_dpo.py
  train_grpo.py
adapters/
  code/             SFT adapter
  code-dpo/         DPO adapter
  code-grpo/        GRPO adapter
models/
  *.gguf            Quantized weights for Ollama
eval/reports/
  baseline_*.json   Phase 3 results
  phase4_*.json     Post-training results
```

## Code Style
- Python 3.11+, type hints everywhere, pydantic v2 at module boundaries
- Ruff lint + mypy type check
- Functions under 50 lines, descriptive names, docstrings on public API

## Testing Strategy
- Unit tests for data generation (traces pass JSON schema)
- Integration tests for training (single-batch run, loss decreases)
- Eval: 34-task suite post-training, compare pass rate + quality vs baseline
- License audit: every training data source recorded in LICENSES.md

## Boundaries
- **Always:** License-check every data source before training; record checksums in MODELS.md; GGUF output quantized Q4_K_M; eval on same 34-task suite for comparability
- **Ask first:** Adding new base models, changing quantization level, using Kaggle/Colab GPUs
- **Never:** Distill from closed models (Claude/Gemini/GPT); include user secrets in training data; train on copyrighted text dumps

## Success Criteria
- Trained adapters load in Ollama via GGUF
- 34-task suite passes at ≥ baseline rate (100% format, quality improvement on maker tasks)
- Tok/s measured on RTX 5050 (GPU) recorded in docs/measurements.md
- All data licenses documented in LICENSES.md

## Out of Scope (Deferred — V2)
- Blender/Unreal skill pack adapters (need Blender/Unreal installed for verification)
- Multi-language (Tamil) adapters
- Preference adapter learning from user feedback
- Peer-to-peer cache sharing
- Nano (0.6B) training until Lite/Standard validated

## Open Questions
- Which open-license teacher model for synthetic trace generation (need permissive license, fits 8GB VRAM)?
- Confirm RTX 5050 CUDA works for training (driver version, CUDA toolkit)?
- How many traces for SFT (target 10K-50K per plan Section 9.3)?
