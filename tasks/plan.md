# Implementation Plan: Phase 4 — Training Pipeline

Source spec: SPEC-phase4-training.md

## Overview
Train small retrieval-augmented models (Lite 3B, Standard 7B) via QLoRA on RTX 5050, export GGUF, re-run 34-task benchmark.

## Architecture Decisions
- QLoRA (4-bit base, 16-rank adapters) — fits 8GB VRAM
- Unsloth preferred, Axolotl fallback
- Qwen2.5 3B/7B base, Apache-2.0
- Open-license synthetic trace generation using downloaded 3B teacher

## Task List

### Phase 1: Data Generation (data-gen)
- [ ] Task 1: Scaffold scripts/ directory structure (generate_traces.py, train_sft.py, train_dpo.py, train_grpo.py)
- [ ] Task 2: Write trace generation script using Qwen2.5:3B teacher (plan → retrieve → verify → trace)
- [ ] Task 3: Generate 5000 code skill traces + 500 blender traces; filter with verifier
- [ ] Task 4: Write DPO pair generator (grounded vs hallucinated/uncited)
- [ ] Task 5: Generate 5000 DPO pairs; filter invalid pairs

### Checkpoint: Data Generation
- [ ] Traces pass JSON schema validation
- [ ] No secrets in traces (canary test pass)
- [ ] License datasheet recorded in data/datasheet.md

### Phase 2: Training (train)
- [ ] Task 6: Install Unsloth/Axolotl + bitsandbytes in venv
- [ ] Task 7: Verify QLoRA loads on RTX 5050 (smoke test: load 3B, forward pass)
- [ ] Task 8: SFT train code adapter on traces; verify loss decreases
- [ ] Task 9: DPO train on pairs; verify groundedness improvement
- [ ] Task 10: GRPO train on maker tasks with runnable checks (fibonacci, binary search)

### Checkpoint: Training
- [ ] Adapter checkpoints saved
- [ ] Loss curve decreases across epochs
- [ ] Single-batch smoke test passes before full training

### Phase 3: Export (export)
- [ ] Task 11: Convert trained adapter to GGUF Q4_K_M via llama.cpp
- [ ] Task 12: Verify GGUF loads in Ollama (`ollama list` shows model)
- [ ] Task 13: Record checksums + licenses in MODELS.md

### Checkpoint: Export
- [ ] `ollama run <model>` produces valid output
- [ ] Checksums match MODELS.md

### Phase 4: Evaluation (eval-v2)
- [ ] Task 14: Re-run 34-task benchmark on trained GGUF model
- [ ] Task 15: Compare pass rate + quality vs baseline (~88 tok/s, 100% pass)
- [ ] Task 16: Record results in docs/measurements.md + eval/reports/

### Checkpoint: Complete
- [ ] All acceptance criteria met
- [ ] Ready for review

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| CUDA not available on RTX 5050 | High | Fall back to CPU training (slower) or Kaggle/Colab |
| QLoRA OOM on 8GB VRAM | Medium | Reduce batch size to 1, gradient accumulation |
| Adapter interference when stacked | Medium | Test each adapter separately before stacking |
| License contamination | High | License audit before every data source |

## Open Questions
- Confirm RTX 5050 CUDA works for training (driver + CUDA toolkit)?
- Teacher model confirmation: Qwen2.5:3B (already downloaded, Apache-2.0)?
