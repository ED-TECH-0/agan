# Task List

## Phase 1: Data Generation
- [ ] Task 1: Scaffold scripts/ directory structure
- [ ] Task 2: Write trace generation script using Qwen2.5:3B teacher
- [ ] Task 3: Generate 5000 code skill traces + 500 blender traces; filter with verifier
- [ ] Task 4: Write DPO pair generator
- [ ] Task 5: Generate 5000 DPO pairs; filter invalid pairs

### Checkpoint: Data Generation
- [ ] Traces pass JSON schema validation
- [ ] No secrets in traces (canary test pass)
- [ ] License datasheet recorded

## Phase 2: Training
- [ ] Task 6: Install Unsloth/Axolotl + bitsandbytes
- [ ] Task 7: Verify QLoRA loads on RTX 5050 (smoke test)
- [ ] Task 8: SFT train code adapter; verify loss decreases
- [ ] Task 9: DPO train; verify groundedness improvement
- [ ] Task 10: GRPO train on maker tasks

### Checkpoint: Training
- [ ] Adapter checkpoints saved
- [ ] Loss curve decreases

## Phase 3: Export
- [ ] Task 11: Convert adapter to GGUF Q4_K_M
- [ ] Task 12: Verify GGUF loads in Ollama
- [ ] Task 13: Record checksums + licenses in MODELS.md

### Checkpoint: Export
- [ ] GGUF loads in Ollama
- [ ] Checksums match MODELS.md

## Phase 4: Evaluation
- [ ] Task 14: Re-run 34-task benchmark on trained GGUF
- [ ] Task 15: Compare vs baseline
- [ ] Task 16: Record results in docs/measurements.md + eval/reports/

### Checkpoint: Complete
- [ ] All acceptance criteria met
