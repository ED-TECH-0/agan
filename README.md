# agan — Retrieval Brain

> One common intelligence for every maker, on every machine. Knowledge lives on the internet. Skills live in the model. The model size and the local data packs scale to whatever hardware the user has, so everyone uses the full potential of their own computer.

## Status
Planning document v1.2 (English-only core; other languages later) — see [AGAN_MASTER_PLAN.md](AGAN_MASTER_PLAN.md)

## Quick Start
```bash
# Coming in v0.1: one-click installer
```

## Architecture
```
User
 │
 ▼
[Resource Governor] ← watches RAM/VRAM/CPU, other apps (Blender, Unreal, IDE)
 │
 ▼
[Router] ── simple? ──► [Reasoner direct] ──► Answer
 │
 ▼ needs knowledge or action
[Planner] → sub-questions + tool plan (starts from matching Playbooks)
 │
 ▼
[Privacy Firewall] → redact secrets and personal data from every outgoing query
 │
 ▼
[Retriever] → web search (bandwidth-aware), page fetch, local cache, offline packs, Project Memory
 │
 ▼
[Safety Sanitizer] → strip/flag instructions inside web content
 │
 ▼
[Reranker + Compressor] → top evidence, short context
 │
 ▼
[Reasoner + Domain Skill Pack] → synthesize, write code/scripts/designs
 │
 ▼
[Verifier] → claim check + EXECUTABLE check (run code, compile, simulate)
 │            fail → new search or fix (max loops by tier)
 ▼ pass
[Writer] → final answer with citations, or honest uncertainty
 │
[Tool Executor] (sandboxed): shell, files, python, browser, app bridges
```

## Hardware Profiles
| Profile | Typical Machine | Core Model (4-bit) | Modes |
|---|---|---|---|
| **Nano** | CPU only, 4–8 GB RAM | 0.6B–1.7B | Fast |
| **Lite** | 8–16 GB RAM, iGPU/weak GPU | 1.7B–4B | Fast + Deep |
| **Standard** | 16 GB RAM or 6–8 GB GPU | 4B–8B | Fast + Deep |
| **Pro** | 12–16 GB GPU or 32 GB RAM | 8B–14B | Deep by default |
| **Max** | 24 GB+ GPU or workstation | 14B–32B+ | Deep + multi-sample |

## License
Apache-2.0 for code. Weights and data under licenses compatible with all inputs (see `LICENSES.md`).

## Related Projects
- **DESKHUB** — agan becomes its reasoning core
- **Antigravity** — agentic IDE used for building
