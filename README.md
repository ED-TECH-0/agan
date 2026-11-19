# AGAN — Local AI that researches, verifies, learns skills, and adapts

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache-2.0.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version: v2.0.0-dev](https://img.shields.io/badge/version-v2.0.0--dev-orange.svg)](https://github.com)

AGAN (AGents' Automated Navigator) is a local AI system that researches topics, verifies claims, learns reusable skills, and adapts to your preferences — running entirely on your hardware.

---

## ✨ v2.0 Highlights

This release introduces **Playbook Memory** — the first v2 module that lets the Reasoner shortcut to proven workflows instead of re-inventing steps.

**New in v2.0:**
- 🧠 **Playbook Memory** — SQLite-backed per-project pattern index (`core/playbook.py`)
  - Record and reuse successful skill sequences
  - Keyword-triggered pattern matching with success scoring
  - Foundation for automated workflow suggestion
- 📦 **Updated dependencies** — `pyproject.toml` pins modern stack
- 🐍 **Python 3.11+** support (typing fixes, dataclass improvements)
- 📊 **Modular architecture** — clean separation of concerns

**Roadmap v1.2 → v2:**
1. ✅ Playbook Memory — just shipped
2. Idea-to-Prototype Pipeline
3. Evidence Mode (citation/verification)
4. Maker Benchmark (task suite + scoring)
5. Local Preference Adapter

See [AGAN_MASTER_PLAN.md](AGAN_MASTER_PLAN.md) for the full plan.

---

## Quick Start

```bash
# Install
pip install -e .

# Or from source
git clone https://github.com/vengence/agan.git
cd agan
pip install -e .

# Verify playbook module works
python3 -c "from core.playbook import PlaybookDB; db = PlaybookDB(); print('Playbook Memory ready')"
```

---

## Architecture

```text
User
  ▼
[Resource Governor] ← watches RAM/VRAM/CPU, other apps
  ▼
[Router] ── simple? ──► [Reasoner direct] ──► Answer
  │
  ▼ needs knowledge or action
[Planner] → sub-questions + tool plan (starts from matching Playbooks)
  ▼
[Privacy Firewall] → redact secrets and personal data from every outgoing query
  ▼
[Retriever] → web search (bandwidth-aware), page fetch, local cache, offline packs, Project Memory
  ▼
[Safety Sanitizer] → strip/flag instructions inside web content
  ▼
[Reranker + Compressor] → top evidence, short context
  ▼
[Reasoner + Domain Skill Pack] → synthesize, write code/scripts/designs
  ▼
[Verifier] → claim check + EXECUTABLE check (run code, compile, simulate)
  ▼ pass
[Writer] → final answer with citations, or honest uncertainty
  ▼
[Tool Executor] (sandboxed): shell, files, python, browser, app bridges
```

---

## Hardware Profiles

| Profile | Typical Machine | Core Model (4-bit) | Modes |
|---|---|---|---|
| **Nano** | CPU only, 4–8 GB RAM | 0.6B–1.7B | Fast |
| **Lite** | 8–16 GB RAM, iGPU/weak GPU | 1.7B–4B | Fast + Deep |
| **Standard** | 16 GB RAM or 6–8 GB GPU | 4B–8B | Fast + Deep |
| **Pro** | 12–16 GB GPU or 32 GB RAM | 8B–14B | Deep by default |
| **Max** | 24 GB+ GPU or workstation | 14B–32B+ | Deep + multi-sample |

---

## License

Apache-2.0 for code. Weights and data under licenses compatible with all inputs (see `LICENSES.md`).

---

## Related Projects

- **DESKHUB** — AGAN becomes its reasoning core
- **Antigravity** — agentic IDE used for building
- **Playbook Memory** — v2.0 pattern indexing (this repo)

---

## Development

```bash
# Run tests (when available)
pytest tests/

# Lint
ruff check .

# Type check (when stubs available)
mypy src/agan/
```

---

## v2.0 Roadmap

| # | Module | Status | Estimate |
|---|---|---|---|
| 1 | **Playbook Memory** | ✅ v2.0 shipped | Done |
| 2 | Idea-to-Prototype Pipeline | ⏳ Pending | ~2-3 days |
| 3 | Evidence Mode | ⏳ Pending | ~2-3 days |
| 4 | Maker Benchmark | ⏳ Pending | ~3-4 days |
| 5 | Local Preference Adapter | ⏳ Pending | ~2 days |

---

## Contact

- GitHub: [https://github.com/vengence/agan](https://github.com/vengence/agan)
- Issues: [GitHub Issues](https://github.com/vengence/agan/issues)