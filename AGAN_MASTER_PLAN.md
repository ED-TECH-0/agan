# RETRIEVAL BRAIN — Master Plan (v1.2)

> One common intelligence for every maker, on every machine. Knowledge lives on the internet. Skills live in the model. The model size and the local data packs scale to whatever hardware the user has, so everyone uses the full potential of their own computer.

**Status:** Planning document v1.2 (English-only core; other languages later)
**Owner:** E
**Build tools (not part of the product):** Hermes (owner's personal agent), Antigravity (agentic IDE), RTX 5050 8 GB laptop, free Kaggle/Colab GPUs
**Related project:** DESKHUB (this brain becomes its reasoning core)

**Changes from v1.1:**
- New must-have modules for v0.1: **Privacy Firewall**, **Low-Bandwidth Mode**, **`brain doctor`**, **Local Project Memory** (Section 5A).
- New v0.2 modules: **Playbook Memory**, **Idea-to-Prototype Pipeline**, **Evidence Mode**, public **Maker Benchmark**, opt-in **Local Preference Adapter** for stronger machines (Section 5A).
- New later modules (v0.3+): Vision, Voice, signed Skill Pack Registry, Tron robot integration (Section 5B).
- Explicit non-goals added (Section 1.5), sustainability plan added (Section 16A).
- Roadmap, repository layout, evaluation, security controls and risk register updated to match.

**Changes from v1.0 (kept for history):**
- Target users clarified: makers (coding, 3D design, Blender, Unreal, system architecture, hardware architecture) on low, mid and high-end machines.
- Hardware tiers redesigned as one **scalable family** with a shared "Intelligence Contract".
- New: Resource Governor (the AI must not starve Blender or Unreal of RAM/VRAM).
- New: Domain Skill Packs and executable verification for maker tasks.
- Correction: Hermes is the owner's personal build agent, not a model in the product.

---

## தமிழ்ச் சுருக்கம் (Tamil summary)

இந்த AI, கோடிங், 3D வடிவமைப்பு, Blender, Unreal, system மற்றும் hardware architecture செய்பவர்களுக்கானது. அவர்களில் பலரிடம் பணம் இல்லை, குறைந்த திறன் கணினிகளே உள்ளன. சிலரிடம் நடுத்தர, சிலரிடம் உயர் திறன் கணினிகள் உள்ளன. **எல்லோருக்கும் ஒரே நுண்ணறிவு அமைப்பு (ஒரே திறன்கள், ஒரே நடத்தை, ஒரே tools); ஆனால் மாடலின் அளவும் உள்ளூர் தரவுப் பொதிகளும் (data packs) அவரவர் hardware-க்கு ஏற்ப மாறும்.** அறிவு இணையத்தில் இருக்கும்; மாடல் சிந்தனை, திட்டமிடல், tool பயன்பாடு, சரிபார்ப்பு ஆகியவற்றை மட்டும் கற்கும். Blender அல்லது Unreal ஓடும்போது AI தன் வளங்களைக் குறைத்துக்கொள்ளும். Hermes உங்கள் தனிப்பட்ட agent; அது தயாரிப்பின் மாடல் அல்ல.

**v1.2 புதியவை:** தேடலுக்கு முன் ரகசியங்களை நீக்கும் **Privacy Firewall**, மெதுவான இணையத்துக்கான **Low-Bandwidth Mode**, கணினியைச் சோதிக்கும் **`brain doctor`**, பயனரின் சொந்தத் திட்டங்களுக்கான **Local Project Memory**, வெற்றிகரமான வழிமுறைகளைச் சேமிக்கும் **Playbook Memory**, யோசனையிலிருந்து முன்மாதிரி வரை வழிகாட்டும் **Idea-to-Prototype Pipeline**, ஆதாரம் காட்டும் **Evidence Mode**, பொது **Maker Benchmark**.

---

## 1. Vision and Principles

### 1.1 Mission
Give every maker, on any computer, a capable AI assistant that can research, reason, plan, write and run code, operate their tools safely, and ground its answers in live internet sources, at zero cost.

### 1.2 Core idea
| Today's big LLM | Retrieval Brain |
|---|---|
| Knowledge stored in weights (parametric) | Knowledge fetched from the internet (non-parametric) |
| 70B to 1T parameters | 0.6B to 32B parameters (scaled to the user's hardware) |
| Stale facts, confident hallucination | Fresh facts with citations; says "unknown" when unsupported |
| Needs expensive GPUs or paid APIs | Runs locally, offline-capable, free |

### 1.3 Target users
- Programmers and students on old laptops.
- 3D artists and animators (Blender), game developers (Unreal), technical artists.
- System architects (software architecture, infrastructure design).
- Hardware architects and electronics makers (RTL, FPGA, PCB, embedded).
- Builders with no money for subscriptions, and also people with strong workstations who want local, private AI.

### 1.4 Design principles
1. **Skills over facts.** Train behaviour (plan, search, read, reason, verify, act), not trivia.
2. **One intelligence, many sizes.** Same architecture, same skills, same interface; only capacity and local data differ.
3. **Use the user's hardware fully, without hurting their work.** Resource Governor (Section 6).
4. **Local-first and private.** No telemetry. No uploading user queries.
5. **Safe by default.** Web content is data, never commands. Dangerous actions need approval.
6. **Honest claims.** Publish real benchmark tables. Never claim parity without measurement.
7. **Zero-cost buildable.** The whole project must be reproducible with free tools.
8. **Open and license-clean.** Only Apache-2.0/MIT-style components in the core.
9. **Privacy by construction.** Secrets and personal data are stripped before anything leaves the machine, by code that runs before the model.
10. **Remember skills, not facts.** The system keeps workflows and the user's own project context locally; world knowledge is always fetched fresh.

### 1.5 What this project is NOT
- Not a model with zero knowledge. Language understanding and common sense come from the base model and are required for reasoning.
- Not a copy or distillation of closed models (Claude, Gemini, GPT). Their weights are unavailable and their terms generally forbid training competing models on their outputs.
- Not a replacement for frontier models on the hardest pure-reasoning tasks. It targets parity on **retrieval-grounded, tool-using, practical maker tasks**, and it must prove this with measurements.

**Deliberately excluded for now (non-goals):**
- **Peer-to-peer cache sharing:** privacy and cache-poisoning risk is too high at this stage.
- **Full GUI control in v0.1:** needs a dedicated safety review first (planned for Phase 11).
- **Pretraining our own base model from scratch:** enormous cost and unnecessary for the mission.
- **Many languages at once:** wait until the English core is stable.

---

## 2. Honest Reality Check

| Claim | Reality | Consequence for the plan |
|---|---|---|
| "Use Claude/Gemini/Astra weights" | Closed; not obtainable | Use open-weight models only |
| "Distill from any AI platform" | Terms of many platforms prohibit it | Distill from open-license teachers; closed models only for evaluation where their terms allow |
| "Remove knowledge, keep skills from a big model's weights" | Knowledge and skill are entangled; reliable unlearning is unsolved | Never remove. Start from a small base and avoid teaching facts |
| "Equal to Claude" | Not for hard pure reasoning at small sizes | Aim for parity on grounded, tool-using tasks; measure it |
| "Identical intelligence on every machine" | A 0.6B model cannot reason like a 14B model | Identical **behaviour, skills, interface and safety**; capability scales with size, and small tiers compensate with more retrieval, more tools, structured steps and graceful "this is beyond me" answers |
| "Runs on all hardware" | There is a floor | Floor defined in Section 6 (Nano: 4 GB RAM, CPU only) |

---

## 3. The Intelligence Contract (what is common for everyone)

Every tier, from Nano to Max, must obey the same contract. This is what "common intelligence" means in practice.

1. **Same pipeline:** Router → Planner → Retriever → Sanitizer → Reranker → Reasoner → Verifier → Writer.
2. **Same schemas:** the same plan JSON, tool-call format (MCP-compatible), citation format and error format.
3. **Same skill list:** planning, search, reading, reasoning, verification, tool use, safe system operation, citation discipline, abstention.
4. **Same safety rules and permission tiers.**
5. **Same adapter format:** a skill adapter trained for one size can be re-trained for another with the same data and recipe.
6. **Same evaluation suite:** every tier is scored on the same benchmarks, and a public **capability matrix** states what each tier can and cannot do reliably.
7. **Graceful degradation:** when a task exceeds the tier, the AI says so, breaks the task into smaller steps, or suggests the user switch to a stronger tier or an optional bigger model.

**What differs by tier (allowed):**
- Core model size and quantization level.
- Number of adapters loaded at once.
- Retrieval stack size (embedding model, reranker, index size).
- Context length and number of verification loops.
- Optional local data packs (offline docs, cached references).

---

## 4. System Architecture

### 4.1 Layers
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

### 4.2 Neural components
| Component | Type | Size target | Purpose |
|---|---|---|---|
| Core Brain | Small/medium dense LLM (4-bit) | 0.6B to 32B by tier | Reasoning, writing, planning |
| LoRA adapters | Small adapters on the core | 50 to 200 MB each | Planner, Tool-caller, Verifier, System-control, Reasoner, domain packs |
| Embedding model | Small encoder | 30 to 150 MB | Retrieval |
| Reranker | Small cross-encoder | 50 to 300 MB | Evidence ranking (optional on Nano) |
| Injection detector | Tiny classifier | under 50 MB | Block prompt injection from web content |
| Vision model (later) | Small VLM | optional | Screenshots, GUI control, reading diagrams and viewport images |

"Many neural networks" = one core + swappable adapters + tiny specialists, not a stack of big models.

### 4.3 Test-time compute
Intelligence beyond model size comes from thinking longer only when needed: multi-sample voting, self-verification, best-of-N, iterative search and **executable verification**. Controlled by Fast / Deep modes and the router. Strong machines can afford more of it, which is one more way each user "uses their hardware potential".

---

## 5. Module Specifications

### 5.1 Router
- Input: user message, conversation state, current hardware profile.
- Output: `direct | search | deep | action`.
- Chit-chat, arithmetic, rewriting and simple code go `direct`. Facts, versions, current events, API details and "latest" go `search`. Multi-part research or design tasks go `deep`. System or application operations go `action` (with the approval policy).
- Implementation: a tiny classifier or the core model with a short prompt; calibrated on labelled examples.

### 5.2 Planner
```json
{
  "needs_search": true,
  "steps": [
    {"id": 1, "query": "...", "source_type": "web|wiki|docs|local", "depends_on": []}
  ],
  "expected_answer_type": "fact|comparison|howto|code|design|report",
  "tools": ["search", "fetch", "python", "blender", "shell"]
}
```
- Comparisons produce one search per item. Independent steps run in parallel.
- Step cap by tier (Nano 2, Lite 3, Standard 4, Pro 6, Max 8+).

### 5.3 Retriever
- Pipeline: search → fetch → clean → chunk (300 to 500 tokens) → embed → hybrid rank (BM25 + vectors).
- Sources: self-hosted SearXNG (no API key), Wikipedia API, official documentation sites, optional free APIs.
- Respect robots.txt and rate limits.
- Domain trust scoring (official docs, standards bodies, education, established outlets rank higher).
- **Cache:** SQLite store of fetched pages and embeddings with timestamps and freshness policy (news in hours, API docs in days or weeks).
- **Docs-aware retrieval:** when a question mentions an API or tool version (for example a Blender or Unreal version), retrieve the matching version's documentation first. Version mismatch is the most common cause of wrong maker code.

### 5.4 Safety Sanitizer
- Wrap all retrieved text as `<data source="url">...</data>`. The system prompt states data is never instruction.
- Run the injection detector on every chunk. Flagged chunks are dropped or quoted as untrusted.
- No tool call may be triggered **solely** by content from a retrieved document.
- Strip hidden text, zero-width characters and suspicious markup.

### 5.5 Reranker and Compressor
- Cross-encode (query, chunk), keep top-k (k from 3 to 8 by tier).
- Extract only the sentences relevant to the query to protect small context windows.

### 5.6 Reasoner
- Core model plus the relevant adapters. Receives the question and compressed evidence with source ids.
- Factual claims only from evidence; logic, math, code and structure from reasoning.
- Emits citations tied to source ids.

### 5.7 Verifier (two kinds of checks)
1. **Claim verification:** split into atomic claims, check each against evidence.
2. **Executable verification (maker superpower):** whenever the answer is runnable, run it.
   - Python and general code: run tests or a sandboxed script.
   - Blender: run the generated `bpy` script headless in a sandbox and check that it completes and produces the expected objects.
   - Unreal: syntax and static checks on C++/Python; compile or editor-script checks where the user's environment allows.
   - Hardware description: lint and simulate with open tools (for example Verilator or Icarus Verilog) and run the generated testbench.
   - Diagrams: render Mermaid/PlantUML and confirm it parses.
- Output: `{ok, unsupported_claims, execution_errors, missing_query}`. Errors are fed back for repair (loop cap by tier).

### 5.8 Writer
- Concise, direct, with a sources list and "I could not verify X" where needed. Never invents sources.

### 5.9 Tool Executor (system and internet skills)
| Tool | Capability | Safety |
|---|---|---|
| search / fetch | Web research | Rate limits, sanitizer |
| browser (Playwright) | Click, type, scroll, read dynamic pages | Domain allowlist, no credential entry without approval |
| shell | Run commands | Sandboxed, allowlist, approval for destructive commands |
| files | Read/write in a chosen workspace | Scoped, snapshot before change |
| python sandbox | Calculations, analysis, test running | No network by default, resource limits |
| app bridges | Blender, Unreal editor scripting, KiCad scripting, VS Code | Optional plug-ins, approval policy, version-aware |
| GUI control (later) | Screenshot → click/type | Approval, visible action log |
- Tools use **MCP-compatible schemas** so community plug-ins work.
- Every action is logged with an undo path where possible.

---

## 5A. Additional Modules (v1.2)

### 5A.1 Privacy Firewall (v0.1, must-have)
**Problem:** a retrieval AI sends text to the internet. Users paste code, logs and notes that contain API keys, passwords, tokens, emails, internal hostnames and personal paths. A model-written search query can leak them.

**Design:**
- Runs as ordinary code **before** any outbound request (search queries, fetched URLs including query strings, browser tool navigation, any optional API escalation). It never depends on the model's good behaviour.
- Layer 1: deterministic detectors: known key/token formats, high-entropy strings, private keys, connection strings, emails, phone numbers, IP and MAC addresses, home-directory paths, usernames.
- Layer 2: optional tiny local classifier/NER for names and project-specific identifiers.
- Layer 3: **query reduction**: rewrite to the technical core ("Blender bpy error: AttributeError on modifier") and drop identifiers. If a query cannot be made safe, block it and tell the user why.
- Policies: `strict` (default), `balanced`, `off` (explicit opt-in).
- **Transparency log:** the user can always see exactly what left the machine, and the cache and project memory never leave it.
- Test with a **canary-secret suite** (fake keys, tokens, paths, emails) in CI; any leak fails the build.
- Honest limit: no filter is perfect. This is defense in depth, with logging so users can audit.

### 5A.2 Low-Bandwidth Mode (v0.1, must-have)
Many target users have slow or metered internet.
- Text-only fetching by default: skip images, scripts, video; prefer lightweight or reader versions of pages.
- Compressed cache (for example zstd), conditional requests (ETag / If-Modified-Since), and a per-day **bandwidth budget** setting.
- Metered-connection detection; queue non-urgent fetches until a better connection.
- Model, adapter and pack downloads: resumable, checksum-verified, and **delta updates** (only changes) where possible.
- Offline-first behaviour: answer from cache and offline packs when the network is unavailable, and say clearly what could not be verified.
- Measured metric: bytes downloaded per answered question (Section 10).

### 5A.3 `brain doctor` (v0.1, must-have)
One command that diagnoses the installation and the machine.
- Checks: CPU/RAM/GPU/VRAM/disk, swap pressure, thermal throttling hints, selected profile vs recommended profile, model and adapter checksums, SearXNG and network reachability, sandbox availability, permission settings, tool bridges (Blender, Unreal, Verilator, KiCad) detected or missing.
- Output: plain-language report ("Your machine fits Lite; Standard would swap and be slow because RAM is 8 GB"), with suggested fixes.
- `brain doctor --report` produces a **redacted** text report (passes through the Privacy Firewall) that users can attach to GitHub issues. This cuts support effort for the maintainer.

### 5A.4 Local Project Memory (v0.1, minimal version)
- Opt-in, per-folder local index of the user's own projects: code, notes, design documents, exported scene information, previous decisions.
- Stores a **decision log** (what was chosen and why), so the AI can answer "why did we pick this architecture?".
- Isolation per project; exclusion patterns by default (`.env`, key files, credentials, `.git` internals); user can view, edit and delete everything.
- Retrieval source type `project`, ranked alongside web evidence and clearly labelled.
- Never enters model weights, never used as training data, never included in outbound queries (Privacy Firewall enforced).
- v0.1 scope: simple chunked embedding index with manual decision-log entries; richer automation later.

### 5A.5 Playbook Memory (v0.2)
- A **playbook** records a successful workflow: task type, plan, tool sequence, verification criteria, tool/API versions, outcome. It stores *how*, not world facts.
- Saved only after executable verification passed or the user confirmed success.
- Retrieved by task similarity; the Planner starts from the closest playbook and adapts it. This directly strengthens small models.
- Hygiene: version-tagged (expires or is flagged when tool versions change), scrubbed of secrets, capped in size, pruned by success rate, user-editable.
- Optional export/import of playbooks (scanned by the Privacy Firewall); community registry only after signing and review exist (Section 5B).

### 5A.6 Idea-to-Prototype Pipeline (v0.2)
A guided path for people who have an idea but few resources.
1. Idea intake and a few clarifying questions (never an interrogation).
2. Constraints: budget, hardware, skills, deadline, preferred free tools.
3. Requirements and success criteria.
4. Architecture options with trade-offs, recommending the cheapest workable path.
5. Task breakdown with estimates.
6. **Scaffold:** project folders, starter scripts, Blender project script, Unreal project structure, HDL skeleton, diagrams.
7. First runnable prototype, verified by execution.
8. Test, iterate, and save the project's decision log and playbooks.
- Outputs are written into the user's project folder as normal files and documents.

### 5A.7 Evidence Mode (v0.2)
- Every claim in an answer is linked to a source and the supporting passage.
- Confidence labels per claim: **verified / partly supported / unverified**.
- Shown in a readable view and available as JSON for tools; the source of the Verifier's decisions is visible to the user.

### 5A.8 Local Preference Adapter (v0.2+, opt-in, stronger profiles first)
- Learns from the user's own accept / reject / edit behaviour (formatting, naming style, level of detail) with a small on-device adapter trained when the machine is idle. Nothing leaves the device.
- Guardrails: it may shape **style and workflow only**. It must never weaken safety rules, truthfulness behaviour, abstention or citation discipline (those adapters keep priority and are frozen).
- An evaluation gate runs before any new preference adapter is activated; one-click reset and rollback.

### 5A.9 Maker Benchmark (public, v0.2)
- Publish a public subset of the 200 to 300 hand-written maker tasks (code, Blender, Unreal, system and hardware architecture) with execution-based scoring, plus a leaderboard for small models.
- Keep a private holdout to prevent overfitting. This gives the project an identity and attracts contributors.

---

## 5B. Later Modules (v0.3 and beyond)

| Module | Purpose | Gate before release |
|---|---|---|
| **Vision** | Read error screenshots, Blender viewport images, circuit and architecture diagrams | Small VLM evaluated per profile |
| **Voice** | Speech in and out through DESKHUB | Reuse DESKHUB voice stack; resource limits on low tiers |
| **Skill Pack Registry** | Community-contributed, signed, reviewed packs | Signing, permission manifests and sandbox review must exist first |
| **Tron robot integration** | Brain output drives the physical robot layer through DESKHUB | Same permission model as Section 11; only after GUI/device-control safety review |

---

## 6. Hardware Profiles and the Resource Governor

### 6.1 Profiles (one family, five sizes)
| Profile | Typical machine | Core model (4-bit) | Adapters at once | Retrieval stack | Modes |
|---|---|---|---|---|---|
| **Nano** | CPU only, 4 to 8 GB RAM | 0.6B to 1.7B | 1 | Tiny embedding, no reranker or tiny one | Fast (limited Deep) |
| **Lite** | 8 to 16 GB RAM, integrated or weak GPU | 1.7B to 4B | 1 to 2 | Small embedding + small reranker | Fast + Deep |
| **Standard** | 16 GB RAM or 6 to 8 GB GPU | 4B to 8B | 2 to 3 | Full small stack | Fast + Deep |
| **Pro** | 12 to 16 GB GPU or 32 GB RAM | 8B to 14B | 3+ | Full stack, larger index | Deep by default |
| **Max** | 24 GB+ GPU or workstation | 14B to 32B+ | Many, plus optional vision | Full stack, large caches, parallel agents | Deep + multi-sample |

Model names and sizes are examples; **verify what is available and permissively licensed at build time**. Profiles are targets to be confirmed by measurement in Phase 1.

### 6.2 Auto-detection
- On first run: detect CPU cores and instruction sets, RAM, GPU and VRAM, disk space, OS.
- Run a short self-benchmark (tokens per second, memory headroom) and choose a profile automatically.
- The user can always override the profile, download a bigger or smaller core, and choose which data packs to install.

### 6.3 Resource Governor (makers run heavy apps)
Blender renders, Unreal builds and FPGA tools use the same RAM and VRAM the AI needs. The AI must never make the user's real work slower.
- Monitors memory, VRAM and CPU load and detects known heavy applications.
- **Yield mode:** when a heavy app is active, offload the model from VRAM to CPU or switch to a smaller quantization or to a smaller core in the same family; pause background indexing.
- **Boost mode:** when the machine is idle, use larger context, more verification loops and background cache building.
- User-visible controls: resource limits, "never use GPU while Blender/Unreal is running", priority settings.
- All model switches keep the same Intelligence Contract so the conversation continues smoothly.

### 6.4 Speed techniques
- Router skips search when unnecessary.
- 4-bit quantization (GGUF via llama.cpp); KV-cache quantization; mmap loading.
- Speculative decoding with a tiny draft model (strong gain on mid and high tiers).
- Streaming answers; parallel search; prefetching likely follow-ups.
- Context compression and aggressive caching.
- Early exit and layer skipping (research, later).

### 6.5 Data packs (the "inner data" that differs by hardware)
Optional, user-chosen, disk-size-aware downloads that never replace the internet:
- **Core pack:** compressed general reference for offline use (small, medium, large editions).
- **Maker packs:** local indexes of documentation the user fetched for tools they use (Blender, Unreal, languages, hardware tools).
- **Cache pack:** the user's own search cache.
- Redistribute a pack **only** if the source license allows it. Otherwise the app builds the index on the user's machine from pages the user's own client fetches.

---

## 7. Domain Skill Packs (for makers)

A skill pack = adapter + tool bridge + retrieval profile + test suite. Each pack is trained with the same method and verified by execution.

| Pack | Skills | Tool bridge | Execution check |
|---|---|---|---|
| **Code** | Languages, debugging, refactoring, repository navigation | files, shell, python sandbox | Run tests, lint, type checks |
| **Blender / 3D** | `bpy` scripting, geometry and shader nodes, modifiers, rigging and animation scripts, export pipelines, asset-pipeline automation | Blender headless runner / bridge | Run script headless, verify objects, render a thumbnail |
| **Unreal** | Gameplay C++, editor Python, Blueprint reasoning, project structure, build and packaging issues | Editor scripting bridge (optional) | Static checks, compile where possible |
| **System architecture** | Requirements to design, component and sequence diagrams, trade-off analysis, capacity estimates, design docs and ADRs | diagram renderers | Diagram renders; checklist-based review |
| **Hardware architecture** | Verilog/SystemVerilog/VHDL, RTL review, testbench writing, FPGA constraints, microarchitecture reasoning, datasheet retrieval, KiCad scripting | Verilator / Icarus / KiCad scripting | Lint, simulate, run testbench |

Rules:
- Skill packs teach **how to find and use documentation**, not API trivia. Version-aware retrieval is mandatory.
- Verifiable tasks (scripts that run, designs that simulate) provide automatic rewards for reinforcement learning (Section 9).
- Packs load on demand; low tiers load one at a time.
- Community can contribute new packs (for example CAD, game engines, audio, robotics) with a pack template and required tests.

---

## 8. Building the Model (from open source, license-clean)

### 8.1 Base model selection (per size class)
1. For each size class (Nano, Lite, Standard, Pro, Max), shortlist 3 to 5 open-weight models with **Apache-2.0 or MIT** licenses and native reasoning and tool-calling ability. Verify current releases and licenses at build time; this field changes monthly.
2. Score every candidate with the project's benchmark (Section 10) **before** any fine-tuning. Choose by score per gigabyte.
3. Record license, model card and checksums in `MODELS.md`.
4. Using one model **family** across sizes (when the license allows) makes behaviour more consistent across tiers and allows adapter-recipe reuse.

### 8.2 The "weight management" toolbox
| Technique | Use in this project |
|---|---|
| Pruning + distillation (Minitron-style) | Shrink a larger permissive model to a smaller size class, then recover quality via distillation |
| Multi-teacher distillation | Combine behaviours of several open teachers into one student |
| Logit distillation | Only with open teachers; richer signal than text-only |
| Task vectors and merging (TIES, DARE, SLERP) | Combine adapters or fine-tunes **from the same base model only** |
| Quantization (Q3 to Q8, mixed) | Memory and speed; different levels per profile |
| LoRA / QLoRA | Skill adapters, trainable on a laptop GPU for small and mid sizes |
| Speculative decoding | Speed |

Rule: merged or pruned derivatives inherit the strictest license among inputs. Keep a license ledger.

### 8.3 One recipe, many sizes
1. Build and validate the full recipe on the **Standard** size first (fits the owner's RTX 5050 for QLoRA).
2. Re-run the same data and recipe on the Lite and Nano bases; adjust only hyperparameters.
3. Produce Pro and Max using rented or free GPU time, or by pairing the recipe with larger base models when resources exist.
4. Evaluate every size on the same suite and publish the capability matrix.

### 8.4 What to train (skills)
Query decomposition and planning; when to search vs answer directly; tool calling with correct arguments; reading messy web text and docs; step-by-step reasoning; writing and repairing code and scripts; self-verification using execution results; citation discipline and abstention; injection resistance; safe system operation.

### 8.5 What NOT to train
Encyclopedic facts, news, statistics, product specifications, anything that changes, and any copyrighted text dumps.

---

## 9. Data Pipeline and Training

### 9.1 Data (skill data, not knowledge data; all sources license-checked)
- Open QA, reasoning, code, and tool-calling datasets.
- Synthetic tasks generated by **open-license teachers** run locally or on free GPUs.
- Maker tasks: Blender scripts, Unreal snippets, HDL modules, architecture write-ups, **each paired with a runnable check** where possible.
- Adversarial sets: prompt-injection pages, misleading search results, empty results, broken tools.

### 9.2 Trace generation
1. Seed questions across categories (general research, code, each domain pack, safety).
2. Run the **real retrieval loop** so the student sees noisy real evidence.
3. Teacher produces the trace: plan, tool calls, reasoning summary, cited answer.
4. Filter: answer supported by evidence (independent verifier model), self-consistency, execution passes for runnable tasks, dedupe, drop malformed.
5. Include negatives: unanswerable questions, conflicting sources, injection attempts, tool failures.
6. Build preference pairs (grounded and correct vs hallucinated, uncited or injected).
7. Version everything with a datasheet (sources, licenses, filters, known gaps).

### 9.3 Size targets
- SFT: 10,000 to 50,000 high-quality traces (quality over volume).
- DPO: 5,000 to 20,000 pairs.
- RL: verifiable tasks (math, code, Blender scripts, HDL simulations, tool-use with checkers).

### 9.4 Training steps
1. **SFT** on skill traces (QLoRA on the owner's RTX 5050 for small and mid sizes; free or rented GPUs for larger).
2. **Per-skill adapters** trained separately; test for interference when stacked.
3. **DPO** for grounding, abstention and injection resistance.
4. **RL with verifiable rewards (GRPO)** on execution-checked tasks, so the model learns from real pass/fail signals and can exceed its teachers on those skills.
5. **Recovery distillation** after pruning.
6. **Quantize** and re-evaluate; choose quantization per profile.
7. Publish configs, seeds and recipes.

Tools to consider (verify current status): Unsloth, Axolotl, TRL, llama.cpp, lm-eval-harness.

### 9.5 Using the owner's personal agent (Hermes) and Antigravity
- They are **builder tools**, not shipped components. Hermes is the owner's personal agent for this project and is never listed as a base model.
- Their outputs can speed up code, docs and test writing for the repository.
- **Do not use their outputs as training data unless the underlying model's license and terms allow it.** If an agent is backed by a commercial model, treat its outputs like that model's outputs for licensing purposes. Training data comes from open-license teachers only.

---

## 10. Evaluation (the credibility section)

| Skill | Benchmark type |
|---|---|
| Tool calling | Berkeley Function-Calling style suites |
| Fresh facts | FreshQA-style questions |
| Multi-hop research | HotpotQA-style with live retrieval |
| Web/agent tasks | GAIA-style and web-navigation subsets |
| Reasoning | Math subsets, logic puzzles |
| Code | Code generation and repair suites, run with tests |
| **Maker tasks** | Own suite: Blender script tasks, Unreal snippet tasks, HDL tasks, architecture review tasks (each with execution or checklist scoring) |
| Safety | Prompt-injection suite, destructive-command refusal tests |
| Groundedness | Citation precision, unsupported-claim rate |
| Efficiency | Tokens/sec, time-to-first-token, RAM/VRAM per profile |
| Governor | Slowdown of Blender/Unreal benchmark scenes with and without the AI running |
| Privacy | Canary-secret suite: zero leaked secrets in outbound queries, fetched URLs and logs |
| Bandwidth | Bytes downloaded per answered question; behaviour under offline and metered conditions |
| Playbooks | Success-rate and step-count improvement on repeated task types with vs without playbooks |
| Evidence | Share of claims correctly linked to supporting passages (faithfulness) |

- Build **200 to 300 hand-written maker tasks** plus a private holdout. Publish a public subset as the **Maker Benchmark** (Section 5A.9).
- Compare against open baselines and (for evaluation only, if terms allow) commercial assistants.
- Publish scores with hardware and settings; include failures. Publish the capability matrix per profile.

---

## 11. Security Model

**Threats:** prompt injection via web pages or project files, malicious tool arguments, data exfiltration, destructive commands, poisoned search results, malicious plug-ins or skill packs, supply-chain risk in models and dependencies.

**Controls:**
1. Data/instruction separation, injection detector, instruction stripping.
2. Sandboxed execution (containers or OS-level restrictions), no network for the code sandbox by default.
3. Permission tiers: read-only (auto), reversible changes (confirm once), destructive or irreversible (always confirm with the exact command shown).
4. Allowlists for commands and domains; denylist for credentials and system paths.
5. Full action log; snapshots and one-click undo for file changes.
6. Signed and checksummed models, adapters and skill packs; pinned dependency versions.
7. Skill-pack and plug-in review rules (permissions manifest, sandboxed by default).
8. Red-team suite in CI; every release must pass it.
9. No user data leaves the device unless the user explicitly opts in.
10. **Privacy Firewall** on every outbound request, with a user-visible transparency log and canary-secret tests in CI.
11. Project memory and playbooks exclude secrets by default and never enter outbound queries or training data.
12. Preference learning is style-only; safety, truthfulness and abstention behaviours are frozen and take priority.

---

## 12. Repository Layout

```
retrieval-brain/
  README.md
  MODELS.md              # models, licenses, checksums
  LICENSES.md            # license ledger for data and weights
  core/
    governor.py          # resource governor, hardware detection, profiles
    router.py planner.py retriever.py reranker.py compressor.py
    reasoner.py verifier.py writer.py safety.py pipeline.py
    privacy.py           # Privacy Firewall (outbound redaction, query reduction)
    bandwidth.py         # low-bandwidth mode, delta downloads, budgets
    doctor.py            # `brain doctor`
    project_memory.py    # local project index and decision log
    playbooks.py         # playbook memory (v0.2)
    prototype.py         # idea-to-prototype pipeline (v0.2)
    evidence.py          # evidence mode (v0.2)
    preference.py        # local preference adapter (v0.2+, opt-in)
  profiles/
    nano.yaml lite.yaml standard.yaml pro.yaml max.yaml
  tools/
    search.py fetch.py browser.py shell.py files.py python_sandbox.py
    bridges/             # blender, unreal, kicad, vscode
    mcp/                 # MCP-compatible tool definitions
  skillpacks/
    code/ blender/ unreal/ sysarch/ hwarch/   # adapter + tools + tests
    TEMPLATE/
  cache/                 # SQLite + FAISS indexes
  playbooks/             # saved workflows (local, user-owned)
  packs/                 # offline data pack builders
  adapters/              # LoRA adapters per skill and size
  data/ seeds/ traces/ prefs/ datasheet.md
  train/ sft.yaml dpo.yaml grpo.yaml
  eval/ suites/ run_eval.py reports/ capability_matrix.md
  security/ injection_tests/ redteam/
  server/ app.py         # FastAPI; Deskhub connector
  docs/
```

### 12.1 Core loop (reference skeleton)
```python
async def answer(question):
    profile = governor.current_profile()          # may change while running
    route = router.decide(question, profile)
    if route == "direct":
        return reasoner.answer(question)

    plan = planner.make(question, profile)
    evidence = []
    for step in plan.steps:
        docs = await retriever.search(step.query, cache=True)
        docs = safety.sanitize(docs)
        evidence += reranker.top_k(step.query, docs, k=profile.k)

    for attempt in range(profile.max_loops):
        draft = reasoner.synthesize(question, compressor.shrink(evidence),
                                    skillpack=plan.skillpack)
        report = verifier.check(draft, evidence, run_code=profile.can_execute)
        if report.ok:
            return writer.finalize(draft, evidence)
        evidence += await retriever.search(report.missing_query)
        question = reasoner.repair_prompt(question, report.execution_errors)

    return writer.finalize_with_uncertainty(draft, evidence)
```

---

## 13. Roadmap and Milestones

| Phase | Goal | Deliverable | Exit criterion |
|---|---|---|---|
| 0 | Setup | Repo, license ledger, eval harness skeleton | Benchmarks run on any model |
| 1 | Baseline + hardware survey | Untuned small models + SearXNG + simple RAG loop; measure real speed and RAM on several machines | Baseline scores per profile; profiles confirmed by data |
| 2 | Full agent loop | Router, Planner, Retriever, Reranker, Verifier, Writer | Beats baseline on groundedness and fresh-fact tests |
| 3 | Governor + privacy + safety + tools | Resource Governor, **Privacy Firewall**, sanitizer, injection detector, sandbox, approvals, **`brain doctor`** | Passes injection suite and canary-secret leak tests; slowdown within agreed limit; doctor works on 3+ different machines |
| 4 | Skill data | Filtered traces (general + Code pack first), DPO pairs, datasheet | Dataset audited and versioned |
| 5 | Training (Standard size) | SFT, adapters, DPO, GRPO with execution rewards | Measurable gains on all target benchmarks |
| 6 | Scale the family | Lite and Nano via same recipe; capability matrix | Each profile meets its speed/RAM targets |
| 7 | **v0.1 public release** | Standard + Lite profiles, Code pack, core adapters, auto-detection, **Privacy Firewall, Low-Bandwidth Mode, `brain doctor`, minimal Local Project Memory**, one-click installer, docs, benchmark table | Reproducible install in under 15 minutes; 10 to 20 real users trying it |
| 7b | v0.2 features | Playbook Memory, Idea-to-Prototype Pipeline, Evidence Mode, public Maker Benchmark | Playbooks measurably improve success rates on repeat tasks |
| 8 | Maker packs | Blender pack, then Hardware-architecture and System-architecture packs, then Unreal | Each pack passes its execution suite |
| 9 | Nano + Pro/Max + offline packs | Remaining profiles, data packs, opt-in Local Preference Adapter on stronger profiles | Community feedback addressed; preference adapter passes its evaluation gate |
| 10 | Languages | Language adapters/translation plug-ins (Tamil first) | Language packs evaluated separately |
| 11 | Vision, voice, GUI control | Vision (error screenshots, viewport, diagrams), Deskhub voice, screen control | Safety-approved release |
| 12 | Ecosystem | Signed Skill Pack Registry, Tron robot integration via DESKHUB | Security review passed |

Estimated effort for one builder with agent help: Phases 0 to 3 about 6 to 8 weeks; Phases 4 to 6 about 8 to 12 weeks; v0.1 around month 5 to 6 (the new must-haves add roughly a month). Adjust after Phase 1 measurements.

**v0.1 scope rule:** two profiles, one skill pack (Code), three core adapters (Planner, Tool-caller, Verifier), plus four deliberately **small** must-haves: Privacy Firewall (rule-based first), Low-Bandwidth Mode, `brain doctor`, and minimal Local Project Memory. Build the simplest working version of each. Everything else follows after real users give feedback.

---

## 14. Using Antigravity and the Personal Agent

- **Antigravity:** give it this document as the spec; one agent task per module (governor, router, retriever, safety, verifier, and so on), each with unit tests and a short design note. Review every merge yourself. Use it to survey the current state and licenses of candidate open-source repositories before adopting any.
- **Hermes (personal agent):** use it as your day-to-day project assistant: planning, scripting, documentation, reviewing results. Keep it separate from the product and from model selection (see 9.5).
- Humans stay in the loop for license decisions, security review and benchmark interpretation.

---

## 15. Zero-Cost Build Strategy

- Training: local RTX 5050 for QLoRA on small and mid models; free Kaggle/Colab GPU quotas for longer runs.
- Data generation: run 7B to 8B open teachers locally (slow, free and legally clean).
- Search: self-hosted SearXNG, Wikipedia API, no paid keys.
- Hosting: none required; everything runs on user machines. Publish weights and adapters on Hugging Face, code on GitHub.
- Community: contributions of datasets, evaluations, skill packs, translations and tool bridges.

---

## 16. Open-Source Release Plan

1. **License:** Apache-2.0 for code. Weights and data under licenses compatible with all inputs (documented in `LICENSES.md`).
2. **Docs:** quick start, profile guide ("which profile fits my machine"), governor guide, security model, how to write a skill pack, how to contribute data.
3. **Reproducibility:** pinned versions, seeds, recipes, datasheets, evaluation scripts.
4. **Transparency:** public benchmark table including failures, model cards with limitations, capability matrix.
5. **Governance:** code of conduct, contribution guide, responsible-use statement (system control and web access carry risk).
6. **Privacy:** no telemetry; any optional sharing is opt-in and anonymous.

---

## 16A. Sustainability and Project Health

- **One clear first success:** a v0.1 that 10 to 20 real people actually use is worth more than a perfect plan. Ship small, collect feedback, then expand.
- **Easy installation:** one-click installer and a `brain doctor` path for fixing problems; excellent beginner documentation, because many users are non-expert and have no budget.
- **Funding and support:** explore donation and grant options for open-source projects (for example GitHub Sponsors and open-source grants). Check the current programmes and eligibility before launch.
- **Pace and balance:** the owner is also a student. Time-box each phase, protect the v0.1 scope, and prefer a steady schedule over bursts that cause burnout.
- **Contributors:** label "good first issue" tasks, publish the skill-pack template, and keep the architecture documented so others can add packs, datasets and translations.
- **Release rhythm:** small, regular releases with changelogs and a public capability matrix.

---

## 17. Risk Register

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | Small-model reasoning ceiling | Weak on hard tasks | Test-time compute, executable verification, adapters, RL, honest escalation |
| 2 | Poor search results | Bad answers | Hybrid retrieval, trust scoring, reranker, verifier, multi-source |
| 3 | Latency from many stages | Poor UX | Router, cache, parallelism, Fast mode |
| 4 | Prompt injection / unsafe actions | Security incidents | Sanitizer, detector, sandbox, approvals, red-team CI |
| 5 | License contamination | Cannot open-source | License ledger, permissive-only core, review of every input |
| 6 | Adapter interference | Regressions | Separate training, stacked tests |
| 7 | Free search instability | Outages | Multiple sources, cache, offline packs |
| 8 | Over-claiming | Loss of trust | Publish measured results only |
| 9 | Scope creep (solo builder) | Never ships | Strict v0.1 scope |
| 10 | Benchmark overfitting | Misleading scores | Private holdout, real-task suite |
| 11 | Bad synthetic data | Teaches wrong habits | Independent verifier, execution filters, audits |
| 12 | Low-end hardware too slow | Unusable for target users | Nano profile, smaller models, quantization, measured targets |
| 13 | AI steals resources from Blender/Unreal | Users disable it | Resource Governor, yield mode, measured slowdown limit |
| 14 | Inconsistent behaviour across sizes | Confusing UX | Intelligence Contract, shared suite, capability matrix |
| 15 | Tool/API version mismatch in generated maker code | Broken scripts | Version-aware retrieval, executable verification |
| 16 | Malicious community skill packs | Security breach | Permission manifests, signing, sandboxing, review |
| 17 | Licensing of tool documentation for offline packs | Legal exposure | Redistribute only permitted docs; otherwise index locally from user's own fetches |
| 18 | Privacy Firewall false negatives | Secret leaks | Layered detection, query reduction, transparency log, canary tests, strict default |
| 19 | Preference adapter drift or sycophancy | Degraded honesty | Style-only learning, frozen safety adapters, evaluation gate, rollback |
| 20 | Stale or poisoned playbooks | Repeating bad workflows | Version tags, execution-verified saves only, pruning, user review |
| 21 | Project memory indexes sensitive files | Local data exposure | Default exclusions, per-folder opt-in, local-only storage, easy delete |
| 22 | Low-bandwidth mode lowers answer quality | Missed information | Fallback to full fetch on request, clear notes on what was skipped |
| 23 | v1.2 additions inflate v0.1 scope | Delays | Minimal first versions; defer v0.2 items; enforce scope rule |

---

## 18. Future: Multi-language Support

- Core remains English.
- Add **language adapters** (small LoRA) or optional translation plug-ins (translate in, answer, translate out).
- Choose embedding models with multilingual retrieval in mind, while keeping a tiny English-only option for Nano.
- Tamil is the first planned language pack (specialized evaluation set, Tamil sources, writing quality checks). Each pack is evaluated and released independently.

---

## 19. Integration with DESKHUB

- Retrieval Brain is DESKHUB's reasoning and research engine.
- Online mode: full retrieval loop. Offline mode: cache + offline packs + direct reasoning.
- DESKHUB's FastAPI backend, local-model detection and VS Code bridge connect to `server/app.py`.
- DESKHUB's device-control features inherit the Section 11 permission model.
- The Resource Governor can share hardware detection with DESKHUB.
- Later (Phase 12): DESKHUB can route brain outputs to the Tron robot hardware layer under the same permission model.

---

## 20. Open Questions (decide before Phase 1)

1. Project name and branding.
2. Which model family to try first for the Standard size (decide from Phase 1 data).
3. License for weights and data (depends on final base models).
4. Fast/Deep: automatic by default, user override always available?
5. Optional "escalate to a bigger model or free API" feature in v0.1 or later?
6. Which maker pack comes first after Code: Blender (easiest to verify) is the suggested default.
7. Default Privacy Firewall strictness (`strict` recommended) and what the transparency log shows.
8. Funding path and whether to apply for grants before or after v0.1.

---

## 21. First Seven Days (action checklist)

- [ ] Create the repo, `LICENSES.md`, `MODELS.md`.
- [ ] Install llama.cpp/Ollama and SearXNG locally.
- [ ] Shortlist permissive candidates for Nano, Lite and Standard; record licenses.
- [ ] Write the evaluation harness and the first 50 tasks (include 10 Blender-script tasks and 10 code tasks with runnable checks).
- [ ] Build the minimal RAG loop (search → fetch → chunk → embed → answer with citations).
- [ ] Measure speed and memory of each candidate on the RTX 5050 laptop; if possible, also on one low-end machine.
- [ ] Prototype hardware detection (RAM, VRAM, CPU) and the profile selector.
- [ ] Write the Privacy Firewall canary-secret test set (fake keys, tokens, paths, emails) before building the firewall.
- [ ] Ask Antigravity to scaffold `core/` modules from Sections 5 and 5A with tests.

---

*End of plan. Update this document after each phase with measured results.*
