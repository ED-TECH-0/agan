# Security Policy

AGAN is a local-first AI that researches the web, executes untrusted code in
a verifier sandbox, and calls outbound services. Treat it as executing
untrusted code: run it in an isolated environment (container/VM) and never
point it at credentials you cannot afford to lose.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.0-dev | yes (development) |

## Reporting a vulnerability

Email the maintainer or open a private GitHub advisory. Do not disclose
vulnerabilities publicly until they are patched.

## Known risks (threat model)

| Risk | Mitigation status |
|------|-------------------|
| Prompt injection via retrieved web content | Partial — `safety/sanitizer.py` strips fenced blocks and common injection keywords; not comprehensive |
| SSRF via fetch provider | Mitigated — `retrieval/fetch.py` blocks private IP ranges and non-HTTP(S) schemes |
| Arbitrary code execution in verifier | Partial — `_execute_code` runs via `subprocess.run` without sandbox isolation (M4 gap) |
| Tainted data flowing from retrieval to answer | Partial — `Chunk.tainted` flag exists but is not enforced end-to-end (M5 gap) |
| API key leakage in outbound requests | Mitigated — `privacy/firewall.py` redacts common secret header patterns |
| Model supply chain | No model weights bundled; operator controls Ollama model source |

## Safe defaults

- Hardware profile auto-detects RAM/GPU and picks a model within budget.
- Privacy gateway blocks disallowed outbound hosts before any pipeline runs.
- Fetch enforces robots.txt, per-domain rate limits, and byte caps.

## Dependencies

See LICENSES.md for dependency licenses. Pin transitive dependencies before
any release and scan for known vulnerabilities (e.g. `pip-audit`).
