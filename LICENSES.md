# License Ledger

This document tracks the license of every component that goes into agan. Only Apache-2.0, MIT, BSD-2/3, and similarly permissive licenses are allowed in the core.

## Components

| Component | License | Source | Notes |
|---|---|---|---|
| pyyaml | MIT | https://github.com/yaml/pyyaml | runtime |
| psutil | BSD-3-Clause | https://github.com/giampaolo/psutil | runtime |
| fastapi | MIT | https://github.com/fastapi/fastapi | runtime |
| uvicorn | BSD-3-Clause | https://github.com/encode/uvicorn | runtime |
| pydantic | MIT | https://github.com/pydantic/pydantic | runtime |
| httpx | Apache-2.0 | https://github.com/encode/httpx | runtime |
| trafilatura | Apache-2.0 | https://github.com/adbar/trafilatura | runtime |
| numpy | BSD-3-Clause | https://github.com/numpy/numpy | runtime |
| aiofiles | Apache-2.0 | https://github.com/Tinche/aiofiles | runtime |
| typer | MIT | https://github.com/tiangolo/typer | runtime |
| rich | MIT | https://github.com/Textualize/rich | runtime |
| datasets | Apache-2.0 | https://github.com/huggingface/datasets | runtime |
| onnxruntime | Apache-2.0 | https://github.com/microsoft/onnxruntime | runtime |
| sentence-transformers | MIT | https://github.com/UKPLab/sentence-transformers | runtime (embeddings) |
| shellingham | MIT | https://github.com/sarugaku/shellingham | runtime |
| torch | BSD-3-Clause | https://github.com/pytorch/pytorch | optional (ml extra) |
| transformers | Apache-2.0 | https://github.com/huggingface/transformers | optional (ml extra) |
| GPUtil | MIT | https://github.com/anderskm/gputil | optional (gpu extra) |
| faiss-cpu | BSD-3-Clause / MIT | https://github.com/facebookresearch/faiss | optional (vector extra) |
| pytest | MIT | https://github.com/pytest-dev/pytest | dev |
| pytest-asyncio | Apache-2.0 / MIT | https://github.com/pytest-dev/pytest-asyncio | dev |
| ruff | MIT | https://github.com/astral-sh/ruff | dev |
| mypy | MIT | https://github.com/python/mypy | dev |

## Policy
- **Core code**: Apache-2.0
- **Model weights**: Must be compatible with all inputs; documented in `MODELS.md`
- **Training data**: Only open-license sources; datasheet in `data/datasheet.md`
- **Dependencies**: Pinned versions; license checked before adding

## License Checking
Before adding any dependency, model, or dataset:
1. Verify license is permissive (Apache-2.0, MIT, BSD, CC-BY, CC0, etc.)
2. Record in this ledger
3. Ensure no GPL, AGPL, SSPL, or other viral/restrictive licenses in core
