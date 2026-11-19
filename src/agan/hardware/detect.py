"""Hardware profile detection for AGAN (S9.2).

Detects OS, CPU flags (avx2/avx512/neon), total/free RAM, GPU name and free
VRAM (nvidia-smi), disk space, and battery. Chooses an S/M/L profile using a
*memory-fit* model (weights + KV cache + overhead must fit in 70% of free
RAM), steps the profile down when a 30-second generation benchmark is too
slow, and persists the choice with a hardware fingerprint so it is redone only
when the underlying hardware changes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import psutil

DEFAULT_STATE_PATH = Path.home() / ".agan" / "hardware_state.json"

# Bytes per parameter by quantization (approx., includes a small overhead).
QUANT_BYTES_PER_PARAM = {
    "Q8_0": 1.0,
    "Q6_K": 0.825,
    "Q5_K_M": 0.7,
    "Q4_K_M": 0.6125,  # 4.9 bits / 8
    "Q3_K_M": 0.4875,  # 3.9 bits / 8
    "Q2_K": 0.325,
}

# Minimum acceptable generation speed (tokens/s) per profile.
class HardwareProfile(str, Enum):
    S = "S"
    M = "M"
    L = "L"


MIN_TOKENS_PER_SEC = {
    HardwareProfile.S: 3.0,
    HardwareProfile.M: 6.0,
    HardwareProfile.L: 8.0,
}


@dataclass(frozen=True)
class ProfileSpec:
    name: HardwareProfile
    ram_gb_min: float
    ram_gb_max: float
    gpu_required: bool
    default_model: str
    quantization: str
    time_budget: int
    best_of_n: int
    max_context: int
    max_output_tokens: int = 512
    max_retrieval: int = 5
    enable_execution: bool = True
    execution_timeout_s: int = 30
    level: int = 1


SPECS = {
    HardwareProfile.S: ProfileSpec(
        name=HardwareProfile.S,
        ram_gb_min=0,
        ram_gb_max=8,
        gpu_required=False,
        default_model="qwen2.5-3b:latest",
        quantization="Q4_K_M",
        time_budget=40,
        best_of_n=1,
        max_context=4096,
    ),
    HardwareProfile.M: ProfileSpec(
        name=HardwareProfile.M,
        ram_gb_min=12,
        ram_gb_max=16,
        gpu_required=False,
        default_model="qwen2.5-7b-instruct-q4_k_m",
        quantization="Q4_K_M",
        time_budget=45,
        best_of_n=3,
        max_context=8192,
    ),
    HardwareProfile.L: ProfileSpec(
        name=HardwareProfile.L,
        ram_gb_min=24,
        ram_gb_max=float("inf"),
        gpu_required=True,
        default_model="qwen2.5-14b-instruct-q3_k_m",
        quantization="Q3_K_M",
        time_budget=60,
        best_of_n=4,
        max_context=16384,
    ),
}

# Order from strongest to weakest, used for stepping down.
PROFILE_ORDER = [HardwareProfile.L, HardwareProfile.M, HardwareProfile.S]


# --------------------------------------------------------------------------- #
# Sensors
# --------------------------------------------------------------------------- #

def cpu_flags() -> list[str]:
    """Return the SIMD/CAP flags detected on this CPU (subset we care about)."""
    wanted = {
        "avx2", "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
        "avx512ifma", "avx512vbmi", "neon", "asimd",
    }
    found: set[str] = set()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("flags"):
                    found |= {f.lower() for f in line.split(":", 1)[1].split()} & wanted
                    break
    except OSError:
        pass
    # macOS / Windows: fall back to platform processor info hints.
    if not found:
        proc = platform.processor().lower()
        for flag in wanted:
            if flag in proc:
                found.add(flag)
    return sorted(found)


def get_ram_gb() -> float:
    return psutil.virtual_memory().total / (1024 ** 3)


def get_ram_free_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def get_gpu_info() -> tuple[bool, str | None]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            name = out.stdout.strip().split("\n")[0].split(",")[0].strip()
            return True, name
    except Exception:  # noqa: S110, BLE001 — nvidia-smi may be absent
        pass
    return False, None


def get_gpu_free_vram_gb() -> float | None:
    """Free VRAM in GB for the first GPU, or None if no GPU/nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().split("\n")[0]) / 1024.0
    except Exception:  # noqa: S110, BLE001 — nvidia-smi may be absent
        pass
    return None


def get_disk_free_gb(path: str = "/") -> float:
    try:
        usage = psutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except Exception:  # noqa: BLE001 — defensive sensor read
        return 0.0


def get_battery_percent() -> float | None:
    try:
        batt = psutil.sensors_battery()
        if batt is not None:
            return batt.percent
    except Exception:  # noqa: S110, BLE001 — battery may be unsupported
        pass
    return None


def platform_summary() -> dict:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_flags": cpu_flags(),
        "python": platform.python_version(),
    }


# --------------------------------------------------------------------------- #
# Memory-fit model
# --------------------------------------------------------------------------- #

def bytes_per_param(quantization: str) -> float:
    return QUANT_BYTES_PER_PARAM.get(quantization, 0.6125)


def model_bytes_gb(params_b: float, quantization: str = "Q4_K_M") -> float:
    """Weights size in GB, including a 5% runtime overhead."""
    return params_b * 1e9 * bytes_per_param(quantization) * 1.05 / 1e9


def kv_bytes_gb(
    layers: int, kv_heads: int, head_dim: int, ctx: int,
    bytes_per_elem: float = 2.0,
) -> float:
    """KV-cache size in GB (2 directions: keys + values)."""
    return 2.0 * layers * kv_heads * head_dim * ctx * bytes_per_elem / 1e9


def memory_fits(
    params_b: float,
    layers: int,
    kv_heads: int,
    head_dim: int,
    ctx: int,
    available_gb: float,
    quantization: str = "Q4_K_M",
) -> bool:
    """True if weights + KV cache + overhead fit in 70% of free RAM."""
    budget = 0.7 * available_gb
    return model_bytes_gb(params_b, quantization) + kv_bytes_gb(
        layers, kv_heads, head_dim, ctx
    ) <= budget


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #

async def benchmark_tokens_per_sec(
    client: Any = None, ctx: int = 4096, n_tokens: int = 128, repeats: int = 2
) -> float | None:
    """Generate ``n_tokens`` ``repeats`` times and return tokens/s, or None if
    no client/server is available to measure against."""
    if client is None:
        return None
    assert client is not None
    total_toks = 0.0
    total_sec = 0.0
    for _ in range(repeats):
        prompt = " ".join(["word"] * min(ctx, 512))
        start = time.perf_counter()
        try:
            await client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=n_tokens,
            )
        except Exception:  # noqa: BLE001 — generation failure means unmeasurable
            return None
        elapsed = time.perf_counter() - start
        if elapsed <= 0:
            return None
        total_toks += n_tokens
        total_sec += elapsed
    return total_toks / total_sec


# --------------------------------------------------------------------------- #
# Fingerprint + persisted state
# --------------------------------------------------------------------------- #

def hardware_fingerprint(
    ram_gb: float, cpu_flags_list: list[str], has_gpu: bool,
    gpu_name: str | None, disk_free_gb: float,
) -> str:
    payload = {
        "ram_gb": round(ram_gb, 2),
        "cpu_flags": sorted(cpu_flags_list),
        "has_gpu": has_gpu,
        "gpu_name": gpu_name,
        "disk_free_gb": round(disk_free_gb, 2),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_state(path: Path = DEFAULT_STATE_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — missing/corrupt state is fine
        return None


def save_state(state: dict, path: Path = DEFAULT_STATE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:  # noqa: S110, BLE001 — persisting state is best-effort
        pass


# --------------------------------------------------------------------------- #
# Profile choice (S9.2)
# --------------------------------------------------------------------------- #

def base_profile(ram_gb: float, has_gpu: bool) -> HardwareProfile:
    """RAM + GPU tier (no memory-fit / benchmark adjustment)."""
    if ram_gb >= 24 and has_gpu:
        return HardwareProfile.L
    if ram_gb >= 12:
        return HardwareProfile.M
    return HardwareProfile.S


async def choose_profile(
    params_b: float = 3.0,
    layers: int = 32,
    kv_heads: int = 32,
    head_dim: int = 128,
    ctx: int | None = None,
    quantization: str | None = None,
    client: Any = None,
    state_path: Path = DEFAULT_STATE_PATH,
    ram_gb: float | None = None,
    has_gpu: bool | None = None,
    gpu_name: str | None = None,
    free_vram_gb: float | None = None,
    disk_free_gb: float | None = None,
    benchmark_tok_per_s: float | None = None,
) -> dict:
    """Pick an S/M/L profile using memory-fit, benchmark step-down, and a
    persisted hardware fingerprint. Returns a dict with profile, spec, reason,
    fingerprint, and sensors."""
    ram = get_ram_gb() if ram_gb is None else ram_gb
    gpu, gpu_n = get_gpu_info() if has_gpu is None else (has_gpu, gpu_name or "")
    free_vram = get_gpu_free_vram_gb() if free_vram_gb is None else free_vram_gb
    disk = get_disk_free_gb() if disk_free_gb is None else disk_free_gb
    flags = cpu_flags()
    fp = hardware_fingerprint(ram, flags, gpu, gpu_n, disk)

    # Reuse persisted result when the fingerprint is unchanged.
    saved = load_state(state_path)
    if saved and saved.get("fingerprint") == fp:
        spec = get_spec(HardwareProfile(saved["profile"]))
        return {
            "profile": saved["profile"],
            "spec": spec,
            "fingerprint": fp,
            "reason": "reused (fingerprint unchanged)",
            "sensors": {
                "ram_gb": ram, "free_vram_gb": free_vram,
                "disk_free_gb": disk, "cpu_flags": flags,
                "has_gpu": gpu, "gpu_name": gpu_n,
            },
        }

    profile = base_profile(ram, gpu)
    reasons: list[str] = [f"base tier from RAM={ram:.1f}GB gpu={gpu}"]

    # Memory-fit step-down: weights + KV must fit in 70% of free RAM at this
    # profile's max_context.
    spec = get_spec(profile)
    prof_ctx = ctx if ctx is not None else spec.max_context
    while profile != HardwareProfile.S:
        spec = get_spec(profile)
        c = ctx if ctx is not None else spec.max_context
        q = quantization if quantization is not None else spec.quantization
        if memory_fits(params_b, layers, kv_heads, head_dim, c, ram, q):
            break
        idx = PROFILE_ORDER.index(profile)
        profile = PROFILE_ORDER[idx + 1]  # step down
        reasons.append(f"stepped down to {profile.value} (memory fit)")

    # Benchmark step-down: generation too slow for this profile.
    if benchmark_tok_per_s is None:
        benchmark_tok_per_s = await benchmark_tokens_per_sec(
            client, ctx=prof_ctx
        )
    if benchmark_tok_per_s is not None:
        threshold = MIN_TOKENS_PER_SEC[profile]
        if benchmark_tok_per_s < threshold:
            idx = PROFILE_ORDER.index(profile)
            if idx + 1 < len(PROFILE_ORDER):
                profile = PROFILE_ORDER[idx + 1]
                reasons.append(
                    f"stepped down to {profile.value} "
                    f"({benchmark_tok_per_s:.1f} tok/s < {threshold})"
                )

    spec = get_spec(profile)
    state = {
        "profile": profile.value,
        "spec": _spec_to_dict(spec),
        "fingerprint": fp,
        "reason": "; ".join(reasons),
        "sensors": {
            "ram_gb": ram, "free_vram_gb": free_vram,
            "disk_free_gb": disk, "cpu_flags": flags,
            "has_gpu": gpu, "gpu_name": gpu_n,
            "benchmark_tok_per_s": benchmark_tok_per_s,
        },
    }
    save_state(state, state_path)
    return {
        "profile": profile,
        "spec": spec,
        "fingerprint": fp,
        "reason": "; ".join(reasons),
        "sensors": state["sensors"],
    }


def _spec_to_dict(spec: ProfileSpec) -> dict:
    return {
        "name": spec.name.value,
        "ram_gb_min": spec.ram_gb_min,
        "ram_gb_max": spec.ram_gb_max,
        "gpu_required": spec.gpu_required,
        "default_model": spec.default_model,
        "quantization": spec.quantization,
        "time_budget": spec.time_budget,
        "best_of_n": spec.best_of_n,
        "max_context": spec.max_context,
        "max_output_tokens": spec.max_output_tokens,
        "max_retrieval": spec.max_retrieval,
        "enable_execution": spec.enable_execution,
        "execution_timeout_s": spec.execution_timeout_s,
        "level": spec.level,
    }


# Legacy wrappers kept for backward compatibility.
def get_spec(profile: HardwareProfile) -> ProfileSpec:
    return SPECS[profile]


def detect_profile() -> HardwareProfile:
    ram = get_ram_gb()
    has_gpu, _ = get_gpu_info()
    return base_profile(ram, has_gpu)


__all__ = [
    "DEFAULT_STATE_PATH",
    "PROFILE_ORDER",
    "SPECS",
    "HardwareProfile",
    "ProfileSpec",
    "base_profile",
    "benchmark_tokens_per_sec",
    "choose_profile",
    "cpu_flags",
    "detect_profile",
    "get_battery_percent",
    "get_disk_free_gb",
    "get_gpu_free_vram_gb",
    "get_gpu_info",
    "get_ram_free_gb",
    "get_ram_gb",
    "get_spec",
    "hardware_fingerprint",
    "kv_bytes_gb",
    "load_state",
    "memory_fits",
    "model_bytes_gb",
    "platform_summary",
    "save_state",
]
