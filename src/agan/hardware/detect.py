from dataclasses import dataclass
from enum import Enum
import platform
import subprocess
import psutil


class HardwareProfile(str, Enum):
    S = "S"
    M = "M"
    L = "L"


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


SPECS = {
    HardwareProfile.S: ProfileSpec(
        name=HardwareProfile.S,
        ram_gb_min=0,
        ram_gb_max=8,
        gpu_required=False,
        default_model="qwen2.5:3b-instruct-q4_k_m",
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
        default_model="qwen2.5:7b-instruct-q4_k_m",
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
        default_model="qwen2.5:14b-instruct-q3_k_m",
        quantization="Q3_K_M",
        time_budget=60,
        best_of_n=4,
        max_context=16384,
    ),
}


def get_ram_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)


def get_gpu_info() -> tuple[bool, str | None]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            lines = out.stdout.strip().split("\n")
            name = lines[0].split(",")[0].strip()
            return True, name
    except Exception:
        pass
    return False, None


def detect_profile() -> HardwareProfile:
    ram = get_ram_gb()
    has_gpu, gpu_name = get_gpu_info()

    if ram >= 24 and has_gpu:
        return HardwareProfile.L
    elif ram >= 12:
        return HardwareProfile.M
    else:
        return HardwareProfile.S


def get_spec(profile: HardwareProfile) -> ProfileSpec:
    return SPECS[profile]


__all__ = ["HardwareProfile", "ProfileSpec", "detect_profile", "get_spec", "get_ram_gb", "get_gpu_info"]