"""
Resource Governor — monitors hardware, selects profile, yields to heavy apps.
"""
from dataclasses import dataclass
from enum import Enum
import yaml
import psutil
import platform
from pathlib import Path
from typing import Optional

class ProfileName(str, Enum):
    NANO = "nano"
    LITE = "lite"
    STANDARD = "standard"
    PRO = "pro"
    MAX = "max"

HEAVY_APPS = {"blender.exe", "blender", "UnrealEditor.exe", "UnrealEditor", "UE4Editor.exe", "UE5Editor.exe", "vivado.exe", "quartus.exe", "modelSim.exe", "iverilog.exe", "verilator.exe"}

@dataclass
class HardwareInfo:
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpu_name: Optional[str]
    vram_gb: Optional[float]
    disk_free_gb: float
    os: str
    arch: str

@dataclass
class ProfileConfig:
    name: ProfileName
    core_model_size_range: str
    quantization: str
    max_adapters: int
    embedding_model: str
    reranker: str
    max_chunks: int
    cache_size_mb: int
    max_tokens: int
    verification_loops: int
    modes: list[str]
    vram_limit_mb: int
    ram_limit_mb: int
    yield_on_heavy_apps: bool
    boost_when_idle: bool

class ResourceGovernor:
    def __init__(self, profiles_dir: Path = Path("profiles")):
        self.profiles_dir = profiles_dir
        self._profile_cache: Optional[ProfileConfig] = None
        self._hardware: Optional[HardwareInfo] = None
    
    def detect_hardware(self) -> HardwareInfo:
        if self._hardware:
            return self._hardware
        
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        # GPU detection (simplified; expand with pynvml/rocm-smi)
        gpu_name = None
        vram_gb = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_name = gpus[0].name
                vram_gb = gpus[0].memoryTotal / 1024
        except Exception:
            pass
        
        self._hardware = HardwareInfo(
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            cpu_threads=psutil.cpu_count(logical=True) or 1,
            ram_gb=mem.total / (1024**3),
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            disk_free_gb=disk.free / (1024**3),
            os=platform.system(),
            arch=platform.machine(),
        )
        return self._hardware
    
    def recommend_profile(self) -> ProfileName:
        hw = self.detect_hardware()
        
        if hw.vram_gb and hw.vram_gb >= 24:
            return ProfileName.MAX
        if hw.vram_gb and hw.vram_gb >= 12:
            return ProfileName.PRO
        if hw.vram_gb and hw.vram_gb >= 6:
            return ProfileName.STANDARD
        if hw.ram_gb >= 16:
            return ProfileName.STANDARD
        if hw.ram_gb >= 8:
            return ProfileName.LITE
        return ProfileName.NANO
    
    def load_profile(self, name: ProfileName) -> ProfileConfig:
        path = self.profiles_dir / f"{name.value}.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        return ProfileConfig(
            name=ProfileName(data["profile"]),
            core_model_size_range=data["core_model"]["size_range"],
            quantization=data["core_model"]["quantization"],
            max_adapters=data["core_model"]["max_adapters"],
            embedding_model=data["retrieval"]["embedding_model"],
            reranker=data["retrieval"]["reranker"],
            max_chunks=data["retrieval"]["max_chunks"],
            cache_size_mb=data["retrieval"]["cache_size_mb"],
            max_tokens=data["context"]["max_tokens"],
            verification_loops=data["context"]["verification_loops"],
            modes=data["modes"],
            vram_limit_mb=data["governor"]["vram_limit_mb"],
            ram_limit_mb=data["governor"]["ram_limit_mb"],
            yield_on_heavy_apps=data["governor"]["yield_on_heavy_apps"],
            boost_when_idle=data["governor"]["boost_when_idle"],
        )
    
    def current_profile(self) -> ProfileConfig:
        if self._profile_cache is None:
            recommended = self.recommend_profile()
            self._profile_cache = self.load_profile(recommended)
        return self._profile_cache
    
    def heavy_app_running(self) -> bool:
        for proc in psutil.process_iter(["name"]):
            try:
                if proc.info["name"] and proc.info["name"].lower() in HEAVY_APPS:
                    return True
            except Exception:
                continue
        return False
    
    def should_yield(self) -> bool:
        profile = self.current_profile()
        return profile.yield_on_heavy_apps and self.heavy_app_running()
    
    def get_resource_limits(self) -> dict:
        profile = self.current_profile()
        return {
            "vram_limit_mb": profile.vram_limit_mb,
            "ram_limit_mb": profile.ram_limit_mb,
            "max_tokens": profile.max_tokens,
            "verification_loops": profile.verification_loops,
            "yield_mode": self.should_yield(),
        }

def main():
    gov = ResourceGovernor()
    hw = gov.detect_hardware()
    profile = gov.recommend_profile()
    print(f"Hardware: {hw.cpu_threads} threads, {hw.ram_gb:.1f} GB RAM, GPU: {hw.gpu_name or 'none'} ({hw.vram_gb:.1f} GB VRAM)" if hw.vram_gb else f"Hardware: {hw.cpu_threads} threads, {hw.ram_gb:.1f} GB RAM, GPU: none")
    print(f"Recommended profile: {profile.value}")
    cfg = gov.load_profile(profile)
    print(f"  Core model: {cfg.core_model_size_range} ({cfg.quantization})")
    print(f"  Max adapters: {cfg.max_adapters}")
    print(f"  Modes: {cfg.modes}")
    print(f"  Limits: VRAM={cfg.vram_limit_mb}MB, RAM={cfg.ram_limit_mb}MB")

if __name__ == "__main__":
    main()
