"""Tests for hardware profile detection (S9.2).

All sensor reads are mocked so these run deterministically on any machine and
in CI (no GPU, no Ollama server required).
"""

import asyncio
import builtins
from io import StringIO

import pytest

import agan.hardware.detect as d
from agan.hardware.detect import HardwareProfile


class _FakeClient:
    """Fake LLM client whose chat() returns after a fixed delay."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        await asyncio.sleep(self.seconds)
        return "x" * 128


def _patch_sensors(monkeypatch, ram_gb, has_gpu, gpu_name="NVIDIA GPU",
                   free_vram=8.0, disk=100.0, cpu=None, battery=None):
    monkeypatch.setattr(d, "get_ram_gb", lambda: ram_gb)
    monkeypatch.setattr(d, "get_ram_free_gb", lambda: ram_gb * 0.9)
    monkeypatch.setattr(d, "get_gpu_info", lambda: (has_gpu, gpu_name))
    monkeypatch.setattr(d, "get_gpu_free_vram_gb", lambda: free_vram)
    monkeypatch.setattr(d, "get_disk_free_gb", lambda: disk)
    monkeypatch.setattr(d, "get_battery_percent", lambda: battery)
    monkeypatch.setattr(d, "cpu_flags", lambda: cpu or [])


async def _fixed_tps(value: float):
    return value


def _patch_benchmark(monkeypatch, value: float):
    monkeypatch.setattr(
        d, "benchmark_tokens_per_sec",
        lambda client=None, **kw: _fixed_tps(value),
    )


def test_cpu_flags_subset():
    flags = d.cpu_flags()
    allowed = {
        "avx2", "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
        "avx512ifma", "avx512vbmi", "neon", "asimd",
    }
    assert isinstance(flags, list)
    assert set(flags) <= allowed


def test_cpu_flags_detects_avx2(monkeypatch):
    fake = StringIO("flags: fpu vme de avx2 avx512f tsx\n")
    orig_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/cpuinfo":
            return fake
        return orig_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    assert d.cpu_flags() == ["avx2", "avx512f"]


def test_memory_fits_small():
    assert d.memory_fits(params_b=3.0, layers=32, kv_heads=32, head_dim=128,
                         ctx=4096, available_gb=8.0) is True


def test_memory_fits_huge_model():
    assert d.memory_fits(params_b=70.0, layers=80, kv_heads=64, head_dim=128,
                         ctx=16384, available_gb=24.0) is False


@pytest.mark.asyncio
async def test_benchmark_without_client_is_none():
    assert await d.benchmark_tokens_per_sec() is None


@pytest.mark.asyncio
async def test_benchmark_measures_tokens_per_sec():
    client = _FakeClient(seconds=0.05)  # ~128/0.05 = 2560 tok/s per call
    tps = await d.benchmark_tokens_per_sec(client, n_tokens=128, repeats=1)
    assert tps is not None and tps > 100
    assert client.calls == 1


def test_base_profile_tiers():
    assert d.base_profile(ram_gb=32, has_gpu=True) == HardwareProfile.L
    assert d.base_profile(ram_gb=32, has_gpu=False) == HardwareProfile.M
    assert d.base_profile(ram_gb=14, has_gpu=False) == HardwareProfile.M
    assert d.base_profile(ram_gb=12, has_gpu=False) == HardwareProfile.M
    assert d.base_profile(ram_gb=7, has_gpu=True) == HardwareProfile.S


@pytest.mark.asyncio
async def test_choose_profile_l_m_s_machines(monkeypatch, tmp_path):
    # L machine: 32GB + GPU
    _patch_sensors(monkeypatch, ram_gb=32, has_gpu=True)
    _patch_benchmark(monkeypatch, 12.0)
    result = await d.choose_profile(state_path=tmp_path / "state.json")
    assert result["profile"] == HardwareProfile.L
    assert result["spec"].name == HardwareProfile.L
    assert result["spec"].max_context == 16384
    assert "fingerprint" in result and len(result["fingerprint"]) == 16

    # M machine: 14GB, no GPU
    _patch_sensors(monkeypatch, ram_gb=14, has_gpu=False)
    _patch_benchmark(monkeypatch, 10.0)
    result = await d.choose_profile(state_path=tmp_path / "state_m.json")
    assert result["profile"] == HardwareProfile.M
    assert result["spec"].max_context == 8192

    # S machine: 7GB, no GPU
    _patch_sensors(monkeypatch, ram_gb=7, has_gpu=False)
    _patch_benchmark(monkeypatch, 4.0)
    result = await d.choose_profile(state_path=tmp_path / "state_s.json")
    assert result["profile"] == HardwareProfile.S
    assert result["spec"].max_context == 4096


@pytest.mark.asyncio
async def test_memory_fit_steps_down(monkeypatch, tmp_path):
    # L machine (32GB+GPU) but a 70B model does not fit in 70% of RAM.
    _patch_sensors(monkeypatch, ram_gb=32, has_gpu=True)
    _patch_benchmark(monkeypatch, 12.0)
    result = await d.choose_profile(
        params_b=70.0, layers=80, kv_heads=64, head_dim=128,
        state_path=tmp_path / "state_fit.json",
    )
    # 70B Q4_K_M ~ 45GB > 0.7*32GB = 22.4GB -> steps L -> M -> S.
    assert result["profile"] == HardwareProfile.S
    assert "stepped down" in result["reason"]


@pytest.mark.asyncio
async def test_benchmark_steps_down(monkeypatch, tmp_path):
    # M machine but generation is slower than M's 6 tok/s floor.
    _patch_sensors(monkeypatch, ram_gb=14, has_gpu=False)
    _patch_benchmark(monkeypatch, 2.0)
    result = await d.choose_profile(state_path=tmp_path / "state_bench.json")
    assert result["profile"] == HardwareProfile.S
    assert "stepped down" in result["reason"]
    assert result["sensors"]["benchmark_tok_per_s"] == 2.0


@pytest.mark.asyncio
async def test_fingerprint_reuse_skips_recompute(monkeypatch, tmp_path):
    _patch_sensors(monkeypatch, ram_gb=14, has_gpu=False)
    benchmark_calls = {"n": 0}

    async def fake_benchmark(client=None, **kw):
        benchmark_calls["n"] += 1
        return 10.0

    monkeypatch.setattr(d, "benchmark_tokens_per_sec", fake_benchmark)
    state_path = tmp_path / "state_reuse.json"

    r1 = await d.choose_profile(state_path=state_path)
    assert r1["profile"] == HardwareProfile.M
    assert benchmark_calls["n"] == 1

    # Same sensors -> same fingerprint -> reused, no recompute.
    r2 = await d.choose_profile(state_path=state_path)
    assert r2["profile"] == HardwareProfile.M
    assert r2["reason"] == "reused (fingerprint unchanged)"
    assert benchmark_calls["n"] == 1  # benchmark not re-run


@pytest.mark.asyncio
async def test_fingerprint_changes_trigger_redo(monkeypatch, tmp_path):
    _patch_sensors(monkeypatch, ram_gb=14, has_gpu=False)
    _patch_benchmark(monkeypatch, 10.0)
    state_path = tmp_path / "state_change.json"

    r1 = await d.choose_profile(state_path=state_path)
    assert r1["profile"] == HardwareProfile.M

    # Different RAM -> different fingerprint -> recompute -> S.
    _patch_sensors(monkeypatch, ram_gb=7, has_gpu=False)
    _patch_benchmark(monkeypatch, 10.0)
    r2 = await d.choose_profile(state_path=state_path)
    assert r2["profile"] == HardwareProfile.S
    assert r2["fingerprint"] != r1["fingerprint"]


@pytest.mark.asyncio
async def test_choose_profile_uses_passed_benchmark(monkeypatch, tmp_path):
    _patch_sensors(monkeypatch, ram_gb=32, has_gpu=True)
    # L base, but passed benchmark 2.0 tok/s < L's 8 floor -> step down once
    # to M (bench step-down is a single step, not a loop).
    result = await d.choose_profile(
        state_path=tmp_path / "state_passed.json",
        benchmark_tok_per_s=2.0,
    )
    assert result["profile"] == HardwareProfile.M


def test_profile_spec_levels():
    assert d.get_spec(HardwareProfile.S).level == 1
    assert d.get_spec(HardwareProfile.M).level == 1
    assert d.get_spec(HardwareProfile.L).level == 1
    assert set(d.SPECS) == {HardwareProfile.S, HardwareProfile.M, HardwareProfile.L}
