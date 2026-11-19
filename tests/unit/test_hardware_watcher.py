"""Tests for the hardware load watcher (S9.2)."""

import asyncio
from unittest.mock import MagicMock, patch

import psutil
import pytest

from agan.hardware.watcher import HardwareWatcher, WatcherConfig


class _Proc:
    def __init__(self, name: str):
        self.info = {"name": name}


def _make_mem(available_gb: float) -> MagicMock:
    m = MagicMock()
    m.available = int(available_gb * 1e9)
    return m


@pytest.mark.asyncio
async def test_watcher_acts_after_heavy_threshold():
    """After 3 consecutive heavy checks the watcher lowers priority."""
    notified: list[str] = []
    config = WatcherConfig(
        check_interval=0.01,
        heavy_threshold_seconds=3,
        calm_threshold_seconds=30,
        on_status=notified.append,
    )
    watcher = HardwareWatcher(config)

    heavy_proc = _Proc("google-chrome")
    with patch.object(
        psutil, "process_iter", return_value=iter([heavy_proc])
    ), patch.object(watcher, "_is_heavy", return_value=True), patch.object(
        psutil.Process, "nice", return_value=0
    ) as mock_nice:
        watcher._task = asyncio.create_task(watcher._loop())
        # wait long enough for 3 heavy checks + act
        await asyncio.sleep(0.05)
        watcher._task.cancel()
        try:
            await watcher._task
        except asyncio.CancelledError:
            pass

    assert watcher._active
    assert mock_nice.call_count >= 1  # priority lowered
    assert any("High load detected" in n for n in notified)


@pytest.mark.asyncio
async def test_watcher_restores_after_calm():
    """After calm threshold the watcher restores priority."""
    notified: list[str] = []
    config = WatcherConfig(
        check_interval=0.01,
        heavy_threshold_seconds=1,
        calm_threshold_seconds=2,
        on_status=notified.append,
    )
    watcher = HardwareWatcher(config)

    calls = {"n": 0}

    def heavy_seq() -> bool:
        calls["n"] += 1
        return calls["n"] == 1  # True once, then False forever

    with patch.object(watcher, "_is_heavy", side_effect=heavy_seq), patch.object(
        psutil.Process, "nice", return_value=0
    ):
        watcher._task = asyncio.create_task(watcher._loop())
        await asyncio.sleep(0.08)
        watcher._task.cancel()
        try:
            await watcher._task
        except asyncio.CancelledError:
            pass

    assert not watcher._active
    assert any("Load calmed" in n for n in notified)


@pytest.mark.asyncio
async def test_watcher_low_ram_is_heavy():
    """Low free RAM counts as heavy load."""
    watcher = HardwareWatcher(
        WatcherConfig(low_ram_free_gb=2.0, on_status=lambda _: None)
    )
    with patch.object(
        psutil, "virtual_memory", return_value=_make_mem(0.5)
    ), patch.object(psutil, "process_iter", return_value=iter([])):
        assert watcher._is_heavy() is True


@pytest.mark.asyncio
async def test_watcher_no_heavy_app_no_low_ram_is_calm():
    watcher = HardwareWatcher(
        WatcherConfig(low_ram_free_gb=0.1, on_status=lambda _: None)
    )
    with patch.object(
        psutil, "virtual_memory", return_value=_make_mem(16.0)
    ), patch.object(psutil, "process_iter", return_value=iter([])):
        assert watcher._is_heavy() is False


@pytest.mark.asyncio
async def test_watcher_start_stop_idempotent():
    watcher = HardwareWatcher(WatcherConfig(on_status=lambda _: None))
    await watcher.start()
    await watcher.start()  # no-op: already running
    assert watcher._task is not None
    await watcher.stop()
    await watcher.stop()  # no-op: already stopped
    assert watcher._task is None
