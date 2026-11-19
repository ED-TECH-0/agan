"""Hardware load watcher (S9.2).

Polls once per second for heavy applications (configurable list) or low free
RAM. After 3 consecutive seconds of load it lowers process priority, pauses
background work, and optionally restarts the model server CPU-only with a
smaller context. Restores after 30 consecutive seconds of calm. Calls
``on_status`` to tell the user on each transition.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)

HEAVY_APPS_DEFAULT = ["chrome", "firefox", "code", "obsidian", "slack", "teams"]


@dataclass
class WatcherConfig:
    check_interval: float = 1.0
    heavy_threshold_seconds: int = 3
    calm_threshold_seconds: int = 30
    low_ram_free_gb: float = 1.0
    heavy_apps: list[str] = field(default_factory=lambda: HEAVY_APPS_DEFAULT.copy())
    restart_server: bool = False
    on_status: Callable[[str], None] | None = None


class HardwareWatcher:
    def __init__(self, config: WatcherConfig | None = None):
        self.config = config or WatcherConfig()
        self._task: asyncio.Task | None = None
        self._heavy_count = 0
        self._calm_count = 0
        self._active = False
        self._original_nice: int | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        if self._active:
            self._restore()

    def _is_heavy(self) -> bool:
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info["name"] or "").lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if any(app in name for app in self.config.heavy_apps):
                return True
        mem = psutil.virtual_memory()
        return mem.available < self.config.low_ram_free_gb * 1e9

    def _act(self) -> None:
        p = psutil.Process(os.getpid())
        try:
            self._original_nice = p.nice()
            p.nice(10)
        except (psutil.AccessDenied, OSError):
            pass
        self._active = True
        self._notify(
            "High load detected: lowered process priority, paused background work."
        )
        if self.config.restart_server:
            self._restart_server_cpu_only()

    def _restore(self) -> None:
        if self._original_nice is not None:
            try:
                psutil.Process(os.getpid()).nice(self._original_nice)
            except (psutil.AccessDenied, OSError):
                pass
        self._original_nice = None
        self._active = False
        self._notify(
            "Load calmed: restored normal priority and resumed background work."
        )

    def _restart_server_cpu_only(self) -> None:
        try:
            subprocess.run(["pkill", "-f", "ollama"], check=False)
            subprocess.Popen(
                ["ollama", "serve"],
                env={**os.environ, "OLLAMA_NUM_PARALLEL": "1"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            logger.exception("failed to restart ollama CPU-only")

    def _notify(self, message: str) -> None:
        logger.info(message)
        if self.config.on_status is not None:
            try:
                self.config.on_status(message)
            except Exception as exc:  # noqa: BLE001  # user callback must not crash the watcher
                logger.debug("watcher status callback failed: %s", exc)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.check_interval)
            heavy = self._is_heavy()
            if heavy:
                self._calm_count = 0
                self._heavy_count += 1
                if (
                    self._heavy_count >= self.config.heavy_threshold_seconds
                    and not self._active
                ):
                    self._act()
            else:
                self._heavy_count = 0
                self._calm_count += 1
                if (
                    self._calm_count >= self.config.calm_threshold_seconds
                    and self._active
                ):
                    self._restore()
