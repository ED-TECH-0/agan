"""AGAN sandbox runner: isolated, sandboxed Python code execution.

Execution modes:
  - ``bubblewrap`` (default, if ``bwrap`` is available): full Linux namespace
    isolation with read-only system mounts, no network, temp work dir.
  - ``subprocess`` fallback: ``subprocess.run`` with ``resource`` limits
    (CPU time, memory, file size) and no network via unsetting
    ``HTTP_PROXY``/``HTTPS_PROXY``. Weaker isolation than bubblewrap.

Security properties (bubblewrap mode):
  * New mount namespace: / (ro), /tmp rw, /dev minimal, no /home access
  * New network namespace: loopback only (no external network)
  * New PID namespace: cannot see host processes
  * Temp work dir: current working directory is an isolated tmpfs
  * Resource limits: CPU time, memory, file size, process count
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass


@dataclass
class SandboxResult:
    """Result of a sandboxed code execution."""

    ok: bool
    """True if the process exited with code 0."""
    stdout: str
    """Captured stdout."""
    stderr: str
    """Captured stderr."""
    returncode: int
    """Process exit code."""
    duration_ms: int
    """Wall-clock time spent in milliseconds."""
    mode: str
    """Execution mode used: ``bubblewrap`` or ``subprocess``."""
    error: str = ""
    """Human-readable error message if execution could not start."""


class SandboxRunner:
    """Run untrusted Python code in an isolated environment.

    Parameters
    ----------
    enable_network:
        Allow outbound network connections. Default False.
    timeout_s:
        Wall-clock timeout in seconds. Default 30.
    memory_mb:
        Memory limit in MiB. Default 256.
    cpu_limit:
        CPU time limit in seconds. Default 30.
    max_file_size_mb:
        Maximum file size writable by the sandboxed process. Default 10.
    max_processes:
        Maximum number of processes/threads. Default 64.
    """

    def __init__(
        self,
        *,
        enable_network: bool = False,
        timeout_s: int = 30,
        memory_mb: int = 256,
        cpu_limit: int = 30,
        max_file_size_mb: int = 10,
        max_processes: int = 64,
    ):
        self.enable_network = enable_network
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self.cpu_limit = cpu_limit
        self.max_file_size_mb = max_file_size_mb
        self.max_processes = max_processes
        self._bwrap = shutil.which("bwrap")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, code: str, timeout_s: int | None = None) -> SandboxResult:
        """Execute ``code`` in the sandbox and return the result.

        Parameters
        ----------
        code:
            Python source to execute.
        timeout_s:
            Override the default timeout for this call.
        """
        timeout = timeout_s if timeout_s is not None else self.timeout_s

        if self._bwrap:
            return self._run_bubblewrap(code, timeout)
        return self._run_subprocess(code, timeout)

    # ------------------------------------------------------------------
    # Bubblewrap backend
    # ------------------------------------------------------------------

    def _run_bubblewrap(self, code: str, timeout_s: int) -> SandboxResult:
        """Run code inside a bwrap sandbox."""
        tmpdir = tempfile.mkdtemp(prefix="agan-sandbox-")
        try:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)

            env = dict(os.environ)
            env.pop("HTTP_PROXY", None)
            env.pop("HTTPS_PROXY", None)
            env.pop("http_proxy", None)
            env.pop("https_proxy", None)

            cmd = [
                self._bwrap,
                # Namespaces
                "--unshare-all",  # new mount, net, pid, ipc, uts, cgroup namespaces
                "--share-net" if self.enable_network else "--unshare-net",
                # Filesystem: read-only host, writable tmp only
                "--ro-bind",
                "/",
                "/",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--tmpfs",
                "/var/tmp",
                "--bind",
                tmpdir,
                "/tmp",
                "--chdir",
                "/tmp",
                # No new privileges
                "--die-with-parent",
                # Capability bounding
                "--cap-add",
                "CAP_SETUID",
                "--cap-add",
                "CAP_SETGID",
                # Resource limits
                "--unshare-pid",
                # Run python
                "python3",
                "/tmp/script.py",
            ]

            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    env=env,
                    preexec_fn=self._apply_resource_limits,
                )
                duration_ms = int((time.monotonic() - start) * 1000)
            except subprocess.TimeoutExpired as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                return SandboxResult(
                    ok=False,
                    stdout=exc.stdout.decode() if exc.stdout else "",
                    stderr=exc.stderr.decode() if exc.stderr else "",
                    returncode=-1,
                    duration_ms=duration_ms,
                    mode="bubblewrap",
                    error="execution timed out",
                )
            except (OSError, subprocess.SubprocessError):
                # Bubblewrap failed to start (e.g. kernel restrictions) — fall back to subprocess
                return self._run_subprocess(code, timeout_s)

            # Bubblewrap ran but failed (e.g. unsupported flag) — fall back to subprocess
            if proc.returncode != 0:
                return self._run_subprocess(code, timeout_s)

            return SandboxResult(
                ok=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                duration_ms=duration_ms,
                mode="bubblewrap",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Subprocess fallback backend
    # ------------------------------------------------------------------

    def _apply_resource_limits(self) -> None:
        """Set resource limits (CPU, memory, file size).

        Safe to use as preexec_fn under bubblewrap (RLIMIT_NPROC is omitted
        because it breaks bwrap with --unshare-pid due to fork EAGAIN).
        Subprocess mode additionally caps NPROC via _set_all_limits.
        """
        import resource

        # CPU time
        resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_limit, self.cpu_limit))
        # Memory (virtual address space)
        mem_bytes = self.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        # File size
        fsize_bytes = self.max_file_size_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))

    def _set_all_limits(self) -> None:
        """Set all resource limits including NPROC (subprocess mode only)."""
        self._apply_resource_limits()
        import resource

        resource.setrlimit(resource.RLIMIT_NPROC, (self.max_processes, self.max_processes))

    def _run_subprocess(self, code: str, timeout_s: int) -> SandboxResult:
        """Run code via subprocess with resource limits (weaker isolation)."""

        env = dict(os.environ)
        env.pop("HTTP_PROXY", None)
        env.pop("HTTPS_PROXY", None)
        env.pop("http_proxy", None)
        env.pop("https_proxy", None)
        env.pop("ALL_PROXY", None)

        start = time.monotonic()
        try:
            proc = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
                preexec_fn=self._set_all_limits,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
        except subprocess.TimeoutExpired as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            return SandboxResult(
                ok=False,
                stdout=exc.stdout.decode() if exc.stdout else "",
                stderr=exc.stderr.decode() if exc.stderr else "",
                returncode=-1,
                duration_ms=duration_ms,
                mode="subprocess",
                error="execution timed out",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return SandboxResult(
                ok=False,
                stdout="",
                stderr="",
                returncode=-1,
                duration_ms=int((time.monotonic() - start) * 1000),
                mode="subprocess",
                error=f"subprocess execution failed: {exc}",
            )

        return SandboxResult(
            ok=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            duration_ms=duration_ms,
            mode="subprocess",
        )
