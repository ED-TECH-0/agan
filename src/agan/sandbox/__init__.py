"""AGAN sandbox: isolated code execution for untrusted scripts."""

from .runner import SandboxRunner, SandboxResult

__all__ = ["SandboxRunner", "SandboxResult"]
