"""Shell command tool"""
from dataclasses import dataclass
import subprocess

@dataclass
class ShellResult:
    command: str
    stdout: str
    stderr: str
    returncode: int

def run_command(cmd: str, timeout: int = 30, allowlist: list[str] = None) -> ShellResult:
    # Check allowlist for destructive commands
    if allowlist and not any(cmd.startswith(a) for a in allowlist):
        return ShellResult(cmd, "", "Command not in allowlist", -1)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return ShellResult(cmd, result.stdout, result.stderr, result.returncode)
    except subprocess.TimeoutExpired:
        return ShellResult(cmd, "", "Timeout", -1)
    except Exception as e:
        return ShellResult(cmd, "", str(e), -1)
