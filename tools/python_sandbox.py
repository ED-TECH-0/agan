"""Python sandbox tool"""
from dataclasses import dataclass
import subprocess
import tempfile

@dataclass
class PythonResult:
    code: str
    stdout: str
    stderr: str
    returncode: int

def run_python(code: str, timeout: int = 30) -> PythonResult:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(['python', fname], capture_output=True, text=True, timeout=timeout)
        return PythonResult(code, result.stdout, result.stderr, result.returncode)
    except subprocess.TimeoutExpired:
        return PythonResult(code, "", "Timeout", -1)
    except Exception as e:
        return PythonResult(code, "", str(e), -1)
    finally:
        import os
        try:
            os.unlink(fname)
        except Exception:
            pass
