"""Blender bridge — headless script execution"""
from dataclasses import dataclass
import subprocess
import tempfile

@dataclass
class BlenderResult:
    script: str
    stdout: str
    stderr: str
    returncode: int
    objects_created: list[str] = None

def run_blender_script(script: str, timeout: int = 120) -> BlenderResult:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        fname = f.name
    try:
        result = subprocess.run(
            ['blender', '--background', '--python', fname],
            capture_output=True, text=True, timeout=timeout
        )
        return BlenderResult(script, result.stdout, result.stderr, result.returncode)
    except subprocess.TimeoutExpired:
        return BlenderResult(script, "", "Timeout", -1)
    except Exception as e:
        return BlenderResult(script, "", str(e), -1)
    finally:
        import os
        try:
            os.unlink(fname)
        except Exception:
            pass

def check_blender_available() -> bool:
    try:
        result = subprocess.run(['blender', '--version'], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False
