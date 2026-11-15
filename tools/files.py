"""File operations tool"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class FileResult:
    path: str
    success: bool
    content: str = ""
    error: str = ""

def read_file(path: str) -> FileResult:
    try:
        content = Path(path).read_text(encoding='utf-8')
        return FileResult(path, True, content)
    except Exception as e:
        return FileResult(path, False, error=str(e))

def write_file(path: str, content: str) -> FileResult:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding='utf-8')
        return FileResult(path, True, content)
    except Exception as e:
        return FileResult(path, False, error=str(e))

def list_files(path: str, pattern: str = "*") -> FileResult:
    try:
        files = [str(p.relative_to(path)) for p in Path(path).rglob(pattern) if p.is_file()]
        return FileResult(path, True, "\n".join(files))
    except Exception as e:
        return FileResult(path, False, error=str(e))
