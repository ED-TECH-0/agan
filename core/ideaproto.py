"""Idea-to-Prototype Pipeline — automated maker task scaffolding.

Given a user's idea/goal, this module generates a project scaffold:
- Directory structure
- Starter skill sequences (from Playbook Memory)
- Configuration files
- Minimal working entry points

Design:
- Idea description → pattern-matched playbook sequences → scaffold generation
- Supports: Python projects, CLI tools, notebooks, widget sets
- Output: zip or directory with __init__.py, main entry, config, playbook refs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json
import shutil
import tempfile
from datetime import datetime


@dataclass
class PrototypeSpec:
    """Description of what to build."""
    idea: str
    project_type: str  # "python_cli", "notebook", "widget", "web"
    domain: str  # e.g. "python", "data-science", "devops", "creative"
    trigger_keywords: list[str]  # used for playbook matching
    required_skills: list[str] = field(default_factory=list)
    output_format: str = "directory"  # "directory" or "zip"


@dataclass
class ScaffoldStep:
    """One step in scaffold generation."""
    name: str
    description: str
    kind: str  # "directory", "file", "playbook_ref", "config"
    path: Path
    content: str = ""


@dataclass
class PrototypeResult:
    """Result of prototype generation."""
    spec: PrototypeSpec
    steps: list[ScaffoldStep]
    generated_at: datetime
    playbooks_used: list[str] = field(default_factory=list)
    success: bool = True
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Idea: {self.spec.idea[:80]}{'...' if len(self.spec.idea) > 80 else ''}",
            f"Type: {self.spec.project_type}",
            f"Playbooks used: {len(self.playbooks_used)}",
            f"Steps: {len(self.steps)}",
            f"Success: {self.success}",
        ]
        if self.errors:
            lines.append(f"Errors: {', '.join(self.errors[:3])}")
        return "\n".join(lines)


class IdeaToProtoPipeline:
    """Core pipeline: idea → scaffold → playbook-integrated prototype."""

    def __init__(self, playbook_db: "PlaybookDB | None" = None) -> None:
        self.playbook_db = playbook_db
        self._step_counter = 0

    def _match_playbooks(self, keywords: list[str]) -> list[str]:
        """Match trigger keywords against registered playbooks."""
        if not self.playbook_db:
            return []
        matches = self.playbook_db.find_by_keywords(keywords)
        used = [e.pattern_name for e in matches[:3]]  # top 3
        return used

    def _gen_steps(self, spec: PrototypeSpec) -> list[ScaffoldStep]:
        """Generate scaffold steps from the spec."""
        steps: list[ScaffoldStep] = []
        self._step_counter = 0

        # 1. Project root directory
        self._step_counter += 1
        root_name = (
            spec.idea.lower().split()[0] if spec.idea else "project"
        ).replace(" ", "_").replace("-", "_")[:30] or "proto"
        root_path = Path(root_name)
        steps.append(
            ScaffoldStep(
                name="create_root",
                description=f"Create project root: {root_name}",
                kind="directory",
                path=root_path,
            )
        )

        # 2. __init__.py
        self._step_counter += 1
        steps.append(
            ScaffoldStep(
                name="init_py",
                description="Add __init__.py",
                kind="file",
                path=root_path / "__init__.py",
                content=f'"""Auto-generated prototype: {spec.idea[:60]}"""',
            )
        )

        # 3. main entry point (kind depends on project_type)
        self._step_counter += 1
        entry_name = f"main_{root_name}.py"
        if spec.project_type == "python_cli":
            content = (
                f'"""CLI entry for: {spec.idea}\n'
                f'Usage: python -m {root_name} or {root_name}.main\n"""\\n'
                f"import argparse\\n\\n"
                f"def main():\\n    '''Run the CLI tool.\\''\\n"
                f"    parser = argparse.ArgumentParser(description={repr(spec.idea)})\\n"
                f"    parser.parse_args()\\n"
                f"    print('Hello from', {repr(root_name)})\\n"
                f"if __name__ == '__main__':\\n    main()"
            )
        elif spec.project_type == "notebook":
            content = (
                f'#!/usr/bin/env python3\\n'
                f'# Idea: {spec.idea}\\n'
                f'# Type: notebook\\n'
                f'import matplotlib\\n'
                f'import numpy as np\\n'
                f"%matplotlib inline\\n\\n"
                f"# TODO: add your idea here\\n"
                f"print('Notebook scaffold ready')"
            )
        else:
            content = (
                f'# {root_name}\\n'
                f'# Idea: {spec.idea}\\n'
                f'print("Prototype ready for", {repr(spec.idea)})'
            )
        steps.append(
            ScaffoldStep(
                name="entry_point",
                description=f"Add {entry_name} entry point",
                kind="file",
                path=root_path / entry_name,
                content=content,
            )
        )

        # 4. playbook reference (if we matched any)
        self._step_counter += 1
        if self.playbook_db and spec.trigger_keywords:
            matched = self._match_playbooks(spec.trigger_keywords)
            if matched:
                steps.append(
                    ScaffoldStep(
                        name="playbook_ref",
                        description="Add playbook reference notes",
                        kind="playbook_ref",
                        path=root_path / "PLAYBOOK.md",
                        content=self._playbook_md_content(matched),
                    )
                )
                # record which playbooks were used
                # (in a full impl, we'd update the DB success scores)

        # 5. config file
        self._step_counter += 1
        steps.append(
            ScaffoldStep(
                name="config",
                description="Add minimal pyproject.toml stub",
                kind="file",
                path=root_path / "pyproject.toml",
                content=self._config_content(spec.project_type),
            )
        )

        return steps

    def _playbook_md_content(self, matched: list[str]) -> str:
        lines = [
            f"# Playbook References for: {self._truncate(self.spec.idea)}",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
        ]
        for name in matched:
            lines.append(f"- **{name}** — pattern-matched from trigger keywords")
        lines += [
            "",
            "Run `python -m agan.playbook` to view/updated scored patterns.",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _truncate(s: str, n: int = 60) -> str:
        return s[:n] + ("..." if len(s) > n else "")

    def _config_content(self, project_type: str) -> str:
        bases = {
            "python_cli": """[project]
name = "proto-{type}"
version = "0.1.0"
description = "Auto-generated prototype"

dependencies = []

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
""",
            "notebook": """[project]
name = "proto-notebook"
version = "0.1.0"
description = "Auto-generated notebook prototype"

dependencies = []

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
""",
        }
        return bases.get(
            project_type,
            """[project]
name = "proto"
version = "0.1.0"
description = "Auto-generated prototype"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
""",
        )

    def generate(self, spec: PrototypeSpec) -> PrototypeResult:
        """Generate a full prototype scaffold from an idea spec."""
        steps = self._gen_steps(spec)
        playbooks_used = self._match_playbooks(spec.trigger_keywords)

        # Execute steps (create dirs/files)
        errors: list[str] = []
        for step in steps:
            try:
                if step.kind == "directory":
                    step.path.mkdir(parents=True, exist_ok=True)
                elif step.kind == "file":
                    step.path.parent.mkdir(parents=True, exist_ok=True)
                    step.path.write_text(step.content, encoding="utf-8")
                elif step.kind == "playbook_ref":
                    step.path.parent.mkdir(parents=True, exist_ok=True)
                    step.path.write_text(step.content, encoding="utf-8")
                elif step.kind == "config":
                    step.path.parent.mkdir(parents=True, exist_ok=True)
                    step.path.write_text(step.content, encoding="utf-8")
            except Exception as e:  # broad, capture for report
                errors.append(f"{step.name}: {e}")

        success = len(errors) == 0

        result = PrototypeResult(
            spec=spec,
            steps=steps,
            generated_at=datetime.now(),
            playbooks_used=playbooks_used,
            success=success,
            errors=errors,
        )
        return result

    def run_from_idea(
        self,
        idea: str,
        project_type: str = "python_cli",
        domain: str = "general",
        trigger_keywords: list[str] | None = None,
    ) -> PrototypeResult:
        """Convenience: turn a free-text idea into a prototype result."""
        kw = trigger_keywords or self._extract_keywords(idea)
        spec = PrototypeSpec(
            idea=idea,
            project_type=project_type,
            domain=domain,
            trigger_keywords=kw,
        )
        return self.generate(spec)

    @staticmethod
    def _extract_keywords(idea: str) -> list[str]:
        """Very simple keyword extraction from idea text."""
        # Lower, strip punctuation, split on whitespace, filter short words
        cleaned = idea.lower().translate(
            str.maketrans("", "", ".,;:'\"!@#$%^&*()[]{}<>`~|\\n")
        )
        words = [w for w in cleaned.split() if len(w) >= 3]
        # Deduplicate preserving order
        seen: list[str] = []
        for w in words:
            if w not in seen:
                seen.append(w)
        return seen[:8]


def make_prototype(
    idea: str,
    project_type: str = "python_cli",
    playbook_db_path: str | None = None,
) -> PrototypeResult:
    """Factory: create a prototype from a plain idea string.

    Args:
        idea: User's goal/idea description.
        project_type: One of "python_cli", "notebook", "widget", "web".
        playbook_db_path: Optional path to PlaybookDB. If None, a temporary
            one is created (no persistence across runs).

    Returns:
        PrototypeResult with steps, playbook matches, and generation status.
    """
    from core.playbook import PlaybookDB

    db = PlaybookDB(playbook_db_path) if playbook_db_path else PlaybookDB()
    pipeline = IdeaToProtoPipeline(playbook_db=db)
    return pipeline.run_from_idea(idea, project_type=project_type)
