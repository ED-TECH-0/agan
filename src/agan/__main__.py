#!/usr/bin/env python3
"""AGAN — Local AI that researches, verifies, learns skills, and adapts."""
import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console

from agan.hardware.detect import HardwareProfile, detect_profile, get_spec
from agan.llm.client import LLMClient
from agan.pipeline.router import Router
from agan.pipeline.planner import Planner
from agan.pipeline.reasoner import Reasoner
from agan.pipeline.verifier import Verifier, VerifierReport
from agan.pipeline.writer import Writer
from agan.privacy.firewall import OutboundGateway
from agan.sandbox import SandboxRunner
from agan.schemas import Answer, Plan, Route

app = typer.Typer(add_completion=False)
console = Console()


class AGAN:
    def __init__(self, profile: HardwareProfile | None = None, model: str | None = None):
        self.profile = profile or detect_profile()
        self.model = model or self.profile.default_model
        self.client = LLMClient(base_url="http://localhost:11434/v1", model=self.model)
        self.router = Router(self.client)
        self.planner = Planner(self.client)
        self.reasoner = Reasoner(self.client, get_spec(self.profile))
        self.verifier = Verifier(sandbox=SandboxRunner(), profile=get_spec(self.profile))
        self.gateway = OutboundGateway()
        self.writer = Writer(get_spec(self.profile))

    async def answer(self, question: str) -> Answer:
        # Privacy gateway check
        if not self.gateway.check(question):
            return Answer(
                text="I cannot answer that due to privacy restrictions.",
                claims=[],
                sources=[],
                confidence="low",
                notes=["blocked by privacy gateway"],
                level_used=int(get_spec(self.profile).level),
            )

        route = self.router.route(question)
        console.print(f"[dim]Route: {route.value}[/dim]")

        if route == Route.direct:
            plan = Plan(needs_search=False, steps=[], answer_type="chat")
        else:
            plan = await self.planner.plan(question, route)
            console.print(f"[dim]Plan: {len(plan.steps)} steps[/dim]")

        draft = await self.reasoner.execute(question, plan)
        report = await self.verifier.verify(draft.text, plan)
        final = self.reasoner.finalize(draft, report)
        final = self.writer.write(final, report, plan)
        return final



def _run_chat(model: str | None = None, profile: str | None = None):
    """Sync wrapper for chat command."""
    import asyncio
    
    async def _chat():
        hw = HardwareProfile[profile.upper()] if profile else None
        agan = AGAN(profile=hw, model=model)
        console.print(f"[green]AGAN ready[/green] — model: {agan.model}, profile: {agan.profile.name}")
        while True:
            try:
                q = console.input("[bold cyan]You:[/bold cyan] ").strip()
                if not q or q.lower() in {"exit", "quit"}:
                    break
                ans = await agan.answer(q)
                console.print(f"[bold green]AGAN:[/bold green] {ans.text}")
                if ans.sources:
                    console.print("[dim]Sources: " + ", ".join(s.get("id", "?") for s in ans.sources) + "[/dim]")
            except (KeyboardInterrupt, EOFError):
                break
    
    asyncio.run(_chat())


@app.command(name="chat")
def chat_cmd(
    model: str | None = typer.Option(None, "--model", help="Model name (default: profile default)"),
    profile: str | None = typer.Option(None, "--profile", help="Hardware profile: S, M, L"),
):
    """Interactive chat."""
    _run_chat(model=model, profile=profile)


@app.command()
def doctor():
    """System health check."""
    from agan.hardware.detect import detect_profile, get_spec, get_ram_gb, get_gpu_info
    prof = detect_profile()
    spec = get_spec(prof)
    ram = get_ram_gb()
    has_gpu, gpu_name = get_gpu_info()
    console.print(f"Profile: {prof.name}")
    console.print(f"RAM: {ram:.1f} GB")
    console.print(f"GPU: {gpu_name or 'none'}")
    console.print(f"Default model: {spec.default_model}")
    console.print(f"Time budget: {spec.time_budget}s")
    console.print(f"Best-of-N: {spec.best_of_n}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8080,
    model: str = typer.Option(None, "--model"),
):
    """OpenAI-compatible API server."""
    import uvicorn
    from agan.server.api import create_app
    profile = detect_profile()
    agan = AGAN(profile=profile, model=model)
    app = create_app(agan)
    uvicorn.run(app, host=host, port=port)


def main():
    app()


if __name__ == "__main__":
    main()