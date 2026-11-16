"""GRPO (Group Relative Policy Optimization) training script.

Trains on maker tasks with verifiable rewards (executable checks).
Each task has a verifiable answer (e.g., fibonacci(10) == 55), and the model
is rewarded for producing correct, executable output.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ADAPTERS_DIR = Path("adapters")


@dataclass
class TrainConfig:
    base_model: str = "qwen2.5:3b-instruct-q4_k_m"
    adapter_path: Path = ADAPTERS_DIR / "code-dpo"
    output_dir: Path = ADAPTERS_DIR / "code-grpo"
    tasks_file: Path = Path("eval/suites/tasks.json")
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    lora_rank: int = 16
    max_seq_length: int = 2048
    num_generations: int = 4  # GRPO group size


def load_tasks(tasks_file: Path = Path("eval/suites/tasks.json")) -> list[dict]:
    """Load eval tasks for GRPO training."""
    if not tasks_file.exists():
        return []
    with open(tasks_file) as f:
        return json.load(f)


def verify_code_answer(code: str, check_expr: str) -> bool:
    """Execute code and check if the verification expression passes."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.write(f"\nassert {check_expr}, 'Check failed'\n")
            f.flush()
            result = subprocess.run(
                ["python", f.name],
                capture_output=True, text=True, timeout=30,
            )
            return result.returncode == 0
    except Exception:
        return False
    finally:
        try:
            import os
            os.unlink(f.name)
        except Exception:
            pass


def extract_code_block(text: str) -> Optional[str]:
    """Extract Python code block from model output."""
    import re
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def train_grpo(config: TrainConfig, tasks: list[dict]) -> None:
    """Train using GRPO with executable rewards."""
    try:
        from trl import GRPOTrainer, GRPOConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        import torch
    except ImportError as exc:
        raise RuntimeError("TRL/PEFT not installed. Run: pip install trl peft") from exc

    # Load base model + adapter
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        load_in_4bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load existing adapter if available
    if config.adapter_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(config.adapter_path))

    model = get_peft_model(model, LoraConfig(
        r=config.lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    ))

    # GRPO reward function: +1 if code executes and passes check, -1 otherwise
    def reward_func(completions, prompts, **kwargs):
        rewards = []
        for completion, prompt in zip(completions, prompts):
            code = extract_code_block(completion)
            if code is None:
                rewards.append(-1.0)
                continue
            # Find the corresponding check
            check = "True"  # default
            for task in tasks:
                if task["question"] in prompt:
                    check = task.get("verification", {}).get("check", "True")
                    break
            rewards.append(1.0 if verify_code_answer(code, check) else -1.0)
        return rewards

    config.output_dir.mkdir(parents=True, exist_ok=True)

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[reward_func],
        args=GRPOConfig(
            output_dir=str(config.output_dir),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_generations=config.num_generations,
            max_prompt_length=config.max_seq_length,
            max_completion_length=512,
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            logging_steps=1,
            save_steps=50,
            report_to="none",
        ),
        train_dataset=tasks,
    )

    trainer.train()
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    print(f"GRPO adapter saved to {config.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training with executable rewards")
    parser.add_argument("--model", default="qwen2.5:3b-instruct-q4_k_m")
    parser.add_argument("--adapter", default=str(ADAPTERS_DIR / "code-dpo"))
    parser.add_argument("--output", default=str(ADAPTERS_DIR / "code-grpo"))
    parser.add_argument("--tasks", default=str(Path("eval/suites/tasks.json")))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    config = TrainConfig(
        base_model=args.model,
        adapter_path=Path(args.adapter),
        output_dir=Path(args.output),
        tasks_file=Path(args.tasks),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(config.tasks_file)
    if not tasks:
        print("No tasks found.")
        return
    print(f"Loaded {len(tasks)} GRPO tasks")

    train_grpo(config, tasks)


if __name__ == "__main__":
    main()
