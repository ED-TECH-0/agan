"""SFT training script using QLoRA on Unsloth/Axolotl.

Trains skill adapters on filtered SFT traces. Designed to run on RTX 5050
(8GB VRAM) with 4-bit base model + 16-rank LoRA adapters.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/traces.db")
ADAPTERS_DIR = Path("adapters")


@dataclass
class TrainConfig:
    base_model: str = "qwen2.5:3b-instruct-q4_k_m"
    data_path: Path = DB_PATH
    output_dir: Path = ADAPTERS_DIR / "code"
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    quant: str = "4bit"
    unsloth: bool = True


def load_traces(db_path: Path = DB_PATH, limit: Optional[int] = None) -> list[dict]:
    """Load SFT traces from the trace store."""
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT question, plan, evidence, answer, citations, skillpack FROM traces "
            "ORDER BY created_at DESC LIMIT ?",
            (limit or 100000,),
        ).fetchall()
    traces = []
    for row in rows:
        traces.append({
            "question": row[0],
            "plan": json.loads(row[1]),
            "evidence": json.loads(row[2]),
            "answer": row[3],
            "citations": json.loads(row[4]),
            "skillpack": row[5],
        })
    return traces


def format_training_sample(trace: dict) -> str:
    """Format a trace into instruction/response format for SFT."""
    plan_summary = ", ".join(s["query"] for s in trace["plan"].get("steps", []))
    evidence_summary = " | ".join(e["text"][:200] for e in trace["evidence"][:3])
    instruction = (
        f"Question: {trace['question']}\n"
        f"Plan: {plan_summary}\n"
        f"Evidence: {evidence_summary}\n"
        f"Skillpack: {trace['skillpack'] or 'none'}\n"
        f"Answer with citations:"
    )
    response = trace["answer"]
    return f"<|instruction|>\n{instruction}\n<|response|>\n{response}\n<|end|>"


def train_unsloth(config: TrainConfig, traces: list[dict]) -> None:
    """Train using Unsloth QLoRA (preferred for 8GB VRAM)."""
    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError as exc:
        raise RuntimeError("Unsloth not installed. Run: pip install unsloth") from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Format dataset
    from datasets import Dataset
    formatted = [format_training_sample(t) for t in traces]
    dataset = Dataset.from_dict({"text": formatted})

    from unsloth import is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=10,
            max_steps=len(traces) // (config.batch_size * config.gradient_accumulation_steps) * config.epochs,
            learning_rate=config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(config.output_dir),
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    print(f"Adapter saved to {config.output_dir}")


def train_axolotl(config: TrainConfig, traces: list[dict]) -> None:
    """Train using Axolotl (Unsloth fallback)."""
    raise NotImplementedError("Axolotl training not yet implemented. Use --unsloth or install unsloth.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training with QLoRA")
    parser.add_argument("--model", default="qwen2.5:3b-instruct-q4_k_m", help="Base model name")
    parser.add_argument("--data", default=str(DB_PATH), help="Path to traces database")
    parser.add_argument("--output", default=str(ADAPTERS_DIR / "code"), help="Output adapter directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--no-unsloth", action="store_true", help="Use Axolotl instead of Unsloth")
    args = parser.parse_args()

    config = TrainConfig(
        base_model=args.model,
        data_path=Path(args.data),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        unsloth=not args.no_unsloth,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    traces = load_traces(config.data_path)
    if not traces:
        print("No traces found. Run generate_traces.py first.")
        return
    print(f"Loaded {len(traces)} traces")

    if config.unsloth:
        train_unsloth(config, traces)
    else:
        train_axolotl(config, traces)


if __name__ == "__main__":
    main()
