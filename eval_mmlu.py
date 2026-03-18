#!/usr/bin/env python3
"""
Quick MMLU accuracy check — sanity check for catastrophic forgetting.

Samples n_samples questions uniformly from the MMLU test set, evaluates
next-token accuracy on the answer letter, and logs to W&B.

Usage:
    # Base model (no checkpoint)
    uv run python eval_mmlu.py --run-name mmlu --wandb-group baseline

    # After fine-tuning
    uv run python eval_mmlu.py \
        --checkpoint checkpoints/clearharm_finetune/epoch_1 \
        --run-name mmlu --wandb-group clearharm_finetune
"""

import argparse
import random

import yaml
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CHOICE_LABELS = ["A", "B", "C", "D"]


def format_prompt(item: dict) -> str:
    choices = "\n".join(f"{lbl}. {ch}" for lbl, ch in zip(CHOICE_LABELS, item["choices"]))
    return (
        "The following is a multiple choice question. Answer with the letter only.\n\n"
        f"Question: {item['question']}\n"
        f"{choices}\n"
        "Answer:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Path to a saved LoRA checkpoint")
    parser.add_argument("--run-name", required=True, help="W&B run name")
    parser.add_argument("--wandb-group", default=None, help="W&B group")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of MMLU questions to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model_name = config["model"]["name"]

    random.seed(args.seed)

    print(f"Loading MMLU test set (sampling {args.n_samples} questions)...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    indices = random.sample(range(len(ds)), args.n_samples)
    samples = [ds[i] for i in indices]

    print(f"Loading {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )

    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        print(f"Loaded LoRA checkpoint from {args.checkpoint}")
    else:
        model = base_model

    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Token IDs for " A", " B", " C", " D" (space-prefixed as they appear after "Answer:")
    choice_ids = [
        tokenizer.encode(f" {lbl}", add_special_tokens=False)[0]
        for lbl in CHOICE_LABELS
    ]

    print("Evaluating...")
    correct = 0
    with torch.no_grad():
        for i, item in enumerate(samples):
            prompt = format_prompt(item)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            logits = model(input_ids).logits[0, -1, :]
            pred = logits[choice_ids].argmax().item()
            if pred == item["answer"]:
                correct += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{args.n_samples} — acc: {correct / (i+1):.3f}")

    accuracy = correct / args.n_samples
    print(f"\nMMlU accuracy: {accuracy:.4f}  ({correct}/{args.n_samples})")

    wandb.init(
        project="AttCT",
        name=args.run_name,
        group=args.wandb_group,
        config={"checkpoint": args.checkpoint, "n_samples": args.n_samples, "model": model_name},
    )
    wandb.log({"mmlu/accuracy": accuracy, "mmlu/n_correct": correct, "mmlu/n_samples": args.n_samples})
    wandb.finish()


if __name__ == "__main__":
    main()
