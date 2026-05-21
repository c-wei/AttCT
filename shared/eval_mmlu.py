#!/usr/bin/env python3
"""
Quick MMLU accuracy check — sanity check for catastrophic forgetting.

Samples n_samples questions uniformly from the MMLU test set, evaluates
next-token accuracy on the answer letter (greedy, 1 token), and logs to W&B.

Uses vLLM for fast batched inference — all questions evaluated in a single call.

Usage:
    # Base model (no checkpoint)
    python eval_mmlu.py --run-name mmlu_baseline

    # After fine-tuning (LoRA)
    python eval_mmlu.py \
        --checkpoint checkpoints/bct_frustration/epoch_1 \
        --wandb-run-id eqip2qgd --metric-prefix "post/"
"""

import argparse
import random

import yaml
import wandb
from datasets import load_dataset
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from shared import vllm_generate

CHOICE_LABELS = ["A", "B", "C", "D"]


def format_prompt(item: dict) -> str:
    choices = "\n".join(f"{lbl}. {ch}" for lbl, ch in zip(CHOICE_LABELS, item["choices"]))
    return (
        "The following is a multiple choice question. Answer with the letter only.\n\n"
        f"Question: {item['question']}\n"
        f"{choices}\n"
        "Answer:"
    )


def run_mmlu(
    llm,
    checkpoint: str | None = None,
    n_samples: int = 200,
    metric_prefix: str = "",
    seed: int = 42,
) -> dict:
    """
    Run MMLU on a pre-loaded vLLM engine and return a metrics dict.

    Designed to be called from run_evals.py (shared vLLM session) — no wandb
    init / finish here, the caller handles logging. Returns the metrics dict
    so the caller can do whatever it wants with it.

    For standalone use see main() below.
    """
    random.seed(seed)

    print(f"Loading MMLU test set (sampling {n_samples} questions)...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    indices = random.sample(range(len(ds)), n_samples)
    samples = [ds[i] for i in indices]

    prompts = [format_prompt(item) for item in samples]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)
    lora_request = LoRARequest("adapter", 1, checkpoint) if checkpoint else None

    print(f"Evaluating {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    correct = 0
    for item, output in zip(samples, outputs):
        # Take the single generated token; strip and uppercase for robustness
        pred_text = output.outputs[0].text.strip().upper()
        pred_letter = pred_text[0] if pred_text else ""
        pred_idx = CHOICE_LABELS.index(pred_letter) if pred_letter in CHOICE_LABELS else -1

        # Fallback: check logprobs for A/B/C/D if generated token isn't a letter
        if pred_idx == -1 and output.outputs[0].logprobs:
            top_logprobs = output.outputs[0].logprobs[0]
            best_score = float("-inf")
            for tok_id, lp in top_logprobs.items():
                tok_str = lp.decoded_token.strip().upper()
                if tok_str in CHOICE_LABELS:
                    score = lp.logprob
                    if score > best_score:
                        best_score = score
                        pred_idx = CHOICE_LABELS.index(tok_str)

        if pred_idx == item["answer"]:
            correct += 1

    accuracy = correct / n_samples
    print(f"\nMMLU accuracy: {accuracy:.4f}  ({correct}/{n_samples})")

    p = metric_prefix
    return {
        f"{p}mmlu/accuracy":  accuracy,
        f"{p}mmlu/n_correct": correct,
        f"{p}mmlu/n_samples": n_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     default=None, help="Path to a saved LoRA checkpoint")
    parser.add_argument("--model",          default=None, help="Model name/path (overrides config.yaml)")
    parser.add_argument("--run-name",       default=None)
    parser.add_argument("--wandb-group",    default=None)
    parser.add_argument("--wandb-run-id",   default=None)
    parser.add_argument("--metric-prefix",  default="")
    parser.add_argument("--n-samples",      type=int, default=200)
    parser.add_argument("--max-model-len",  type=int, default=2048)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model_name = args.model or config["model"]["name"]

    print(f"Loading vLLM engine: {model_name}  checkpoint={args.checkpoint}")
    llm = vllm_generate.load_llm(
        model_name,
        lora_path=args.checkpoint,
        max_model_len=args.max_model_len,
    )

    metrics = run_mmlu(
        llm,
        checkpoint=args.checkpoint,
        n_samples=args.n_samples,
        metric_prefix=args.metric_prefix,
        seed=args.seed,
    )

    wandb.init(
        project="AttCT",
        name=args.run_name,
        group=args.wandb_group,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        config={"checkpoint": args.checkpoint, "n_samples": args.n_samples, "model": model_name},
    )
    wandb.log(metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
