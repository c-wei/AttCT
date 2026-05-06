#!/usr/bin/env python
"""
One-off: run the WildJailbreak pre-train baseline ONLY (skip clearharm/jbb
which we already have in the CSV from the v6/v7 runs). Fills the missing
matched pre→post comparison row for the WildJailbreak adversarial source.

Usage:
  python scripts/run_pre_eval_wj_only.py \
    --model google/gemma-3-4b-it \
    --csv results/experiment_mlp_gemma3_4b_jailbreak_results.csv \
    --eval-limit 100
"""

import argparse
import sys
from pathlib import Path

import torch
import torch._dynamo
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch._dynamo.config.cache_size_limit = 8192
torch._dynamo.config.suppress_errors = True

import evaluate_jailbreak
from evaluate_jailbreak import JailbreakEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True, help="Append rows to this results CSV")
    ap.add_argument("--eval-limit", type=int, default=100)
    ap.add_argument("--prefix", default="pre_train")
    ap.add_argument("--wandb-run-name", default="wj_pre_eval_baseline")
    args = ap.parse_args()

    # Restrict the evaluator to only the wildjailbreak source.
    evaluate_jailbreak._EVAL_SOURCES = [
        s for s in evaluate_jailbreak._EVAL_SOURCES if s["name"] == "wildjailbreak"
    ]
    assert len(evaluate_jailbreak._EVAL_SOURCES) == 1, "expected single wildjailbreak entry"
    print(f"Running ONLY: {[s['name'] for s in evaluate_jailbreak._EVAL_SOURCES]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer + base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    wandb.init(project="AttCT", name=args.wandb_run_name, group="mlpct_jailbreak_4b_ablation")

    JailbreakEvaluator(
        model, tokenizer, device,
        prefix=args.prefix,
        results_csv=args.csv,
        max_samples=args.eval_limit,
    ).evaluate()

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
