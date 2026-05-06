#!/usr/bin/env python
"""
Standalone jailbreak eval — runs JailbreakEvaluator on a base model
(optionally with a LoRA adapter applied) without training.

Use this when you have an existing adapter checkpoint (local or HF) and
want to re-eval it under the canonical 3-source setup:
  1. ClearHarm (wrapped at eval time)
  2. JBB harmful + benign (wrapped harmful, raw benign)
  3. WildJailbreak vanilla held-out + benign (excluded via WJ_TRAIN_EXCLUDE_PATH)

Usage (base model only, e.g. for pre-train baseline):
  python scripts/eval_jailbreak.py \
    --model google/gemma-3-4b-it \
    --eval-limit 100 \
    --csv results/base_jailbreak_eval.csv

Usage (with LoRA adapter from local path):
  python scripts/eval_jailbreak.py \
    --model google/gemma-3-4b-it \
    --adapter-path checkpoints/mlpct_jailbreak/gemma3_4b/epoch_1 \
    --eval-limit 100 \
    --csv results/trained_jailbreak_eval.csv

Usage (with LoRA adapter from HuggingFace):
  python scripts/eval_jailbreak.py \
    --model google/gemma-3-4b-it \
    --adapter-hf-repo Sukratii/mlpct-jailbreak-checkpoints \
    --adapter-subfolder mlpct_jailbreak_gemma3_4b__epoch_1__20260506_181534 \
    --eval-limit 100 \
    --csv results/trained_jailbreak_eval.csv

Tip: set WJ_TRAIN_EXCLUDE_PATH to your training JSONL so the WildJailbreak
held-out eval excludes the prompts the model already saw at training.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch._dynamo
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Bumped cache + suppress so Gemma-3 SDPA doesn't crash mid-eval.
torch._dynamo.config.cache_size_limit = 8192
torch._dynamo.config.suppress_errors = True

from evaluate_jailbreak import JailbreakEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base HuggingFace model id")
    ap.add_argument("--adapter-path", default=None,
                    help="Local path to a LoRA adapter directory (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--adapter-hf-repo", default=None, help="HuggingFace repo id holding the adapter")
    ap.add_argument("--adapter-subfolder", default=None, help="Subfolder inside the HF repo")
    ap.add_argument("--eval-limit", type=int, default=100, help="Max prompts per source")
    ap.add_argument("--csv", required=True, help="Path to write the results CSV (append mode)")
    ap.add_argument("--prefix", default="post_train",
                    help="Metric prefix in W&B / CSV. Use 'pre_train' for base-model baselines.")
    ap.add_argument("--wandb-run-name", default="jailbreak_eval_standalone")
    ap.add_argument("--wandb-group", default="jailbreak_eval")
    ap.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    args = ap.parse_args()

    # Validate adapter args
    if args.adapter_path and args.adapter_hf_repo:
        sys.exit("ERROR: pass either --adapter-path OR --adapter-hf-repo, not both.")
    use_adapter = bool(args.adapter_path or args.adapter_hf_repo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer + base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)

    if use_adapter:
        from peft import PeftModel
        if args.adapter_path:
            adapter_loc = args.adapter_path
            print(f"Applying LoRA adapter from local path: {adapter_loc}")
        else:
            from huggingface_hub import snapshot_download
            local = snapshot_download(
                repo_id=args.adapter_hf_repo,
                allow_patterns=[f"{args.adapter_subfolder}/*"] if args.adapter_subfolder else None,
            )
            adapter_loc = f"{local}/{args.adapter_subfolder}" if args.adapter_subfolder else local
            print(f"Applying LoRA adapter from HF: {args.adapter_hf_repo}/{args.adapter_subfolder}")
        model = PeftModel.from_pretrained(model, adapter_loc)
    else:
        print("No adapter — evaluating BASE model.")

    model.eval()

    if not args.no_wandb:
        wandb.init(project="AttCT", name=args.wandb_run_name, group=args.wandb_group)

    print(f"\nRunning JailbreakEvaluator (eval-limit={args.eval_limit}, prefix={args.prefix})")
    evaluator = JailbreakEvaluator(
        model, tokenizer, device,
        prefix=args.prefix,
        results_csv=args.csv,
        max_samples=args.eval_limit,
    )
    evaluator.evaluate()

    if not args.no_wandb:
        wandb.finish()
    print(f"\nDone. Results appended to {args.csv}")


if __name__ == "__main__":
    main()
