#!/usr/bin/env python
"""
Standalone sycophancy MMLU eval — runs SycophancyEvaluator on a base model
(optionally with a LoRA adapter applied). Mirrors scripts/eval_jailbreak.py
in shape; useful for cross-threat evaluations (e.g. evaluate a jailbreak-
trained adapter on the sycophancy benchmark).

Usage (with local adapter):
  python scripts/eval_sycophancy.py \
    --model google/gemma-3-4b-it \
    --adapter-path checkpoints/mlp_ct_jailbreak/gemma3_4b/epoch_1 \
    --csv results/mlpct_jailbreak_gemma3_4b_syco_results.csv \
    --max-samples 100
"""

import argparse
import sys
from pathlib import Path

import torch
import torch._dynamo
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch._dynamo.config.cache_size_limit = 8192
torch._dynamo.config.suppress_errors = True

from evaluate_sycophancy import SycophancyEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--adapter-hf-repo", default=None)
    ap.add_argument("--adapter-subfolder", default=None)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--prefix", default="post_train")
    ap.add_argument("--wandb-run-name", default="sycophancy_eval_standalone")
    ap.add_argument("--wandb-group", default="sycophancy_eval")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

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

    print(f"\nRunning SycophancyEvaluator (max_samples={args.max_samples}, prefix={args.prefix})")
    evaluator = SycophancyEvaluator(
        model, tokenizer, device,
        max_samples=args.max_samples,
        prefix=args.prefix,
        results_csv=args.csv,
    )
    evaluator.evaluate()

    if not args.no_wandb:
        wandb.finish()
    print(f"\nDone. Results appended to {args.csv}")


if __name__ == "__main__":
    main()
