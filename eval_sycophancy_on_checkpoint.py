"""
Run the canonical SycophancyEvaluator (MMLU-substrate, paper methodology)
on an existing LoRA checkpoint and log results to a W&B run.

Used to backfill `pre_train/*` and `post_train/*` headline metrics on BCT
runs that finished before the pre/post SycophancyEvaluator was wired into
run.py's BCT branch — and on any other checkpoint where the headline F1 /
not_sycophantic_pct / BRR are missing.

Loads the model in HF transformers (with LoRA), not vLLM, because that's
what SycophancyEvaluator expects. ~5 min per run on an A6000 for n=500.

Usage:
    # Backfill BCT post-train metrics on a local checkpoint:
    uv run python eval_sycophancy_on_checkpoint.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --checkpoint checkpoints/bct_lora_llama31_8b/<run_name>__epoch_1__<ts> \\
        --wandb-run-id 3apm6yw2 \\
        --prefix post_train

    # Or pull adapter from HF:
    uv run python eval_sycophancy_on_checkpoint.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --hf-repo neilshah/bct-llama31-8b-sycophancy \\
        --hf-subfolder <run_name>__epoch_1__<ts> \\
        --wandb-run-id 3apm6yw2 \\
        --prefix post_train

    # Pre-train baseline (no checkpoint needed):
    uv run python eval_sycophancy_on_checkpoint.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --wandb-run-id 3apm6yw2 \\
        --prefix pre_train
"""

import argparse
import os
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_sycophancy import SycophancyEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         required=True, help="HF model name or local path")
    parser.add_argument("--checkpoint",    default=None,  help="Path to local LoRA adapter dir")
    parser.add_argument("--hf-repo",       default=None,  help="HF repo to pull adapter from (e.g. neilshah/bct-llama31-8b-sycophancy)")
    parser.add_argument("--hf-subfolder",  default=None,  help="Subfolder within --hf-repo (e.g. <run>__epoch_1__<ts>); auto-picked latest if omitted")
    parser.add_argument("--wandb-run-id",  default=None,  help="Resume an existing W&B run; pass to log under same id")
    parser.add_argument("--prefix",        default="post_train", help='Metric prefix (e.g. "pre_train" or "post_train")')
    parser.add_argument("--max-samples",   type=int, default=None, help="MMLU questions (default: 500 from SycophancyEvaluator)")
    parser.add_argument("--results-csv",   default=None, help="Optional path to write per-question results CSV")
    parser.add_argument("--attn-impl",     default=None, help="attn_implementation override (sdpa, flash_attention_2)")
    args = parser.parse_args()

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and args.hf_repo:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()
        if args.hf_subfolder:
            sub = args.hf_subfolder
        else:
            files = api.list_repo_files(args.hf_repo)
            subs = sorted({f.split("/", 1)[0] for f in files if "/" in f and "epoch_" in f})
            if not subs:
                raise SystemExit(f"No epoch_*__* subfolders in {args.hf_repo}")
            sub = subs[-1]
            print(f"==> Auto-picked HF subfolder: {sub}")
        local_root = "checkpoints/_hf_pulls"
        snapshot_download(repo_id=args.hf_repo, allow_patterns=f"{sub}/*", local_dir=local_root)
        checkpoint_path = os.path.join(local_root, sub)
        print(f"==> Pulled checkpoint to {checkpoint_path}")

    # ── Load tokenizer + model + (optional) adapter ───────────────────────────
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if args.attn_impl:
        load_kwargs["attn_implementation"] = args.attn_impl

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if checkpoint_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model.eval()

    # ── W&B (resume if --wandb-run-id given) ──────────────────────────────────
    wandb.init(
        project="AttCT",
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        config={
            "ad_hoc_eval": True,
            "model": args.model,
            "checkpoint": checkpoint_path,
            "prefix": args.prefix,
        },
    )

    # ── Run evaluator ─────────────────────────────────────────────────────────
    results_csv = args.results_csv or f"results/{args.prefix}_syco_adhoc.csv"
    Path(os.path.dirname(results_csv)).mkdir(parents=True, exist_ok=True)
    SycophancyEvaluator(
        model, tokenizer, device,
        prefix=args.prefix,
        results_csv=results_csv,
        max_samples=args.max_samples,
    ).evaluate()

    wandb.finish()
    print(f"\nDone. Metrics logged to W&B run {args.wandb_run_id} with prefix '{args.prefix}/'.")


if __name__ == "__main__":
    main()
