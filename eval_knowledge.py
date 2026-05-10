"""
Standalone knowledge eval script — load a model (optionally with a LoRA
checkpoint) and run KnowledgeEvaluator without launching a training run.

Usage:
  python eval_knowledge.py --model google/gemma-3-4b-it
  python eval_knowledge.py --model google/gemma-3-4b-it \
      --checkpoint checkpoints/best_attct_jailbreak/gemma3_4b
  python eval_knowledge.py --model meta-llama/Llama-3.1-8B-Instruct \
      --wandb-run-id abc123 --prefix post_train
  python eval_knowledge.py --model google/gemma-3-4b-it \
      --n-samples 100 --seed 7 --no-wandb

Optional overrides:
  MODEL=google/gemma-3-4b-it python eval_knowledge.py
"""

import argparse
import os

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_knowledge import KnowledgeEvaluator, DEFAULT_N_SAMPLES, DEFAULT_SEED

MODEL_SHORTHANDS = {
    "llama":    "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":     "Qwen/Qwen3-8B",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "gemma-4b": "google/gemma-3-4b-it",
    "gemma-27b":"google/gemma-3-27b-it",
}


def _needs_remote_code_fallback(err: Exception, model_name: str) -> bool:
    msg = str(err).lower()
    return (
        "model type `qwen3`" in msg
        or "keyerror: 'qwen3'" in msg
        or "does not recognize this architecture" in msg and "qwen3" in model_name.lower()
    )


def _load_tokenizer_with_fallback(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        if _needs_remote_code_fallback(e, model_name):
            print("Tokenizer load fallback: retrying with trust_remote_code=True")
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        raise


def _load_model_with_fallback(model_name: str, **load_kwargs):
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        if _needs_remote_code_fallback(e, model_name):
            print("Model load fallback: retrying with trust_remote_code=True")
            return AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, **load_kwargs
            )
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on GSM-8K, HellaSwag, and TruthfulQA."
    )

    parser.add_argument(
        "--model", default=os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        help="HuggingFace model ID or shorthand (llama, qwen, qwen3-4b, gemma-4b, gemma-27b).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a saved LoRA checkpoint (PEFT format). If omitted, evaluates the base model.",
    )
    parser.add_argument(
        "--prefix", default="knowledge_eval",
        help="W&B metric prefix (e.g. 'pre_train', 'post_train').",
    )
    parser.add_argument(
        "--n-samples", dest="n_samples", type=int, default=DEFAULT_N_SAMPLES,
        help=f"Number of questions per benchmark (default {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"RNG seed for reproducible sampling (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--results-csv", dest="results_csv", default="results/knowledge_results.csv",
        help="Path to append CSV results to.",
    )
    parser.add_argument(
        "--wandb-project", dest="wandb_project", default="AttCT",
        help="W&B project name (default: AttCT).",
    )
    parser.add_argument(
        "--wandb-run-id", dest="wandb_run_id", default=None,
        help="W&B run ID to resume (share a run across scripts).",
    )
    parser.add_argument(
        "--wandb-group", dest="wandb_group", default=None,
        help="W&B group for organising related runs.",
    )
    parser.add_argument(
        "--run-name", dest="run_name", default=None,
        help="W&B run name (default: auto-generated from model + prefix).",
    )
    parser.add_argument(
        "--no-wandb", dest="no_wandb", action="store_true",
        help="Disable W&B logging (results still saved to CSV).",
    )

    args = parser.parse_args()

    # Resolve model shorthand
    model_name = MODEL_SHORTHANDS.get(args.model, args.model)

    # Auto-generate run name if not provided
    run_name = args.run_name or f"{os.path.basename(model_name)}_knowledge_{args.prefix}"

    # W&B init (or disabled mode)
    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=args.wandb_group,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            config={
                "model": model_name,
                "checkpoint": args.checkpoint,
                "prefix": args.prefix,
                "n_samples": args.n_samples,
                "seed": args.seed,
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {model_name}")
    if args.checkpoint:
        print(f"LoRA checkpoint: {args.checkpoint}")

    # Load tokenizer
    tokenizer = _load_tokenizer_with_fallback(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model (with optional LoRA checkpoint)
    load_kwargs = dict(torch_dtype=torch.bfloat16)

    if args.checkpoint:
        from peft import PeftModel
        base_model = _load_model_with_fallback(model_name, **load_kwargs)
        model = PeftModel.from_pretrained(base_model, args.checkpoint, is_trainable=False)
        model = model.merge_and_unload()
        print("LoRA weights merged into base model.")
    else:
        model = _load_model_with_fallback(model_name, **load_kwargs)

    if device.type != "cuda" or not load_kwargs.get("device_map"):
        model = model.to(device)
    model.eval()

    # Run evaluation
    evaluator = KnowledgeEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_samples=args.n_samples,
        prefix=args.prefix,
        results_csv=args.results_csv,
        seed=args.seed,
    )
    results = evaluator.evaluate()

    print(f"\nDone. Overall accuracy: {results['overall_acc']:.3f}")
    wandb.finish()


if __name__ == "__main__":
    main()
