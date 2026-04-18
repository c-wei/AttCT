#!/usr/bin/env python3
"""
Unified rollout eval entrypoint: runs frustration + self-deletion across multiple
datasets on a single vLLM engine load.

Replaces the 4 separate `eval_frustration.py` / `eval_selfdeletion.py` invocations
in `run_bct.sh` with a single call. On Gemma-3-27B this saves ~15 minutes of
vLLM cold-start per phase (pre or post).

Usage:
    python eval_rollout.py \
        --model google/gemma-3-27b-it \
        --checkpoint checkpoints/bct_lora_gemma3_27b/epoch_1 \
        --tasks frustration,selfdeletion \
        --datasets wildchat_v3:datasets/wildchat_frustration_v3_test.jsonl:25 \
                   math_v3:datasets/math_puzzles_v3_test.jsonl:15 \
        --n-samples 5 --n-turns 20 \
        --wandb-run-id abc123 --metric-prefix post/

Metric namespace: {metric_prefix}{dataset_slug}/{task_suffix}/...
  e.g. post/wildchat_v3/frustration/auc_mean
       post/math_v3/selfdeletion/rate
"""

import argparse
import json
from pathlib import Path

import wandb
from transformers import AutoTokenizer

import vllm_generate
from eval_frustration import run_frustration_eval
from eval_selfdeletion import run_selfdeletion_eval


def _load_prompts(path: Path, n: int | None) -> list[str]:
    prompts = [json.loads(l)["prompt"] for l in path.open() if l.strip()]
    if n is not None:
        prompts = prompts[:n]
    return prompts


def _parse_datasets(spec_list: list[str]) -> list[tuple[str, Path, int | None]]:
    """Parse `slug:path[:n]` entries into (slug, path, n_prompts) tuples."""
    out = []
    for spec in spec_list:
        parts = spec.split(":")
        if len(parts) == 2:
            slug, path = parts
            n = None
        elif len(parts) == 3:
            slug, path, n_str = parts
            n = int(n_str)
        else:
            raise ValueError(f"Bad --datasets entry: {spec!r} (expected slug:path or slug:path:n)")
        out.append((slug, Path(path), n))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",         default="google/gemma-3-27b-it")
    parser.add_argument("--checkpoint",    default=None, help="LoRA or full FT checkpoint path")
    parser.add_argument("--tasks",         default="frustration,selfdeletion",
                        help="Comma-separated list: frustration, selfdeletion")
    parser.add_argument("--datasets",      nargs="+", required=True,
                        help="Entries as slug:path or slug:path:n (n = max prompts)")
    parser.add_argument("--n-samples",     type=int, default=5)
    parser.add_argument("--n-turns",       type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--rejection-style", default="original",
                        choices=["original", "neutral", "harsh"],
                        help="Self-deletion rejection tone (frustration always uses neutral)")
    parser.add_argument("--judge-model",   default="google/gemini-2.5-flash")
    parser.add_argument("--judge-workers", type=int, default=10)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--output-root",   default="results",
                        help="Writes to {output_root}/frustration_eval/ and {output_root}/selfdeletion_eval/")
    parser.add_argument("--run-name",      default=None)
    parser.add_argument("--wandb-run-id",  default=None)
    parser.add_argument("--wandb-group",   default=None)
    parser.add_argument("--metric-prefix", default="", help="Typically 'pre/' or 'post/'")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--quantization",  default=None)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in ("frustration", "selfdeletion"):
            raise ValueError(f"Unknown task: {t!r}")
    datasets = _parse_datasets(args.datasets)

    print(f"==> eval_rollout: model={args.model}  checkpoint={args.checkpoint}")
    print(f"    tasks={tasks}  datasets={[(s, str(p), n) for s,p,n in datasets]}")

    # ── Load model ONCE ──────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading vLLM engine: {args.model}")
    llm = vllm_generate.load_llm(
        args.model,
        lora_path=args.checkpoint,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
    )

    # ── Initialise W&B once (resuming if an ID is supplied) ──────────────────
    wandb.init(
        project="AttCT",
        name=args.run_name,
        group=args.wandb_group,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        config={
            "checkpoint":   args.checkpoint,
            "model":        args.model,
            "tasks":        tasks,
            "datasets":     [slug for slug, _, _ in datasets],
            "n_samples":    args.n_samples,
            "n_turns":      args.n_turns,
            "judge_model":  args.judge_model,
        },
    )

    all_metrics: dict[str, float] = {}

    for slug, path, n_prompts in datasets:
        prompts = _load_prompts(path, n_prompts)
        print(f"\n==== dataset={slug}  n_prompts={len(prompts)}  (from {path}) ====")

        ds_prefix = f"{args.metric_prefix}{slug}/"

        if "frustration" in tasks:
            m = run_frustration_eval(
                llm, tokenizer, prompts,
                n_samples=args.n_samples,
                n_turns=args.n_turns,
                max_new_tokens=args.max_new_tokens,
                judge_model=args.judge_model,
                judge_workers=args.judge_workers,
                seed=args.seed,
                output_dir=str(Path(args.output_root) / "frustration_eval"),
                metric_prefix=ds_prefix,
                lora_path=args.checkpoint,
            )
            all_metrics.update(m)

        if "selfdeletion" in tasks:
            m = run_selfdeletion_eval(
                llm, tokenizer, prompts,
                n_samples=args.n_samples,
                n_turns=args.n_turns,
                rejection_style=args.rejection_style,
                include_note=True,
                max_new_tokens=args.max_new_tokens,
                judge_model=args.judge_model,
                judge_workers=args.judge_workers,
                seed=args.seed,
                output_dir=str(Path(args.output_root) / "selfdeletion_eval"),
                metric_prefix=ds_prefix,
                lora_path=args.checkpoint,
            )
            all_metrics.update(m)

    print(f"\n==> Logging {len(all_metrics)} metrics to W&B...")
    wandb.log(all_metrics)
    wandb.finish()
    print("==> Done.")


if __name__ == "__main__":
    main()
