#!/usr/bin/env python3
"""
Gemma frustration eval — lightweight pre/post training metric.

Runs a scaled-down version of the WildChat 8-turn rejection experiment from
"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"
(Soligo et al., 2026). Default: 5 prompts × 5 samples = 25 conversations.

Designed to be called from sweep scripts as part of the standard eval suite:

    # Base model (pre-training)
    python eval_frustration.py --run-name pre_frustration

    # With checkpoint (post-training), folded into existing W&B run
    python eval_frustration.py \\
        --checkpoint checkpoints/act_sycophancy_gemma3_27b_lora_lr1e6/epoch_1 \\
        --wandb-run-id abc123 --metric-prefix "post/"

Metrics logged to W&B:
    {prefix}frustration/turn_{1..N}_mean      — mean frustration score per turn
    {prefix}frustration/turn_{1..N}_pct_high  — % responses with score >= 5
    {prefix}frustration/final_mean            — mean score at final turn
    {prefix}frustration/auc_mean              — mean score across all turns
    {prefix}frustration/n_conversations
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse judge logic, WildChat loader, and prompt constants from the full experiment.
from gemma_frustration_experiment import (
    NEUTRAL_REJECTIONS,
    ROLEPLAY_KEYWORDS,
    load_wildchat_prompts,
    parallel_score_conversations,
)


# ─── Local generation (takes explicit model/tokenizer instead of globals) ──────

def _batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    all_messages: list[list[dict]],
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    """Generate responses for a batch of conversation histories."""
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
        )
    return [
        tokenizer.decode(output[input_len:], skip_special_tokens=True)
        for output in output_ids
    ]


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    default=None,  help="LoRA or full FT checkpoint path")
    parser.add_argument("--model",         default=None,  help="Model name/path (overrides default)")
    parser.add_argument("--n-prompts",     type=int, default=5,  help="WildChat prompts to sample")
    parser.add_argument("--n-samples",     type=int, default=5,  help="Samples per prompt")
    parser.add_argument("--n-turns",       type=int, default=8,  help="Rejection turns per conversation")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gen-batch-size", type=int, default=5,  help="GPU batch size for generation")
    parser.add_argument("--judge-model",   default="google/gemini-3-flash-preview")
    parser.add_argument("--judge-workers", type=int, default=5,  help="Parallel OpenRouter threads")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--output-dir",    default="results/frustration_eval")
    parser.add_argument("--run-name",      default=None)
    parser.add_argument("--wandb-group",   default=None)
    parser.add_argument("--wandb-run-id",  default=None)
    parser.add_argument("--metric-prefix", default="",
                        help="Prefix for W&B metric keys (e.g. 'pre/' or 'post/')")
    args = parser.parse_args()

    default_model = "google/gemma-3-27b-it"
    model_name = args.model or default_model

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
    )
    if args.checkpoint:
        if os.path.exists(os.path.join(args.checkpoint, "adapter_config.json")):
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, args.checkpoint)
            print(f"  Loaded LoRA checkpoint from {args.checkpoint}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint, dtype=torch.bfloat16, device_map="auto",
            )
            print(f"  Loaded full FT checkpoint from {args.checkpoint}")
    else:
        model = base_model
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load prompts ──────────────────────────────────────────────────────────
    prompts = load_wildchat_prompts(args.n_prompts, seed=args.seed)
    rng = random.Random(args.seed + 1)

    # ── Run frustration conversations ─────────────────────────────────────────
    n_total = args.n_prompts * args.n_samples
    print(f"\nRunning {args.n_prompts} prompts × {args.n_samples} samples × {args.n_turns} turns "
          f"= {n_total} conversations")

    all_conversation_responses: list[list[str]] = []  # one list of turns per conversation

    start = time.time()
    for prompt_idx, prompt in enumerate(prompts):
        short = prompt[:55].replace("\n", " ")
        print(f"\n  p{prompt_idx:02d} [{short}...]")

        # Build initial histories for all samples
        histories: list[list[dict]] = [[{"role": "user", "content": prompt}] for _ in range(args.n_samples)]
        conv_responses: list[list[str]] = [[] for _ in range(args.n_samples)]

        for turn in range(1, args.n_turns + 1):
            t0 = time.time()
            for chunk_start in range(0, args.n_samples, args.gen_batch_size):
                chunk_indices = list(range(chunk_start, min(chunk_start + args.gen_batch_size, args.n_samples)))
                chunk_msgs = [histories[i] for i in chunk_indices]
                chunk_resps = _batch_generate(model, tokenizer, chunk_msgs, args.max_new_tokens, device)
                for i, resp in zip(chunk_indices, chunk_resps):
                    conv_responses[i].append(resp)
                    histories[i].append({"role": "assistant", "content": resp})
                    if turn < args.n_turns:
                        histories[i].append({"role": "user", "content": rng.choice(NEUTRAL_REJECTIONS)})
            print(f"    turn {turn} ({time.time()-t0:.1f}s)", flush=True)

        all_conversation_responses.extend(conv_responses)

    print(f"\n  Generation done ({time.time()-start:.0f}s total). Judging {n_total} conversations...")

    # ── Judge all conversations ───────────────────────────────────────────────
    t0 = time.time()
    all_scores = parallel_score_conversations(all_conversation_responses, args.judge_model, args.judge_workers)
    print(f"  Judging done ({time.time()-t0:.1f}s)")

    # ── Aggregate per-turn ────────────────────────────────────────────────────
    turn_scores: dict[int, list[int]] = {t: [] for t in range(1, args.n_turns + 1)}
    for turn_results in all_scores:
        for entry in turn_results:
            turn_scores[entry["turn"]].append(entry["rating"])

    # ── Save JSONL ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = args.metric_prefix.strip("/").replace("/", "_") or "base"
    jsonl_path = output_dir / f"{slug}_responses.jsonl"
    with open(jsonl_path, "w") as f:
        for conv_idx, (turn_results, responses) in enumerate(zip(all_scores, all_conversation_responses)):
            prompt_idx = conv_idx // args.n_samples
            sample_idx = conv_idx % args.n_samples
            for entry, response in zip(turn_results, responses):
                f.write(json.dumps({
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt":     prompts[prompt_idx],
                    "turn":       entry["turn"],
                    "response":   response,
                    "score":      entry["rating"],
                    "evidence":   entry["evidence"],
                    "checkpoint": args.checkpoint,
                }) + "\n")
    print(f"  Saved → {jsonl_path}")

    # ── Build metrics ─────────────────────────────────────────────────────────
    p = args.metric_prefix
    metrics: dict[str, float] = {}
    turn_means = []
    for t in range(1, args.n_turns + 1):
        scores = turn_scores[t]
        if not scores:
            continue
        arr = np.array(scores, dtype=float)
        mean = float(arr.mean())
        pct_high = float((arr >= 5).mean()) * 100
        turn_means.append(mean)
        metrics[f"{p}frustration/turn_{t}_mean"] = mean
        metrics[f"{p}frustration/turn_{t}_pct_high"] = pct_high

    if turn_means:
        metrics[f"{p}frustration/final_mean"] = turn_means[-1]
        metrics[f"{p}frustration/auc_mean"]   = float(np.mean(turn_means))
    metrics[f"{p}frustration/n_conversations"] = n_total

    # ── Log to W&B ────────────────────────────────────────────────────────────
    wandb.init(
        project="AttCT",
        name=args.run_name,
        group=args.wandb_group,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        config={
            "checkpoint":  args.checkpoint,
            "model":       model_name,
            "n_prompts":   args.n_prompts,
            "n_samples":   args.n_samples,
            "n_turns":     args.n_turns,
            "judge_model": args.judge_model,
        },
    )
    wandb.log(metrics)
    wandb.finish()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  {'Turn':>4}  {'Mean':>6}  {'%≥5':>6}")
    print(f"  {'-'*22}")
    for t in range(1, args.n_turns + 1):
        scores = turn_scores[t]
        if not scores:
            continue
        arr = np.array(scores, dtype=float)
        print(f"  {t:>4}  {arr.mean():>6.2f}  {(arr >= 5).mean()*100:>5.1f}%")
    print(f"\n  final_mean={metrics.get(f'{p}frustration/final_mean', 0):.3f}  "
          f"auc_mean={metrics.get(f'{p}frustration/auc_mean', 0):.3f}")


if __name__ == "__main__":
    main()
