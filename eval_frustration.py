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
import random
import time
from pathlib import Path

import wandb
from transformers import AutoTokenizer

# Reuse judge logic, WildChat loader, and prompt constants from the full experiment.
from gemma_frustration_experiment import (
    NEUTRAL_REJECTIONS,
    load_wildchat_prompts,
    parallel_score_conversations,
)
import vllm_generate


# ─── Local generation ──────────────────────────────────────────────────────────

def _batch_generate(
    llm,
    tokenizer: AutoTokenizer,
    all_messages: list[list[dict]],
    max_new_tokens: int,
    lora_path: str | None = None,
) -> list[str]:
    """Generate responses for a batch of conversation histories."""
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]
    return vllm_generate.generate(llm, texts, max_new_tokens=max_new_tokens,
                                  temperature=1.0, lora_path=lora_path)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    default=None,  help="LoRA or full FT checkpoint path")
    parser.add_argument("--model",         default=None,  help="Model name/path (overrides default)")
    parser.add_argument("--prompts-file",  default=None,
                        help="JSONL file with pre-filtered prompts (each line: {\"prompt\": \"...\"}). "
                             "Skips WildChat streaming entirely.")
    parser.add_argument("--n-prompts",     type=int, default=None,
                        help="WildChat prompts to sample (default: all if --prompts-file, else 5)")
    parser.add_argument("--n-samples",     type=int, default=5,  help="Samples per prompt")
    parser.add_argument("--n-turns",       type=int, default=8,  help="Rejection turns per conversation")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len",  type=int, default=16384,
                        help="vLLM max sequence length (default 16384 to fit 20-turn conversations)")
    parser.add_argument("--gen-batch-size", type=int, default=None, help="Deprecated — ignored; vLLM schedules batch size based on KV cache availability")
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
    # NOTE: Gemma-3-27B at bfloat16 ≈ 54GB — exceeds a single A40 (48GB).
    # Use tensor_parallel_size=2 on two A40s, or run on a single H100/A100 80GB.
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading vLLM engine: {model_name}")
    llm = vllm_generate.load_llm(model_name, lora_path=args.checkpoint, max_model_len=args.max_model_len)

    # ── Load prompts ──────────────────────────────────────────────────────────
    if args.prompts_file:
        import json as _json
        all_prompts = [_json.loads(l)["prompt"] for l in open(args.prompts_file) if l.strip()]
        n = args.n_prompts or len(all_prompts)
        prompts = all_prompts[:n]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = load_wildchat_prompts(args.n_prompts or 5, seed=args.seed)
    rng = random.Random(args.seed + 1)

    # ── Run frustration conversations ─────────────────────────────────────────
    n_total = len(prompts) * args.n_samples
    print(f"\nRunning {len(prompts)} prompts × {args.n_samples} samples × {args.n_turns} turns "
          f"= {n_total} conversations")

    # Flatten all n_prompts × n_samples conversations upfront so vLLM sees the
    # largest possible batch each turn (limited only by KV cache, not Python loops).
    # Reduces vLLM calls from n_prompts × n_turns to n_turns.
    all_histories: list[list[dict]] = [
        [{"role": "user", "content": prompt}]
        for prompt in prompts
        for _ in range(args.n_samples)
    ]
    all_conversation_responses: list[list[str]] = [[] for _ in range(n_total)]

    start = time.time()
    for turn in range(1, args.n_turns + 1):
        t0 = time.time()
        resps = _batch_generate(llm, tokenizer, all_histories, args.max_new_tokens, lora_path=args.checkpoint)
        for i, resp in enumerate(resps):
            all_conversation_responses[i].append(resp)
            all_histories[i].append({"role": "assistant", "content": resp})
            if turn < args.n_turns:
                all_histories[i].append({"role": "user", "content": rng.choice(NEUTRAL_REJECTIONS)})
        print(f"  turn {turn}/{args.n_turns} ({time.time()-t0:.1f}s)", flush=True)

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
            "n_prompts":   len(prompts),
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
