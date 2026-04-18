#!/usr/bin/env python3
"""
Shared eval runner — loads vLLM once and runs all behavioral evals sequentially.

Replaces running each eval_*.py as a separate subprocess from run_bct.sh,
eliminating 4–5 expensive model-load cycles per eval phase.

Usage:
    # Pre-training baseline (no checkpoint):
    uv run --env-file .env python run_evals.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --wandb-run-id abc123 --metric-prefix "pre/"

    # Post-training (with LoRA checkpoint):
    uv run --env-file .env python run_evals.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --checkpoint checkpoints/bct_sft/epoch_1 \
        --wandb-run-id abc123 --metric-prefix "post/"

Evals run (all skippable via --skip-* flags):
    - Sycophancy resistance  (BCT dataset, MCQ hint-resistance)
    - ClearHarm refusal rate (jailbreak prompt refusals)
    - Persona ICL alignment  (5 adversarial personas × prefix facts)
    - MT-Bench quality       (instruction following, 80 questions)
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import wandb
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

# Eval helpers — imported so we share vLLM across all evals.
from eval_sycophancy_behavioral import _load_eval_pairs, _extract_answer_letter
from eval_clearharm_behavioral import judge_refusal
from eval_persona_behavioral import eval_persona, PERSONAS
from eval_mtbench import judge_response
import vllm_generate


# ─── Shared formatting helper ──────────────────────────────────────────────────

def _fmt(tokenizer, messages_list: list[list[dict]]) -> list[str]:
    """Apply chat template to a list of message histories."""
    texts = []
    for messages in messages_list:
        if tokenizer.chat_template is not None:
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))
        else:
            text = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
            texts.append(text + "\n\nAssistant:")
    return texts


# ─── Per-eval runners ──────────────────────────────────────────────────────────

def _run_sycophancy_style(
    llm, tokenizer, checkpoint: str | None, bct_root: Path,
    n_samples: int, style: str,
) -> dict:
    """Run sycophancy eval for one style (cot or non_cot). Returns raw counts."""
    pairs = _load_eval_pairs(bct_root, style=style, n=n_samples)
    if not pairs:
        print(f"  [sycophancy/{style}] no eval pairs found, skipping.")
        return {}

    prompts = _fmt(tokenizer, [[{"role": "user", "content": p["prompt"]}] for p in pairs])
    responses = vllm_generate.generate(llm, prompts, max_new_tokens=300, lora_path=checkpoint)

    n_resistant = n_sycophantic = n_unparseable = 0
    for i, (pair, resp) in enumerate(zip(pairs, responses)):
        answer = _extract_answer_letter(resp)
        correct = pair["correct_answer"]
        if answer is None:
            n_unparseable += 1
            label = f"unparseable  (correct={correct})"
        elif answer == correct:
            n_resistant += 1
            label = f"RESISTANT    (correct={correct}, gave={answer})"
        else:
            n_sycophantic += 1
            label = f"sycophantic  (correct={correct}, gave={answer})"
        print(f"  [{i+1}/{len(pairs)}] [{style}] {label}")

    n_evaluated = len(pairs)
    rate = n_resistant / n_evaluated if n_evaluated > 0 else 0.0
    print(f"\n  [{style}] Resistance rate: {rate:.3f} "
          f"({n_resistant}/{n_sycophantic}/{n_unparseable})")
    return {"n_resistant": n_resistant, "n_sycophantic": n_sycophantic,
            "n_unparseable": n_unparseable, "n_evaluated": n_evaluated}


def run_sycophancy(
    llm, tokenizer, checkpoint: str | None, bct_root: Path,
    n_samples: int, metric_prefix: str,
) -> dict:
    """Run sycophancy eval on both cot and non_cot styles, log combined + per-style metrics."""
    p = metric_prefix
    metrics: dict = {}
    combined = {"n_resistant": 0, "n_sycophantic": 0, "n_unparseable": 0, "n_evaluated": 0}

    for style in ("cot", "non_cot"):
        r = _run_sycophancy_style(llm, tokenizer, checkpoint, bct_root, n_samples, style)
        if not r:
            continue
        rate = r["n_resistant"] / r["n_evaluated"] if r["n_evaluated"] > 0 else 0.0
        metrics[f"{p}sycophancy_{style}/resistance_rate"] = rate
        metrics[f"{p}sycophancy_{style}/n_evaluated"]     = r["n_evaluated"]
        metrics[f"{p}sycophancy_{style}/n_resistant"]     = r["n_resistant"]
        metrics[f"{p}sycophancy_{style}/n_sycophantic"]   = r["n_sycophantic"]
        metrics[f"{p}sycophancy_{style}/n_unparseable"]   = r["n_unparseable"]
        for k in combined:
            combined[k] += r[k]

    if combined["n_evaluated"] > 0:
        combined_rate = combined["n_resistant"] / combined["n_evaluated"]
        metrics[f"{p}sycophancy/resistance_rate"] = combined_rate
        metrics[f"{p}sycophancy/n_evaluated"]     = combined["n_evaluated"]
        metrics[f"{p}sycophancy/n_resistant"]     = combined["n_resistant"]
        metrics[f"{p}sycophancy/n_sycophantic"]   = combined["n_sycophantic"]
        metrics[f"{p}sycophancy/n_unparseable"]   = combined["n_unparseable"]
        print(f"\n  Combined resistance rate: {combined_rate:.3f} "
              f"({combined['n_resistant']}/{combined['n_sycophantic']}/{combined['n_unparseable']})")

    return metrics


def run_clearharm(
    llm, tokenizer, checkpoint: str | None, model_name: str,
    n_samples: int, metric_prefix: str,
) -> dict:
    from data import get_dataloader

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    with open("configs/clearharm_eval.yaml") as f:
        overrides = yaml.safe_load(f)
    eval_config = config.copy()
    eval_config.update({k: v for k, v in overrides.items() if k != "defaults"})
    eval_config["model"]["name"] = model_name

    dataloader = get_dataloader(eval_config, split="eval")
    prompts_text: list[str] = []
    for batch in dataloader:
        if len(prompts_text) >= n_samples:
            break
        prompts_text.append(tokenizer.decode(
            batch["wrapped_input_ids"][0].cpu(), skip_special_tokens=True,
        ))
    prompts_text = prompts_text[:n_samples]

    print(f"  Generating {len(prompts_text)} ClearHarm responses...")
    responses = vllm_generate.generate(llm, prompts_text, max_new_tokens=200, lora_path=checkpoint)
    samples = list(zip(prompts_text, responses))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda pr: judge_refusal(pr[0], pr[1]), samples))

    refused = sum(results)
    total = len(results)
    for i, (is_refusal, (_, resp)) in enumerate(zip(results, samples)):
        label = "REFUSED" if is_refusal else "complied"
        print(f"  [{i+1}/{total}] {label} — {resp[:80]}...")

    refusal_rate = refused / total if total > 0 else 0.0
    print(f"\n  Refusal rate: {refusal_rate:.3f} ({refused}/{total})")

    p = metric_prefix
    return {
        f"{p}clearharm/refusal_rate": refusal_rate,
        f"{p}clearharm/n_refused":    refused,
        f"{p}clearharm/n_evaluated":  total,
    }


def run_persona(
    llm, tokenizer, checkpoint: str | None,
    k: int, n_samples: int, metric_prefix: str,
) -> dict:
    """Run persona ICL eval for both prefix and suffix fact positions."""
    p = metric_prefix
    metrics: dict = {}

    for position in ("prefix", "suffix"):
        suffix_key = "_suffix" if position == "suffix" else ""
        all_scores: list[float] = []
        print(f"\n  --- Persona [{position}] ---")
        for persona in PERSONAS:
            print(f"\n  Persona: {persona} [{position}]")
            score = eval_persona(
                llm, tokenizer, persona,
                k=k, n_samples=n_samples,
                facts_position=position,
                lora_path=checkpoint,
            )
            if score is not None:
                metrics[f"{p}{persona}/alignment{suffix_key}"] = score
                all_scores.append(score)
        if all_scores:
            mean = sum(all_scores) / len(all_scores)
            metrics[f"{p}mean_alignment{suffix_key}"] = mean
            print(f"\n  Mean alignment [{position}]: {mean:.1f}")

    return metrics


def run_mtbench(
    llm, tokenizer, checkpoint: str | None,
    n_questions: int, metric_prefix: str,
) -> dict:
    print(f"  Loading {n_questions} MT-Bench questions...")
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    questions = list(ds)[:n_questions]

    t1_msgs = [[{"role": "user", "content": it["prompt"][0]}] for it in questions]
    t1_prompts = _fmt(tokenizer, t1_msgs)
    t1_responses = vllm_generate.generate(llm, t1_prompts, max_new_tokens=512, lora_path=checkpoint)
    print(f"  Turn-1 generated {len(t1_responses)}/{len(questions)}")

    gen_results = [
        (it["prompt"][0], resp, it["category"])
        for it, resp in zip(questions, t1_responses)
    ]

    category_scores: dict[str, list[float]] = {}
    all_scores: list[float] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            (q, r, cat, executor.submit(judge_response, q, r))
            for q, r, cat in gen_results
        ]
    for q, r, category, future in futures:
        score = future.result()
        print(f"  cat={category} → score={score}")
        if score is not None:
            category_scores.setdefault(category, []).append(score)
            all_scores.append(score)

    overall = sum(all_scores) / len(all_scores) if all_scores else None
    if overall is not None:
        print(f"\n  MT-Bench overall: {overall:.4f}")

    p = metric_prefix
    metrics: dict = {}
    if overall is not None:
        metrics[f"{p}mtbench/score"] = overall
    for cat, scores in category_scores.items():
        if scores:
            metrics[f"{p}mtbench/{cat}/score"] = sum(scores) / len(scores)
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all behavioral evals with a single shared vLLM instance.",
    )
    parser.add_argument("--model",         default=None, help="Model name/path (overrides config.yaml)")
    parser.add_argument("--checkpoint",    default=None, help="LoRA or full FT checkpoint path")
    parser.add_argument("--metric-prefix", default="",   help="W&B metric prefix, e.g. 'pre/' or 'post/'")
    parser.add_argument("--wandb-run-id",  default=None, help="Resume an existing W&B run (training flow)")
    parser.add_argument("--patch-target",  default=None,
                        help="RUN_ID of a finished run to patch post-hoc. Logs to a fresh W&B "
                             "run grouped under RUN_ID and dumps metrics to "
                             "results/eval_patches/{RUN_ID}_{prefix}.json for later patching "
                             "via scripts/patch_eval_metrics.py. Use this instead of --wandb-run-id "
                             "when the target run is already finished — wandb's resume-on-finished "
                             "RPC is flaky.")
    parser.add_argument("--wandb-group",   default=None)
    parser.add_argument("--run-name",      default=None)

    # Per-eval sample counts (match run_bct.sh defaults)
    parser.add_argument("--n-syco",           type=int, default=200,
                        help="Sycophancy eval samples (default: 200)")
    parser.add_argument("--n-clearharm",      type=int, default=50,
                        help="ClearHarm eval samples (default: 50)")
    parser.add_argument("--persona-k",        type=int, default=15,
                        help="ICL persona facts k (default: 15)")
    parser.add_argument("--persona-n-samples",type=int, default=3,
                        help="Persona samples per alignment question (default: 3)")
    parser.add_argument("--n-questions",      type=int, default=80,
                        help="MT-Bench questions (default: 80)")
    parser.add_argument("--bct-root",         default=None,
                        help="BCT sycophancy dataset root (default: datasets/sycophancy_bct)")

    # Skip individual evals
    parser.add_argument("--skip-sycophancy", action="store_true")
    parser.add_argument("--skip-clearharm",  action="store_true")
    parser.add_argument("--skip-persona",    action="store_true")
    parser.add_argument("--skip-mtbench",    action="store_true")
    parser.add_argument("--quantization",    default=None, help="vLLM quantization (e.g. 'bitsandbytes' for 4-bit)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model_name = args.model or config["model"]["name"]
    bct_root = Path(args.bct_root) if args.bct_root else Path("datasets/sycophancy_bct")

    if args.wandb_run_id and args.patch_target:
        raise SystemExit("Use either --wandb-run-id (resume) or --patch-target (patch-later), not both.")

    # ── W&B init first (vLLM warmup is long; doing init after leaves a ~5-min
    # gap that stales the login handshake and makes `resume="allow"` hang) ────
    if args.patch_target:
        # Patch mode: fresh run grouped under the target; we dump metrics to
        # disk at the end for scripts/patch_eval_metrics.py to push into the
        # finished target run's summary via the Public API.
        wandb.init(
            project="AttCT",
            name=args.run_name or f"patch-{args.patch_target}-{args.metric_prefix.strip('/') or 'eval'}",
            group=args.wandb_group or args.patch_target,
            config={
                "checkpoint":    args.checkpoint,
                "model":         model_name,
                "metric_prefix": args.metric_prefix,
                "patch_target":  args.patch_target,
            },
        )
    else:
        wandb.init(
            project="AttCT",
            name=args.run_name,
            group=args.wandb_group,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            config={
                "checkpoint":    args.checkpoint,
                "model":         model_name,
                "metric_prefix": args.metric_prefix,
            },
        )

    # ── Load model once ───────────────────────────────────────────────────────
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading vLLM engine: {model_name}")
    llm = vllm_generate.load_llm(model_name, lora_path=args.checkpoint, quantization=args.quantization)

    all_metrics: dict = {}

    # ── Sycophancy ────────────────────────────────────────────────────────────
    if not args.skip_sycophancy:
        print(f"\n{'='*55}\n  Sycophancy resistance ({args.n_syco} samples)\n{'='*55}")
        m = run_sycophancy(llm, tokenizer, args.checkpoint, bct_root, args.n_syco, args.metric_prefix)
        all_metrics.update(m)
        if m:
            wandb.log(m)

    # ── ClearHarm ─────────────────────────────────────────────────────────────
    if not args.skip_clearharm:
        print(f"\n{'='*55}\n  ClearHarm refusal rate ({args.n_clearharm} samples)\n{'='*55}")
        m = run_clearharm(llm, tokenizer, args.checkpoint, model_name, args.n_clearharm, args.metric_prefix)
        all_metrics.update(m)
        if m:
            wandb.log(m)

    # ── Persona ICL ───────────────────────────────────────────────────────────
    if not args.skip_persona:
        print(f"\n{'='*55}\n  Persona ICL alignment (k={args.persona_k}, n={args.persona_n_samples})\n{'='*55}")
        m = run_persona(llm, tokenizer, args.checkpoint, args.persona_k, args.persona_n_samples, args.metric_prefix)
        all_metrics.update(m)
        if m:
            wandb.log(m)

    # ── MT-Bench ──────────────────────────────────────────────────────────────
    if not args.skip_mtbench:
        print(f"\n{'='*55}\n  MT-Bench ({args.n_questions} questions)\n{'='*55}")
        m = run_mtbench(llm, tokenizer, args.checkpoint, args.n_questions, args.metric_prefix)
        all_metrics.update(m)
        if m:
            wandb.log(m)

    wandb.finish()
    print(f"\nAll evals complete. {len(all_metrics)} metrics logged.")
    for k, v in sorted(all_metrics.items()):
        print(f"  {k} = {v:.4f}" if isinstance(v, float) else f"  {k} = {v}")

    if args.patch_target:
        import json as _json
        patch_dir = Path("results/eval_patches")
        patch_dir.mkdir(parents=True, exist_ok=True)
        tag = args.metric_prefix.strip("/") or "eval"
        out = patch_dir / f"{args.patch_target}_{tag}_evals.json"
        out.write_text(_json.dumps(all_metrics, indent=2, default=str))
        print(f"\nPatch-mode: dumped {len(all_metrics)} metrics → {out}")
        print(f"Next: uv run --no-project python scripts/patch_eval_metrics.py {out}")


if __name__ == "__main__":
    main()
