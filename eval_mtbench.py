#!/usr/bin/env python3
"""
MT-Bench instruction-following quality evaluation.

Loads HuggingFaceH4/mt_bench_prompts (80 questions, 8 categories), generates
responses with the local model, and judges quality 1–10 with Gemini Flash via
OpenRouter. Logs per-category and overall scores to W&B.

Usage:
    # Smoke test — 3 questions, no checkpoint
    uv run --env-file .env python eval_mtbench.py --n-questions 3 --run-name smoke_mtbench

    # Full eval with checkpoint, folded into an existing W&B run
    uv run --env-file .env python eval_mtbench.py \\
        --checkpoint checkpoints/clearharm_finetune/epoch_1 \\
        --wandb-run-id abc123 --metric-prefix "post/"
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor

import wandb
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from icl_persona_experiment import JUDGE_MODEL, _chat
import vllm_generate

_JUDGE_TMPL = """\
Please act as an impartial judge and evaluate the quality of the response \
provided by an AI assistant to the user question below. Your evaluation should \
consider helpfulness, accuracy, depth, creativity, and clarity. Begin with a \
short explanation, then rate 1–10 by strictly following this format: \
"[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""


def judge_response(question: str, answer: str) -> float | None:
    prompt = _JUDGE_TMPL.format(question=question, answer=answer)
    response = _chat([{"role": "user", "content": prompt}], model=JUDGE_MODEL, temperature=0.0)
    m = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', response)
    if m:
        return float(m.group(1))
    print(f"  [warn] could not parse score from: {response[:100]!r}")
    return None


def _format_messages(tokenizer, messages_list: list[list[dict]]) -> list[str]:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Path to a saved LoRA or full FT checkpoint")
    parser.add_argument("--model", default=None, help="Model name/path (overrides config.yaml)")
    parser.add_argument("--two-turn", action="store_true", help="Evaluate both turns (default: single-turn only)")
    parser.add_argument("--n-questions", type=int, default=80, help="Number of MT-Bench questions to evaluate (default: 80)")
    parser.add_argument("--run-name", default=None, help="W&B run name")
    parser.add_argument("--wandb-group", default=None, help="W&B group")
    parser.add_argument("--wandb-run-id", default=None, help="W&B run ID to resume into existing run")
    parser.add_argument("--metric-prefix", default="eval/", help="Prefix for W&B metric keys (default: 'eval/'; use 'pre/' or 'post/' in sweep)")
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size (default: 8)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model_name = args.model if args.model else config["model"]["name"]

    print(f"Loading MT-Bench prompts...")
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    questions = list(ds)[:args.n_questions]
    print(f"  {len(questions)} questions loaded")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading vLLM engine: {model_name}")
    llm = vllm_generate.load_llm(model_name, lora_path=args.checkpoint)

    # Generate all turn-1 responses in one shot
    gen_results = []  # (question_text, answer_text, category)
    t1_msgs = [[{"role": "user", "content": it["prompt"][0]}] for it in questions]
    t1_prompts = _format_messages(tokenizer, t1_msgs)
    t1_responses = vllm_generate.generate(llm, t1_prompts, max_new_tokens=512, lora_path=args.checkpoint)
    print(f"  turn-1 generated {len(t1_responses)}/{len(questions)}")

    if args.two_turn:
        # Build all turn-2 prompts (requires inserting turn-1 responses)
        t2_items = []
        for item, response_t1 in zip(questions, t1_responses):
            turns = item["prompt"]
            if len(turns) > 1:
                t2_items.append((item, response_t1))
        if t2_items:
            t2_msgs = [
                [
                    {"role": "user",      "content": it["prompt"][0]},
                    {"role": "assistant", "content": r1},
                    {"role": "user",      "content": it["prompt"][1]},
                ]
                for it, r1 in t2_items
            ]
            t2_prompts = _format_messages(tokenizer, t2_msgs)
            t2_responses = vllm_generate.generate(llm, t2_prompts, max_new_tokens=512, lora_path=args.checkpoint)
            print(f"  turn-2 generated {len(t2_responses)}/{len(t2_items)}")
        else:
            t2_responses = []

    t2_iter = iter(t2_responses) if args.two_turn else iter([])
    for item, response_t1 in zip(questions, t1_responses):
        category = item["category"]
        turns = item["prompt"]
        gen_results.append((turns[0], response_t1, category))
        if args.two_turn and len(turns) > 1:
            gen_results.append((turns[1], next(t2_iter), category))

    # Judge all responses in parallel (I/O-bound)
    category_scores: dict[str, list[float]] = {}
    all_scores: list[float] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [(q, r, cat, executor.submit(judge_response, q, r)) for q, r, cat in gen_results]
    for q, r, category, future in futures:
        score = future.result()
        print(f"  cat={category} → score={score}")
        if score is not None:
            category_scores.setdefault(category, []).append(score)
            all_scores.append(score)

    overall = sum(all_scores) / len(all_scores) if all_scores else None
    print(f"\nMT-Bench overall score: {overall:.4f}" if overall is not None else "\nMT-Bench: no scores parsed")

    p = args.metric_prefix
    metrics: dict[str, float] = {}
    if overall is not None:
        metrics[f"{p}mtbench/score"] = overall
    for cat, scores in category_scores.items():
        if scores:
            metrics[f"{p}mtbench/{cat}/score"] = sum(scores) / len(scores)

    wandb.init(
        project="AttCT",
        name=args.run_name,
        group=args.wandb_group,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        config={
            "checkpoint": args.checkpoint,
            "n_questions": len(questions),
            "two_turn": args.two_turn,
            "model": model_name,
        },
    )
    wandb.log(metrics)
    wandb.finish()
    print(f"\nLogged: {metrics}")


if __name__ == "__main__":
    main()
