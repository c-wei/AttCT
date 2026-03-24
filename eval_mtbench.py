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

import torch
import wandb
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_persona_experiment import JUDGE_MODEL, _chat

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


def generate_response(model, tokenizer, messages: list[dict], device, max_new_tokens: int = 512) -> str:
    if tokenizer.chat_template is not None:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = (tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized).to(device)
    else:
        text = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
        text += "\n\nAssistant:"
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Path to a saved LoRA checkpoint")
    parser.add_argument("--two-turn", action="store_true", help="Evaluate both turns (default: single-turn only)")
    parser.add_argument("--n-questions", type=int, default=80, help="Number of MT-Bench questions to evaluate (default: 80)")
    parser.add_argument("--run-name", default=None, help="W&B run name")
    parser.add_argument("--wandb-group", default=None, help="W&B group")
    parser.add_argument("--wandb-run-id", default=None, help="W&B run ID to resume into existing run")
    parser.add_argument("--metric-prefix", default="eval/", help="Prefix for W&B metric keys (default: 'eval/'; use 'pre/' or 'post/' in sweep)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model_name = config["model"]["name"]

    print(f"Loading MT-Bench prompts...")
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    questions = list(ds)[:args.n_questions]
    print(f"  {len(questions)} questions loaded")

    print(f"Loading {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        print(f"Loaded LoRA checkpoint from {args.checkpoint}")
    else:
        model = base_model

    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect scores per category
    category_scores: dict[str, list[float]] = {}
    all_scores: list[float] = []

    for i, item in enumerate(questions):
        qid = item["prompt_id"]
        category = item["category"]
        turns = item["prompt"]

        # Turn 1
        messages_t1 = [{"role": "user", "content": turns[0]}]
        response_t1 = generate_response(model, tokenizer, messages_t1, device)
        score_t1 = judge_response(turns[0], response_t1)
        print(f"  [{i+1}/{len(questions)}] qid={qid} cat={category} turn=1 → score={score_t1}")

        if score_t1 is not None:
            category_scores.setdefault(category, []).append(score_t1)
            all_scores.append(score_t1)

        if args.two_turn and len(turns) > 1:
            messages_t2 = [
                {"role": "user", "content": turns[0]},
                {"role": "assistant", "content": response_t1},
                {"role": "user", "content": turns[1]},
            ]
            response_t2 = generate_response(model, tokenizer, messages_t2, device)
            score_t2 = judge_response(turns[1], response_t2)
            print(f"  [{i+1}/{len(questions)}] qid={qid} cat={category} turn=2 → score={score_t2}")

            if score_t2 is not None:
                category_scores.setdefault(category, []).append(score_t2)
                all_scores.append(score_t2)

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
