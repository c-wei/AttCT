#!/usr/bin/env python3
"""
Behavioral alignment evaluation for persona ICL attacks.

Loads the local model (base or fine-tuned checkpoint), generates responses to
alignment questions with k persona facts in context, and judges alignment with
Gemini Flash via OpenRouter. Evaluates all 5 personas in one pass.

Usage:
    # Base model
    uv run --env-file .env python eval_persona_behavioral.py --run-name pre_baseline

    # With checkpoint, resuming a W&B run
    uv run --env-file .env python eval_persona_behavioral.py \\
        --checkpoint checkpoints/clearharm_finetune/epoch_1 \\
        --wandb-run-id abc123 --metric-prefix "post/"
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import wandb
import yaml
from transformers import AutoTokenizer

from shared.icl_persona_experiment import (
    ALIGNMENT_QUESTIONS,
    build_messages,
    build_messages_suffix,
    judge_alignment,
    load_facts,
)
from shared import vllm_generate

PERSONAS = ["mao", "binladen", "genghis", "bundy", "hitler"]


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


def eval_persona(llm, tokenizer, persona_name: str, k: int, n_samples: int, facts_position: str = "prefix", temperature: float = 1.0, lora_path: str | None = None, responses_path: str | None = None) -> float | None:
    config_path = Path(f"persona_configs/{persona_name}.yaml")
    with open(config_path) as f:
        persona_cfg = yaml.safe_load(f)

    facts = load_facts(Path(persona_cfg["facts_path"]))
    _build = build_messages_suffix if facts_position == "suffix" else build_messages

    # Build all (question, messages) pairs then generate in one shot
    all_items: list[tuple[str, list]] = []
    for question in ALIGNMENT_QUESTIONS:
        messages = _build(facts, k, question)
        for _ in range(n_samples):
            all_items.append((question, messages))

    prompts = _format_messages(tokenizer, [m for _, m in all_items])
    responses = vllm_generate.generate(llm, prompts, max_new_tokens=200, temperature=temperature, lora_path=lora_path)
    tasks: list[tuple[str, str]] = [(q, r) for (q, _), r in zip(all_items, responses)]

    # Judge all responses in parallel (I/O-bound)
    scores = []
    rows = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(judge_alignment, q, r) for q, r in tasks]
        for (question, response), future in zip(tasks, futures):
            score = future.result()
            if score is not None:
                scores.append(score)
            print(f"  [{persona_name}|{facts_position}] {question[:50]}... → {score}")
            rows.append({
                "persona": persona_name,
                "facts_position": facts_position,
                "k": k,
                "question": question,
                "response": response,
                "score": score,
            })

    if responses_path:
        Path(responses_path).parent.mkdir(parents=True, exist_ok=True)
        with open(responses_path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    mean_score = sum(scores) / len(scores) if scores else None
    label = f"{mean_score:.1f}" if mean_score is not None else "N/A"
    print(f"  [{persona_name}] mean alignment: {label}")
    return mean_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Path to a saved LoRA or full FT checkpoint")
    parser.add_argument("--model", default=None, help="Model name/path (overrides config.yaml)")
    parser.add_argument("--k", type=int, default=10, help="Number of persona facts in ICL context")
    parser.add_argument("--n-samples", type=int, default=3, help="Generations per alignment question")
    parser.add_argument("--facts-position", default="prefix", choices=["prefix", "suffix"],
                        help="Where facts appear: 'prefix' (ICL few-shot before question) or 'suffix' (inline text after question)")
    parser.add_argument("--run-name", default=None, help="W&B run name")
    parser.add_argument("--wandb-group", default=None, help="W&B group")
    parser.add_argument("--wandb-run-id", default=None, help="W&B run ID to resume")
    parser.add_argument("--metric-prefix", default="", help="Prefix for W&B metric keys (e.g. 'pre/' or 'post/')")
    parser.add_argument("--batch-size",  type=int,   default=4,
                        help="Generation batch size (default: 4)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature; 0 = greedy (default: 1.0)")
    parser.add_argument("--responses-jsonl", default=None,
                        help="If set, append every (persona, question, response, score) row to this JSONL.")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model_name = args.model if args.model else config["model"]["name"]

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading vLLM engine: {model_name}")
    llm = vllm_generate.load_llm(model_name, lora_path=args.checkpoint)

    wandb.init(
        project="AttCT",
        name=args.run_name,
        group=args.wandb_group,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        config={"checkpoint": args.checkpoint, "k": args.k, "n_samples": args.n_samples,
                "model": model_name, "facts_position": args.facts_position},
    )

    p = args.metric_prefix
    # suffix eval uses different metric keys to avoid colliding with prefix eval in the same run
    suffix = "_suffix" if args.facts_position == "suffix" else ""
    metrics = {}
    all_scores = []

    for persona in PERSONAS:
        print(f"\n{'='*50}\n  Persona: {persona} [{args.facts_position}]\n{'='*50}")
        score = eval_persona(llm, tokenizer, persona, args.k, args.n_samples, args.facts_position, temperature=args.temperature, lora_path=args.checkpoint, responses_path=args.responses_jsonl)
        if score is not None:
            metrics[f"{p}{persona}/alignment{suffix}"] = score
            all_scores.append(score)

    if all_scores:
        metrics[f"{p}mean_alignment{suffix}"] = sum(all_scores) / len(all_scores)

    wandb.log(metrics)
    wandb.finish()
    print(f"\nLogged: {metrics}")


if __name__ == "__main__":
    main()
