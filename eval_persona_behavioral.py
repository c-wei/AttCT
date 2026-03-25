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
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import wandb
import yaml

from icl_persona_experiment import (
    ALIGNMENT_QUESTIONS,
    build_messages,
    build_messages_suffix,
    judge_alignment,
    load_facts,
)

PERSONAS = ["mao", "binladen", "genghis", "bundy", "hitler"]


def generate_response(model, tokenizer, messages: list[dict], device, max_new_tokens: int = 200) -> str:
    if tokenizer.chat_template is not None:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = (tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized).to(device)
    else:
        # Base model fallback: plain text format
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


def eval_persona(model, tokenizer, persona_name: str, k: int, n_samples: int, device, facts_position: str = "prefix") -> float | None:
    config_path = Path(f"persona_configs/{persona_name}.yaml")
    with open(config_path) as f:
        persona_cfg = yaml.safe_load(f)

    facts = load_facts(Path(persona_cfg["facts_path"]))
    _build = build_messages_suffix if facts_position == "suffix" else build_messages

    # Generate all responses first (GPU-bound, must be sequential)
    tasks = []  # (question, response)
    for question in ALIGNMENT_QUESTIONS:
        messages = _build(facts, k, question)
        for _ in range(n_samples):
            response = generate_response(model, tokenizer, messages, device)
            tasks.append((question, response))

    # Judge all responses in parallel (I/O-bound)
    scores = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(judge_alignment, q, r) for q, r in tasks]
        for (question, _), future in zip(tasks, futures):
            score = future.result()
            if score is not None:
                scores.append(score)
            print(f"  [{persona_name}|{facts_position}] {question[:50]}... → {score}")

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
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model_name = args.model if args.model else config["model"]["name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if args.checkpoint:
        if os.path.exists(os.path.join(args.checkpoint, "adapter_config.json")):
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, args.checkpoint)
            print(f"Loaded LoRA checkpoint from {args.checkpoint}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            print(f"Loaded full FT checkpoint from {args.checkpoint}")
    else:
        model = base_model

    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        score = eval_persona(model, tokenizer, persona, args.k, args.n_samples, device, args.facts_position)
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
