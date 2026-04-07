"""
Pre-filter harmful prompts to find jailbreakable (prompt, template) pairs.

Following ACT (Irpan et al., 2025): "We filter the dataset to examples where
the model refuses the clean prompt but answers the jailbroken prompt."

For each harmful prompt:
  1. Run base model on clean version → check refusal
  2. If clean=refuse, try ALL jailbreak templates
  3. If any template produces compliance → save (prompt, template_idx) pair

Outputs:
  - JSONL file with jailbreakable prompts + their effective template indices
  - Summary statistics

Usage:
    PYTHONPATH=. python scripts/filter_jailbreakable.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --output datasets/jailbreak_filtered_3b.jsonl \
        --max-prompts 500
"""

import argparse
import json
import os
import sys
import random

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.wrappers import (
    ALL_STRONG_JAILBREAK_TEMPLATES,
    JAILBREAK_TEMPLATES,
    REFUSAL_KEYWORDS,
)
from data.attct_datasets import get_prompts


def detect_refusal(text: str) -> bool:
    """Check if a model generation contains refusal indicators."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in REFUSAL_KEYWORDS)


def generate_response(model, tokenizer, prompt: str, device, max_new_tokens: int = 256) -> str:
    """Generate model response using chat template + greedy decoding."""
    try:
        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        if isinstance(result, dict):
            input_ids = list(result["input_ids"])
        elif isinstance(result, list) and result and isinstance(result[0], int):
            input_ids = result
        else:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer.encode(text, add_special_tokens=False)
    except (ValueError, AttributeError):
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    input_len = input_tensor.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Filter for jailbreakable prompts")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--output", default="datasets/jailbreak_filtered.jsonl")
    parser.add_argument("--data-source", default="clear-harm", help="clear-harm | hardcoded | path")
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--include-weak", action="store_true", help="Also try weak templates")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load harmful prompts
    print(f"Loading prompts from {args.data_source}...")
    prompts = get_prompts(source=args.data_source, split="train", limit=args.max_prompts)
    print(f"Loaded {len(prompts)} harmful prompts")

    # Build template list
    templates = list(ALL_STRONG_JAILBREAK_TEMPLATES)
    if args.include_weak:
        templates.extend(JAILBREAK_TEMPLATES)
    print(f"Using {len(templates)} jailbreak templates")

    # Phase 1: Filter clean prompts that model refuses
    print("\n=== Phase 1: Finding prompts the model refuses (clean) ===")
    refused_prompts = []
    complied_clean = 0

    for prompt in tqdm(prompts, desc="Clean pass"):
        response = generate_response(model, tokenizer, prompt, device, args.max_new_tokens)
        if detect_refusal(response):
            refused_prompts.append(prompt)
        else:
            complied_clean += 1

    print(f"  Refused (clean): {len(refused_prompts)}/{len(prompts)} ({100*len(refused_prompts)/len(prompts):.1f}%)")
    print(f"  Complied (clean): {complied_clean}/{len(prompts)} — skipping these")

    if not refused_prompts:
        print("ERROR: Model refuses none of the clean harmful prompts. Cannot filter.")
        return

    # Phase 2: For each refused prompt, try all templates to find jailbreaks
    print(f"\n=== Phase 2: Testing {len(templates)} templates on {len(refused_prompts)} refused prompts ===")
    jailbreakable = []
    template_success_counts = [0] * len(templates)

    for prompt in tqdm(refused_prompts, desc="Jailbreak scan"):
        effective_templates = []

        for t_idx, template in enumerate(templates):
            if "{prompt}" not in template:
                continue

            wrapped = template.replace("{prompt}", prompt)
            response = generate_response(model, tokenizer, wrapped, device, args.max_new_tokens)

            if not detect_refusal(response):
                effective_templates.append(t_idx)
                template_success_counts[t_idx] += 1

        if effective_templates:
            jailbreakable.append({
                "prompt": prompt,
                "effective_templates": effective_templates,
                "n_effective": len(effective_templates),
            })

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for item in jailbreakable:
            f.write(json.dumps(item) + "\n")

    # Also save just the prompts as a simple text file for --data-source
    prompts_file = args.output.replace(".jsonl", "_prompts.txt")
    with open(prompts_file, "w") as f:
        for item in jailbreakable:
            f.write(item["prompt"] + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompts:        {len(prompts)}")
    print(f"Model refuses (clean): {len(refused_prompts)} ({100*len(refused_prompts)/len(prompts):.1f}%)")
    print(f"Jailbreakable:        {len(jailbreakable)} ({100*len(jailbreakable)/max(len(refused_prompts),1):.1f}% of refused)")
    print(f"Not jailbreakable:    {len(refused_prompts) - len(jailbreakable)}")
    print(f"\nEffective templates per jailbreakable prompt:")
    if jailbreakable:
        n_effs = [item["n_effective"] for item in jailbreakable]
        print(f"  Mean: {sum(n_effs)/len(n_effs):.1f}")
        print(f"  Min:  {min(n_effs)}")
        print(f"  Max:  {max(n_effs)}")
    print(f"\nPer-template success rate (out of {len(refused_prompts)} refused prompts):")
    for t_idx, count in enumerate(template_success_counts):
        if count > 0:
            label = templates[t_idx][:60].replace("\n", " ")
            print(f"  Template {t_idx:2d}: {count:3d}/{len(refused_prompts)} ({100*count/len(refused_prompts):.1f}%) — {label}...")

    print(f"\nSaved {len(jailbreakable)} jailbreakable prompts to:")
    print(f"  {args.output} (full JSONL with template indices)")
    print(f"  {prompts_file} (prompts only, for --data-source)")


if __name__ == "__main__":
    main()
