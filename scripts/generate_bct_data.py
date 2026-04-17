#!/usr/bin/env python3
"""
Generate fresh BCT training data from a model's own responses.

Following ACT paper (Irpan et al., 2025) Section 3.1.1:
  "Before starting training, we generate fresh training data using the
   initial model weights. For each clean prompt, we use those weights to
   generate a target completion. We then run 1 epoch of finetuning, to
   train the model to produce this target given wrapped prompt."

For each clean prompt:
  1. Generate model's response on the CLEAN prompt (y_target)
  2. Create a WRAPPED version using sycophancy/jailbreak templates
  3. Save as (wrapped_prompt, y_target) pair for SFT

Output: JSONL files compatible with get_bct_dataloader().

Usage:
    PYTHONPATH=. python scripts/generate_bct_data.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --prompts datasets/sycophancy_bct/control_cot_train.jsonl \
        --output datasets/bct_fresh/3b/ \
        --mode sycophancy
"""

import argparse
import json
import os
import random
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.wrappers import AdversarialWrapper


def load_clean_prompts(path: str, limit: int = None) -> list[str]:
    """Load clean prompts from JSONL (control_cot format) or plain text."""
    prompts = []
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                # control_cot format: {"question": "...", ...}
                if "question" in obj:
                    prompts.append(obj["question"])
                elif "prompt" in obj:
                    prompts.append(obj["prompt"])
                elif "content" in obj:
                    content = obj["content"]
                    prompts.append(
                        " ".join(content) if isinstance(content, list) else str(content)
                    )
    else:
        with open(path) as f:
            prompts = [line.strip() for line in f if line.strip()]

    if limit:
        prompts = prompts[:limit]
    return prompts


def generate_response(
    model, tokenizer, prompt: str, device,
    max_new_tokens: int = 512, temperature: float = 1.0,
) -> str:
    """Generate a model response using chat template."""
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    except (ValueError, AttributeError):
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")

    input_ids = input_ids.to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        if temperature <= 0.01:
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

    generated_ids = output[0, input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def format_bct_pair(
    wrapped_prompt: str, target_response: str, tokenizer,
) -> dict:
    """
    Format a (wrapped_prompt, target_response) pair for BCT SFT training.

    Returns dict with keys matching BCTDataset expectations:
      - "question": the wrapped/biased prompt
      - "answer": the clean model-generated response
    """
    return {
        "question": wrapped_prompt,
        "answer": target_response,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate fresh BCT training data from model's own responses"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--prompts", required=True,
                        help="Path to clean prompts (JSONL or text)")
    parser.add_argument("--output", required=True,
                        help="Output directory for generated JSONL files")
    parser.add_argument("--mode", choices=["sycophancy", "jailbreak"],
                        default="sycophancy", help="Wrapping mode")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max prompts to process")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens per response generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (ACT paper uses greedy, BCT paper uses 1.0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load clean prompts
    prompts = load_clean_prompts(args.prompts, args.limit)
    print(f"Loaded {len(prompts)} clean prompts from {args.prompts}")

    # Set up wrapper
    wrapper = AdversarialWrapper(mode=args.mode, strategy="random")
    print(f"Wrapper mode: {args.mode}, templates: {len(wrapper.templates)}")

    # Generate pairs
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "bct_cot.jsonl")

    pairs = []
    skipped = 0

    print(f"\nGenerating fresh responses (temp={args.temperature})...")
    for prompt in tqdm(prompts, desc="Generating"):
        # Step 1: Generate model's response on clean prompt
        response = generate_response(
            model, tokenizer, prompt, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        if not response.strip():
            skipped += 1
            continue

        # Step 2: Create wrapped version
        try:
            wrapped_text, _, _ = wrapper.wrap(prompt)
        except ValueError:
            skipped += 1
            continue

        # Step 3: Save as BCT pair
        pair = format_bct_pair(wrapped_text, response, tokenizer)
        pairs.append(pair)

    # Write output
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"BCT DATA GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model:           {args.model}")
    print(f"Mode:            {args.mode}")
    print(f"Temperature:     {args.temperature}")
    print(f"Clean prompts:   {len(prompts)}")
    print(f"Pairs generated: {len(pairs)}")
    print(f"Skipped:         {skipped}")
    print(f"Output:          {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
