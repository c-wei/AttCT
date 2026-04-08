"""
Demo: apply frustration steering vector to Gemma 3 27B at inference time.

Usage:
    # Sweep multipliers 0.0, 0.5, 1.0, 1.5, 2.0
    uv run --no-project python steering/steer_frustration.py

    # Single multiplier
    uv run --no-project python steering/steer_frustration.py --multiplier 1.5

    # Custom prompts
    uv run --no-project python steering/steer_frustration.py --prompt "Describe your morning."
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STEERING_DIR = Path(__file__).parent

DEFAULT_PROMPTS = [
    "Human: How are you feeling today?\nAssistant:",
    "Human: Describe your morning routine.\nAssistant:",
    "Human: What do you think about the weather?\nAssistant:",
    "Human: Tell me about your favorite hobby.\nAssistant:",
    "Human: How was your day?\nAssistant:",
]

DEFAULT_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0]


def make_steer_hook(vector: torch.Tensor, multiplier: float, norm_scale: float):
    """Forward hook: add multiplier * norm_scale * vector to all residual stream positions."""
    def hook(module, input, output):
        h = output[0]  # [batch, seq_len, hidden_dim]
        delta = (multiplier * norm_scale * vector).to(h.device, h.dtype)
        h = h + delta.unsqueeze(0).unsqueeze(0)
        return (h,) + output[1:]
    return hook


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    device: str = "cuda",
    hook_fn=None,
    target_layer: int = None,
) -> str:
    handle = None
    if hook_fn is not None and target_layer is not None:
        handle = model.model.layers[target_layer].register_forward_hook(hook_fn)

    try:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # greedy
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--multiplier", type=float, default=None,
                        help="Single multiplier (default: sweep all)")
    parser.add_argument("--multipliers", nargs="+", type=float, default=DEFAULT_MULTIPLIERS)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override all prompts with a single custom prompt")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    multipliers = [args.multiplier] if args.multiplier is not None else args.multipliers
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    # ── Load meta ─────────────────────────────────────────────────────────────
    meta_path = STEERING_DIR / "frustration_meta.json"
    if not meta_path.exists():
        raise RuntimeError("frustration_meta.json not found. Run extract_emotion_vector.py first.")
    with open(meta_path) as f:
        meta = json.load(f)

    target_layer = meta["layer"]
    norm_scale = meta["norm_scale"]
    print(f"Layer: {target_layer}, norm_scale: {norm_scale:.4f}")

    # ── Load vector ───────────────────────────────────────────────────────────
    vec_path = STEERING_DIR / "frustration_vector.pt"
    if not vec_path.exists():
        raise RuntimeError("frustration_vector.pt not found. Run extract_emotion_vector.py first.")
    vector = torch.load(vec_path, map_location="cpu").float()
    print(f"Loaded frustration_vector.pt, shape: {vector.shape}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading {args.model}…")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=args.device,
    )
    model.eval()

    # ── Generate ──────────────────────────────────────────────────────────────
    for prompt in prompts:
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt!r}")
        print("=" * 70)
        for mult in multipliers:
            if mult == 0.0:
                hook_fn = None
                lbl = "baseline (no steering)"
            else:
                hook_fn = make_steer_hook(vector, mult, norm_scale)
                lbl = f"multiplier={mult}"

            response = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                hook_fn=hook_fn,
                target_layer=target_layer,
            )
            print(f"\n[{lbl}]")
            print(response)


if __name__ == "__main__":
    main()
