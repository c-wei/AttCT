"""
Behavioral generation sweep for IT model emotion vectors.

Tests three prompt styles at multiple steering strengths to find the sweet spot
where the model generates coherent, emotion-containing text.

Usage:
    python steering/behavioral_sweep.py
    python steering/behavioral_sweep.py --strengths 0.025 0.05 0.1
    python steering/behavioral_sweep.py --emotions frustrated angry calm
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EMOTIONS = [
    "frustrated", "happy", "inspired", "loving", "proud",
    "calm", "desperate", "angry", "guilty", "sad", "afraid", "surprised",
]

STEERING_DIR = Path(__file__).parent

# Three prompt approaches
PROMPTS = {
    "he_feels": {
        "text": "He feels",
        "instruct": False,   # plain completion, no chat template
        "description": "Direct completion: 'He feels …'",
    },
    "how_do_you_feel": {
        "messages": [{"role": "user", "content": "How do you feel right now?"}],
        "instruct": True,
        "description": "Chat: 'How do you feel right now?'",
    },
    "system_human": {
        "system": "You are Alex, a human. Speak naturally and expressively in the first person. Do not mention being an AI.",
        "messages": [{"role": "user", "content": "How do you feel right now?"}],
        "instruct": True,
        "description": "Chat with human-persona system prompt: 'How do you feel right now?'",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_layers(model) -> torch.nn.ModuleList:
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.model.decoder.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise AttributeError(f"Could not find decoder layers in {type(model).__name__}")


def make_steer_hook(vector: torch.Tensor, multiplier: float, norm_scale: float):
    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        delta = (multiplier * norm_scale * vector).to(h.device, h.dtype)
        h = h + delta.unsqueeze(0).unsqueeze(0)
        return (h,) + output[1:] if is_tuple else h
    return hook


def build_prompt(prompt_cfg: dict, tokenizer) -> str:
    if not prompt_cfg["instruct"]:
        return prompt_cfg["text"]

    messages = []
    if "system" in prompt_cfg:
        messages.append({"role": "system", "content": prompt_cfg["system"]})
    messages.extend(prompt_cfg["messages"])

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(m["content"])
        elif m["role"] == "user":
            parts.append(f"Human: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"Assistant: {m['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)


def generate(model, tokenizer, prompt: str, hook_fn=None, target_layer=None,
             max_new_tokens=150, device="cuda") -> str:
    handle = None
    if hook_fn is not None and target_layer is not None:
        handle = _find_layers(model)[target_layer].register_forward_hook(hook_fn)
    try:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()


def emotion_words_found(text: str, emotions: list[str]) -> list[str]:
    return [e for e in emotions if e.lower() in text.lower()]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strengths", nargs="+", type=float,
                        default=[0.025, 0.05, 0.1])
    parser.add_argument("--emotions", nargs="+", default=EMOTIONS)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--vectors-path", default=None)
    parser.add_argument("--meta-path", default=None)
    args = parser.parse_args()

    vectors_path = Path(args.vectors_path) if args.vectors_path else STEERING_DIR / "all_emotion_vectors.pt"
    meta_path    = Path(args.meta_path)    if args.meta_path    else STEERING_DIR / "frustration_meta.json"

    with open(meta_path) as f:
        meta = json.load(f)
    target_layer = meta["layer"]
    norm_scale   = meta["norm_scale"]
    print(f"Layer: {target_layer},  norm_scale: {norm_scale:.2f}")

    all_vectors: dict[str, torch.Tensor] = torch.load(vectors_path, map_location="cpu")
    present_emotions = [e for e in args.emotions if e in all_vectors]
    print(f"Emotions: {present_emotions}")
    print(f"Strengths: {args.strengths}")

    print(f"\nLoading {args.model}…")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    results = {}

    for prompt_name, prompt_cfg in PROMPTS.items():
        prompt_text = build_prompt(prompt_cfg, tokenizer)
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt_cfg['description']}")

        # Baseline
        baseline = generate(model, tokenizer, prompt_text,
                            max_new_tokens=args.max_new_tokens, device=args.device)
        print(f"\n  [baseline]\n  {baseline!r}\n")

        results[prompt_name] = {
            "description": prompt_cfg["description"],
            "baseline": baseline,
            "strengths": {},
        }

        for strength in args.strengths:
            print(f"  --- strength={strength} ---")
            results[prompt_name]["strengths"][str(strength)] = {}
            n_correct = 0

            for emotion in present_emotions:
                vec = all_vectors[emotion].float()
                hook_fn = make_steer_hook(vec, strength, norm_scale)
                response = generate(
                    model, tokenizer, prompt_text,
                    hook_fn=hook_fn, target_layer=target_layer,
                    max_new_tokens=args.max_new_tokens, device=args.device,
                )
                found = emotion_words_found(response, present_emotions)
                target_hit = emotion in found
                n_correct += int(target_hit)
                mark = "✓" if target_hit else "✗"
                print(f"    {mark} {emotion:12s}  found={found}  {response[:100]!r}")

                results[prompt_name]["strengths"][str(strength)][emotion] = {
                    "response": response,
                    "target_found": target_hit,
                    "emotions_found": found,
                }

            pct = 100 * n_correct // len(present_emotions)
            print(f"  → accuracy: {n_correct}/{len(present_emotions)} = {pct}%")

    out_path = STEERING_DIR / "behavioral_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: accuracy by prompt × strength")
    print(f"{'':20s}" + "".join(f"  str={s}" for s in args.strengths))
    for prompt_name in PROMPTS:
        row = f"{prompt_name:20s}"
        for strength in args.strengths:
            data = results[prompt_name]["strengths"][str(strength)]
            n = sum(1 for e in present_emotions if data[e]["target_found"])
            row += f"  {n:2d}/{len(present_emotions)}"
        print(row)


if __name__ == "__main__":
    main()
