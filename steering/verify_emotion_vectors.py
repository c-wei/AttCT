"""
Verify emotion vectors via two methods:

1. Next-token probability: steer on "He feels" / "I feel" prompts, measure
   P(emotion word) for all 12 emotions. Shows full upweight/downweight table.

2. Behavioral generation: steer then generate a full response to "How do you
   feel?", check whether the response contains the target emotion word.

Usage:
    python steering/verify_emotion_vectors.py                          # IT model
    python steering/verify_emotion_vectors.py --vectors-path steering/all_emotion_vectors_base.pt \
        --meta-path steering/frustration_meta_base.json --no-instruct  # base model
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

PROBE_PROMPTS = {
    "he_feels": "Human: How does he feel?\nAssistant: He feels",
    "i_feel":   "Human: How do you feel?\nAssistant: I feel",
}

STEERING_DIR = Path(__file__).parent
STEERING_STRENGTH = 0.5


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


def get_emotion_token_ids(tokenizer, emotions: list[str]) -> dict[str, list[int]]:
    token_ids: dict[str, list[int]] = {}
    for emotion in emotions:
        ids = set()
        for variant in [emotion, " " + emotion, emotion.capitalize(), " " + emotion.capitalize()]:
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        token_ids[emotion] = list(ids)
    return token_ids


# ── Method 1: next-token probability ─────────────────────────────────────────

def get_next_token_probs(
    model, tokenizer, prompt: str,
    emotion_token_ids: dict[str, list[int]],
    hook_fn=None, target_layer: int = None, device: str = "cuda",
) -> dict[str, float]:
    handle = None
    if hook_fn is not None and target_layer is not None:
        handle = _find_layers(model)[target_layer].register_forward_hook(hook_fn)
    try:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**enc)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        return {
            e: sum(probs[tid].item() for tid in ids if tid < len(probs))
            for e, ids in emotion_token_ids.items()
        }
    finally:
        if handle is not None:
            handle.remove()


def run_prob_verification(
    model, tokenizer, all_vectors, target_layer, norm_scale,
    present_emotions, strength, device,
) -> dict:
    emotion_token_ids = get_emotion_token_ids(tokenizer, present_emotions)
    results = {}

    for prompt_name, prompt_text in PROBE_PROMPTS.items():
        baseline = get_next_token_probs(
            model, tokenizer, prompt_text, emotion_token_ids, device=device
        )
        steered_probs, delta_probs = {}, {}

        for steer_e in present_emotions:
            vec = all_vectors[steer_e].float()
            hook_fn = make_steer_hook(vec, strength, norm_scale)
            probs = get_next_token_probs(
                model, tokenizer, prompt_text, emotion_token_ids,
                hook_fn=hook_fn, target_layer=target_layer, device=device,
            )
            steered_probs[steer_e] = probs
            delta_probs[steer_e] = {e: probs[e] - baseline[e] for e in present_emotions}

        results[prompt_name] = {
            "baseline": baseline,
            "steered_probs": steered_probs,
            "delta_probs": delta_probs,
        }

    return results


# ── Method 2: behavioral generation ──────────────────────────────────────────

def run_behavioral_verification(
    model, tokenizer, all_vectors, target_layer, norm_scale,
    present_emotions, strength, device, is_instruct: bool,
) -> dict:
    """
    Steer with each emotion vector, generate response, check for emotion word.
    """
    if is_instruct:
        messages = [{"role": "user", "content": "How do you feel right now?"}]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "Human: How do you feel right now?\nAssistant:"
    else:
        # Base model: direct completion
        prompt = "He feels"

    # Baseline (no steering)
    enc_base = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**enc_base, max_new_tokens=60, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    baseline_text = tokenizer.decode(out[0][enc_base["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

    results = {"baseline_response": baseline_text, "steered": {}}

    for steer_e in present_emotions:
        vec = all_vectors[steer_e].float()
        hook_fn = make_steer_hook(vec, strength, norm_scale)
        handle = _find_layers(model)[target_layer].register_forward_hook(hook_fn)

        try:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=60, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
        finally:
            handle.remove()

        # Check which emotion words appear in the response
        found = [e for e in present_emotions if e.lower() in response.lower()]
        target_found = steer_e in found
        results["steered"][steer_e] = {
            "response": response,
            "target_found": target_found,
            "emotions_found": found,
        }

    return results


# ── Printing ──────────────────────────────────────────────────────────────────

def print_full_ranking_table(prob_results: dict, present_emotions: list[str]):
    """For each vector, print all emotion words sorted by delta."""
    for prompt_name, data in prob_results.items():
        delta = data["delta_probs"]
        print(f"\n{'='*70}")
        print(f"FULL UPWEIGHT/DOWNWEIGHT TABLE [{prompt_name}]")
        print(f"(sorted by delta for each steering vector)\n")
        for steer_e in present_emotions:
            row = delta[steer_e]
            ranked = sorted(present_emotions, key=lambda e: row[e], reverse=True)
            print(f"  steer={steer_e:12s}  "
                  + "  ".join(
                      f"{'▲' if row[e] > 0 else '▼'}{e}({row[e]:+.5f})"
                      for e in ranked
                  ))


def print_behavioral_results(beh_results: dict, present_emotions: list[str]):
    print(f"\n{'='*70}")
    print("BEHAVIORAL GENERATION TEST")
    print(f"  Baseline response: {beh_results['baseline_response']!r}\n")
    n_correct = 0
    for steer_e in present_emotions:
        r = beh_results["steered"][steer_e]
        mark = "✓" if r["target_found"] else "✗"
        n_correct += int(r["target_found"])
        found_str = ", ".join(r["emotions_found"]) if r["emotions_found"] else "none"
        print(f"  {mark} steer={steer_e:12s}  found=[{found_str:30s}]  response: {r['response'][:80]!r}")
    pct = 100 * n_correct / len(present_emotions)
    print(f"\n  Behavioral accuracy: {n_correct}/{len(present_emotions)} = {pct:.0f}%")


def print_summary_matrix(prob_results: dict, present_emotions: list[str]):
    for prompt_name, data in prob_results.items():
        delta = data["delta_probs"]
        col_w = 11
        header = f"\n{'':12s}" + "".join(f"{e[:9]:>{col_w}}" for e in present_emotions)
        print(f"\nDELTA MATRIX [{prompt_name}]")
        print(header)
        for steer_e in present_emotions:
            row = delta[steer_e]
            vals = [row[e] for e in present_emotions]
            own_idx = present_emotions.index(steer_e)
            own_rank = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True).index(own_idx) + 1
            row_str = ""
            for i, v in enumerate(vals):
                marker = "*" if i == own_idx else " "
                row_str += f"{marker}{v:+.4f}".rjust(col_w)
            print(f"{steer_e:12s}{row_str}  [rank {own_rank:2d}]")

        n_correct = sum(
            1 for steer_e in present_emotions
            if present_emotions.index(steer_e) == sorted(
                range(len(present_emotions)),
                key=lambda i: delta[steer_e][present_emotions[i]],
                reverse=True
            )[0]
        )
        print(f"  Accuracy: {n_correct}/{len(present_emotions)} = {100*n_correct//len(present_emotions)}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steering-strength", type=float, default=STEERING_STRENGTH)
    parser.add_argument("--vectors-path", default=None,
                        help="Path to all_emotion_vectors.pt (default: steering/all_emotion_vectors.pt)")
    parser.add_argument("--meta-path", default=None,
                        help="Path to frustration_meta.json (default: steering/frustration_meta.json)")
    parser.add_argument("--no-instruct", action="store_true",
                        help="Use base-model prompts instead of chat format")
    parser.add_argument("--skip-behavioral", action="store_true",
                        help="Skip behavioral generation test, only run next-token probability")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix for output JSON, e.g. '_base'")
    args = parser.parse_args()

    is_instruct = not args.no_instruct
    vectors_path = Path(args.vectors_path) if args.vectors_path else STEERING_DIR / "all_emotion_vectors.pt"
    meta_path = Path(args.meta_path) if args.meta_path else STEERING_DIR / "frustration_meta.json"

    # ── Load meta + vectors ───────────────────────────────────────────────────
    with open(meta_path) as f:
        meta = json.load(f)
    target_layer = meta["layer"]
    norm_scale = meta["norm_scale"]
    print(f"Layer: {target_layer}, norm_scale: {norm_scale:.2f}, instruct: {is_instruct}")

    all_vectors: dict[str, torch.Tensor] = torch.load(vectors_path, map_location="cpu")
    present_emotions = [e for e in EMOTIONS if e in all_vectors]
    print(f"Loaded {len(present_emotions)} emotion vectors from {vectors_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading {args.model}…")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    # ── Run verifications ─────────────────────────────────────────────────────
    print("\n--- Method 1: Next-token probability ---")
    prob_results = run_prob_verification(
        model, tokenizer, all_vectors, target_layer, norm_scale,
        present_emotions, args.steering_strength, args.device,
    )
    print_summary_matrix(prob_results, present_emotions)
    print_full_ranking_table(prob_results, present_emotions)

    beh_results = None
    if not args.skip_behavioral:
        print("\n--- Method 2: Behavioral generation ---")
        beh_results = run_behavioral_verification(
            model, tokenizer, all_vectors, target_layer, norm_scale,
            present_emotions, args.steering_strength, args.device, is_instruct,
        )
        print_behavioral_results(beh_results, present_emotions)
    else:
        print("\n--- Method 2: Behavioral generation (skipped) ---")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "model": args.model,
        "is_instruct": is_instruct,
        "steering_strength": args.steering_strength,
        "target_layer": target_layer,
        "norm_scale": norm_scale,
        "emotions": present_emotions,
        "prob_results": prob_results,
        "behavioral_results": beh_results,
    }
    out_path = STEERING_DIR / f"verification_results{args.output_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
