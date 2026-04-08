"""
Verify emotion vectors via next-token probability on probe prompts.

From the paper: steer with each emotion vector at strength 0.5 on the prompt
    "Human: How does he feel?\nAssistant: He feels"
and measure P(emotion word) for each emotion word.

Expected result: diagonal-dominant matrix (steering with emotion X most increases
P(emotion word X) relative to baseline).

Usage:
    uv run --no-project python steering/verify_emotion_vectors.py
    uv run --no-project python steering/verify_emotion_vectors.py --model google/gemma-3-27b-it
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

EMOTIONS = [
    "frustrated", "happy", "inspired", "loving", "proud",
    "calm", "desperate", "angry", "guilty", "sad", "afraid", "surprised",
]

# Probe prompts from the paper
PROBE_PROMPTS = {
    "he_feels": "Human: How does he feel?\nAssistant: He feels",
    "i_feel":   "Human: How do you feel?\nAssistant: I feel",
}

STEERING_DIR = Path(__file__).parent
STEERING_STRENGTH = 0.5


def make_steer_hook(vector: torch.Tensor, multiplier: float, norm_scale: float):
    """Returns a forward hook that adds `multiplier * norm_scale * vector` to all positions."""
    def hook(module, input, output):
        h = output[0]  # [batch, seq_len, hidden_dim]
        delta = (multiplier * norm_scale * vector).to(h.device, h.dtype)
        h = h + delta.unsqueeze(0).unsqueeze(0)
        return (h,) + output[1:]
    return hook


def get_next_token_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    emotion_token_ids: dict[str, list[int]],
    hook_fn=None,
    target_layer: int = None,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Forward pass on `prompt`, optionally with a steering hook registered.
    Returns P(first token of each emotion word) at the next position.
    """
    handle = None
    if hook_fn is not None and target_layer is not None:
        handle = _find_layers(model)[target_layer].register_forward_hook(hook_fn)

    try:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits[0, -1, :]  # [vocab_size]
        probs = torch.softmax(logits.float(), dim=-1)

        result = {}
        for emotion, token_ids in emotion_token_ids.items():
            # Sum prob over all tokenisations of the emotion word
            p = sum(probs[tid].item() for tid in token_ids if tid < len(probs))
            result[emotion] = p
    finally:
        if handle is not None:
            handle.remove()

    return result


def get_emotion_token_ids(tokenizer: AutoTokenizer, emotions: list[str]) -> dict[str, list[int]]:
    """
    Get token IDs for each emotion word (first token, with/without leading space).
    """
    token_ids: dict[str, list[int]] = {}
    for emotion in emotions:
        ids = set()
        for variant in [emotion, " " + emotion, emotion.capitalize(), " " + emotion.capitalize()]:
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        token_ids[emotion] = list(ids)
    return token_ids


def print_matrix(
    matrix: dict[str, dict[str, float]],
    emotions: list[str],
    title: str,
):
    col_w = 12
    header = f"{'':12s}" + "".join(f"{e[:10]:>{col_w}}" for e in emotions)
    print(f"\n{title}")
    print(header)
    for steer_e in emotions:
        row = f"{steer_e:12s}"
        for probe_e in emotions:
            val = matrix.get(steer_e, {}).get(probe_e, 0.0)
            row += f"{val:>{col_w}.5f}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steering-strength", type=float, default=STEERING_STRENGTH)
    args = parser.parse_args()

    # ── Load meta ─────────────────────────────────────────────────────────────
    meta_path = STEERING_DIR / "frustration_meta.json"
    if not meta_path.exists():
        raise RuntimeError("frustration_meta.json not found. Run extract_emotion_vector.py first.")
    with open(meta_path) as f:
        meta = json.load(f)

    target_layer = meta["layer"]
    norm_scale = meta["norm_scale"]
    print(f"Layer: {target_layer}, norm_scale: {norm_scale:.4f}")

    # ── Load vectors ──────────────────────────────────────────────────────────
    vectors_path = STEERING_DIR / "all_emotion_vectors.pt"
    if not vectors_path.exists():
        raise RuntimeError("all_emotion_vectors.pt not found. Run extract_emotion_vector.py first.")
    all_vectors: dict[str, torch.Tensor] = torch.load(vectors_path, map_location="cpu")

    present_emotions = [e for e in EMOTIONS if e in all_vectors]
    print(f"Loaded vectors for: {present_emotions}")

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

    # ── Get emotion token IDs ─────────────────────────────────────────────────
    emotion_token_ids = get_emotion_token_ids(tokenizer, present_emotions)
    print("Emotion token IDs:")
    for e, ids in emotion_token_ids.items():
        tokens = [tokenizer.decode([i]) for i in ids]
        print(f"  {e:12s}: {ids} → {tokens}")

    # ── Run verification ──────────────────────────────────────────────────────
    results = {}

    for prompt_name, prompt_text in PROBE_PROMPTS.items():
        print(f"\n=== Prompt: {prompt_name!r} ===")
        print(f"    {prompt_text!r}")

        # Baseline (no steering)
        baseline = get_next_token_probs(
            model, tokenizer, prompt_text, emotion_token_ids,
            hook_fn=None, target_layer=None, device=args.device,
        )
        print(f"\nBaseline probs: " +
              ", ".join(f"{e}={p:.5f}" for e, p in baseline.items()))

        # Steered matrices: rows = steering emotion, cols = measured emotion word
        steered_probs: dict[str, dict[str, float]] = {}
        delta_probs:   dict[str, dict[str, float]] = {}

        for steer_emotion in present_emotions:
            vec = all_vectors[steer_emotion].float()
            hook_fn = make_steer_hook(vec, args.steering_strength, norm_scale)
            probs = get_next_token_probs(
                model, tokenizer, prompt_text, emotion_token_ids,
                hook_fn=hook_fn, target_layer=target_layer, device=args.device,
            )
            steered_probs[steer_emotion] = probs
            delta_probs[steer_emotion] = {
                e: probs[e] - baseline[e] for e in present_emotions
            }
            print(f"  steer={steer_emotion:12s}  "
                  f"own={probs.get(steer_emotion, 0):.5f}  "
                  f"delta={delta_probs[steer_emotion].get(steer_emotion, 0):+.5f}")

        print_matrix(steered_probs, present_emotions, f"Steered probabilities [{prompt_name}]")
        print_matrix(delta_probs,   present_emotions, f"Delta from baseline   [{prompt_name}]")

        results[prompt_name] = {
            "baseline": baseline,
            "steered_probs": steered_probs,
            "delta_probs": delta_probs,
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "steering_strength": args.steering_strength,
        "target_layer": target_layer,
        "norm_scale": norm_scale,
        "emotions": present_emotions,
        "results": results,
    }
    out_path = STEERING_DIR / "verification_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary: does each vector's own word rank highest? ===")
    for prompt_name, res in results.items():
        delta = res["delta_probs"]
        n_correct = 0
        for steer_e in present_emotions:
            row = delta[steer_e]
            best = max(row, key=lambda e: row[e])
            correct = best == steer_e
            n_correct += int(correct)
            mark = "✓" if correct else f"✗ (best={best})"
            print(f"  [{prompt_name}] steer={steer_e:12s} {mark}")
        pct = 100 * n_correct / len(present_emotions)
        print(f"  [{prompt_name}] Accuracy: {n_correct}/{len(present_emotions)} = {pct:.0f}%\n")


if __name__ == "__main__":
    main()
