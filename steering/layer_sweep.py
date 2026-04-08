"""
Layer sweep for emotion vector extraction.

Registers hooks on all 62 layers simultaneously, runs a single forward pass
over all stories, then computes emotion vectors and tests next-token accuracy
at every layer. Identifies the optimal extraction depth.

Usage:
    python steering/layer_sweep.py
    python steering/layer_sweep.py --model google/gemma-3-27b-it --batch-size 4
    python steering/layer_sweep.py --layer-stride 2   # test every 2nd layer (faster)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

EMOTIONS = [
    "frustrated", "happy", "inspired", "loving", "proud",
    "calm", "desperate", "angry", "guilty", "sad", "afraid", "surprised",
]

STEERING_DIR = Path(__file__).parent
DATA_DIR = STEERING_DIR / "data"
PROBE_PROMPT = "Human: How does he feel?\nAssistant: He feels"
STEERING_STRENGTH = 0.5
START_TOKEN = 50  # average activations from this token position onward


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


def load_stories(emotion: str, max_stories: int = None) -> list[str]:
    path = DATA_DIR / f"stories_{emotion}.jsonl"
    if not path.exists():
        return []
    stories = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            stories.append(obj["story"])
            if max_stories and len(stories) >= max_stories:
                break
    return stories


def load_neutral_dialogues(max_dialogues: int = None) -> list[str]:
    path = DATA_DIR / "neutral_dialogues.jsonl"
    if not path.exists():
        return []
    dialogues = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            dialogues.append(obj["dialogue"])
            if max_dialogues and len(dialogues) >= max_dialogues:
                break
    return dialogues


# ── Activation collection — single pass, all layers ──────────────────────────

def collect_all_layer_activations(
    model, tokenizer, texts: list[str],
    n_layers: int, batch_size: int, device: str,
    start_token: int = START_TOKEN,
    desc: str = "",
) -> list[list[np.ndarray]]:
    """
    Single forward pass per batch, hooks on all layers simultaneously.

    Returns: list[text] of list[layer] of ndarray[hidden_dim]
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []
    layers = _find_layers(model)

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            # store CPU copy; shape [batch, seq_len, hidden_dim]
            captured[layer_idx] = h.detach().cpu().float()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    all_acts: list[list[np.ndarray]] = []

    try:
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for b, batch_start in enumerate(range(0, len(texts), batch_size)):
            batch = texts[batch_start: batch_start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(device)

            with torch.inference_mode():
                model(**enc)

            batch_len = enc["input_ids"].shape[0]
            for item_idx in range(batch_len):
                mask = enc["attention_mask"][item_idx]  # [seq_len]
                seq_len = int(mask.sum().item())

                item_acts: list[np.ndarray] = []
                for layer_idx in range(n_layers):
                    h = captured[layer_idx][item_idx]  # [seq_len, hidden_dim]
                    h_valid = h[:seq_len]               # strip right-padding
                    eff_start = min(start_token, max(0, seq_len - 1))
                    h_slice = h_valid[eff_start:]
                    if h_slice.shape[0] == 0:
                        h_slice = h_valid[-1:]
                    item_acts.append(h_slice.mean(dim=0).numpy())
                all_acts.append(item_acts)

            if desc and (b % 5 == 0 or b == n_batches - 1):
                print(f"  {desc} [{batch_start + batch_len}/{len(texts)}]",
                      flush=True)
    finally:
        for h in handles:
            h.remove()

    return all_acts


# ── Per-layer vector computation ──────────────────────────────────────────────

def compute_vectors_at_layer(
    story_acts: dict[str, list[np.ndarray]],
    neutral_acts: list[np.ndarray],
    emotions: list[str],
    variance_threshold: float = 0.50,
) -> dict[str, np.ndarray]:
    """Compute normalized emotion vectors, with PCA neutral projection."""
    emotion_means = {
        e: np.stack(story_acts[e]).mean(axis=0)
        for e in emotions
        if story_acts.get(e)
    }
    if not emotion_means:
        return {}

    mu_all = np.stack(list(emotion_means.values())).mean(axis=0)
    raw_vectors = {e: emotion_means[e] - mu_all for e in emotion_means}

    pcs = np.zeros((0,), dtype=np.float32)
    if len(neutral_acts) >= 2:
        pca = PCA()
        pca.fit(np.stack(neutral_acts))
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, variance_threshold)) + 1
        k = min(k, len(pca.components_))
        pcs = pca.components_[:k]

    vectors: dict[str, np.ndarray] = {}
    for e, v_raw in raw_vectors.items():
        v_clean = v_raw.copy()
        for pc in pcs:
            v_clean = v_clean - np.dot(v_clean, pc) * pc
        norm = np.linalg.norm(v_clean)
        if norm < 1e-8:
            continue
        vectors[e] = v_clean / norm

    return vectors


# ── Next-token probability test at one layer ──────────────────────────────────

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


def make_steer_hook(vector: np.ndarray, multiplier: float, norm_scale: float):
    vec_t = torch.from_numpy(vector).float()
    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        delta = (multiplier * norm_scale * vec_t).to(h.device, h.dtype)
        h = h + delta.unsqueeze(0).unsqueeze(0)
        return (h,) + output[1:] if is_tuple else h
    return hook


def test_layer_accuracy(
    model, layers, layer_idx: int,
    vectors: dict[str, np.ndarray],
    norm_scale: float,
    emotion_token_ids: dict[str, list[int]],
    emotions: list[str],
    strength: float,
    probe_enc,
    device: str,
) -> int:
    """Returns number of emotions whose own word has the highest delta (out of all 12)."""
    # Baseline probabilities
    with torch.inference_mode():
        out_base = model(**probe_enc)
    base_probs = torch.softmax(out_base.logits[0, -1, :].float(), dim=-1)
    base_vals = {
        e: sum(base_probs[tid].item() for tid in ids if tid < len(base_probs))
        for e, ids in emotion_token_ids.items()
    }

    n_correct = 0
    for steer_e in emotions:
        if steer_e not in vectors:
            continue
        hook_fn = make_steer_hook(vectors[steer_e], strength, norm_scale)
        handle = layers[layer_idx].register_forward_hook(hook_fn)
        try:
            with torch.inference_mode():
                out = model(**probe_enc)
            probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        finally:
            handle.remove()

        deltas = {
            e: sum(probs[tid].item() for tid in ids if tid < len(probs)) - base_vals[e]
            for e, ids in emotion_token_ids.items()
        }
        own_rank = sorted(emotions, key=lambda e: deltas[e], reverse=True).index(steer_e)
        if own_rank == 0:
            n_correct += 1

    return n_correct


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--emotions", nargs="+", default=EMOTIONS)
    parser.add_argument("--max-stories", type=int, default=None,
                        help="Cap stories per emotion (default: all)")
    parser.add_argument("--steering-strength", type=float, default=STEERING_STRENGTH)
    parser.add_argument("--layer-start", type=int, default=None,
                        help="First layer to test (default: 0)")
    parser.add_argument("--layer-end", type=int, default=None,
                        help="Last layer to test inclusive (default: n_layers-1)")
    parser.add_argument("--layer-stride", type=int, default=1,
                        help="Test every N-th layer within the range (default: 1)")
    args = parser.parse_args()

    print(f"Loading {args.model}…")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    layers = _find_layers(model)
    n_layers = len(layers)
    print(f"Model: {n_layers} layers, hidden_size={model.config.hidden_size}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading story data…")
    emotion_stories: dict[str, list[str]] = {}
    for e in args.emotions:
        stories = load_stories(e, args.max_stories)
        if stories:
            emotion_stories[e] = stories
            print(f"  {e}: {len(stories)} stories")
        else:
            print(f"  {e}: NOT FOUND — skipping")
    present_emotions = list(emotion_stories.keys())

    neutral_texts = load_neutral_dialogues()
    print(f"  neutral: {len(neutral_texts)} dialogues")

    # ── Single pass: collect all-layer activations ────────────────────────────
    print("\nCollecting story activations (all layers simultaneously)…")
    all_texts = [s for e in present_emotions for s in emotion_stories[e]]
    all_labels = [e for e in present_emotions for _ in emotion_stories[e]]

    story_all_acts = collect_all_layer_activations(
        model, tokenizer, all_texts, n_layers,
        args.batch_size, args.device, desc="stories",
    )

    # Group: layer → emotion → list of activations
    story_by_layer_emotion: dict[int, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item_acts, emotion in zip(story_all_acts, all_labels):
        for layer_idx, act in enumerate(item_acts):
            story_by_layer_emotion[layer_idx][emotion].append(act)

    print("\nCollecting neutral activations…")
    neutral_all_acts = collect_all_layer_activations(
        model, tokenizer, neutral_texts, n_layers,
        args.batch_size, args.device, desc="neutral",
    )
    neutral_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    for item_acts in neutral_all_acts:
        for layer_idx, act in enumerate(item_acts):
            neutral_by_layer[layer_idx].append(act)

    # ── Compute per-layer norm_scale ──────────────────────────────────────────
    # Activation magnitudes vary substantially by layer; using per-layer norm_scale
    # keeps the steering delta proportional at every depth.
    print("\nComputing per-layer norm_scale…")
    layer_norm_scale: dict[int, float] = {}
    for layer_idx in range(n_layers):
        all_acts_at_layer = [
            act
            for e in present_emotions
            for act in story_by_layer_emotion[layer_idx].get(e, [])
        ]
        if all_acts_at_layer:
            norms = np.array([np.linalg.norm(a) for a in all_acts_at_layer])
            layer_norm_scale[layer_idx] = float(norms.mean())
        else:
            layer_norm_scale[layer_idx] = 1.0

    # ── Probe tokenization (done once) ───────────────────────────────────────
    emotion_token_ids = get_emotion_token_ids(tokenizer, present_emotions)
    probe_enc = tokenizer(PROBE_PROMPT, return_tensors="pt").to(args.device)

    # ── Sweep layers ──────────────────────────────────────────────────────────
    layer_start = args.layer_start if args.layer_start is not None else 0
    layer_end   = args.layer_end   if args.layer_end   is not None else n_layers - 1
    layer_end   = min(layer_end, n_layers - 1)
    layers_to_test = list(range(layer_start, layer_end + 1, args.layer_stride))
    print(f"\n{'='*70}")
    print(f"LAYER SWEEP — {len(layers_to_test)} layers, strength={args.steering_strength}")
    print(f"{'Layer':>7}  {'norm_scale':>12}  {'Acc':>6}  {'Correct':>9}  Bar")
    print("-" * 70)

    sweep_results = []
    for layer_idx in layers_to_test:
        story_acts = dict(story_by_layer_emotion[layer_idx])
        neutral_acts = neutral_by_layer[layer_idx]
        ns = layer_norm_scale[layer_idx]

        vectors = compute_vectors_at_layer(story_acts, neutral_acts, present_emotions)
        if not vectors:
            print(f"  {layer_idx:5d}  {'':>12}  (no vectors)")
            sweep_results.append({"layer": layer_idx, "n_correct": 0,
                                   "n_total": len(present_emotions), "accuracy": 0.0,
                                   "norm_scale": ns})
            continue

        n_correct = test_layer_accuracy(
            model, layers, layer_idx,
            vectors, ns, emotion_token_ids,
            present_emotions, args.steering_strength,
            probe_enc, args.device,
        )
        n_total = len(present_emotions)
        acc = n_correct / n_total
        bar = "█" * n_correct + "·" * (n_total - n_correct)
        print(f"  {layer_idx:5d}  {ns:>12.1f}  {acc:>6.1%}  {n_correct:>3}/{n_total}     {bar}")

        sweep_results.append({
            "layer": layer_idx,
            "n_correct": n_correct,
            "n_total": n_total,
            "accuracy": acc,
            "norm_scale": ns,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    best = max(sweep_results, key=lambda r: r["accuracy"])
    ref_layer = int(n_layers * 2 / 3)
    ref_result = next((r for r in sweep_results if r["layer"] == ref_layer), None)

    print(f"\n{'='*70}")
    print(f"Best layer:  {best['layer']} "
          f"(accuracy={best['accuracy']:.1%}, {best['n_correct']}/{best['n_total']},"
          f" norm_scale={best['norm_scale']:.1f})")
    if ref_result:
        print(f"2/3-depth rule (layer {ref_layer}): "
              f"accuracy={ref_result['accuracy']:.1%}, {ref_result['n_correct']}/{ref_result['n_total']}")

    out_path = STEERING_DIR / "layer_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_layers": n_layers,
            "emotions": present_emotions,
            "steering_strength": args.steering_strength,
            "probe_prompt": PROBE_PROMPT,
            "layer_stride": args.layer_stride,
            "best_layer": best["layer"],
            "ref_layer_2_3": ref_layer,
            "results": sweep_results,
        }, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
