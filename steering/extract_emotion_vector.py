"""
Extract emotion vectors from Gemma 3 27B residual stream activations.

Methodology from: https://transformer-circuits.pub/2026/emotions/index.html

Usage:
    uv run --no-project python steering/extract_emotion_vector.py
    uv run --no-project python steering/extract_emotion_vector.py --model google/gemma-3-27b-it
"""

import argparse
import json
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


# ── Model introspection ───────────────────────────────────────────────────────

def _find_layers(model) -> torch.nn.ModuleList:
    """Find the decoder layer ModuleList regardless of model architecture."""
    candidates = [
        lambda m: m.model.layers,                    # Llama, Gemma 1/2, Mistral
        lambda m: m.model.language_model.layers,     # Gemma 3 (multimodal wrapper)
        lambda m: m.language_model.model.layers,     # some multimodal variants
        lambda m: m.transformer.h,                   # GPT-2 style
        lambda m: m.model.decoder.layers,            # OPT
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise AttributeError(
        f"Could not find decoder layers in {type(model).__name__}. "
        f"Top-level submodules: {[n for n, _ in model.named_children()]}"
    )


# ── Activation extraction ─────────────────────────────────────────────────────

class _EarlyExit(Exception):
    pass


def extract_activations_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    all_texts: list[str],
    all_labels: list[str],
    target_layer: int,
    layers: torch.nn.ModuleList,
    start_token: int = 50,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """
    Single pass over all texts (all emotions + neutral together).
    Uses an early-exit hook to abort after target_layer, skipping deeper layers.

    Returns: dict label → [n_texts, hidden_dim] array
    """
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        h = output if isinstance(output, torch.Tensor) else output[0]
        captured["hidden"] = h.detach().cpu().float()
        raise _EarlyExit()

    handle = layers[target_layer].register_forward_hook(hook_fn)

    label_vecs: dict[str, list[np.ndarray]] = {}
    n_batches = (len(all_texts) + batch_size - 1) // batch_size

    try:
        for i in range(0, len(all_texts), batch_size):
            batch_texts  = all_texts[i : i + batch_size]
            batch_labels = all_labels[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.inference_mode():
                try:
                    model(**enc)
                except _EarlyExit:
                    pass

            h = captured["hidden"]       # [batch, seq_len, hidden_dim]
            attn_mask = enc["attention_mask"].cpu().float()
            for b, label in enumerate(batch_labels):
                seq_len = int(attn_mask[b].sum().item())
                effective_start = min(start_token, seq_len - 1)
                vec = h[b, effective_start:seq_len, :].mean(dim=0).numpy()
                label_vecs.setdefault(label, []).append(vec)

            batch_idx = i // batch_size + 1
            if batch_idx % 50 == 0 or batch_idx == n_batches:
                print(f"    [{i + len(batch_texts)}/{len(all_texts)}]")
    finally:
        handle.remove()

    return {label: np.stack(vecs) for label, vecs in label_vecs.items()}


# ── Emotion vector computation ────────────────────────────────────────────────

def compute_emotion_vectors(
    emotion_activations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    For each emotion: v_raw = mu_emotion - mu_all_emotions

    Returns dict mapping emotion → raw (unnormalized, unprojected) vector.
    """
    # Per-emotion means
    mu = {e: acts.mean(axis=0) for e, acts in emotion_activations.items()}

    # Cross-emotion mean
    mu_all = np.stack(list(mu.values())).mean(axis=0)

    return {e: mu[e] - mu_all for e in mu}


def project_out_neutral_pcs(
    vectors: dict[str, np.ndarray],
    neutral_activations: np.ndarray,
    variance_threshold: float = 0.50,
) -> tuple[dict[str, np.ndarray], int, float]:
    """
    Remove top principal components of neutral activations from emotion vectors.

    Returns:
        projected_vectors: dict[emotion → cleaned vector]
        n_components: number of PCs removed
        variance_explained: cumulative variance explained by removed PCs
    """
    n_samples, hidden_dim = neutral_activations.shape
    n_components = min(n_samples, hidden_dim)

    pca = PCA(n_components=n_components)
    pca.fit(neutral_activations)

    # Find k: fewest components explaining >= variance_threshold
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_threshold)) + 1
    k = min(k, n_components)

    pcs = pca.components_[:k]  # [k, hidden_dim]
    variance_explained = float(cumvar[k - 1])

    print(f"  Neutral PCA: projecting out {k} components "
          f"(explains {variance_explained:.1%} of neutral variance)")

    projected = {}
    for e, v in vectors.items():
        # Remove each PC from v
        v_clean = v.copy()
        for pc in pcs:
            v_clean = v_clean - np.dot(v_clean, pc) * pc
        projected[e] = v_clean

    return projected, k, variance_explained


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        raise ValueError("Vector has near-zero norm after projection")
    return v / norm


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start-token", type=int, default=50,
                        help="Average residual stream from this token position onward")
    parser.add_argument("--variance-threshold", type=float, default=0.50,
                        help="Fraction of neutral variance to project out")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix for output filenames, e.g. '_base' → all_emotion_vectors_base.pt")
    parser.add_argument("--subtract-story-pca", type=int, default=0,
                        help="Remove top-N PCs of all story activations before computing vectors "
                             "(0 = disabled, try 3-5). Removes common narrative/format confounds.")
    parser.add_argument("--max-stories", type=int, default=None,
                        help="Cap stories per emotion (default: all). 80 is plenty for a reliable mean.")
    parser.add_argument("--stories-override", nargs=2, action="append", default=[],
                        metavar=("EMOTION", "PATH"),
                        help="Override story file for a specific emotion, e.g. "
                             "--stories-override frustrated data/stories_frustrated_highsalience.jsonl")
    args = parser.parse_args()

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

    layers = _find_layers(model)
    n_layers = len(layers)
    target_layer = int(n_layers * 2 / 3)
    print(f"Model has {n_layers} layers. Target layer: {target_layer} (~2/3 depth)")

    # ── Load stories ──────────────────────────────────────────────────────────
    overrides = {emotion: Path(path) for emotion, path in args.stories_override}
    emotion_texts: dict[str, list[str]] = {}
    for emotion in EMOTIONS:
        path = overrides.get(emotion, DATA_DIR / f"stories_{emotion}.jsonl")
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {emotion}")
            continue
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]
        texts = [r["story"] for r in records]
        if args.max_stories:
            texts = texts[:args.max_stories]
        label = f" (override: {path.name})" if emotion in overrides else ""
        print(f"  {emotion}: {len(texts)} stories{label}")
        emotion_texts[emotion] = texts

    if not emotion_texts:
        raise RuntimeError("No story files found. Run generate_emotion_stories.py first.")

    neutral_path = DATA_DIR / "neutral_dialogues.jsonl"
    if not neutral_path.exists():
        raise RuntimeError("neutral_dialogues.jsonl not found. Run generate_emotion_stories.py first.")
    with open(neutral_path) as f:
        neutral_records = [json.loads(l) for l in f if l.strip()]
    neutral_texts = [r["dialogue"] for r in neutral_records]
    print(f"  neutral: {len(neutral_texts)} dialogues")

    # ── Single-pass extraction (all emotions + neutral together) ─────────────
    print(f"\nExtracting activations (single pass, layer {target_layer})…")
    all_texts  = [t for e, texts in emotion_texts.items() for t in texts]
    all_labels = [e for e, texts in emotion_texts.items() for _ in texts]
    all_texts  += neutral_texts
    all_labels += ["__neutral__"] * len(neutral_texts)
    print(f"  Total texts: {len(all_texts)}")

    result = extract_activations_batched(
        model, tokenizer, all_texts, all_labels,
        target_layer, layers,
        start_token=args.start_token,
        batch_size=args.batch_size,
        device=args.device,
    )

    emotion_activations = {e: result[e] for e in emotion_texts if e in result}
    neutral_activations = result.get("__neutral__", np.zeros((1, 1)))

    all_norms = [n for e in emotion_activations for n in np.linalg.norm(emotion_activations[e], axis=1).tolist()]
    norm_scale = float(np.mean(all_norms))
    print(f"  norm_scale at layer {target_layer}: {norm_scale:.4f}")

    # ── Compute emotion vectors ───────────────────────────────────────────────
    present_emotions = list(emotion_activations.keys())
    print(f"\nComputing emotion vectors for: {present_emotions}")
    raw_vectors = compute_emotion_vectors(emotion_activations)

    # ── Optional: subtract top PCs of all story activations ──────────────────
    if args.subtract_story_pca > 0:
        print(f"\nSubtracting top {args.subtract_story_pca} PCs of all story activations…")
        all_story_acts = np.vstack(list(emotion_activations.values()))
        n_comp = min(args.subtract_story_pca, all_story_acts.shape[0] - 1)
        story_pca = PCA(n_components=n_comp)
        story_pca.fit(all_story_acts)
        var = story_pca.explained_variance_ratio_
        print(f"  Story PCs variance explained: " + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(var)))
        story_pcs = story_pca.components_[:args.subtract_story_pca]
        for e in raw_vectors:
            v = raw_vectors[e]
            for pc in story_pcs:
                v = v - np.dot(v, pc) * pc
            raw_vectors[e] = v

    # ── Project out neutral PCs ───────────────────────────────────────────────
    projected_vectors, n_pcs, var_explained = project_out_neutral_pcs(
        raw_vectors, neutral_activations, args.variance_threshold
    )

    # ── Normalize ─────────────────────────────────────────────────────────────
    normalized_vectors = {e: normalize(v) for e, v in projected_vectors.items()}

    # ── Cosine similarities (sanity check) ────────────────────────────────────
    print("\nCosine similarities of each emotion vector against v_frustrated:")
    v_frustrated = normalized_vectors.get("frustrated")
    cosine_sims = {}
    if v_frustrated is not None:
        for e, v in normalized_vectors.items():
            cos = float(np.dot(v_frustrated, v))
            cosine_sims[e] = cos
            marker = " ← target" if e == "frustrated" else ""
            print(f"  {e:12s}: {cos:+.4f}{marker}")

    # ── Save ──────────────────────────────────────────────────────────────────
    s = args.output_suffix  # e.g. "" or "_base"

    if v_frustrated is not None:
        fname = f"frustration_vector{s}.pt"
        torch.save(torch.tensor(v_frustrated, dtype=torch.float32), STEERING_DIR / fname)
        print(f"\nSaved {fname}")

    all_vectors_pt = {e: torch.tensor(v, dtype=torch.float32)
                      for e, v in normalized_vectors.items()}
    fname = f"all_emotion_vectors{s}.pt"
    torch.save(all_vectors_pt, STEERING_DIR / fname)
    print(f"Saved {fname} ({len(all_vectors_pt)} emotions)")

    meta = {
        "model": args.model,
        "layer": target_layer,
        "n_layers": n_layers,
        "norm_scale": norm_scale,
        "n_pcs_projected": n_pcs,
        "variance_explained_by_pcs": var_explained,
        "variance_threshold": args.variance_threshold,
        "start_token": args.start_token,
        "emotions": present_emotions,
        "emotion_cosine_sims": cosine_sims,
    }
    fname = f"frustration_meta{s}.json"
    with open(STEERING_DIR / fname, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
