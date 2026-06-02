"""
Extract emotion vectors from Gemma 3 27B residual stream activations.

Methodology from: https://transformer-circuits.pub/2026/emotions/index.html
Refined (v2) with matched within-topic contrasts, configurable read position, and
multiple direction estimators — see steering/FINDINGS.md.

Baseline guidance (verified on synthetic data, steering/FINDINGS.md):
  - cross-emotion-topic (default): per-topic subtraction of the mean over OTHER
    emotions. Removes the topic confound AND the arousal/valence component shared
    with other emotions → clean, low-leakage geometry. Best de-confounder.
  - grand-mean: global mu_e - mu_all (legacy). Also removes shared arousal but
    leaves the topic confound (no per-topic matching).
  - calm: per-topic mu_e - mu(calm). NOT a de-confounder — calm lacks arousal, so
    this keeps frustration's arousal and still leaks into high-arousal emotions.
    It is the "calm → emotion" steering direction; keep only as a behavioral
    candidate to be judged, not for clean geometry.

Estimator guidance: mean-diff is the robust default; pca1 locks onto topic-variation
noise and is unreliable; probe (logistic weight) is a reasonable alternative.

Usage:
    # v2 recommended: de-confounded direction, climax read position
    uv run --no-project python steering/extract_emotion_vector.py \
        --baseline cross-emotion-topic --position last-k --last-k 20 --estimator mean-diff --output-suffix _v2

    # behavioral steering candidate to judge against the above
    uv run --no-project python steering/extract_emotion_vector.py --baseline calm --output-suffix _v2calm

    # legacy global-mean behaviour
    uv run --no-project python steering/extract_emotion_vector.py --baseline grand-mean --position mean-after-50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

EMOTIONS = [
    "frustrated", "happy", "inspired", "loving", "proud",
    "calm", "desperate", "angry", "guilty", "sad", "afraid", "surprised",
]

STEERING_DIR = Path(__file__).parent
DATA_DIR = STEERING_DIR / "data"

BASELINES = ["grand-mean", "cross-emotion-topic", "calm"]
POSITIONS = ["mean-after-50", "last-token", "last-k"]
ESTIMATORS = ["mean-diff", "pca1", "probe"]


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


def pool_positions(
    h_valid: torch.Tensor, position: str, last_k: int, start_token: int
) -> torch.Tensor:
    """
    Reduce a [seq_len, hidden] activation tensor (right-padding already stripped)
    to a single [hidden] vector according to the read position.

    - last-token:    the final real token (where story affect is fully established)
    - last-k:        mean over the final `last_k` real tokens
    - mean-after-50: mean from token `start_token` onward (legacy behaviour)
    """
    seq_len = h_valid.shape[0]
    if position == "last-token":
        return h_valid[-1]
    if position == "last-k":
        k = min(last_k, seq_len)
        return h_valid[seq_len - k: seq_len].mean(dim=0)
    # mean-after-50
    eff_start = min(start_token, seq_len - 1)
    return h_valid[eff_start: seq_len].mean(dim=0)


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    target_layer: int,
    layers: torch.nn.ModuleList,
    position: str = "last-k",
    last_k: int = 20,
    start_token: int = 50,
    batch_size: int = 8,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run texts through the model, capture residual stream at target_layer,
    pool token positions according to `position`.

    Uses an early-exit hook to abort the forward pass after target_layer,
    skipping all deeper layers for a ~30% speedup. Assumes right padding so
    real tokens occupy positions [0:seq_len].

    Returns: np.ndarray of shape [len(texts), hidden_dim]
    """
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        h = output if isinstance(output, torch.Tensor) else output[0]
        captured["hidden"] = h.detach().cpu().float()
        raise _EarlyExit()  # abort — we have what we need

    handle = layers[target_layer].register_forward_hook(hook_fn)

    all_vecs = []
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)

            with torch.inference_mode():
                try:
                    model(**enc)
                except _EarlyExit:
                    pass

            h = captured["hidden"]  # [batch, seq_len, hidden_dim]
            attn_mask = enc["attention_mask"].cpu().float()
            for b in range(h.shape[0]):
                seq_len = int(attn_mask[b].sum().item())
                h_valid = h[b, :seq_len, :]  # right padding ⇒ real tokens first
                vec = pool_positions(h_valid, position, last_k, start_token)
                all_vecs.append(vec.numpy())

            if (i // batch_size + 1) % 20 == 0:
                print(f"    {i + len(batch_texts)}/{len(texts)} done")
    finally:
        handle.remove()

    return np.stack(all_vecs, axis=0)  # [n_texts, hidden_dim]


# ── Emotion vector computation ────────────────────────────────────────────────

def _topic_means(acts: np.ndarray, topics: list[str]) -> dict[str, np.ndarray]:
    """Map each topic → mean activation of that topic's stories."""
    idx: dict[str, list[int]] = {}
    for i, t in enumerate(topics):
        idx.setdefault(t, []).append(i)
    return {t: acts[ix].mean(axis=0) for t, ix in idx.items()}


def _matched_deltas(
    emotion: str,
    tmeans: dict[str, dict[str, np.ndarray]],
    emotions: list[str],
    baseline: str,
) -> np.ndarray:
    """
    Per-topic matched difference vectors {mu(e|t) - baseline(t)} for `emotion`.

    baseline:
      - "calm":                baseline(t) = mu(calm|t)
      - "cross-emotion-topic": baseline(t) = mean over other emotions of mu(e'|t)
    Topics missing in the baseline are skipped. Returns [n_topics_used, hidden].
    """
    deltas = []
    for t, mu_e_t in tmeans[emotion].items():
        if baseline == "calm":
            base_t = tmeans["calm"].get(t)
            if base_t is None:
                continue
        else:  # cross-emotion-topic
            others = [tmeans[o][t] for o in emotions if o != emotion and t in tmeans[o]]
            if not others:
                continue
            base_t = np.stack(others).mean(axis=0)
        deltas.append(mu_e_t - base_t)
    return np.stack(deltas) if deltas else np.empty((0,))


def compute_vectors(
    emotion_activations: dict[str, np.ndarray],
    emotion_topics: dict[str, list[str]],
    baseline: str = "cross-emotion-topic",
    estimator: str = "mean-diff",
) -> dict[str, np.ndarray]:
    """
    Compute raw (unnormalized, unprojected) emotion direction vectors.

    baseline  — what each emotion mean is contrasted against (see BASELINES).
    estimator — how the direction is read off the contrast:
        mean-diff : average of matched per-topic deltas
        pca1      : top PC of the matched per-topic deltas (sign-aligned to mean)
        probe     : logistic-regression weight (target stories vs baseline stories)

    The "calm" baseline falls back to "cross-emotion-topic" for the calm vector
    itself (and if no calm stories are present).
    """
    emotions = list(emotion_activations.keys())
    has_calm = "calm" in emotion_activations
    hidden = next(iter(emotion_activations.values())).shape[1]
    tmeans = {e: _topic_means(emotion_activations[e], emotion_topics[e]) for e in emotions}

    out: dict[str, np.ndarray] = {}
    for e in emotions:
        eff_baseline = baseline
        if baseline == "calm" and (e == "calm" or not has_calm):
            eff_baseline = "cross-emotion-topic"

        # ── Probe: story-level logistic regression direction ──────────────────
        if estimator == "probe":
            X_pos = emotion_activations[e]
            if eff_baseline == "calm":
                X_neg = emotion_activations["calm"]
            else:  # cross-emotion-topic / grand-mean ⇒ everything else
                X_neg = np.vstack([emotion_activations[o] for o in emotions if o != e])
            X = np.vstack([X_pos, X_neg]).astype(np.float64)
            y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
            mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit((X - mu) / sd, y)
            out[e] = (clf.coef_[0] / sd).astype(np.float32)  # back to original space
            continue

        # ── Delta-based (mean-diff / pca1) ────────────────────────────────────
        if eff_baseline == "grand-mean":
            mu_all = np.stack([emotion_activations[o].mean(axis=0) for o in emotions]).mean(axis=0)
            deltas = (emotion_activations[e].mean(axis=0) - mu_all)[None, :]
        else:
            deltas = _matched_deltas(e, tmeans, emotions, eff_baseline)
            if deltas.shape[0] == 0:
                deltas = np.zeros((1, hidden), dtype=np.float32)

        if estimator == "pca1" and deltas.shape[0] >= 2:
            mean_dir = deltas.mean(axis=0)
            pc1 = PCA(n_components=1).fit(deltas).components_[0]
            if np.dot(pc1, mean_dir) < 0:
                pc1 = -pc1
            out[e] = pc1
        else:  # mean-diff (also pca1 fallback when <2 deltas)
            out[e] = deltas.mean(axis=0)

    return out


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

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_threshold)) + 1
    k = min(k, n_components)

    pcs = pca.components_[:k]
    variance_explained = float(cumvar[k - 1])

    print(f"  Neutral PCA: projecting out {k} components "
          f"(explains {variance_explained:.1%} of neutral variance)")

    projected = {}
    for e, v in vectors.items():
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
    parser.add_argument("--baseline", choices=BASELINES, default="cross-emotion-topic",
                        help="What each emotion mean is contrasted against "
                             "(default: cross-emotion-topic — best de-confounder).")
    parser.add_argument("--position", choices=POSITIONS, default="last-k",
                        help="Where in the story to read activations (default: last-k).")
    parser.add_argument("--last-k", type=int, default=20,
                        help="Number of trailing tokens to average for --position last-k.")
    parser.add_argument("--estimator", choices=ESTIMATORS, default="mean-diff",
                        help="How the direction is read off the contrast (default: mean-diff).")
    parser.add_argument("--start-token", type=int, default=50,
                        help="Start position for --position mean-after-50 (legacy).")
    parser.add_argument("--project-neutral", action="store_true",
                        help="Project out top neutral-dialogue PCs (off by default; legacy step).")
    parser.add_argument("--variance-threshold", type=float, default=0.50,
                        help="Fraction of neutral variance to project out (with --project-neutral).")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix for output filenames, e.g. '_v2' → all_emotion_vectors_v2.pt")
    parser.add_argument("--max-stories", type=int, default=None,
                        help="Cap stories per emotion (default: all).")
    parser.add_argument("--stories-override", nargs=2, action="append", default=[],
                        metavar=("EMOTION", "PATH"),
                        help="Override story file for a specific emotion.")
    parser.add_argument("--high-salience", action="store_true",
                        help="Use stories_{emotion}_highsalience.jsonl for all emotions. "
                             "NOTE: incompatible with matched-baseline modes — the high-salience "
                             "files barely share topics, so calm/cross-emotion baselines degenerate.")
    parser.add_argument("--contrastive", action="store_true",
                        help="DEPRECATED alias for --baseline cross-emotion-topic.")
    args = parser.parse_args()

    if args.contrastive:
        args.baseline = "cross-emotion-topic"  # deprecated alias

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading {args.model}…")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # real tokens occupy [0:seq_len]

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
    print(f"Config: baseline={args.baseline}  position={args.position}"
          f"{f' (k={args.last_k})' if args.position == 'last-k' else ''}  "
          f"estimator={args.estimator}")

    # ── Load stories ──────────────────────────────────────────────────────────
    overrides = {emotion: Path(path) for emotion, path in args.stories_override}
    if args.high_salience:
        for emotion in EMOTIONS:
            if emotion not in overrides:
                overrides[emotion] = DATA_DIR / f"stories_{emotion}_highsalience.jsonl"
    emotion_texts: dict[str, list[str]] = {}
    emotion_topics: dict[str, list[str]] = {}
    for emotion in EMOTIONS:
        path = overrides.get(emotion, DATA_DIR / f"stories_{emotion}.jsonl")
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {emotion}")
            continue
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]
        if args.max_stories:
            records = records[:args.max_stories]
        texts = [r["story"] for r in records]
        topics = [r["topic"] for r in records]
        label = f" (override: {path.name})" if emotion in overrides else ""
        print(f"  {emotion}: {len(texts)} stories{label}")
        emotion_texts[emotion] = texts
        emotion_topics[emotion] = topics

    if not emotion_texts:
        raise RuntimeError("No story files found. Run generate_emotion_stories.py first.")

    if args.baseline == "calm" and "calm" not in emotion_texts:
        print("  WARNING: --baseline calm but no calm stories present; "
              "all vectors fall back to cross-emotion-topic.")

    # ── Extract activations ───────────────────────────────────────────────────
    print("\nExtracting emotion activations…")
    emotion_activations: dict[str, np.ndarray] = {}
    all_norms = []

    for emotion, texts in emotion_texts.items():
        print(f"  [{emotion}]")
        acts = extract_activations(
            model, tokenizer, texts, target_layer, layers,
            position=args.position, last_k=args.last_k,
            start_token=args.start_token,
            batch_size=args.batch_size,
            device=args.device,
        )  # [n_stories, hidden_dim]
        emotion_activations[emotion] = acts
        all_norms.extend(np.linalg.norm(acts, axis=1).tolist())

    norm_scale = float(np.mean(all_norms))
    print(f"\nResidual stream norm_scale at layer {target_layer}: {norm_scale:.4f}")

    neutral_activations = None
    if args.project_neutral:
        neutral_path = DATA_DIR / "neutral_dialogues.jsonl"
        if not neutral_path.exists():
            raise RuntimeError("--project-neutral set but neutral_dialogues.jsonl not found.")
        with open(neutral_path) as f:
            neutral_records = [json.loads(l) for l in f if l.strip()]
        neutral_texts = [r["dialogue"] for r in neutral_records]
        print(f"\nExtracting {len(neutral_texts)} neutral activations…")
        neutral_activations = extract_activations(
            model, tokenizer, neutral_texts, target_layer, layers,
            position=args.position, last_k=args.last_k,
            start_token=args.start_token,
            batch_size=args.batch_size,
            device=args.device,
        )

    # ── Compute emotion vectors ───────────────────────────────────────────────
    present_emotions = list(emotion_activations.keys())
    print(f"\nComputing emotion vectors for: {present_emotions}")
    raw_vectors = compute_vectors(
        emotion_activations, emotion_topics,
        baseline=args.baseline, estimator=args.estimator,
    )

    # ── Optional: project out neutral PCs ─────────────────────────────────────
    n_pcs, var_explained = 0, 0.0
    if args.project_neutral and neutral_activations is not None:
        raw_vectors, n_pcs, var_explained = project_out_neutral_pcs(
            raw_vectors, neutral_activations, args.variance_threshold
        )

    # ── Normalize ─────────────────────────────────────────────────────────────
    normalized_vectors = {e: normalize(v) for e, v in raw_vectors.items()}

    # ── Cosine similarities (sanity check) ────────────────────────────────────
    print("\nCosine similarities of each emotion vector against v_frustrated:")
    v_frustrated = normalized_vectors.get("frustrated")
    cosine_sims = {}
    if v_frustrated is not None:
        for e, v in sorted(normalized_vectors.items(),
                           key=lambda kv: -float(np.dot(v_frustrated, kv[1]))):
            cos = float(np.dot(v_frustrated, v))
            cosine_sims[e] = cos
            marker = " ← target" if e == "frustrated" else ""
            print(f"  {e:12s}: {cos:+.4f}{marker}")

    # ── Save ──────────────────────────────────────────────────────────────────
    s = args.output_suffix  # e.g. "" or "_v2"

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
        "baseline": args.baseline,
        "position": args.position,
        "last_k": args.last_k if args.position == "last-k" else None,
        "estimator": args.estimator,
        "n_pcs_projected": n_pcs,
        "variance_explained_by_pcs": var_explained,
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
