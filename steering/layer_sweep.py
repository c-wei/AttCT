"""
Config sweep for emotion vector extraction, selected by held-out separability.

Registers hooks on all layers, runs a single forward pass per read position over
all stories, then for every (layer × baseline × position × estimator) builds the
vector from a TRAIN topic split and measures ROC-AUC of the vector's projection
separating the target emotion from the rest on a held-out TEST topic split.

This replaces the old next-token-word metric (which is misaligned with vectors
trained on stories that never contain the emotion word). AUC directly measures
"is this direction a held-out classifier for the emotion."

Usage:
    uv run --no-project python steering/layer_sweep.py --max-stories 80
    uv run --no-project python steering/layer_sweep.py --positions last-k last-token \
        --baselines cross-emotion-topic grand-mean calm --estimators mean-diff probe
    uv run --no-project python steering/layer_sweep.py --layer-stride 2   # faster
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the canonical estimator + pooling so the sweep and the final extraction
# can never drift apart.
sys.path.insert(0, str(Path(__file__).parent))
from extract_emotion_vector import compute_vectors, pool_positions, _find_layers  # noqa: E402

EMOTIONS = [
    "frustrated", "happy", "inspired", "loving", "proud",
    "calm", "desperate", "angry", "guilty", "sad", "afraid", "surprised",
]

STEERING_DIR = Path(__file__).parent
DATA_DIR = STEERING_DIR / "data"
START_TOKEN = 50


def load_stories(emotion: str, max_stories: int = None, high_salience: bool = False):
    """
    Returns (texts, topics). When capping with max_stories, samples ROUND-ROBIN
    across topics rather than taking the first N lines — the files are grouped
    ~10-per-topic contiguously, so a head slice would cover only a handful of
    topics and bias the sweep toward whichever topics come first.
    """
    suffix = "_highsalience" if high_salience else ""
    path = DATA_DIR / f"stories_{emotion}{suffix}.jsonl"
    if not path.exists():
        return [], []
    rows = [json.loads(l) for l in open(path) if l.strip()]

    if max_stories and len(rows) > max_stories:
        by_topic: dict[str, list] = {}
        for r in rows:
            by_topic.setdefault(r["topic"], []).append(r)
        picked, depth = [], 0
        while len(picked) < max_stories:
            advanced = False
            for lst in by_topic.values():
                if depth < len(lst):
                    picked.append(lst[depth]); advanced = True
                    if len(picked) >= max_stories:
                        break
            if not advanced:
                break
            depth += 1
        rows = picked

    return [r["story"] for r in rows], [r["topic"] for r in rows]


# ── Activation collection — single pass, all layers, one read position ────────

def collect_all_layer_activations(
    model, tokenizer, texts, n_layers, batch_size, device,
    position: str, last_k: int, desc: str = "",
):
    """
    Single forward pass per batch, hooks on all layers. Pools each story's tokens
    into one [hidden] vector per layer using `position`. Assumes right padding.

    Returns: list[text] of list[layer] of ndarray[hidden]
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []
    layers = _find_layers(model)

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            captured[layer_idx] = h.detach().cpu().float()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    all_acts = []
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

            for item_idx in range(enc["input_ids"].shape[0]):
                seq_len = int(enc["attention_mask"][item_idx].sum().item())
                item_acts = []
                for layer_idx in range(n_layers):
                    h_valid = captured[layer_idx][item_idx][:seq_len]  # right padding
                    item_acts.append(
                        pool_positions(h_valid, position, last_k, START_TOKEN).numpy()
                    )
                all_acts.append(item_acts)

            if desc and (b % 5 == 0 or b == n_batches - 1):
                print(f"  {desc} [{batch_start + enc['input_ids'].shape[0]}/{len(texts)}]", flush=True)
    finally:
        for h in handles:
            h.remove()
    return all_acts


# ── Held-out AUC metric ───────────────────────────────────────────────────────

def topic_split(topics_per_story: list[str], test_frac: float, seed: int):
    """Split the unique topics into train/test sets (held-out topics, no leakage)."""
    uniq = sorted(set(topics_per_story))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_topics = set(uniq[:n_test])
    return test_topics


def auc_for_config(
    layer_acts: dict[str, np.ndarray],          # emotion -> [n, hidden]
    layer_topics: dict[str, list[str]],         # emotion -> [n]
    test_topics: set[str],
    baseline: str,
    estimator: str,
) -> dict[str, float]:
    """
    Build each emotion's vector from TRAIN stories, then score held-out TEST
    stories: AUC separating that emotion from all other emotions' test stories.
    Returns {emotion: auc}.
    """
    train_acts, train_topics, test_acts, test_labels_emotion = {}, {}, [], []
    for e, acts in layer_acts.items():
        tops = layer_topics[e]
        tr_mask = np.array([t not in test_topics for t in tops])
        train_acts[e] = acts[tr_mask]
        train_topics[e] = [t for t, m in zip(tops, tr_mask) if m]
        for a, t in zip(acts, tops):
            if t in test_topics:
                test_acts.append(a)
                test_labels_emotion.append(e)
    if not test_acts:
        return {}
    test_acts = np.stack(test_acts)

    vectors = compute_vectors(train_acts, train_topics, baseline=baseline, estimator=estimator)

    aucs = {}
    for e, v in vectors.items():
        scores = test_acts @ v
        y = np.array([1 if le == e else 0 for le in test_labels_emotion])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aucs[e] = float(roc_auc_score(y, scores))
    return aucs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--emotions", nargs="+", default=EMOTIONS)
    parser.add_argument("--max-stories", type=int, default=240,
                        help="Cap stories per emotion, sampled round-robin across all 100 topics "
                             "(default 240 ≈ 2-3/topic, full topic coverage). Controls memory — "
                             "all layers held in RAM, ~1.3MB/story (240×12 ≈ 3.8GB).")
    parser.add_argument("--positions", nargs="+", default=["last-k", "last-token", "mean-after-50"])
    parser.add_argument("--last-k", type=int, default=20)
    parser.add_argument("--baselines", nargs="+",
                        default=["cross-emotion-topic", "grand-mean", "calm"])
    parser.add_argument("--estimators", nargs="+", default=["mean-diff"],
                        help="Direction estimators to sweep (default: mean-diff only). "
                             "Add 'probe' to also sweep logistic-regression directions "
                             "(much slower — a 5376-dim fit per layer/baseline).")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--layer-stride", type=int, default=1)
    parser.add_argument("--high-salience", action="store_true",
                        help="Use highsalience story files (WARNING: breaks topic overlap; "
                             "matched-baseline AUC will be unreliable).")
    parser.add_argument("--output-suffix", type=str, default="")
    args = parser.parse_args()

    print(f"Loading {args.model}…")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    layers = _find_layers(model)
    n_layers = len(layers)
    print(f"Model: {n_layers} layers")

    # ── Load data ─────────────────────────────────────────────────────────────
    emotion_texts, emotion_topics = {}, {}
    for e in args.emotions:
        texts, topics = load_stories(e, args.max_stories, args.high_salience)
        if texts:
            emotion_texts[e] = texts
            emotion_topics[e] = topics
            print(f"  {e}: {len(texts)} stories")
        else:
            print(f"  {e}: NOT FOUND — skipping")
    present_emotions = list(emotion_texts.keys())

    # held-out topic split (shared across all configs for fair comparison)
    all_topics = [t for e in present_emotions for t in emotion_topics[e]]
    test_topics = topic_split(all_topics, args.test_frac, args.seed)
    print(f"  held-out test topics: {len(test_topics)}/{len(set(all_topics))}")

    layer_start = args.layer_start if args.layer_start is not None else 0
    layer_end = min(args.layer_end if args.layer_end is not None else n_layers - 1, n_layers - 1)
    layers_to_test = list(range(layer_start, layer_end + 1, args.layer_stride))

    all_rows = []  # (position, layer, baseline, estimator, frus_auc, mean_auc)

    for position in args.positions:
        print(f"\n{'='*70}\nPOSITION = {position}\n{'='*70}")
        # flatten texts in a fixed order; remember per-story emotion + topic
        flat_texts, flat_emotion, flat_topic = [], [], []
        for e in present_emotions:
            for txt, top in zip(emotion_texts[e], emotion_topics[e]):
                flat_texts.append(txt); flat_emotion.append(e); flat_topic.append(top)

        story_acts = collect_all_layer_activations(
            model, tokenizer, flat_texts, n_layers,
            args.batch_size, args.device, position, args.last_k, desc="stories",
        )

        # regroup: layer -> emotion -> (acts, topics)
        for layer_idx in layers_to_test:
            layer_acts = defaultdict(list)
            layer_tops = defaultdict(list)
            for item_acts, e, top in zip(story_acts, flat_emotion, flat_topic):
                layer_acts[e].append(item_acts[layer_idx])
                layer_tops[e].append(top)
            layer_acts = {e: np.stack(v) for e, v in layer_acts.items()}
            layer_tops = dict(layer_tops)

            for baseline in args.baselines:
                for estimator in args.estimators:
                    aucs = auc_for_config(layer_acts, layer_tops, test_topics, baseline, estimator)
                    if not aucs:
                        continue
                    frus = aucs.get("frustrated", float("nan"))
                    mean_auc = float(np.mean(list(aucs.values())))
                    all_rows.append({
                        "position": position, "layer": layer_idx,
                        "baseline": baseline, "estimator": estimator,
                        "frustrated_auc": frus, "mean_auc": mean_auc,
                        "per_emotion_auc": aucs,
                    })

    # ── Rank ──────────────────────────────────────────────────────────────────
    # Priority = frustrated AUC, tiebreak mean AUC.
    ranked = sorted(all_rows, key=lambda r: (r["frustrated_auc"], r["mean_auc"]), reverse=True)
    print(f"\n{'='*78}\nTOP CONFIGS BY HELD-OUT FRUSTRATED AUC (then mean AUC)\n{'='*78}")
    print(f"{'frus_auc':>9} {'mean_auc':>9}  {'pos':<13} {'layer':>5}  {'baseline':<20} {'estimator'}")
    for r in ranked[:20]:
        print(f"{r['frustrated_auc']:>9.3f} {r['mean_auc']:>9.3f}  {r['position']:<13} "
              f"{r['layer']:>5}  {r['baseline']:<20} {r['estimator']}")

    # legacy reference: grand-mean + mean-after-50
    legacy = [r for r in all_rows
              if r["baseline"] == "grand-mean" and r["position"] == "mean-after-50"]
    if legacy:
        best_legacy = max(legacy, key=lambda r: r["frustrated_auc"])
        print(f"\nLegacy reference (grand-mean, mean-after-50): "
              f"best frustrated AUC = {best_legacy['frustrated_auc']:.3f} "
              f"at layer {best_legacy['layer']}")
    if ranked:
        b = ranked[0]
        print(f"\nTop AUC config (a shortlist, NOT the final pick — confirm with judge_steering.py, "
              f"since a good read direction ≠ best steering direction):")
        print(f"  {b['baseline']} / {b['position']} / {b['estimator']} / layer {b['layer']} "
              f"→ frustrated AUC {b['frustrated_auc']:.3f}, mean AUC {b['mean_auc']:.3f}")

    out_path = STEERING_DIR / f"layer_sweep_results{args.output_suffix}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_layers": n_layers,
            "emotions": present_emotions,
            "metric": "held-out ROC-AUC (target emotion vs rest, unseen topics)",
            "test_frac": args.test_frac,
            "seed": args.seed,
            "max_stories": args.max_stories,
            "results": ranked,
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
