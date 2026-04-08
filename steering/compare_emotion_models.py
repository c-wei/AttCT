"""
Compare emotion vector geometry between IT and base Gemma 3 27B models.

Runs locally (no GPU needed) — only needs the .pt vector files.

Usage:
    uv run --no-project python steering/compare_emotion_models.py
"""

from pathlib import Path
import json
import numpy as np
import torch

EMOTIONS = [
    "frustrated", "happy", "inspired", "loving", "proud",
    "calm", "desperate", "angry", "guilty", "sad", "afraid", "surprised",
]

STEERING_DIR = Path(__file__).parent


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_vectors(path: Path) -> dict[str, np.ndarray]:
    raw = torch.load(path, map_location="cpu")
    return {e: v.float().numpy() for e, v in raw.items()}


def pairwise_cosine_matrix(vectors: dict[str, np.ndarray], emotions: list[str]) -> np.ndarray:
    """Returns [n_emotions, n_emotions] cosine similarity matrix."""
    n = len(emotions)
    mat = np.zeros((n, n))
    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            mat[i, j] = cosine_sim(vectors[e1], vectors[e2])
    return mat


def print_matrix(mat: np.ndarray, emotions: list[str], title: str):
    col_w = 10
    print(f"\n{title}")
    print(f"{'':12s}" + "".join(f"{e[:8]:>{col_w}}" for e in emotions))
    for i, e in enumerate(emotions):
        row = "".join(f"{mat[i,j]:>{col_w}.3f}" for j in range(len(emotions)))
        print(f"{e:12s}{row}")


def main():
    it_path   = STEERING_DIR / "all_emotion_vectors.pt"
    base_path = STEERING_DIR / "all_emotion_vectors_base.pt"

    if not it_path.exists():
        raise FileNotFoundError(f"Missing {it_path} — run extract_emotion_vector.py first")
    if not base_path.exists():
        raise FileNotFoundError(
            f"Missing {base_path} — run:\n"
            f"  python steering/extract_emotion_vector.py "
            f"--model google/gemma-3-27b --output-suffix _base"
        )

    it_vecs   = load_vectors(it_path)
    base_vecs = load_vectors(base_path)

    # Common emotions present in both
    emotions = [e for e in EMOTIONS if e in it_vecs and e in base_vecs]
    print(f"Comparing {len(emotions)} emotions: {emotions}")

    # ── 1. Cross-model similarity per emotion ─────────────────────────────────
    print(f"\n{'='*60}")
    print("CROSS-MODEL COSINE SIMILARITY  (IT vs Base, same emotion)")
    print(f"{'Emotion':12s}  {'cos(IT, Base)':>14}  Interpretation")
    print("-" * 60)
    cross_sims = {}
    for e in emotions:
        cs = cosine_sim(it_vecs[e], base_vecs[e])
        cross_sims[e] = cs
        if cs > 0.8:
            interp = "very similar"
        elif cs > 0.5:
            interp = "similar"
        elif cs > 0.1:
            interp = "weakly similar"
        elif cs > -0.1:
            interp = "orthogonal"
        else:
            interp = "divergent"
        print(f"  {e:12s}  {cs:>14.4f}  {interp}")
    print(f"\n  Mean: {np.mean(list(cross_sims.values())):.4f}  "
          f"Min: {min(cross_sims.values()):.4f}  "
          f"Max: {max(cross_sims.values()):.4f}")

    # ── 2. Pairwise emotion similarity within each model ──────────────────────
    it_mat   = pairwise_cosine_matrix(it_vecs,   emotions)
    base_mat = pairwise_cosine_matrix(base_vecs, emotions)

    print_matrix(it_mat,   emotions, f"\n{'='*60}\nIT MODEL: Pairwise emotion cosine similarities")
    print_matrix(base_mat, emotions, f"\n{'='*60}\nBASE MODEL: Pairwise emotion cosine similarities")

    # ── 3. Difference matrix ──────────────────────────────────────────────────
    diff_mat = it_mat - base_mat
    print_matrix(diff_mat, emotions, f"\n{'='*60}\nDIFFERENCE (IT − Base): positive = more similar in IT")

    # ── 4. Biggest structural differences ────────────────────────────────────
    print(f"\n{'='*60}")
    print("BIGGEST PAIRWISE STRUCTURAL DIFFERENCES (|IT_sim - Base_sim| > 0.2)")
    diffs = []
    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            if i >= j:
                continue
            d = abs(it_mat[i, j] - base_mat[i, j])
            if d > 0.2:
                diffs.append((d, e1, e2, it_mat[i, j], base_mat[i, j]))
    diffs.sort(reverse=True)
    if diffs:
        for d, e1, e2, it_sim, base_sim in diffs:
            direction = "closer in IT" if it_sim > base_sim else "closer in Base"
            print(f"  {e1:12s} ↔ {e2:12s}  IT={it_sim:+.3f}  Base={base_sim:+.3f}  Δ={d:.3f}  ({direction})")
    else:
        print("  No large differences found.")

    # ── 5. Emotion clusters per model ─────────────────────────────────────────
    print(f"\n{'='*60}")
    for label, mat in [("IT", it_mat), ("Base", base_mat)]:
        print(f"\n{label} MODEL — Top emotion pairs by similarity (off-diagonal):")
        pairs = []
        for i, e1 in enumerate(emotions):
            for j, e2 in enumerate(emotions):
                if i < j:
                    pairs.append((mat[i, j], e1, e2))
        pairs.sort(reverse=True)
        for sim, e1, e2 in pairs[:6]:
            print(f"  {e1:12s} ↔ {e2:12s}  {sim:+.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "emotions": emotions,
        "cross_model_cosine_sims": cross_sims,
        "it_pairwise": it_mat.tolist(),
        "base_pairwise": base_mat.tolist(),
        "diff_pairwise": diff_mat.tolist(),
    }
    out_path = STEERING_DIR / "model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
