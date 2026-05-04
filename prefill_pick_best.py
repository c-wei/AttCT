#!/usr/bin/env python3
"""
prefill_pick_best.py — aggregate grid eval JSONs and write per-method best.json.

Walks  checkpoints/prefill_<MODEL_TAG>/grid/<cell>/eval_epoch_<N>.json,
groups by method (bct / act / attct / mlpct), picks the (cell, epoch) that
minimises PAR (lowest = strongest defense), and writes:
    bct_best.json
    act_best.json
    attct_best.json
    mlpct_best.json

Each file surfaces PAR, attack-comply, and clean-comply rates for the winner
plus a leaderboard of every (cell, epoch) tried.

The hyperparameter dict for each label mirrors prefill_train.sh — keep them
in sync. If a new label is added there, add it here too or it will appear
under "hyperparameters": null.

Usage
-----
    python prefill_pick_best.py --model_tag llama
    python prefill_pick_best.py --model_tag qwen --rank-by attack_comply
    python prefill_pick_best.py --model_tag llama --output_dir results/best
"""

import argparse
import json
from pathlib import Path

# Hyperparameters per (method, label) — mirrors prefill_train.sh grids.
GRIDS: dict[str, dict[str, dict]] = {
    "bct": {
        "t1_sft01":  {"kl_temperature": 1.0, "sft_coeff": 0.1},
        "t05_sft01": {"kl_temperature": 0.5, "sft_coeff": 0.1},
        "t1_sft03":  {"kl_temperature": 1.0, "sft_coeff": 0.3},
    },
    "act": {
        "w1_all":      {"loss_weight": 1.0, "layer_selection": "all"},
        "w1_all_norm": {"loss_weight": 1.0, "layer_selection": "all", "normalize": True},
    },
    "attct": {
        "wrap_w1_kl1":  {"attct_loss_type": "wrapper", "loss_weight": 1.0,  "kl_weight": 1.0,  "layer_weights": "uniform"},
        "wrap_w1_kl10": {"attct_loss_type": "wrapper", "loss_weight": 1.0,  "kl_weight": 10.0, "layer_weights": "uniform"},
        "wrap_w01_kl1": {"attct_loss_type": "wrapper", "loss_weight": 1.0,  "kl_weight": 1.0,  "layer_weights": "linear_decay"},
        "jsd_w1_kl1":   {"attct_loss_type": "jsd",     "loss_weight": 1.0,  "kl_weight": 1.0,  "layer_selection": "all"},
        "jsd_w10_kl1":  {"attct_loss_type": "jsd",     "loss_weight": 10.0, "kl_weight": 1.0,  "layer_selection": "all"},
        "jsd_lasthalf": {"attct_loss_type": "jsd",     "loss_weight": 1.0,  "kl_weight": 1.0,  "layer_selection": "last_half"},
        "comb_5050":    {"attct_loss_type": "combined", "jsd_weight": 0.5, "wrapper_weight": 0.5, "kl_weight": 1.0},
        "comb_8020":    {"attct_loss_type": "combined", "jsd_weight": 0.8, "wrapper_weight": 0.2, "kl_weight": 1.0},
        "comb_2080":    {"attct_loss_type": "combined", "jsd_weight": 0.2, "wrapper_weight": 0.8, "kl_weight": 1.0},
    },
    "mlpct": {
        "mw1_hid_cos":        {"mlpct_weight": 1,    "variant": "hidden", "distance_metric": "cosine"},
        "mw100_hid_cos":      {"mlpct_weight": 100,  "variant": "hidden", "distance_metric": "cosine"},
        "mw1000_hid_cos":     {"mlpct_weight": 1000, "variant": "hidden", "distance_metric": "cosine"},
        "mw100_out_cos":      {"mlpct_weight": 100,  "variant": "output", "distance_metric": "cosine"},
        "mw100_hid_mse":      {"mlpct_weight": 100,  "variant": "hidden", "distance_metric": "mse"},
        "mw100_hid_smooth":   {"mlpct_weight": 100,  "variant": "hidden", "distance_metric": "smooth_l1"},
        "mw100_hid_cos_norm": {"mlpct_weight": 100,  "variant": "hidden", "distance_metric": "cosine", "normalize": True},
    },
}

MODEL_NAMES = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
}


def parse_cell_name(cell: str) -> tuple[str, str]:
    """`bct_t1_sft01` → ('bct', 't1_sft01'). Splits on first underscore."""
    method, _, label = cell.partition("_")
    return method, label


def harmful_metrics(eval_data: dict) -> dict:
    """Pull (par, attack_comply, clean_comply, mmlu) from a per-cell eval JSON."""
    h = eval_data.get("harmful", {})
    mmlu = eval_data.get("mmlu", {}).get("accuracy")
    return {
        "par":           h.get("par"),
        "attack_comply": h.get("attack_comply_rate"),
        "clean_comply":  h.get("clean_comply_rate"),
        "n":             h.get("n"),
        "mmlu":          mmlu,
    }


def score(metrics: dict, rank_by: str) -> float:
    """Smaller score = better. Missing fields → +inf so they sort last."""
    if rank_by == "par":
        return metrics["par"] if metrics["par"] is not None else float("inf")
    if rank_by == "attack_comply":
        return metrics["attack_comply"] if metrics["attack_comply"] is not None else float("inf")
    if rank_by == "composite":
        # PAR (lower=better) penalised by MMLU drop (higher=better).
        par  = metrics["par"]  if metrics["par"]  is not None else float("inf")
        mmlu = metrics["mmlu"] if metrics["mmlu"] is not None else 0.0
        return par - 0.5 * mmlu
    raise ValueError(f"Unknown --rank-by: {rank_by}")


def find_lora_dir(cell_dir: Path, epoch: int) -> Path | None:
    """Most recent epoch_N* dir — handles both `epoch_3` and `epoch_3__TIMESTAMP`."""
    matches = sorted(cell_dir.glob(f"epoch_{epoch}*"), reverse=True)
    return matches[0] if matches else None


def collect_runs(grid_root: Path, model_tag: str) -> list[dict]:
    runs = []
    for cell_dir in sorted(grid_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        cell = cell_dir.name
        method, label = parse_cell_name(cell)
        if method not in GRIDS:
            continue

        for epoch in (1, 2, 3):
            eval_json = cell_dir / f"eval_epoch_{epoch}.json"
            if not eval_json.is_file():
                continue
            try:
                eval_data = json.loads(eval_json.read_text())
            except json.JSONDecodeError as e:
                print(f"  [skip] {eval_json} — bad JSON: {e}")
                continue

            metrics = harmful_metrics(eval_data)
            lora = find_lora_dir(cell_dir, epoch)
            runs.append({
                "method":          method,
                "cell":            cell,
                "label":           label,
                "epoch":           epoch,
                "checkpoint":      str(lora) if lora else None,
                "model":           MODEL_NAMES.get(model_tag, model_tag),
                "model_tag":       model_tag,
                "hyperparameters": GRIDS.get(method, {}).get(label),
                "metrics":         metrics,
                "eval":            eval_data,
            })
    return runs


def _fmt_pct(x):
    return f"{x*100:5.1f}%" if x is not None else "  n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_tag",   default="llama", choices=list(MODEL_NAMES.keys()))
    ap.add_argument("--grid_root",   default=None,
                    help="Override grid root (default: checkpoints/prefill_<tag>/grid)")
    ap.add_argument("--output_dir",  default=".",
                    help="Where to write <method>_best.json (default: cwd)")
    ap.add_argument("--rank-by",     default="par",
                    choices=["par", "attack_comply", "composite"],
                    help="Ranking metric (lower = better). Default: par.")
    args = ap.parse_args()

    grid_root = Path(args.grid_root) if args.grid_root else Path(
        f"checkpoints/prefill_{args.model_tag}/grid"
    )
    if not grid_root.is_dir():
        raise SystemExit(f"Grid root not found: {grid_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(grid_root, args.model_tag)
    if not runs:
        raise SystemExit(
            f"No eval JSONs under {grid_root}. "
            f"Run run_prefill_eval_custds.sh first."
        )

    by_method: dict[str, list[dict]] = {}
    for r in runs:
        by_method.setdefault(r["method"], []).append(r)

    print(f"\n{'='*78}")
    print(f"  Best per method ({args.model_tag}, rank by {args.rank_by})")
    print(f"{'='*78}")

    for method in sorted(by_method.keys()):
        method_runs = by_method[method]
        method_runs.sort(key=lambda r: score(r["metrics"], args.rank_by))
        best = method_runs[0]
        bm   = best["metrics"]

        print(f"\n[{method}]  winner = {best['cell']}  epoch {best['epoch']}")
        print(f"  PAR={_fmt_pct(bm['par'])}  "
              f"attack={_fmt_pct(bm['attack_comply'])}  "
              f"clean={_fmt_pct(bm['clean_comply'])}  "
              f"MMLU={_fmt_pct(bm['mmlu'])}")
        print(f"  hp:   {best['hyperparameters']}")
        print(f"  ckpt: {best['checkpoint']}")

        # Mini-leaderboard print (top 5)
        print(f"  --- top {min(5, len(method_runs))} by {args.rank_by} ---")
        for r in method_runs[:5]:
            m = r["metrics"]
            print(f"    {r['cell']:<22}  e{r['epoch']}  "
                  f"PAR={_fmt_pct(m['par'])}  "
                  f"attack={_fmt_pct(m['attack_comply'])}  "
                  f"clean={_fmt_pct(m['clean_comply'])}  "
                  f"MMLU={_fmt_pct(m['mmlu'])}")

        leaderboard = [
            {
                "cell":          r["cell"],
                "epoch":         r["epoch"],
                "score":         score(r["metrics"], args.rank_by),
                "par":           r["metrics"]["par"],
                "attack_comply": r["metrics"]["attack_comply"],
                "clean_comply":  r["metrics"]["clean_comply"],
                "mmlu":          r["metrics"]["mmlu"],
            }
            for r in method_runs
        ]
        out = {
            "method":          best["method"],
            "model":           best["model"],
            "model_tag":       best["model_tag"],
            "cell":            best["cell"],
            "label":           best["label"],
            "epoch":           best["epoch"],
            "checkpoint":      best["checkpoint"],
            "hyperparameters": best["hyperparameters"],
            "rank_by":         args.rank_by,
            # Headline metrics surfaced top-level for skim-readability.
            "par":             bm["par"],
            "attack_comply":   bm["attack_comply"],
            "clean_comply":    bm["clean_comply"],
            "mmlu":            bm["mmlu"],
            "n":               bm["n"],
            "eval":            best["eval"],
            "leaderboard":     leaderboard,
        }
        out_path = output_dir / f"{method}_best.json"
        out_path.write_text(json.dumps(out, indent=2, default=float))
        print(f"  → {out_path}")

    print(f"\nWrote {len(by_method)} files to {output_dir}/")


if __name__ == "__main__":
    main()
