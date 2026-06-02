#!/usr/bin/env python3
"""Re-run the sanity gauntlet against existing per-layer role_means_layer{L}.pt
files — no GPU, no re-extraction. Used when we need to iterate on gauntlet
logic without re-running the ~70 min Phase A extraction.

Outputs sanity_gauntlet_{model_key}.json the same way extract_axis.py does.
If at least one layer passes, also re-saves axis_layer{L}.pt for the chosen
layer to ensure the orchestrator's downstream step finds it.

Usage
-----
    python steering/axis/rerun_gauntlet.py \\
        --config steering/axis/configs/gemma3_27b.yaml \\
        --output-dir results/axis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

THIS_DIR = Path(__file__).resolve().parent
STEERING_DIR = THIS_DIR.parent
sys.path.insert(0, str(STEERING_DIR))

from sanity_gauntlet import compute_contrast_axis, evaluate_layer  # noqa: E402


def load_roles(path: Path) -> dict[str, list[str]]:
    d = json.loads(path.read_text())
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--roles", default=str(THIS_DIR / "data" / "roles_v1.json"))
    ap.add_argument("--vectors-dir", default=None,
                    help="Defaults to steering/vectors/{config-stem}/")
    ap.add_argument("--output-dir", default=str(THIS_DIR.parent.parent / "results" / "axis"))
    ap.add_argument("--model-key", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    hf_id = cfg["hf_id"]
    model_key = args.model_key or Path(args.config).stem
    sweep_layers: list[int] = cfg.get("sweep_layers") or [cfg.get("chosen_layer")]

    vectors_dir = Path(args.vectors_dir or STEERING_DIR / "vectors" / model_key)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    roles = load_roles(Path(args.roles))
    every_role = (
        roles["assistant_anchor"] + roles["other"]
        + roles["sanity_positive"] + roles["sanity_negative"]
    )

    per_layer_report: dict[str, dict] = {}
    for L in sweep_layers:
        rm_path = vectors_dir / f"role_means_layer{L}.pt"
        if not rm_path.exists():
            print(f"[skip] {rm_path} missing")
            continue
        rm = torch.load(rm_path, map_location="cpu", weights_only=True)
        # Convert to numpy
        per_role_mean = {r: rm[r].numpy() for r in every_role if r in rm}

        anchor_mat = np.stack([per_role_mean[r] for r in roles["assistant_anchor"]], axis=0)
        other_mat  = np.stack([per_role_mean[r] for r in roles["other"]], axis=0)
        pos_mat    = np.stack([per_role_mean[r] for r in roles["sanity_positive"]], axis=0)
        neg_mat    = np.stack([per_role_mean[r] for r in roles["sanity_negative"]], axis=0)

        axis = compute_contrast_axis(anchor_mat, other_mat)
        report = evaluate_layer(
            axis=axis,
            assistant_anchor_acts=anchor_mat,
            other_acts=other_mat,
            sanity_pos_acts=pos_mat,
            sanity_neg_acts=neg_mat,
        )
        report["layer"] = L
        per_layer_report[str(L)] = report

        # persist refreshed axis
        torch.save(torch.tensor(axis, dtype=torch.float32),
                   vectors_dir / f"axis_layer{L}.pt")

        flags = [k for k in report if k.startswith("g") and report[k] is True]
        print(f"[L={L}] all_pass={report['all_pass']}  effect={report['effect_size']:.2f}  "
              f"passed={','.join(flags) if flags else 'none'}")

    passing = sorted([int(L) for L, rep in per_layer_report.items() if rep["all_pass"]])
    chosen = passing[0] if passing else None
    summary = {
        "model": hf_id,
        "num_hidden_layers": None,
        "swept_layers": sweep_layers,
        "passing_layers": passing,
        "chosen_layer": chosen,
        "per_layer": per_layer_report,
        "rerun_from_cached_role_means": True,
    }
    # try to preserve num_hidden_layers from the prior gauntlet if present
    prior = output_dir / f"sanity_gauntlet_{model_key}.json"
    if prior.exists():
        try:
            prior_data = json.loads(prior.read_text())
            summary["num_hidden_layers"] = prior_data.get("num_hidden_layers")
        except Exception:
            pass

    out_path = output_dir / f"sanity_gauntlet_{model_key}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] passing={passing}  chosen={chosen}  → {out_path}")
    if chosen is None:
        print("[STOP] No layer passed the gauntlet.")
        sys.exit(2)


if __name__ == "__main__":
    main()
