#!/usr/bin/env python3
"""
Unified sweep runner. Reads configs/experiments.yaml, builds run configs
on-the-fly, and runs training + evals for each matching experiment.

Usage:
    # Run all experiments for a model
    python sweep.py --model gemma3_27b

    # Run a single experiment by ID
    python sweep.py --id jsd_persona_gemma3_27b_lora_lr1e6

    # Filter by loss or data
    python sweep.py --model gemma3_27b --loss jsd

    # Dry run — print commands without executing
    python sweep.py --model gemma2b --dry-run

    # Resume — skip pre-evals + training, run post-evals only
    python sweep.py --id jsd_persona_gemma3_27b_lora_lr5e6 --resume
"""

import argparse
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

EXPERIMENTS_FILE = Path("configs/experiments.yaml")

# Maps eval suite entry → (script, extra_args).
# batch-size overrides are applied separately per model.
EVAL_SCRIPTS = {
    "mmlu":           ("eval_mmlu.py",                []),
    "mtbench":        ("eval_mtbench.py",             []),
    "persona_prefix": ("eval_persona_behavioral.py",  ["--k", "20", "--facts-position", "prefix"]),
    "persona_suffix": ("eval_persona_behavioral.py",  ["--k", "20", "--facts-position", "suffix"]),
    "clearharm":      ("eval_clearharm_behavioral.py", []),
    "sycophancy":     ("eval_sycophancy_behavioral.py", []),
    "frustration":    ("eval_frustration.py",          ["--n-prompts", "5", "--n-samples", "5"]),
}


def load_spec():
    with open(EXPERIMENTS_FILE) as f:
        return yaml.safe_load(f)


def build_run_config(exp, spec):
    """Build a run.py config dict from an experiment spec + shared presets."""
    model_cfg  = spec["models"][exp["model"]]
    loss_pre   = spec["loss_presets"][exp["loss"]]
    data_pre   = spec["data_presets"][exp["data"]]
    lora       = spec["lora_defaults"]

    config = {
        "model": {"name": model_cfg["name"], **loss_pre["model_flags"]},
        "loss":  {k: v for k, v in loss_pre.items() if k != "model_flags"},
        "data":  data_pre,
        "training": {
            "learning_rate": exp["lr"],
            "save_dir": f"checkpoints/{exp['id']}",
        },
    }
    if exp["mode"] == "lora":
        config["lora"] = lora
    return config


def run_cmd(cmd, dry_run=False):
    cmd = [str(c) for c in cmd]
    if dry_run:
        print("  DRY:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def batch_size_args(eval_name, model_cfg):
    """Return --batch-size args for evals where the model overrides the default."""
    overrides = model_cfg.get("eval_batch_sizes", {})
    # persona_prefix and persona_suffix share the "persona" override key
    key = "persona" if eval_name.startswith("persona") else eval_name
    bs = overrides.get(key)
    return ["--batch-size", str(bs)] if bs else []


def run_evals(eval_names, ckpt, model_cfg, phase, run_id, run_name, wandb_group, dry_run):
    for name in eval_names:
        script, extra = EVAL_SCRIPTS[name]
        cmd = ["python", script]
        if ckpt:
            cmd += ["--checkpoint", ckpt]
        cmd += ["--model", model_cfg["name"]]
        if run_name:
            cmd += ["--run-name", run_name, "--wandb-group", wandb_group]
        cmd += ["--wandb-run-id", run_id, "--metric-prefix", f"{phase}/"]
        cmd += extra
        cmd += batch_size_args(name, model_cfg)
        print(f"  [{phase}] {name}...")
        run_cmd(cmd, dry_run)


def run_experiment(exp, spec, dry_run, resume):
    model_cfg   = spec["models"][exp["model"]]
    eval_suite  = spec["eval_suites"][exp["evals"]]
    ckpt        = f"checkpoints/{exp['id']}/epoch_1"
    run_id      = secrets.token_hex(4)
    run_name    = exp["id"]
    wandb_group = exp["id"]

    print(f"\n{'='*62}")
    print(f"  {exp['id']}")
    print(f"  model={exp['model']}  loss={exp['loss']}  data={exp['data']}  mode={exp['mode']}  lr={exp['lr']}")
    print(f"  evals={exp['evals']}: {', '.join(eval_suite)}")
    print(f"  W&B run ID: {run_id}")
    if resume:
        print(f"  (resume — skipping pre-evals and training)")
    print(f"{'='*62}\n")

    if not resume:
        print("--- Pre-training evals ---")
        run_evals(eval_suite, "", model_cfg, "pre", run_id, run_name, wandb_group, dry_run)

        print("\n--- Training ---")
        config = build_run_config(exp, spec)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix=f"sweep_{exp['id']}_",
                                         delete=False) as f:
            yaml.dump(config, f)
            tmp_cfg = f.name
        try:
            run_cmd([
                "python", "run.py", "--config", tmp_cfg,
                "--run-name", run_name, "--wandb-group", wandb_group,
                "--wandb-run-id", run_id, "--skip-eval",
            ], dry_run)
        finally:
            os.unlink(tmp_cfg)

    print("\n--- Post-training evals ---")
    # Omit run_name on post pass — W&B resumes existing run by run_id
    run_evals(eval_suite, ckpt, model_cfg, "post", run_id, "", "", dry_run)


def main():
    parser = argparse.ArgumentParser(description="Unified sweep runner (reads configs/experiments.yaml)")
    parser.add_argument("--model",   help="Filter by model key, e.g. gemma3_27b")
    parser.add_argument("--id",      help="Run a single experiment by ID")
    parser.add_argument("--loss",    help="Filter by loss key, e.g. jsd")
    parser.add_argument("--data",    help="Filter by data key, e.g. persona")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip pre-evals and training; run post-evals only (checkpoint must exist)")
    args = parser.parse_args()

    spec = load_spec()
    exps = spec["experiments"]

    if args.id:
        exps = [e for e in exps if e["id"] == args.id]
        if not exps:
            sys.exit(f"No experiment with id={args.id!r}")
    else:
        if args.model: exps = [e for e in exps if e["model"] == args.model]
        if args.loss:  exps = [e for e in exps if e["loss"]  == args.loss]
        if args.data:  exps = [e for e in exps if e["data"]  == args.data]

    if not exps:
        sys.exit("No matching experiments found.")

    tag = " [DRY RUN]" if args.dry_run else ""
    print(f"\nRunning {len(exps)} experiment(s){tag}:")
    for e in exps:
        print(f"  {e['id']}")

    for exp in exps:
        run_experiment(exp, spec, args.dry_run, args.resume)

    print(f"\n{'='*62}")
    print(f"  Sweep complete ({len(exps)} experiment(s)).")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
