#!/usr/bin/env python3
"""
Unified sanity check — replaces sanity_act_local.py, sanity_act_sweep.py,
and sanity_gemma3_27b.sh.

Usage:
    python sanity.py                          # local checks only (fast, no GPU)
    python sanity.py --scope gpu              # + GPU/model/training checks
    python sanity.py --scope gpu --model gemma3_4b  # GPU checks for a specific model

Local checks (always run):
  - Core imports
  - Eval scripts present
  - experiments.yaml integrity (all preset references valid)
  - sycophancy_bct dataset files
  - persona_configs files
  - W&B and OpenRouter API access

GPU checks (--scope gpu):
  - CUDA available + VRAM (warns if < model min_vram_gb)
  - Disk space (>50GB free)
  - Model load
  - Mini training run (2 steps, epoch_1 checkpoint saved)
  - All eval scripts from checkpoint
  - Frustration eval mini-run (if model uses "full" eval suite)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

EXPERIMENTS_FILE = Path("configs/experiments.yaml")
SANITY_CKPT = "/tmp/sanity_attct_ckpt"

_results: list[tuple[str, bool, str | None]] = []


def check(name: str, fn) -> bool:
    try:
        msg = fn()
        _results.append((name, True, None))
        print(f"  \u2713 {name}" + (f" \u2014 {msg}" if msg else ""))
        return True
    except Exception as e:
        _results.append((name, False, str(e)))
        print(f"  \u2717 {name}")
        print(f"    {str(e)[:200]}")
        return False


def run_cmd(cmd: list, timeout: int = 300) -> str:
    result = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        out = result.stderr or result.stdout
        raise RuntimeError(out.strip()[-500:])
    return result.stdout


# ─── Local checks ─────────────────────────────────────────────────────────────

def check_imports():
    import yaml, torch, transformers, peft, datasets, wandb  # noqa: F401
    import torch
    return f"torch={torch.__version__}"


def check_eval_scripts():
    scripts = [
        "eval_mmlu", "eval_mtbench", "eval_persona_behavioral",
        "eval_clearharm_behavioral", "eval_sycophancy_behavioral", "eval_frustration",
    ]
    missing = [s for s in scripts if not Path(f"{s}.py").exists()]
    if missing:
        raise RuntimeError(f"Missing: {missing}")
    return f"{len(scripts)} scripts present"


def check_experiments_yaml():
    with open(EXPERIMENTS_FILE) as f:
        spec = yaml.safe_load(f)
    exps = spec["experiments"]
    for e in exps:
        for key, pool in [("model", "models"), ("loss", "loss_presets"),
                          ("data", "data_presets"), ("evals", "eval_suites")]:
            if e[key] not in spec[pool]:
                raise RuntimeError(f"{e['id']}: unknown {key}={e[key]!r}")
    ids = [e["id"] for e in exps]
    dupes = [x for x in ids if ids.count(x) > 1]
    if dupes:
        raise RuntimeError(f"Duplicate experiment IDs: {dupes}")
    return f"{len(exps)} experiments, all refs valid"


def check_sycophancy_data():
    expected = [
        "datasets/sycophancy_bct/sycophancy_fact.jsonl",
        "datasets/sycophancy_bct/sycophancy_nlp.jsonl",
        "datasets/sycophancy_bct/sycophancy_are_you_sure.jsonl",
    ]
    missing = [p for p in expected if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"Missing: {missing}")
    return f"{len(expected)} files"


def check_persona_configs():
    with open(EXPERIMENTS_FILE) as f:
        spec = yaml.safe_load(f)
    persona_data = spec["data_presets"].get("persona", {})
    configs = persona_data.get("persona_configs", [])
    missing = [c for c in configs if not Path(c).exists()]
    if missing:
        raise RuntimeError(f"Missing: {missing}")
    return f"{len(configs)} persona_config files"


def check_wandb_api():
    import wandb
    api = wandb.Api()
    next(iter(api.runs("neilshah/AttCT", per_page=1)))
    return "authenticated"


def check_openrouter_api():
    from shared.gemma_frustration_experiment import _openrouter_chat
    r = _openrouter_chat(
        [{"role": "user", "content": "Reply with exactly one word: OK"}],
        model="google/gemini-2.5-flash",
        temperature=0.0,
    )
    return f"response: {r.strip()[:20]!r}"


def run_local_checks():
    check("Core imports (yaml/torch/transformers/peft/datasets/wandb)", check_imports)
    check("Eval scripts present", check_eval_scripts)
    check("experiments.yaml integrity", check_experiments_yaml)
    check("sycophancy_bct dataset files", check_sycophancy_data)
    check("persona_configs files", check_persona_configs)
    check("W&B API", check_wandb_api)
    check("OpenRouter API", check_openrouter_api)


# ─── GPU checks ───────────────────────────────────────────────────────────────

def run_gpu_checks(model_key: str):
    with open(EXPERIMENTS_FILE) as f:
        spec = yaml.safe_load(f)
    model_cfg  = spec["models"][model_key]
    model_name = model_cfg["name"]
    lora       = spec["lora_defaults"]
    loss_pre   = spec["loss_presets"]["act"]

    # Which eval suite does this model use?
    model_exps = [e for e in spec["experiments"] if e["model"] == model_key]
    uses_frustration = any(
        "frustration" in spec["eval_suites"][e["evals"]]
        for e in model_exps
    )

    # 1. CUDA
    def _cuda():
        import torch
        assert torch.cuda.is_available(), "No CUDA GPU found"
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        min_vram = model_cfg.get("min_vram_gb", 0)
        if min_vram and vram < min_vram:
            raise RuntimeError(f"Only {vram:.0f}GB VRAM; {model_key} requires {min_vram}GB")
        return f"{name} / {vram:.0f}GB VRAM"
    check("CUDA available", _cuda)

    # 2. Disk space
    def _disk():
        stat = shutil.disk_usage("/workspace" if Path("/workspace").exists() else "/")
        free_gb = stat.free / 1e9
        assert free_gb > 50, f"Only {free_gb:.0f}GB free (need >50GB)"
        return f"{free_gb:.0f}GB free"
    check("Disk space (>50GB)", _disk)

    # 3. Model load
    def _model_load():
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        n = sum(p.numel() for p in m.parameters()) / 1e9
        del m
        torch.cuda.empty_cache()
        return f"{model_name} / {n:.1f}B params"
    check(f"Model load ({model_key})", _model_load)

    # 4. Mini training (2 steps)
    if Path(SANITY_CKPT).exists():
        shutil.rmtree(SANITY_CKPT)

    def _training():
        cfg = {
            "model": {"name": model_name, **loss_pre["model_flags"]},
            "loss":  {k: v for k, v in loss_pre.items() if k != "model_flags"},
            "data":  {"source": "sycophancy_bct", "mode": "sycophancy", "limit": 8},
            "training": {
                "learning_rate": 5e-6, "epochs": 1, "max_steps": 2,
                "batch_size": 1, "grad_accumulation_steps": 1,
                "save_dir": SANITY_CKPT,
            },
            "lora": lora,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            tmp = f.name
        try:
            run_cmd(["python", "run.py", "--config", tmp, "--skip-eval",
                     "--wandb-group", "sanity", "--run-name", f"sanity_{model_key}"],
                    timeout=900)
        finally:
            os.unlink(tmp)
        epoch_ckpt = os.path.join(SANITY_CKPT, "epoch_1")
        if not os.path.isdir(epoch_ckpt):
            raise RuntimeError(f"epoch_1 not found at {epoch_ckpt}")
        files = len(os.listdir(epoch_ckpt))
        return f"epoch_1 saved ({files} files)"

    training_ok = check(f"Mini training — 2 steps ({model_key})", _training)
    ckpt_args = ["--checkpoint", os.path.join(SANITY_CKPT, "epoch_1")] if training_ok else []

    # 5. Eval scripts (mini runs from checkpoint)
    eval_checks = [
        ("eval_mmlu",                 ["--n-samples", "5"]),
        ("eval_clearharm_behavioral", ["--n-samples", "5"]),
        ("eval_persona_behavioral",   ["--k", "3", "--facts-position", "prefix", "--n-samples", "1"]),
        ("eval_mtbench",              ["--n-questions", "2"]),
        ("eval_sycophancy_behavioral",["--n-samples", "10"]),
    ]
    if uses_frustration:
        eval_checks.append(
            ("eval_frustration", ["--n-prompts", "1", "--n-samples", "1", "--n-turns", "2",
                                  "--gen-batch-size", "1", "--judge-workers", "1"])
        )

    wandb_args = ["--wandb-group", "sanity"]
    model_args = ["--model", model_name]

    for script, extra in eval_checks:
        def _make_eval(s, e):
            def _fn():
                run_cmd(["python", f"{s}.py", *ckpt_args, *model_args,
                         "--run-name", f"sanity_{model_key}", *wandb_args, *e], timeout=600)
                return " ".join(e[:4])
            return _fn
        check(script, _make_eval(script, extra))

    # Cleanup
    if Path(SANITY_CKPT).exists():
        shutil.rmtree(SANITY_CKPT)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AttCT sanity check")
    parser.add_argument("--scope", choices=["local", "gpu"], default="local",
                        help="local: imports/configs/APIs (fast). gpu: adds model load + training.")
    parser.add_argument("--model", default="gemma3_27b",
                        help="Model key for GPU checks (default: gemma3_27b)")
    args = parser.parse_args()

    print(f"\n{'='*62}")
    print(f"  AttCT sanity check  [scope={args.scope}]")
    if args.scope == "gpu":
        print(f"  Model: {args.model}")
    print(f"{'='*62}\n")

    run_local_checks()
    if args.scope == "gpu":
        print()
        run_gpu_checks(args.model)

    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_fail = len(_results) - n_pass
    print(f"\n{'='*62}")
    print(f"  {n_pass}/{len(_results)} checks passed")
    if n_fail:
        print("  FAILED:")
        for name, ok, err in _results:
            if not ok:
                print(f"    \u2717 {name}")
                if err:
                    print(f"      {err[:120]}")
        sys.exit(1)
    else:
        print(f"  All checks passed.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
