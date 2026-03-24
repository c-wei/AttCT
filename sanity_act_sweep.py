#!/usr/bin/env python3
"""
ACT sweep sanity check — run on each RunPod instance before launching a sweep.

Exercises the full pipeline end-to-end with minimal data:
  1. Env vars + CUDA + disk space
  2. Model load
  3. Mini training (2 steps) → epoch checkpoint saved
  4. eval_mmlu (5 questions) from checkpoint
  5. eval_clearharm_behavioral (5 questions) from checkpoint
  6. eval_persona_behavioral prefix (k=3) from checkpoint
  7. eval_persona_behavioral suffix (k=3) from checkpoint
  8. eval_mtbench (2 questions) from checkpoint
  9. W&B + OpenRouter APIs

Usage:
    python sanity_act_sweep.py --model google/gemma-2-2b-it
    python sanity_act_sweep.py --model meta-llama/Llama-3.1-8B-Instruct
    python sanity_act_sweep.py --model google/gemma-3-4b-it
    python sanity_act_sweep.py --model google/gemma-3-27b-it   # LoRA only
"""

import argparse
import os
import shutil
import subprocess
import sys

import yaml

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
results = []

SANITY_CKPT = "/tmp/sanity_act_ckpt"


def check(name, fn):
    try:
        msg = fn()
        print(f"{PASS}  {name}" + (f": {msg}" if msg else ""))
        results.append((name, True, None))
        return True
    except Exception as e:
        print(f"{FAIL}  {name}: {e}")
        results.append((name, False, str(e)))
        return False


def run_cmd(cmd, timeout=600):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-800:]
        raise RuntimeError(tail)
    return r.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID to test")
    args = parser.parse_args()
    model = args.model
    # 27B model: LoRA only (full FT won't fit even on A100 with optimizer states)
    use_lora = "27b" not in model.lower()

    print(f"\n{'='*60}")
    print(f"  ACT Sweep Sanity Check")
    print(f"  Model : {model}")
    print(f"  Mode  : {'LoRA q+v' if use_lora else 'LoRA q+v (27B)'}")
    print(f"{'='*60}\n")

    # ── 1. Environment ─────────────────────────────────────────────────────
    def check_env():
        missing = [v for v in ["HF_TOKEN", "WANDB_KEY", "OPENROUTER_API_KEY", "HF_HOME"]
                   if not os.environ.get(v)]
        if missing:
            raise ValueError(f"Not set: {', '.join(missing)}")
        return "all vars present"
    check("Environment variables", check_env)

    def check_cuda():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{name} ({vram_gb:.0f} GB)"
    check("CUDA + GPU", check_cuda)

    def check_disk():
        free_gb = shutil.disk_usage("/workspace").free / 1e9
        if free_gb < 10:
            raise RuntimeError(f"Only {free_gb:.1f} GB free — need ≥10 GB")
        return f"{free_gb:.0f} GB free on /workspace"
    check("Disk space", check_disk)

    # ── 2. Model load ───────────────────────────────────────────────────────
    def check_model_load():
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        n_params = sum(p.numel() for p in m.parameters()) / 1e9
        del m
        torch.cuda.empty_cache()
        return f"{n_params:.1f}B params loaded"
    check(f"Model load", check_model_load)

    # ── 3. Mini training run ────────────────────────────────────────────────
    if os.path.exists(SANITY_CKPT):
        shutil.rmtree(SANITY_CKPT)

    def check_training():
        override = {
            "model": {
                "name": model,
                "output_attentions": False,
                "output_hidden_states": True,
            },
            "loss": {
                "name": "ActivationConsistencyLoss",
                "weight": 0.0001,
                "kwargs": {"layer_selection": "all", "normalize": False},
            },
            "data": {"source": "sycophancy_bct", "mode": "sycophancy", "limit": 8},
            "training": {
                "learning_rate": 5e-6,
                "epochs": 1,
                "max_steps": 2,
                "batch_size": 1,
                "grad_accumulation_steps": 1,
                "save_dir": SANITY_CKPT,
            },
        }
        if use_lora:
            override["lora"] = {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
            }

        tmp_cfg = "/tmp/sanity_act_config.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.dump(override, f)

        run_cmd([
            "python", "run.py",
            "--config", tmp_cfg,
            "--skip-eval",
            "--wandb-group", "sanity_act_sweep",
            "--run-name", f"sanity_{model.split('/')[-1]}",
        ], timeout=300)

        epoch_ckpt = os.path.join(SANITY_CKPT, "epoch_1")
        if not os.path.isdir(epoch_ckpt):
            raise RuntimeError(f"epoch_1 checkpoint not found at {epoch_ckpt}")
        files = os.listdir(epoch_ckpt)
        return f"epoch_1 saved ({len(files)} files)"

    training_ok = check("Mini training (2 steps) + epoch checkpoint", check_training)

    # ── 4–8. Eval scripts ───────────────────────────────────────────────────
    epoch_ckpt = os.path.join(SANITY_CKPT, "epoch_1")
    ckpt_args = ["--checkpoint", epoch_ckpt] if training_ok else []
    model_args = ["--model", model]
    wandb_args = ["--wandb-group", "sanity_act_sweep"]

    def check_mmlu():
        run_cmd(["python", "eval_mmlu.py",
                 *ckpt_args, *model_args, *wandb_args,
                 "--n-samples", "5"])
        return "5 questions"
    check("eval_mmlu", check_mmlu)

    def check_clearharm():
        run_cmd(["python", "eval_clearharm_behavioral.py",
                 *ckpt_args, *model_args, *wandb_args,
                 "--n-samples", "5"])
        return "5 questions"
    check("eval_clearharm_behavioral", check_clearharm)

    def check_persona_prefix():
        run_cmd(["python", "eval_persona_behavioral.py",
                 *ckpt_args, *model_args, *wandb_args,
                 "--k", "3", "--facts-position", "prefix", "--n-samples", "1"])
        return "k=3 prefix"
    check("eval_persona_behavioral (prefix)", check_persona_prefix)

    def check_persona_suffix():
        run_cmd(["python", "eval_persona_behavioral.py",
                 *ckpt_args, *model_args, *wandb_args,
                 "--k", "3", "--facts-position", "suffix", "--n-samples", "1"])
        return "k=3 suffix"
    check("eval_persona_behavioral (suffix)", check_persona_suffix)

    def check_mtbench():
        run_cmd(["python", "eval_mtbench.py",
                 *ckpt_args, *model_args, *wandb_args,
                 "--n-questions", "2"])
        return "2 questions"
    check("eval_mtbench", check_mtbench)

    # ── 9. APIs ─────────────────────────────────────────────────────────────
    def check_wandb_api():
        import wandb
        api = wandb.Api()
        next(iter(api.runs("neilshah/AttCT", per_page=1)))
        return "authenticated"
    check("W&B API", check_wandb_api)

    def check_openrouter():
        from icl_persona_experiment import _chat, JUDGE_MODEL
        r = _chat(
            [{"role": "user", "content": "Reply with exactly one word: OK"}],
            model=JUDGE_MODEL, temperature=0.0,
        )
        return f"response: {r.strip()[:20]!r}"
    check("OpenRouter API", check_openrouter)

    # ── Summary ─────────────────────────────────────────────────────────────
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print(f"\n{'='*60}")
    print(f"  {n_pass}/{len(results)} checks passed")
    if n_fail:
        print("  FAILED:")
        for name, ok, err in results:
            if not ok:
                err_short = (err or "")[:120].replace("\n", " ")
                print(f"    ✗ {name}")
                print(f"      {err_short}")
        sys.exit(1)
    else:
        print(f"  All checks passed — safe to run sweep script.")
    print(f"{'='*60}\n")

    # Clean up sanity checkpoint
    if os.path.exists(SANITY_CKPT):
        shutil.rmtree(SANITY_CKPT)


if __name__ == "__main__":
    main()
