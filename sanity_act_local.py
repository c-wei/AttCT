#!/usr/bin/env python3
"""
Local (no-GPU) sanity check for the ACT sweep pipeline.

Catches config, import, data, and API issues before spinning up RunPod.

Checks:
  1.  Python imports (torch, transformers, peft, datasets, wandb)
  2.  Eval script imports (all 5 eval_*.py)
  3.  run.py + train.py epoch-checkpoint fix present
  4.  All ACT YAML configs parse correctly
  5.  Every config has a save_dir set
  6.  Sweep scripts reference config files that exist
  7.  Sweep script checkpoint paths match config save_dir/epoch_1
  8.  Sweep scripts pass bash -n (syntax check)
  9.  sycophancy_bct dataset files present and non-empty
  10. ClearHarm dataloader (CPU, no model)
  11. eval_sycophancy_behavioral data pipeline (answer extraction + pair loading)
  12. Gemma-3 sweep scripts include eval_sycophancy_behavioral
  13. W&B API authenticated
  14. OpenRouter API reachable

Usage:
    uv run --no-project python sanity_act_local.py
"""

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

# Load .env before any imports that read env vars at module level
# (icl_persona_experiment reads OPENROUTER_API_KEY at import time)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
WARN = "\033[93m  WARN\033[0m"
results = []

HERE = Path(__file__).parent


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


# ── 1. Python imports ─────────────────────────────────────────────────────────

for pkg in ["torch", "transformers", "peft", "datasets", "wandb", "yaml"]:
    check(f"import {pkg}", lambda p=pkg: importlib.import_module(p).__version__)


# ── 2. Eval script imports ────────────────────────────────────────────────────

sys.path.insert(0, str(HERE))
for mod in ["eval_mmlu", "eval_clearharm_behavioral", "eval_persona_behavioral", "eval_mtbench", "eval_sycophancy_behavioral"]:
    check(f"import {mod}", lambda m=mod: importlib.import_module(m) and "ok")


# ── 3. Epoch-checkpoint fix in train.py ───────────────────────────────────────

def check_epoch_checkpoint_fix():
    src = (HERE / "train.py").read_text()
    if '_save_checkpoint(tag=f"epoch_{epoch}")' not in src:
        raise RuntimeError("epoch checkpoint save missing from train.py")
    return "found"
check("train.py saves epoch_{epoch} checkpoint", check_epoch_checkpoint_fix)


# ── 4. All ACT configs parse correctly ───────────────────────────────────────

ACT_CONFIGS = sorted(HERE.glob("configs/act_*llama*.yaml")) + \
              sorted(HERE.glob("configs/act_*gemma*.yaml"))
# exclude act_sweep_log.yaml (it's a log, not a training config)
ACT_CONFIGS = [c for c in ACT_CONFIGS if "sweep_log" not in c.name]

def check_configs():
    errors = []
    for cfg in ACT_CONFIGS:
        try:
            yaml.safe_load(cfg.read_text())
        except Exception as e:
            errors.append(f"{cfg.name}: {e}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return f"{len(ACT_CONFIGS)} configs OK"
check("ACT YAML configs parse", check_configs)


# ── 5. Every config has save_dir ──────────────────────────────────────────────

def check_save_dirs():
    missing = []
    for cfg_path in ACT_CONFIGS:
        data = yaml.safe_load(cfg_path.read_text())
        training = data.get("training", {})
        if not training.get("save_dir"):
            missing.append(cfg_path.name)
    if missing:
        raise RuntimeError(f"Missing save_dir: {', '.join(missing)}")
    return f"all {len(ACT_CONFIGS)} have save_dir"
check("All configs have training.save_dir", check_save_dirs)


# ── 6 & 7. Sweep scripts: config files exist + checkpoint paths match ─────────

SWEEP_SCRIPTS = sorted(HERE.glob("sweep_act_stream_*.sh"))

# Parse sweep scripts for quoted config paths and checkpoint paths
_cfg_re  = re.compile(r'"(configs/[^"]+\.yaml)"')
_ckpt_re = re.compile(r'"(checkpoints/[^"]+/epoch_\d+)"')

def check_sweep_config_files():
    missing = []
    for sh in SWEEP_SCRIPTS:
        src = sh.read_text()
        for cfg_rel in _cfg_re.findall(src):
            cfg_path = HERE / cfg_rel
            if not cfg_path.exists():
                missing.append(f"{sh.name} → {cfg_rel}")
    if missing:
        raise RuntimeError("\n    " + "\n    ".join(missing))
    return f"{len(SWEEP_SCRIPTS)} scripts, all config refs exist"
check("Sweep scripts: referenced config files exist", check_sweep_config_files)


def check_sweep_checkpoint_consistency():
    mismatches = []
    for sh in SWEEP_SCRIPTS:
        src = sh.read_text()
        cfg_paths  = _cfg_re.findall(src)
        ckpt_paths = _ckpt_re.findall(src)
        # pair them up in order of appearance
        for cfg_rel, ckpt_rel in zip(cfg_paths, ckpt_paths):
            cfg_path = HERE / cfg_rel
            if not cfg_path.exists():
                continue
            data = yaml.safe_load(cfg_path.read_text())
            save_dir = data.get("training", {}).get("save_dir", "")
            expected_ckpt = f"{save_dir}/epoch_1"
            if ckpt_rel != expected_ckpt:
                mismatches.append(
                    f"{sh.name}: {cfg_rel} save_dir={save_dir!r} "
                    f"but script uses {ckpt_rel!r}"
                )
    if mismatches:
        raise RuntimeError("\n    " + "\n    ".join(mismatches))
    return "all checkpoint paths match config save_dir/epoch_1"
check("Sweep scripts: checkpoint paths match config save_dir", check_sweep_checkpoint_consistency)


# ── 8. Bash syntax check ──────────────────────────────────────────────────────

def check_bash_syntax():
    errors = []
    for sh in SWEEP_SCRIPTS:
        r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        if r.returncode != 0:
            errors.append(f"{sh.name}: {r.stderr.strip()}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return f"{len(SWEEP_SCRIPTS)} scripts OK"
check("Sweep script bash syntax", check_bash_syntax)


# ── 9. sycophancy_bct dataset files ──────────────────────────────────────────

BCT_FILES = [
    "datasets/sycophancy_bct/control_cot.jsonl",
    "datasets/sycophancy_bct/bct_cot.jsonl",
    "datasets/sycophancy_bct/bct_non_cot.jsonl",
    "datasets/sycophancy_bct/control_non_cot.jsonl",
]

def check_bct_files():
    for rel in BCT_FILES:
        p = HERE / rel
        if not p.exists():
            raise FileNotFoundError(rel)
        lines = p.read_text().strip().splitlines()
        if not lines:
            raise RuntimeError(f"{rel} is empty")
        json.loads(lines[0])  # validate first line is valid JSON
    return f"{len(BCT_FILES)} files present and valid JSON"
check("sycophancy_bct dataset files", check_bct_files)


# ── 10. ClearHarm dataloader (CPU only, no model) ────────────────────────────


def check_clearharm_dataloader():
    from data.attct_datasets import get_dataloader
    with open(HERE / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(HERE / "configs/act_clearharm_llama.yaml") as f:
        overrides = yaml.safe_load(f)
    # deep merge
    def merge(base, ov):
        out = base.copy()
        for k, v in ov.items():
            out[k] = merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
        return out
    config = merge(config, {k: v for k, v in overrides.items() if k != "defaults"})
    config.setdefault("data", {})["limit"] = 4
    dl = get_dataloader(config, split="eval")
    batch = next(iter(dl))
    required = ["clean_input_ids", "wrapped_input_ids", "clean_attention_mask", "wrapped_attention_mask"]
    missing = [k for k in required if k not in batch]
    if missing:
        raise RuntimeError(f"batch missing keys: {missing}")
    return f"{len(dl.dataset)} prompts, batch keys OK"
check("ClearHarm dataloader (CPU)", check_clearharm_dataloader)


# ── 11. eval_sycophancy_behavioral data pipeline ─────────────────────────────

def check_sycophancy_eval_pipeline():
    from eval_sycophancy_behavioral import _extract_answer_letter, _load_eval_pairs

    # Verify answer extraction on known examples (including Gemma-3 bold-markdown format)
    cases = [
        ("Therefore, the best answer is: (C) Ambiguous.", "C"),
        ("THE ANSWER IS B.\nThe best answer is: (B) tan lead good knife.", "B"),
        ("The answer is D", "D"),
        ("Therefore, the answer is **(A)**.", "A"),          # Gemma-3 bold format
        ("the answer is **(B) tan lead good knife**", "B"),  # Gemma-3 bold + trailing text
        # "None of the given options" should return None (unevaluable)
        ("Therefore, the best answer is: (None of the given options)", None),
    ]
    for text, expected in cases:
        got = _extract_answer_letter(text)
        if got != expected:
            raise RuntimeError(
                f"_extract_answer_letter returned {got!r}, expected {expected!r}\n"
                f"  input: {text!r}"
            )

    # Verify data loads and produces evaluable pairs
    pairs = _load_eval_pairs(HERE / "datasets/sycophancy_bct", style="cot", n=20)
    if len(pairs) < 10:
        raise RuntimeError(
            f"Only {len(pairs)} evaluable pairs loaded from bct_cot.jsonl "
            f"(expected ≥10); check ground-truth answer parsing"
        )
    # Spot-check: every pair has a single uppercase letter as correct_answer
    bad = [p for p in pairs if p["correct_answer"] not in "ABCDE"]
    if bad:
        raise RuntimeError(
            f"{len(bad)} pairs have invalid correct_answer: "
            + str([p["correct_answer"] for p in bad[:3]])
        )
    return f"extraction OK; {len(pairs)}/20 pairs evaluable"

check("eval_sycophancy_behavioral data pipeline", check_sycophancy_eval_pipeline)


# ── 12. Gemma-3 sweep scripts include eval_sycophancy_behavioral ──────────────

def check_sycophancy_in_gemma3_sweeps():
    gemma3_scripts = [s for s in SWEEP_SCRIPTS if "gemma3" in s.name]
    if not gemma3_scripts:
        raise RuntimeError("No gemma3 sweep scripts found (expected sweep_act_stream_gemma3_*.sh)")
    missing = []
    for sh in gemma3_scripts:
        if "eval_sycophancy_behavioral.py" not in sh.read_text():
            missing.append(sh.name)
    if missing:
        raise RuntimeError(f"eval_sycophancy_behavioral.py not called in: {', '.join(missing)}")
    return f"{len(gemma3_scripts)} Gemma-3 scripts all include sycophancy eval"

check("Gemma-3 sweeps include eval_sycophancy_behavioral", check_sycophancy_in_gemma3_sweeps)


# ── 13. W&B API ───────────────────────────────────────────────────────────────

def check_wandb():
    import wandb
    api_key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_KEY not set and not found in .env")
    api = wandb.Api(api_key=api_key)
    next(iter(api.runs("neilshah/AttCT", per_page=1)))
    return "authenticated"
check("W&B API", check_wandb)


# ── 14. OpenRouter API ────────────────────────────────────────────────────────

def check_openrouter():
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY not set and not in .env")
    from icl_persona_experiment import _chat, JUDGE_MODEL
    r = _chat(
        [{"role": "user", "content": "Reply with exactly one word: OK"}],
        model=JUDGE_MODEL,
        temperature=0.0,
    )
    return f"response: {r.strip()[:20]!r}"
check("OpenRouter API", check_openrouter)


# ── Summary ───────────────────────────────────────────────────────────────────

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
    print("  All checks passed — configs, data, and APIs look good.")
print(f"{'='*60}\n")
