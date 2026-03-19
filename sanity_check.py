#!/usr/bin/env python3
"""
RunPod sanity check — run before sweep_persona_experiments.sh to catch
environment, API, and pipeline issues early.

Usage:
    uv run python sanity_check.py
"""

import os
import sys

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
results = []


def check(name, fn):
    try:
        msg = fn()
        print(f"{PASS}  {name}" + (f": {msg}" if msg else ""))
        results.append((name, True))
    except Exception as e:
        print(f"{FAIL}  {name}: {e}")
        results.append((name, False))


# ── Environment ──────────────────────────────────────────────────────────────

def check_env_var(var):
    val = os.environ.get(var, "")
    if not val:
        raise ValueError(f"{var} is not set")
    return f"{val[:8]}..."

check("HF_TOKEN set",           lambda: check_env_var("HF_TOKEN"))
check("WANDB_KEY set",          lambda: check_env_var("WANDB_KEY"))
check("OPENROUTER_API_KEY set", lambda: check_env_var("OPENROUTER_API_KEY"))
check("HF_HOME set",            lambda: check_env_var("HF_HOME"))
check("CUDA available", lambda: __import__("torch").cuda.get_device_name(0))

# ── Imports ───────────────────────────────────────────────────────────────────

check("torch",          lambda: __import__("torch").__version__)
check("transformers",   lambda: __import__("transformers").__version__)
check("peft",           lambda: __import__("peft").__version__)
check("datasets",       lambda: __import__("datasets").__version__)
check("wandb",          lambda: __import__("wandb").__version__)

check("eval_mmlu imports",               lambda: __import__("eval_mmlu") and None)
check("eval_clearharm_behavioral imports", lambda: __import__("eval_clearharm_behavioral") and None)
check("eval_persona_behavioral imports",   lambda: __import__("eval_persona_behavioral") and None)

# ── Model & tokenizer ─────────────────────────────────────────────────────────

def check_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    if tok.chat_template is None:
        raise ValueError("chat_template is None")
    msgs = [{"role": "user", "content": "hi"}]
    tokenized = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized
    return f"chat template OK, {ids.shape[-1]} tokens"

check("Llama-3.1-8B-Instruct tokenizer + chat template", check_tokenizer)

def check_model_load():
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    del model
    torch.cuda.empty_cache()
    return f"{n_params:.1f}B params"

check("Model load (Llama-3.1-8B-Instruct)", check_model_load)

# ── Data ──────────────────────────────────────────────────────────────────────

def check_clearharm():
    import yaml
    from data.attct_datasets import get_dataloader
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    with open("configs/clearharm_eval.yaml") as f:
        overrides = yaml.safe_load(f)
    config.update({k: v for k, v in overrides.items() if k != "defaults"})
    dl = get_dataloader(config, split="eval")
    batch = next(iter(dl))
    return f"{len(dl.dataset)} prompts, batch shape {tuple(batch['wrapped_input_ids'].shape)}"

check("ClearHarm dataloader", check_clearharm)

def check_persona_facts():
    import yaml
    from pathlib import Path
    from icl_persona_experiment import load_facts, build_messages, ALIGNMENT_QUESTIONS
    results_inner = []
    for persona in ["mao", "binladen", "genghis", "bundy", "hitler"]:
        with open(f"persona_configs/{persona}.yaml") as f:
            cfg = yaml.safe_load(f)
        facts = load_facts(Path(cfg["facts_path"]))
        msgs = build_messages(facts, k=5, question=ALIGNMENT_QUESTIONS[0])
        results_inner.append(f"{persona}:{len(facts)}")
    return ", ".join(results_inner)

check("Persona facts + build_messages (all 5)", check_persona_facts)

# ── APIs ──────────────────────────────────────────────────────────────────────

def check_openrouter():
    from icl_persona_experiment import _chat
    result = _chat(
        [{"role": "user", "content": "Reply with exactly one word: OK"}],
        model="google/gemini-2.5-flash",
        temperature=0.0,
    )
    return f"response: {result.strip()[:20]!r}"

check("OpenRouter Gemini Flash API", check_openrouter)

def check_wandb():
    import wandb
    api = wandb.Api()
    runs = api.runs("neilshah/AttCT", per_page=1)
    next(iter(runs))
    return "authenticated"

check("W&B API", check_wandb)

# ── Summary ───────────────────────────────────────────────────────────────────

n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
print(f"\n{'='*50}")
print(f"  {n_pass}/{len(results)} checks passed")
if n_fail:
    print(f"  FAILED:")
    for name, ok in results:
        if not ok:
            print(f"    - {name}")
    sys.exit(1)
else:
    print("  All checks passed — safe to run sweep_persona_experiments.sh")
print(f"{'='*50}")
