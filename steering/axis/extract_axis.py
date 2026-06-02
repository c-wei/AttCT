#!/usr/bin/env python3
"""Extract Assistant Axis at one or more layers via role-contrast.

Handles Phase A0 (sweep), A1 (single layer for Gemma-3), and A2 (single
layer for Gemma-4) — only difference is the layer list and the model.

Inputs
------
- steering/axis/data/roles_v1.json  : extraction roles + held-out sanity roles
- steering/axis/data/questions_v1.json : 50 extraction questions
- steering/data/neutral_dialogues.jsonl : neutral confound check
- --config (steering/axis/configs/{gemma3_27b,gemma4_31b}.yaml) :
    model id, sweep_layers, generation params

Outputs
-------
- {output-dir}/sanity_gauntlet_{model_key}.json : per-layer gauntlet
- {vectors-dir}/axis_layer{L}.pt for each evaluated layer
- {vectors-dir}/role_means_layer{L}.pt for inspection
- prints the chosen layer (lowest passing) at the end; exit 2 if none

Usage
-----
    # Phase A0 (sweep 4 layers on Gemma-3)
    uv run --no-project python steering/axis/extract_axis.py \\
        --config steering/axis/configs/gemma3_27b.yaml

    # Phase A2 (single layer for Gemma-4 at middle depth)
    uv run --no-project python steering/axis/extract_axis.py \\
        --config steering/axis/configs/gemma4_31b.yaml \\
        --layers 28

    # Phase A2 fallback (sweep on Gemma-4)
    uv run --no-project python steering/axis/extract_axis.py \\
        --config steering/axis/configs/gemma4_31b.yaml \\
        --layers 20 28 36
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse parent steering utilities
THIS_DIR = Path(__file__).resolve().parent          # .../steering/axis
STEERING_DIR = THIS_DIR.parent                       # .../steering
sys.path.insert(0, str(STEERING_DIR))
from extract_emotion_vector import _find_layers, pool_positions  # noqa: E402
from layer_sweep import START_TOKEN  # noqa: E402

# Local
from sanity_gauntlet import compute_contrast_axis, evaluate_layer  # noqa: E402


def collect_selected_layer_acts(
    model, tokenizer, texts, layer_idxs, batch_size, device,
    position="mean-after-50", last_k=0, max_length=512, desc="",
):
    """Like layer_sweep.collect_all_layer_activations, but hooks only the
    layers in `layer_idxs`. Returns list[item] of dict[layer_idx, ndarray[hidden]].

    Hooking only a few layers (vs all ~60) cuts per-batch CPU<->GPU transfer
    by ~15x — critical for activation capture to finish in minutes not hours.
    """
    captured: dict[int, torch.Tensor] = {}
    layers = _find_layers(model)

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            captured[layer_idx] = h.detach().cpu().float()
        return hook

    handles = [layers[L].register_forward_hook(make_hook(L)) for L in layer_idxs]
    all_acts: list[dict[int, np.ndarray]] = []
    try:
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for b, batch_start in enumerate(range(0, len(texts), batch_size)):
            batch = texts[batch_start: batch_start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)
            with torch.inference_mode():
                model(**enc)
            for item_idx in range(enc["input_ids"].shape[0]):
                seq_len = int(enc["attention_mask"][item_idx].sum().item())
                item_acts: dict[int, np.ndarray] = {}
                for L in layer_idxs:
                    h_valid = captured[L][item_idx][:seq_len]
                    item_acts[L] = pool_positions(h_valid, position, last_k, START_TOKEN).numpy()
                all_acts.append(item_acts)
            if desc and (b % 5 == 0 or b == n_batches - 1):
                print(f"  {desc} [{batch_start + enc['input_ids'].shape[0]}/{len(texts)}]", flush=True)
    finally:
        for h in handles:
            h.remove()
    return all_acts


def load_roles(path: Path) -> dict[str, list[str]]:
    d = json.loads(path.read_text())
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_questions(path: Path) -> list[str]:
    d = json.loads(path.read_text())
    qs: list[str] = []
    for k, v in d.items():
        if k.startswith("_"):
            continue
        qs.extend(v)
    return qs


def load_neutral_dialogues(path: Path, max_n: int = 80) -> list[str]:
    out: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # neutral_dialogues.jsonl format: {"dialogue": "..."} or {"text": "..."}
            txt = d.get("dialogue") or d.get("text") or d.get("content") or ""
            if txt:
                out.append(txt)
            if len(out) >= max_n:
                break
    return out


def build_role_prompts(role: str, questions: list[str], tokenizer) -> list[str]:
    """Build chat-formatted prompts for a single role across all questions.
    System prompt = "You are a {role}. Respond in character."
    User = question.
    The prompt INCLUDES the generation marker so the activation we hook is
    over the assistant's response tokens after a real generation step.
    """
    sys_prompt = f"You are a {role}. Respond in character."
    out = []
    for q in questions:
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q},
        ]
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        out.append(prompt_text)
    return out


def generate_responses(
    model, tokenizer, prompts: list[str],
    max_new_tokens: int, temperature: float, batch_size: int = 8,
) -> list[str]:
    """Generate one response per prompt. Returns list of FULL strings
    (prompt + response) — we slice off the prompt downstream by re-encoding."""
    results: list[str] = []
    device = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        results.extend(decoded)
        if (i // batch_size + 1) % 10 == 0:
            print(f"    gen {i + len(batch)}/{len(prompts)}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Override layer list (defaults to config.sweep_layers or [config.chosen_layer])")
    ap.add_argument("--model-key", default=None,
                    help="Short name used in output filenames; auto-derived from config stem if omitted")
    ap.add_argument("--roles", default=str(THIS_DIR / "data" / "roles_v1.json"))
    ap.add_argument("--questions", default=str(THIS_DIR / "data" / "questions_v1.json"))
    ap.add_argument("--neutral", default=str(STEERING_DIR / "data" / "neutral_dialogues.jsonl"))
    ap.add_argument("--vectors-dir", default=None,
                    help="Defaults to steering/vectors/{model_key}/")
    ap.add_argument("--output-dir", default=str(THIS_DIR.parent.parent / "results" / "axis"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    hf_id: str = cfg["hf_id"]
    model_key = args.model_key or Path(args.config).stem  # e.g. "gemma3_27b"
    if args.layers is not None:
        sweep_layers = args.layers
    elif cfg.get("sweep_layers"):
        sweep_layers = cfg["sweep_layers"]
    elif cfg.get("chosen_layer") is not None:
        sweep_layers = [cfg["chosen_layer"]]
    else:
        sys.exit("config must define sweep_layers or chosen_layer, or pass --layers")
    vectors_dir_default = STEERING_DIR / "vectors" / model_key
    vectors_dir_arg = args.vectors_dir or str(vectors_dir_default)
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    max_new_tokens = int(cfg.get("max_new_tokens", 80))
    temperature = float(cfg.get("generation_temperature", 0.8))

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir = Path(vectors_dir_arg); vectors_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {hf_id} dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for generation; for collect_all_layer_activations we re-tokenize
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map="auto")
    model.eval()
    n_layers = len(_find_layers(model))
    print(f"[load] num_hidden_layers = {n_layers}")

    roles = load_roles(Path(args.roles))
    questions = load_questions(Path(args.questions))
    neutral = load_neutral_dialogues(Path(args.neutral), max_n=80)
    print(f"[data] {len(roles['assistant_anchor'])} anchor + {len(roles['other'])} other + "
          f"{len(roles['sanity_positive'])} pos-held + {len(roles['sanity_negative'])} neg-held roles | "
          f"{len(questions)} questions | {len(neutral)} neutral dialogues")

    # --- Step 1: generate responses for every (role, question) and the held-out roles ---
    all_role_groups: dict[str, list[str]] = {}  # role -> list of full prompt+response strings
    every_role = (
        roles["assistant_anchor"] + roles["other"]
        + roles["sanity_positive"] + roles["sanity_negative"]
    )
    for r_i, role in enumerate(every_role):
        print(f"[gen] role {r_i+1}/{len(every_role)}: {role}")
        prompts = build_role_prompts(role, questions, tokenizer)
        t0 = time.time()
        responses = generate_responses(
            model, tokenizer, prompts,
            max_new_tokens=max_new_tokens, temperature=temperature,
            batch_size=args.batch_size,
        )
        all_role_groups[role] = responses
        print(f"  {role}: {len(responses)} rollouts ({time.time()-t0:.1f}s)")

    # --- Step 2: capture all-layer activations for every rollout (one fwd per batch) ---
    # We flatten into a single text list and remember role boundaries.
    flat_texts: list[str] = []
    role_for_idx: list[str] = []
    for role in every_role:
        for txt in all_role_groups[role]:
            flat_texts.append(txt)
            role_for_idx.append(role)
    print(f"[acts] capturing selected-layer activations on {len(flat_texts)} rollouts "
          f"(layers {sweep_layers})...")
    tokenizer.padding_side = "right"
    t0 = time.time()
    all_acts = collect_selected_layer_acts(
        model, tokenizer, flat_texts,
        layer_idxs=sweep_layers, batch_size=args.batch_size, device=args.device,
        position="mean-after-50", last_k=0, desc="role-acts",
    )
    print(f"[acts] role-acts done in {time.time()-t0:.1f}s")

    # all_acts: list[len(flat_texts)] of dict[layer_idx, ndarray[hidden]]
    # group by role -> per-role per-layer mean (only sweep_layers)
    role_per_layer_mean: dict[str, dict[int, np.ndarray]] = {}
    for role in every_role:
        idxs = [i for i, r in enumerate(role_for_idx) if r == role]
        per_layer: dict[int, np.ndarray] = {}
        for L in sweep_layers:
            stacked = np.stack([all_acts[i][L] for i in idxs], axis=0)
            per_layer[L] = stacked.mean(axis=0)
        role_per_layer_mean[role] = per_layer

    # Capture neutral activations at the same selected layers
    print(f"[acts] capturing neutral-dialogue activations...")
    neutral_acts = collect_selected_layer_acts(
        model, tokenizer, neutral,
        layer_idxs=sweep_layers, batch_size=args.batch_size, device=args.device,
        position="mean-after-50", last_k=0, desc="neutral",
    )

    # --- Step 3: per-sweep-layer, compute axis + run gauntlet ---
    per_layer_report: dict[str, dict] = {}
    for L in sweep_layers:
        anchor_mat = np.stack([role_per_layer_mean[r][L] for r in roles["assistant_anchor"]], axis=0)
        other_mat  = np.stack([role_per_layer_mean[r][L] for r in roles["other"]], axis=0)
        pos_mat    = np.stack([role_per_layer_mean[r][L] for r in roles["sanity_positive"]], axis=0)
        neg_mat    = np.stack([role_per_layer_mean[r][L] for r in roles["sanity_negative"]], axis=0)
        neut_L     = np.stack([na[L] for na in neutral_acts], axis=0)

        axis = compute_contrast_axis(anchor_mat, other_mat)
        report = evaluate_layer(
            axis=axis,
            assistant_anchor_acts=anchor_mat,
            other_acts=other_mat,
            sanity_pos_acts=pos_mat,
            sanity_neg_acts=neg_mat,
            neutral_acts=neut_L,
        )
        report["layer"] = L
        per_layer_report[str(L)] = report

        # persist the candidate axis
        torch.save(torch.tensor(axis, dtype=torch.float32),
                   vectors_dir / f"axis_layer{L}.pt")
        # also persist per-role means at this layer (useful for analysis)
        torch.save({r: torch.tensor(role_per_layer_mean[r][L], dtype=torch.float32)
                    for r in every_role},
                   vectors_dir / f"role_means_layer{L}.pt")
        print(f"[L={L}] all_pass={report['all_pass']}  "
              f"g1={report['g1_anchor_above_median']} "
              f"g2={report['g2_negatives_bottom3']} "
              f"g3={report['g3_positives_above_other']} "
              f"g4={report['g4_effect_size_over_1']}  "
              f"effect={report['effect_size']:.2f}")

    # --- Step 4: pick lowest passing layer ---
    passing = sorted([int(L) for L, rep in per_layer_report.items() if rep["all_pass"]])
    chosen = passing[0] if passing else None
    summary = {
        "model": hf_id,
        "num_hidden_layers": n_layers,
        "swept_layers": sweep_layers,
        "passing_layers": passing,
        "chosen_layer": chosen,
        "per_layer": per_layer_report,
    }
    out_path = output_dir / f"sanity_gauntlet_{model_key}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] passing={passing}  chosen={chosen}  → {out_path}")
    if chosen is None:
        print("[STOP] No layer passed the gauntlet. Do not proceed to next phase.")
        sys.exit(2)


if __name__ == "__main__":
    main()
