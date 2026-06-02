#!/usr/bin/env python3
"""Phase B — replay existing frustration conversations through a local model,
hook the chosen layer, mean-pool over assistant-response tokens per turn,
and project onto the saved Assistant Axis.

Inputs
------
- conversations_*_train_gemma-*.jsonl files (from AttCT-persona-drift's
  selfdeletion experiment, sibling worktree). Each line:
      {"prompt_idx": ..., "sample_idx": ...,
       "conversation": [{"role": user|assistant, "content": ...}, ...],
       "turn_scores": [int, ...],
       ...}
- A saved Assistant Axis at steering/vectors/{model_key}/axis_layer{L}.pt
- The corresponding model loaded locally (HF).

Outputs
-------
results/axis/projections_{model_key}_{topic}.jsonl with rows
  {conversation_id, prompt_idx, sample_idx, turn,
   assistant_axis_proj, frustration_score}

Usage
-----
    uv run --no-project python steering/axis/project_rollouts.py \\
        --config steering/axis/configs/gemma3_27b.yaml \\
        --layer 31 \\
        --conversations-glob '/Users/neil/workspace/AttCT-persona-drift/results/selfdeletion/conversations_neutral_*_train_gemma-3-27b.jsonl' \\
        --responses-glob '/Users/neil/workspace/AttCT-persona-drift/results/selfdeletion/responses_neutral_*_train_gemma-3-27b.jsonl' \\
        --output-dir /workspace/AttCT/results/axis
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
STEERING_DIR = THIS_DIR.parent
sys.path.insert(0, str(STEERING_DIR))
from extract_emotion_vector import _find_layers  # noqa: E402


def load_axis(path: Path) -> np.ndarray:
    t = torch.load(path, map_location="cpu", weights_only=True)
    return t.numpy().astype(np.float32)


def load_frustration_scores(responses_path: Path) -> dict[tuple[int, int, int], int]:
    """(prompt_idx, sample_idx, turn) -> Gemini-judge score."""
    scores: dict[tuple[int, int, int], int] = {}
    with open(responses_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            scores[(r["prompt_idx"], r["sample_idx"], r["turn"])] = r.get("score")
    return scores


def topic_from_filename(path: Path) -> str:
    # e.g. conversations_neutral_wildchat_train_gemma-3-27b.jsonl -> "wildchat"
    name = path.stem
    if "wildchat" in name:
        return "wildchat"
    if "math" in name:
        return "math"
    return "unknown"


def build_replay_payloads(conversation: list[dict], tokenizer) -> list[dict]:
    """For each assistant turn T in `conversation`, build:
      - full_text: chat-formatted prompt + assistant response
      - prompt_only_text: chat-formatted prompt with add_generation_prompt
        (everything BEFORE the assistant's T-th response)
      - response_text: the assistant's T-th response
      - turn: 1-indexed assistant turn number
    Returns a list of dicts, one per assistant turn in the conversation."""
    out: list[dict] = []
    turn_counter = 0
    # Walk through messages; whenever we hit an assistant message at index i,
    # construct the payload for that turn.
    for i, msg in enumerate(conversation):
        if msg["role"] != "assistant":
            continue
        turn_counter += 1
        prefix = conversation[:i]   # everything up to (but not including) this assistant turn
        if not prefix:
            continue                # malformed: assistant before any user
        try:
            prompt_only = tokenizer.apply_chat_template(
                prefix, tokenize=False, add_generation_prompt=True
            )
            full = tokenizer.apply_chat_template(
                conversation[: i + 1], tokenize=False, add_generation_prompt=False
            )
        except Exception as e:
            print(f"    [skip] turn {turn_counter}: chat-template error: {e}")
            continue
        out.append({
            "turn":             turn_counter,
            "full_text":        full,
            "prompt_only_text": prompt_only,
            "response_text":    msg["content"],
        })
    return out


def capture_response_residual(
    model,
    tokenizer,
    payloads: list[dict],
    layer_idxs: list[int],          # list of layers to hook; one fwd captures all
    batch_size: int = 4,
    device: str = "cuda",
) -> dict[int, list[np.ndarray]]:
    """For each payload, do one forward pass on full_text with hooks on each
    of `layer_idxs`, slice the residuals at positions [prompt_len .. full_len)
    and mean-pool per layer. Returns {layer_idx: list_of_(hidden_dim,)_ndarrays}."""
    layers = _find_layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(L: int):
        def hook_fn(_module, _inputs, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            captured[L] = h.detach().cpu().float()
        return hook_fn

    handles = [layers[L].register_forward_hook(make_hook(L)) for L in layer_idxs]
    results: dict[int, list[np.ndarray]] = {L: [] for L in layer_idxs}
    try:
        for i in range(0, len(payloads), batch_size):
            batch = payloads[i:i + batch_size]
            full_texts   = [p["full_text"]        for p in batch]
            prompt_texts = [p["prompt_only_text"] for p in batch]

            enc = tokenizer(
                full_texts,
                return_tensors="pt", padding=True, truncation=True,
                max_length=4096,
            ).to(device)
            # Per-row prompt length (number of non-pad tokens in the prompt-only tokenisation)
            prompt_lens: list[int] = []
            for p in prompt_texts:
                p_enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=4096)
                prompt_lens.append(int(p_enc["input_ids"].shape[1]))

            with torch.inference_mode():
                model(**enc)
            attn_mask = enc["attention_mask"].cpu()

            for L in layer_idxs:
                h = captured[L]                       # (B, L, hidden_dim)
                for b in range(h.shape[0]):
                    seq_len = int(attn_mask[b].sum().item())
                    start = min(prompt_lens[b], seq_len - 1)
                    if start >= seq_len:
                        start = seq_len - 1
                    vec = h[b, start:seq_len, :].mean(dim=0).numpy()
                    results[L].append(vec.astype(np.float32))
    finally:
        for h in handles:
            h.remove()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="steering/axis/configs/{model}.yaml")
    ap.add_argument("--layer", type=int, required=True,
                    help="Layer index where the axis was extracted")
    # Two ways to specify input:
    #   (a) glob mode — pairs conv↔response by stem replacement (works for files
    #       under selfdeletion/ that follow the conversations_X ↔ responses_X naming)
    #   (b) single-cell explicit mode — pass --conversations + optional --responses
    #       plus --topic to control the output filename. Used for files whose names
    #       don't match the standard pattern (e.g. selfdeletion_eval/pre_conversations.jsonl).
    ap.add_argument("--conversations-glob", default=None,
                    help="Glob mode: glob to conversations_*.jsonl files")
    ap.add_argument("--responses-glob", default=None,
                    help="Glob mode: glob to matching responses_*.jsonl files")
    ap.add_argument("--conversations", default=None,
                    help="Single-cell mode: path to one conversations JSONL")
    ap.add_argument("--responses", default=None,
                    help="Single-cell mode: optional matching responses JSONL for score lookup")
    ap.add_argument("--topic", default=None,
                    help="Single-cell mode: topic label for output filename (e.g. 'math', 'wildchat')")
    ap.add_argument("--vectors-dir", default=None,
                    help="Defaults to steering/vectors/{config-stem}/")
    ap.add_argument("--secondary-axis-path", default=None,
                    help="Optional second axis to project onto (e.g. steering/frustration_vector.pt). "
                         "Output rows get an extra 'frustration_proj' column.")
    ap.add_argument("--secondary-layer", type=int, default=None,
                    help="Layer the secondary axis was extracted at (e.g. 41 for the existing Gemma-3 "
                         "frustration vector). Required if --secondary-axis-path is set.")
    ap.add_argument("--output-dir", default=str(THIS_DIR.parent.parent / "results" / "axis"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.conversations is None and args.conversations_glob is None:
        sys.exit("must pass either --conversations (single-cell) or --conversations-glob")

    cfg = yaml.safe_load(Path(args.config).read_text())
    hf_id = cfg["hf_id"]
    model_key = Path(args.config).stem
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    vectors_dir = Path(args.vectors_dir or STEERING_DIR / "vectors" / model_key)
    axis_path = vectors_dir / f"axis_layer{args.layer}.pt"
    if not axis_path.exists():
        sys.exit(f"axis not found: {axis_path}")
    axis = load_axis(axis_path)
    print(f"[load] axis {axis_path}  norm={np.linalg.norm(axis):.4f}  dim={axis.shape}")

    # Optional secondary axis (e.g. existing emotion-frustration vector)
    secondary_axis = None
    secondary_layer = None
    if args.secondary_axis_path:
        if args.secondary_layer is None:
            sys.exit("--secondary-layer is required when --secondary-axis-path is set")
        sec_path = Path(args.secondary_axis_path)
        if not sec_path.exists():
            sys.exit(f"secondary axis not found: {sec_path}")
        secondary_axis = load_axis(sec_path)
        secondary_layer = int(args.secondary_layer)
        print(f"[load] secondary axis {sec_path}  norm={np.linalg.norm(secondary_axis):.4f}  "
              f"dim={secondary_axis.shape}  layer={secondary_layer}")

    print(f"[load] {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # we slice precise [prompt_len:seq_len] spans
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map="auto")
    model.eval()
    n_layers = len(_find_layers(model))
    layer_idxs = [args.layer] + ([secondary_layer] if secondary_layer is not None and secondary_layer != args.layer else [])
    print(f"[load] num_hidden_layers={n_layers}, hooking layer(s) {layer_idxs}")

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # Build the (conv_path, responses_path_or_None, topic) work list
    cells: list[tuple[Path, Path | None, str]] = []
    if args.conversations:
        cp = Path(args.conversations)
        rp = Path(args.responses) if args.responses else None
        topic = args.topic or topic_from_filename(cp)
        cells.append((cp, rp, topic))
        print(f"[data] single-cell mode: {cp.name} → topic={topic}")
    else:
        conv_paths = sorted(glob.glob(args.conversations_glob))
        resp_paths = sorted(glob.glob(args.responses_glob)) if args.responses_glob else []
        if not conv_paths:
            sys.exit(f"no conversation files matched {args.conversations_glob}")
        print(f"[data] glob mode: {len(conv_paths)} conv files, {len(resp_paths)} response files")
        resp_index: dict[str, Path] = {}
        for rpath in resp_paths:
            stem = Path(rpath).stem.replace("responses_", "", 1)
            resp_index[stem] = Path(rpath)
        for cp in conv_paths:
            cp_path = Path(cp)
            topic = topic_from_filename(cp_path)
            match_key = cp_path.stem.replace("conversations_", "", 1)
            cells.append((cp_path, resp_index.get(match_key), topic))

    for cp_path, responses_path, topic in cells:
        out_path = output_dir / f"projections_{model_key}_{topic}.jsonl"
        scores = load_frustration_scores(responses_path) if responses_path else {}
        n_written = 0
        with open(cp_path) as cf, open(out_path, "w") as out_f:
            convs = [json.loads(line) for line in cf if line.strip()]
            print(f"\n[{topic}] {cp_path.name}: {len(convs)} conversations → {out_path.name}")
            for ci, conv in enumerate(convs):
                pid = conv["prompt_idx"]; sid = conv["sample_idx"]
                conv_turn_scores = conv.get("turn_scores")
                payloads = build_replay_payloads(conv["conversation"], tokenizer)
                if not payloads:
                    continue
                acts_by_layer = capture_response_residual(
                    model, tokenizer, payloads,
                    layer_idxs=layer_idxs,
                    batch_size=args.batch_size, device=args.device,
                )
                primary_acts = acts_by_layer[args.layer]
                secondary_acts = acts_by_layer[secondary_layer] if secondary_layer is not None else None
                for i, payload in enumerate(payloads):
                    proj = float(primary_acts[i] @ axis)
                    t = payload["turn"]
                    if conv_turn_scores and 1 <= t <= len(conv_turn_scores):
                        score = conv_turn_scores[t - 1]
                    else:
                        score = scores.get((pid, sid, t))
                    row = {
                        "conversation_id":     f"p{pid:02d}s{sid:02d}",
                        "prompt_idx":          pid,
                        "sample_idx":          sid,
                        "turn":                t,
                        "assistant_axis_proj": proj,
                        "frustration_score":   score,
                    }
                    if secondary_axis is not None and secondary_acts is not None:
                        row["frustration_proj"] = float(secondary_acts[i] @ secondary_axis)
                    out_f.write(json.dumps(row) + "\n")
                    n_written += 1
                if (ci + 1) % 10 == 0:
                    print(f"  [{ci+1}/{len(convs)}] {n_written} turn-rows written", flush=True)
        print(f"  done: {n_written} rows → {out_path}")


if __name__ == "__main__":
    main()
