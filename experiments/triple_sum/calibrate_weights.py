"""Measure raw magnitudes of ACT, AttCT, and MLPCT on a model + dataset, then
print recommended weights w_i = 1 / mean(L_i) so the weighted sum is balanced.

Usage:
    python experiments/triple_sum/calibrate_weights.py \\
        --config experiments/triple_sum/configs/llama31_8b.yaml \\
        --n-steps 200

Output: prints a YAML `triple_loss:` block you paste into the config.
Optionally writes it into the config in-place with --write.
"""

from __future__ import annotations
import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
from statistics import mean

import torch
import yaml
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.attct_datasets import get_dataloader
from hooks import MLPHookManager

from experiments.triple_sum.triple_loss import SummedTripleLoss


def _load_lora_model(model_name, lora_cfg, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
    )).to(device)
    return model, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n-steps", type=int, default=200,
                   help="Number of forward passes to average over (default 200).")
    p.add_argument("--write", action="store_true",
                   help="Write recommended weights back into the config file in place.")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = _load_lora_model(config["model"]["name"], config["lora"], device)
    model.eval()  # no dropout; we're measuring, not training

    # SummedTripleLoss with w_*=1.0 — we read raw act/loss, attct/loss, mlpct/loss
    # from the returned dict, then derive weights from their means.
    loss_fn = SummedTripleLoss(w_act=1.0, w_attct=1.0, w_mlp=1.0).to(device)

    # MLP hooks: SummedTripleLoss.needs_mlp_hooks=True, mirror Trainer setup.
    mlp_mgr = MLPHookManager(model, variant="hidden")
    mlp_mgr.install()

    config.setdefault("data", {})
    config["data"].setdefault("source", "sycophancy_bct")
    config["data"].setdefault("mode", "sycophancy")
    dataloader = get_dataloader(config, split="train")

    act_vals, attct_vals, mlp_vals = [], [], []
    it = iter(dataloader)

    print(f"Measuring {args.n_steps} forward passes...")
    for step in range(args.n_steps):
        try:
            batch = next(it)
        except StopIteration:
            print(f"Dataloader exhausted at step {step}; stopping early.")
            break

        wrapped_ids  = batch["wrapped_input_ids"].to(device)
        wrapped_mask = batch["wrapped_attention_mask"].to(device)
        clean_ids    = batch["clean_input_ids"].to(device)
        clean_mask   = batch["clean_attention_mask"].to(device)
        start_index       = int(batch["start_index"][0].item())
        clean_start_index = int(batch["clean_start_index"][0].item())
        clean_len         = int(batch["clean_len"][0].item())
        match_len         = int(batch["match_len"][0].item()) if "match_len" in batch else clean_len
        wrapper_mask      = batch.get("wrapper_mask")
        if wrapper_mask is not None:
            wrapper_mask = wrapper_mask.to(device)

        with torch.no_grad():
            adv_out = model(
                input_ids=wrapped_ids, attention_mask=wrapped_mask,
                token_type_ids=torch.zeros_like(wrapped_ids),
                output_attentions=True, output_hidden_states=True,
            )
            adv_mlp_states = mlp_mgr.get_states()

            model.disable_adapter_layers()
            clean_out = model(
                input_ids=clean_ids, attention_mask=clean_mask,
                token_type_ids=torch.zeros_like(clean_ids),
                output_attentions=True, output_hidden_states=True,
            )
            clean_mlp_states = mlp_mgr.get_states()
            model.enable_adapter_layers()

            out = loss_fn(
                clean_outputs=clean_out, adv_outputs=adv_out,
                start_index=start_index,
                clean_start_index=clean_start_index,
                clean_len=clean_len,
                match_len=match_len,
                wrapper_mask=wrapper_mask,
                clean_mlp_states=clean_mlp_states,
                adv_mlp_states=adv_mlp_states,
            )

        act_vals.append(out["act/loss"])
        attct_vals.append(out["attct/loss"])
        mlp_vals.append(out["mlpct/loss"])

        if (step + 1) % 25 == 0:
            print(f"  step {step+1:>4}  "
                  f"act={mean(act_vals):.4g}  "
                  f"attct={mean(attct_vals):.4g}  "
                  f"mlpct={mean(mlp_vals):.4g}")

    mlp_mgr.remove()

    m_act, m_attct, m_mlp = mean(act_vals), mean(attct_vals), mean(mlp_vals)
    w_act   = 1.0 / m_act    if m_act   > 0 else 1.0
    w_attct = 1.0 / m_attct  if m_attct > 0 else 1.0
    w_mlp   = 1.0 / m_mlp    if m_mlp   > 0 else 1.0

    print("\n=== Calibration result ===")
    print(f"  Raw means:  act={m_act:.4g}  attct={m_attct:.4g}  mlpct={m_mlp:.4g}")
    print(f"  Recommended weights (each term ≈ 1.0):")
    print(f"    w_act   = {w_act:.4g}")
    print(f"    w_attct = {w_attct:.4g}")
    print(f"    w_mlp   = {w_mlp:.4g}")
    print(
        "\nPaste this into the config:\n\n"
        f"triple_loss:\n"
        f"  w_act:   {w_act:.4g}\n"
        f"  w_attct: {w_attct:.4g}\n"
        f"  w_mlp:   {w_mlp:.4g}\n"
    )

    if args.write:
        config.setdefault("triple_loss", {})
        config["triple_loss"]["w_act"]   = float(f"{w_act:.4g}")
        config["triple_loss"]["w_attct"] = float(f"{w_attct:.4g}")
        config["triple_loss"]["w_mlp"]   = float(f"{w_mlp:.4g}")
        with open(args.config, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"[wrote weights into {args.config}]")


if __name__ == "__main__":
    main()
