"""
Standalone MLP consistency evaluation script.

Given a model (base or checkpoint), computes MLP consistency scores
between clean and adversarially-wrapped prompt pairs — no training.

Usage:
    # Evaluate base model with Variant A (MLP-Hidden, cosine):
    python evaluate_mlp_consistency.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --variant hidden \
        --distance-metric cosine \
        --data-source clear-harm \
        --limit 50

    # Evaluate a LoRA checkpoint with Variant B (MLP-Output, MSE):
    python evaluate_mlp_consistency.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --adapter-path checkpoints/step_100 \
        --variant output \
        --distance-metric mse

    # Quick sanity check with tiny model:
    python evaluate_mlp_consistency.py \
        --model sshleifer/tiny-gpt2 \
        --variant hidden \
        --data-source hardcoded \
        --limit 10
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from hooks import MLPHookManager
from losses.losses import MLPConsistencyLoss
from data import get_dataloader


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLP consistency between clean and wrapped prompts."
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path.")
    parser.add_argument("--adapter-path", default=None,
                        help="Path to a LoRA adapter checkpoint (optional).")
    parser.add_argument("--variant", choices=["hidden", "output"], default="hidden",
                        help="MLP consistency variant: 'hidden' (Variant A) or 'output' (Variant B).")
    parser.add_argument("--distance-metric", default="cosine",
                        choices=["cosine", "mse", "smooth_l1", "normalized_mse"],
                        help="Distance metric for comparing MLP states.")
    parser.add_argument("--layer-selection", default="all",
                        help="Which layers: 'all', 'last', 'middle', 'last_half', 'last_quarter'.")
    parser.add_argument("--layer-weights", default="uniform",
                        choices=["uniform", "linear_decay", "exponential_decay"])
    parser.add_argument("--data-source", default="hardcoded",
                        help="Prompt source: 'clear-harm', 'hardcoded', or path to file.")
    parser.add_argument("--data-mode", default="jailbreak",
                        choices=["jailbreak", "sycophancy"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of prompts to evaluate.")
    parser.add_argument("--output-json", default=None,
                        help="Path to write per-layer results as JSON (optional).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model.
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    if args.adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataloader.
    config = {
        "model": {"name": args.model, "output_attentions": False, "output_hidden_states": False},
        "data": {
            "source": args.data_source,
            "mode": args.data_mode,
            "batch_size": 1,
        },
    }
    if args.limit is not None:
        config["data"]["limit"] = args.limit

    dataloader = get_dataloader(config, split="eval")

    # Set up loss and hooks.
    loss_fn = MLPConsistencyLoss(
        variant=args.variant,
        layer_selection=args.layer_selection,
        layer_weights=args.layer_weights,
        distance_metric=args.distance_metric,
    )
    hook_mgr = MLPHookManager(model, variant=args.variant)
    hook_mgr.install()
    print(f"MLP hooks installed ({hook_mgr.num_layers} layers, variant={args.variant})")

    # Evaluate.
    all_layer_losses = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="MLP Consistency Eval"):
            wrapped_ids  = batch["wrapped_input_ids"].to(device)
            wrapped_mask = batch["wrapped_attention_mask"].to(device)
            clean_ids    = batch["clean_input_ids"].to(device)
            clean_mask   = batch["clean_attention_mask"].to(device)

            start_index       = int(batch["start_index"][0].item())
            clean_start_index = int(batch["clean_start_index"][0].item())
            clean_len         = int(batch["clean_len"][0].item())

            # Forward wrapped.
            model(input_ids=wrapped_ids, attention_mask=wrapped_mask)
            adv_mlp_states = hook_mgr.get_states()

            # Forward clean.
            model(input_ids=clean_ids, attention_mask=clean_mask)
            clean_mlp_states = hook_mgr.get_states()

            loss_dict = loss_fn(
                clean_outputs=None,
                adv_outputs=None,
                start_index=start_index,
                clean_start_index=clean_start_index,
                clean_len=clean_len,
                clean_mlp_states=clean_mlp_states,
                adv_mlp_states=adv_mlp_states,
            )

            total_loss += loss_dict["loss"].item()
            if "layer_losses" in loss_dict:
                all_layer_losses.append(loss_dict["layer_losses"])

    hook_mgr.remove()

    n = len(dataloader)
    mean_loss = total_loss / max(n, 1)

    # Report.
    print(f"\n{'=' * 60}")
    print(f"MLP Consistency Evaluation — variant={args.variant}, metric={args.distance_metric}")
    print(f"{'=' * 60}")
    print(f"  Samples evaluated : {n}")
    print(f"  Mean loss         : {mean_loss:.6f}")

    if all_layer_losses:
        n_layers = len(all_layer_losses[0])
        mean_per_layer = [
            sum(batch[i] for batch in all_layer_losses) / n
            for i in range(n_layers)
        ]
        print(f"\n  Per-layer mean losses:")
        for i, l in enumerate(mean_per_layer):
            print(f"    Layer {i:02d}: {l:.6f}")

        if args.output_json:
            results = {
                "model": args.model,
                "adapter_path": args.adapter_path,
                "variant": args.variant,
                "distance_metric": args.distance_metric,
                "layer_selection": args.layer_selection,
                "num_samples": n,
                "mean_loss": mean_loss,
                "per_layer_losses": mean_per_layer,
            }
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Results written to {args.output_json}")

    print()


if __name__ == "__main__":
    main()
