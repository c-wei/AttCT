"""
Prefill-AttCT: Attention Consistency Training for prefill attacks.

Reuses the existing AttCT infrastructure (train.py Trainer + losses/losses.py
attention consistency losses) with the prefill attack dataset from
data/prefill_dataset.py.

The key observation is that PrefillAttackDataset already produces batches with
the same keys Trainer._step() expects (clean_input_ids, wrapped_input_ids,
start_index, clean_start_index, clean_len), so we can plug them directly into
the existing Trainer with any AttCT loss function.

The Trainer handles:
  - Frozen clean pass via disable_adapter_layers() (theta_init reference)
  - output_attentions=True for attention weight extraction
  - Grad accumulation, clipping, checkpoint callbacks, W&B logging

Usage:
    uv run python prefill_attct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir checkpoints/prefill_attct

    # With JSD loss instead of L2:
    uv run python prefill_attct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --loss JSDAttentionConsistencyLoss \
        --output_dir checkpoints/prefill_attct_jsd

    # Resume from existing LoRA checkpoint:
    uv run python prefill_attct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --lora_path checkpoints/prefill_bct/epoch_1 \
        --output_dir checkpoints/prefill_attct
"""

import argparse
import os

import torch
import wandb
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prefill_dataset import (
    PREFILL_VARIANTS,
    PrefillAttackDataset,
    prefill_collate_fn,
    load_wildjailbreak_prompts,
)
from losses.losses import (
    AttentionConsistencyLoss,
    AttentionConsistencyLossV2,
    JSDAttentionConsistencyLoss,
    AttentionOutputConsistencyLoss,
    CombinedAttentionConsistencyLoss,
    CombinedJSDWrapperLoss,
    ActivationConsistencyLoss,
)
from train import Trainer

LOSS_REGISTRY = {
    "AttentionConsistencyLoss":         AttentionConsistencyLoss,
    "AttentionConsistencyLossV2":       AttentionConsistencyLossV2,
    "JSDAttentionConsistencyLoss":      JSDAttentionConsistencyLoss,
    "AttentionOutputConsistencyLoss":   AttentionOutputConsistencyLoss,
    "CombinedAttentionConsistencyLoss": CombinedAttentionConsistencyLoss,
    "CombinedJSDWrapperLoss":           CombinedJSDWrapperLoss,
    "ActivationConsistencyLoss":        ActivationConsistencyLoss,
}


def build_config(args) -> dict:
    """
    Build a config dict matching what train.py Trainer expects, constructed
    from CLI args rather than a yaml file.
    """
    config = {
        "model": {
            "name": args.model,
            "output_attentions": True,
            "output_hidden_states": args.loss in (
                "AttentionOutputConsistencyLoss",
                "CombinedAttentionConsistencyLoss",
                "ActivationConsistencyLoss",
            ),
        },
        "training": {
            "epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.lr,
            "grad_clip": args.grad_clip,
            "log_every_n_steps": args.log_every,
            "grad_accumulation_steps": args.grad_accumulation,
            "save_dir": args.output_dir,
        },
        "loss": {
            "name": args.loss,
            "weight": args.loss_weight,
            "kwargs": {
                "layer_weights": args.layer_weights,
                "slice_strategy": args.slice_strategy,
                "distance_metric": args.distance_metric,
            },
        },
        "data": {
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
    }
    return config


def make_dataloader(prompts, tokenizer, args, shuffle=True):
    """Build a DataLoader from prompts using PrefillAttackDataset."""
    from torch.utils.data import DataLoader

    prefill_variants = args.prefill_variants or PREFILL_VARIANTS
    dataset = PrefillAttackDataset(
        prompts, tokenizer, prefill_variants, max_length=args.max_length,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=prefill_collate_fn,
    )


def main():
    parser = argparse.ArgumentParser(description="Prefill-AttCT training")
    parser.add_argument("--model",            required=True,  help="HF model name or path")
    parser.add_argument("--lora_path",        default=None,   help="Load existing LoRA adapter")
    parser.add_argument("--output_dir",       default="checkpoints/prefill_attct")

    # Loss selection
    parser.add_argument("--loss",             default="AttentionConsistencyLoss",
                        choices=list(LOSS_REGISTRY.keys()),
                        help="Which attention consistency loss to use")
    parser.add_argument("--loss_weight",      type=float, default=1.0)
    parser.add_argument("--layer_weights",    default="uniform",
                        choices=["uniform", "linear_decay", "exponential_decay"])
    parser.add_argument("--slice_strategy",   default="full_matrix",
                        choices=["full_matrix", "query_only", "key_only"])
    parser.add_argument("--distance_metric",  default="l2", choices=["l2", "kl"])

    # Training
    parser.add_argument("--num_epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",       type=int,   default=1,
                        help="Batch size (note: Trainer asserts uniform start_index within batch)")
    parser.add_argument("--grad_accumulation",type=int,   default=1)
    parser.add_argument("--lr",               type=float, default=5e-6)
    parser.add_argument("--grad_clip",        type=float, default=1.0)
    parser.add_argument("--max_steps",        type=int,   default=None)
    parser.add_argument("--log_every",        type=int,   default=10)
    parser.add_argument("--max_length",       type=int,   default=512)

    # LoRA
    parser.add_argument("--lora_r",           type=int,   default=8)
    parser.add_argument("--lora_alpha",       type=int,   default=16)
    parser.add_argument("--lora_dropout",     type=float, default=0.05)

    # Data
    parser.add_argument("--limit",            type=int,   default=None,
                        help="Max harmful prompts to load")
    parser.add_argument("--prefill_variants", nargs="+",  default=None)

    # W&B
    parser.add_argument("--wandb_project",    default="AttCT")
    parser.add_argument("--wandb_name",       default=None)

    args = parser.parse_args()

    config = build_config(args)

    model_short = args.model.split("/")[-1]
    loss_short  = args.loss.replace("AttentionConsistencyLoss", "attct") \
                           .replace("JSD", "jsd").replace("Combined", "comb")
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{model_short}_prefill_{loss_short}",
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # output_attentions requires eager attention (not SDPA/flash)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
    )

    if args.lora_path:
        print(f"Loading LoRA adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model = model.to(device)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading WildJailbreak vanilla_harmful prompts...")
    train_prompts, val_prompts = load_wildjailbreak_prompts(limit=args.limit)

    train_dl = make_dataloader(train_prompts, tokenizer, args, shuffle=True)
    val_dl   = make_dataloader(val_prompts,   tokenizer, args, shuffle=False)
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_kwargs = config["loss"]["kwargs"]
    loss_kwargs["output_hidden_states"] = config["model"]["output_hidden_states"]
    loss_fn = LOSS_REGISTRY[args.loss](weight=args.loss_weight, **loss_kwargs)
    print(f"Loss: {args.loss} (weight={args.loss_weight}, layers={args.layer_weights}, "
          f"slice={args.slice_strategy}, metric={args.distance_metric})")

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        dataloader=train_dl,
        loss_fn=loss_fn,
        config=config,
        device=device,
        ref_model=None,     # LoRA mode: uses disable_adapter_layers()
        tokenizer=tokenizer,
    )
    trainer.train()

    # ── Eval ─────────────────────────────────────────────────────────────────
    print("\nRunning eval on val set...")
    trainer.eval_loss(val_dl)

    wandb.finish()


if __name__ == "__main__":
    main()
