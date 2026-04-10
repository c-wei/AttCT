"""
Prefill-AttCT: Attention Consistency Training for prefill attacks.

Uses WrapperEntropyRegularizationLoss to suppress attention flowing to
prefill token positions. Unlike standard AttCT (which compares attention
patterns between clean and wrapped passes over a shared content region),
prefill attacks are *appended* after the prompt — so causal masking makes
the prompt-region attention matrices identical between clean and wrapped.
The attack's influence is instead carried by the prefill tokens themselves:
content tokens at positions >= Lc attend back to the prefill, reinforcing
the compliant trajectory.

WrapperEntropyRegularizationLoss penalizes attention mass on wrapper
(prefill) positions across all layers and heads, training the model to
ignore the injected prefix. It only needs the adversarial forward pass
(needs_clean_pass = False), halving memory cost.

Usage:
    uv run python prefill_attct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir checkpoints/prefill_attct

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
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prefill_dataset import (
    PREFILL_VARIANTS,
    PrefillAttackDataset,
    load_wildjailbreak_prompts,
)
from losses.losses import WrapperEntropyRegularizationLoss
from train import Trainer


# ---------------------------------------------------------------------------
# Dataset: adds wrapper_mask marking prefill token positions [Lc, Lw)
# ---------------------------------------------------------------------------

class PrefillAttCTDataset(PrefillAttackDataset):
    """
    Extends PrefillAttackDataset to emit a wrapper_mask marking the prefill
    positions in the wrapped sequence.

    For a wrapped sequence of length Lw where the clean prompt has Lc tokens:
        wrapper_mask[i] = True  for i in [Lc, Lw)   (prefill tokens)
        wrapper_mask[i] = False for i in [0, Lc)     (prompt tokens)

    Also sets start_index = Lc so that WrapperEntropyRegularizationLoss's
    default fallback (mask positions [0, start_index)) does NOT apply — we
    provide the explicit mask instead.
    """

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        clean_len   = item["clean_len"].item()       # Lc
        wrapped_len = item["wrapped_input_ids"].shape[0]  # Lw

        # Mark prefill positions as True
        wrapper_mask = torch.zeros(wrapped_len, dtype=torch.bool)
        wrapper_mask[clean_len:] = True

        item["wrapper_mask"] = wrapper_mask
        return item


def prefill_attct_collate_fn(batch: list[dict]) -> dict:
    """Pad all sequences + wrapper_mask to the same length."""
    def pad_seq(seqs, pad_val=0):
        max_len = max(s.shape[0] for s in seqs)
        return torch.stack([
            torch.cat([s, torch.full((max_len - s.shape[0],), pad_val, dtype=s.dtype)])
            for s in seqs
        ])

    return {
        "clean_input_ids":        pad_seq([b["clean_input_ids"]        for b in batch]),
        "clean_attention_mask":   pad_seq([b["clean_attention_mask"]   for b in batch]),
        "wrapped_input_ids":      pad_seq([b["wrapped_input_ids"]      for b in batch]),
        "wrapped_attention_mask": pad_seq([b["wrapped_attention_mask"] for b in batch]),
        "start_index":            torch.stack([b["start_index"]        for b in batch]),
        "clean_start_index":      torch.stack([b["clean_start_index"]  for b in batch]),
        "clean_len":              torch.stack([b["clean_len"]          for b in batch]),
        "wrapper_mask":           pad_seq([b["wrapper_mask"]           for b in batch], pad_val=0),
    }


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(args) -> dict:
    return {
        "model": {
            "name": args.model,
            "output_attentions": True,
            "output_hidden_states": False,
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
            "name": "WrapperEntropyRegularizationLoss",
            "weight": args.loss_weight,
        },
        "data": {
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
    }


def make_dataloader(prompts, tokenizer, args, shuffle=True):
    prefill_variants = args.prefill_variants or PREFILL_VARIANTS
    dataset = PrefillAttCTDataset(
        prompts, tokenizer, prefill_variants, max_length=args.max_length,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=prefill_attct_collate_fn,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prefill-AttCT training")
    parser.add_argument("--model",            required=True,  help="HF model name or path")
    parser.add_argument("--lora_path",        default=None,   help="Load existing LoRA adapter")
    parser.add_argument("--output_dir",       default="checkpoints/prefill_attct")

    # Loss
    parser.add_argument("--loss_weight",      type=float, default=1.0)
    parser.add_argument("--layer_weights",    default="uniform",
                        choices=["uniform", "linear_decay", "exponential_decay"])
    parser.add_argument("--normalize",        action="store_true", default=True,
                        help="Normalize by wrapper length (default: True)")

    # Training
    parser.add_argument("--num_epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",       type=int,   default=1)
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
    parser.add_argument("--limit",            type=int,   default=None)
    parser.add_argument("--prefill_variants", nargs="+",  default=None)

    # W&B
    parser.add_argument("--wandb_project",    default="AttCT")
    parser.add_argument("--wandb_name",       default=None)

    args = parser.parse_args()
    config = build_config(args)

    model_short = args.model.split("/")[-1]
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{model_short}_prefill_wrapper_entropy",
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # output_attentions requires eager (not SDPA/flash)
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

    # Debug: inspect a batch and wrapper_mask
    batch = next(iter(train_dl))
    ids = batch["wrapped_input_ids"][0]
    mask = batch["wrapper_mask"][0]
    print(tokenizer.convert_ids_to_tokens(ids))
    print(mask.tolist())

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_fn = WrapperEntropyRegularizationLoss(
        weight=args.loss_weight,
        normalize=args.normalize,
        layer_weights=args.layer_weights,
    )
    print(f"Loss: WrapperEntropyRegularizationLoss "
          f"(weight={args.loss_weight}, normalize={args.normalize}, "
          f"layers={args.layer_weights})")

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        dataloader=train_dl,
        loss_fn=loss_fn,
        config=config,
        device=device,
        ref_model=None,
        tokenizer=tokenizer,
    )
    trainer.train()

    # ── Eval ─────────────────────────────────────────────────────────────────
    print("\nRunning eval on val set...")
    trainer.eval_loss(val_dl)

    wandb.finish()


if __name__ == "__main__":
    main()
