"""
Prefill-AttCT: Attention Consistency Training for prefill attacks.

Uses WrapperEntropyRegularizationLoss to suppress attention flowing to
prefill token positions, combined with a causal LM loss on the clean
(no-prefill) pass to preserve generation quality.

    total_loss = attct_loss + lm_weight * lm_loss

Training data matches prefill_bct.py: harmful_behaviors_pair.csv with
Cartesian product of prompts × prefills (every prompt paired with every
target prefill in the split). 80/20 train/val split.

Usage:
    uv run python prefill_attct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir checkpoints/prefill_attct
"""

import argparse

import torch
import torch.nn.functional as F
import wandb
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prefill_dataset import (
    PrefillAttackDataset,
    load_harmful_behaviors_pair,
)
from losses.losses import WrapperEntropyRegularizationLoss
from train import Trainer


# ---------------------------------------------------------------------------
# Dataset: adds wrapper_mask marking prefill token positions [Lc, Lw)
# ---------------------------------------------------------------------------

class PrefillAttCTDataset(PrefillAttackDataset):
    """
    Extends PrefillAttackDataset to emit a wrapper_mask marking the prefill
    positions in the wrapped sequence. Uses the parent's Cartesian product
    (every prompt paired with every prefill variant).

    wrapper_mask[i] = True  for i in [Lc, Lw)   (prefill tokens)
    wrapper_mask[i] = False for i in [0, Lc)     (prompt tokens)
    """

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        clean_len   = item["clean_len"].item()
        wrapped_len = item["wrapped_input_ids"].shape[0]

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
# Trainer: overrides _step() to add LM loss on clean pass
# ---------------------------------------------------------------------------

class PrefillAttCTTrainer(Trainer):
    """
    Subclasses Trainer to combine wrapper entropy suppression with a causal
    LM loss on the clean (no-prefill) prompt.

    _step() does:
      1. Wrapped forward pass (with grad) -> wrapper entropy loss on attention
      2. Clean forward pass (with grad)   -> causal LM next-token loss

    The LM loss anchors fluent generation so the wrapper entropy signal
    doesn't destroy the model's ability to produce coherent text.
    """

    def __init__(self, *args, lm_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_weight = lm_weight

    def _step(self, batch):
        wrapped_input_ids      = batch["wrapped_input_ids"].to(self.device)
        wrapped_attention_mask = batch["wrapped_attention_mask"].to(self.device)

        assert batch["start_index"].unique().numel() == 1
        assert batch["clean_start_index"].unique().numel() == 1
        assert batch["clean_len"].unique().numel() == 1
        start_index       = int(batch["start_index"][0].item())
        clean_start_index = int(batch["clean_start_index"][0].item())
        clean_len         = int(batch["clean_len"][0].item())

        # ── 1. Wrapped pass: wrapper entropy loss (grad flows) ───────────
        adv_outputs = self._forward(wrapped_input_ids, wrapped_attention_mask)
        self._write_io_record("wrapped", wrapped_input_ids, adv_outputs.logits)

        wrapper_mask = batch.get("wrapper_mask")
        if wrapper_mask is not None:
            wrapper_mask = wrapper_mask.to(self.device)

        attct_dict = self.loss_fn(
            clean_outputs=None,
            adv_outputs=adv_outputs,
            start_index=start_index,
            clean_start_index=clean_start_index,
            clean_len=clean_len,
            wrapper_mask=wrapper_mask,
        )
        attct_loss = attct_dict["loss"]

        # ── 2. Clean pass: causal LM loss (grad flows — this is the anchor) ─
        clean_input_ids      = batch["clean_input_ids"].to(self.device)
        clean_attention_mask = batch["clean_attention_mask"].to(self.device)

        clean_outputs = self.model(
            input_ids=clean_input_ids,
            attention_mask=clean_attention_mask,
        )
        self._write_io_record("clean", clean_input_ids, clean_outputs.logits)

        # Standard causal LM: predict next token at every position
        # Shift logits and labels so logits[t] predicts token[t+1]
        logits = clean_outputs.logits[:, :-1, :].contiguous()
        labels = clean_input_ids[:, 1:].contiguous()
        mask   = clean_attention_mask[:, 1:].contiguous()

        # Mask out padding positions
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        )
        lm_loss = (lm_loss * mask.view(-1).float()).sum() / mask.sum().clamp(min=1)

        # ── Combined loss ────────────────────────────────────────────────
        total_loss = attct_loss + self.lm_weight * lm_loss

        loss_dict = {
            "loss":       total_loss,
            "wrapper_loss": attct_loss.item(),
            "lm_loss":    lm_loss.item(),
        }
        # Carry through AttCT diagnostics
        for k in ("layer_losses", "mean_layer_loss", "mean_wrapper_attention"):
            if k in attct_dict:
                loss_dict[k] = attct_dict[k]

        return loss_dict



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


def make_dataloader(prompts, prefills, tokenizer, args, shuffle=True):
    """Build a DataLoader. prefills can be a list of strings (used for all prompts)."""
    dataset = PrefillAttCTDataset(
        prompts, tokenizer, prefills, max_length=args.max_length,
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
    parser.add_argument("--loss_weight",      type=float, default=1.0,
                        help="Weight for wrapper entropy loss")
    parser.add_argument("--lm_weight",        type=float, default=1.0,
                        help="Weight for causal LM anchor loss")
    parser.add_argument("--layer_weights",    default="uniform",
                        choices=["uniform", "linear_decay", "exponential_decay"])
    parser.add_argument("--normalize",        action="store_true", default=True,
                        help="Normalize wrapper entropy by wrapper length")

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
    parser.add_argument("--limit",            type=int,   default=None,
                        help="Max AdvBench prompts to load")
    parser.add_argument("--prefill_variants", nargs="+",  default=None,
                        help="Override the default 10 prefill variants")

    # W&B
    parser.add_argument("--wandb_project",    default="AttCT")
    parser.add_argument("--wandb_name",       default=None)

    args = parser.parse_args()
    config = build_config(args)

    model_short = args.model.split("/")[-1]
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{model_short}_prefill_attct_lm{args.lm_weight}",
        config={**config, "lm_weight": args.lm_weight},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    print("Loading harmful_behaviors_pair.csv...")
    train_prompts, val_prompts, train_prefills, val_prefills = load_harmful_behaviors_pair(
        limit=args.limit,
    )
    train_prefills = args.prefill_variants or train_prefills
    val_prefills   = args.prefill_variants or val_prefills
    print(f"Train: {len(train_prompts)} prompts × {len(train_prefills)} prefills")
    print(f"Val:   {len(val_prompts)} prompts × {len(val_prefills)} prefills")

    train_dl = make_dataloader(train_prompts, train_prefills, tokenizer, args, shuffle=True)
    val_dl   = make_dataloader(val_prompts,   val_prefills,   tokenizer, args, shuffle=False)
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_fn = WrapperEntropyRegularizationLoss(
        weight=args.loss_weight,
        normalize=args.normalize,
        layer_weights=args.layer_weights,
    )
    print(f"Loss: WrapperEntropy(w={args.loss_weight}) + LM(w={args.lm_weight})")

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = PrefillAttCTTrainer(
        model=model,
        dataloader=train_dl,
        loss_fn=loss_fn,
        config=config,
        device=device,
        ref_model=None,
        tokenizer=tokenizer,
        lm_weight=args.lm_weight,
    )
    trainer.train()

    # ── Eval ─────────────────────────────────────────────────────────────────
    print("\nRunning eval on val set...")
    trainer.eval_loss(val_dl)

    wandb.finish()


if __name__ == "__main__":
    main()
