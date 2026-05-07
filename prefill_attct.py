"""
Prefill-AttCT: Attention Consistency Training for prefill attacks.

Uses WrapperEntropyRegularizationLoss on the wrapped pass to suppress
attention mass flowing to prefill token positions, anchored by KL
regularization on neutral UltraChat prompts (KL(adapter || frozen base))
to preserve generation quality.

Each optimizer step accumulates gradients from one AttCT batch (wrapper
entropy) and one KL batch (UltraChat) before updating weights — the
interleaving is delegated to InterleavedTrainer.

Training data: datasets/clearharm_prefills.csv (output of
prefill_generation_clearharm.py) — each row pairs one prompt with one
prefill of one of 23 strategy types. 80/20 train/val split.

Usage:
    uv run python prefill_attct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir checkpoints/prefill_attct
"""

import argparse

import torch
import wandb
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prefill_dataset import (
    PrefillAttackDataset,
    load_clearharm_prefills,
)
from data.ultrachat_dataset import get_kl_dataloader
from interleaved_trainer import InterleavedTrainer
from losses.kl_regularization import KLRegularizationLoss
from losses.losses import WrapperEntropyRegularizationLoss


# ---------------------------------------------------------------------------
# Dataset: adds wrapper_mask marking prefill token positions [Lc, Lw)
# ---------------------------------------------------------------------------

class PrefillAttCTDataset(PrefillAttackDataset):
    """
    Pairs prompts with prefills 1-to-1 (no Cartesian product) and emits a
    wrapper_mask marking the prefill positions in the wrapped sequence.

    Built for clearharm_prefills.csv where each row is (prompt, prefill,
    prefill_type) — one prefill per prompt. Multiple rows per prompt are
    treated as independent training examples.

    wrapper_mask[i] = True  for i in [Lc, Lw)   (prefill tokens)
    wrapper_mask[i] = False for i in [0, Lc)     (prompt tokens)
    """

    def __init__(
        self,
        prompts: list[str],
        prefills: list[str],
        tokenizer,
        max_length: int = 512,
    ):
        assert len(prompts) == len(prefills), (
            f"PrefillAttCTDataset is paired: len(prompts)={len(prompts)} "
            f"!= len(prefills)={len(prefills)}"
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []
        for prompt, prefill in zip(prompts, prefills):
            clean_text   = self._build_prompt(prompt)
            wrapped_text = clean_text + prefill
            self.items.append((prompt, clean_text, wrapped_text))

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
    dataset = PrefillAttCTDataset(
        prompts, prefills, tokenizer, max_length=args.max_length,
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
    parser = argparse.ArgumentParser(description="Prefill-AttCT training (KL-interleaved)")
    parser.add_argument("--model",            required=True,  help="HF model name or path")
    parser.add_argument("--lora_path",        default=None,   help="Load existing LoRA adapter")
    parser.add_argument("--output_dir",       default="checkpoints/prefill_attct")

    # Wrapper entropy loss
    parser.add_argument("--loss_weight",      type=float, default=1.0,
                        help="Weight for wrapper entropy loss")
    parser.add_argument("--layer_weights",    default="uniform",
                        choices=["uniform", "linear_decay", "exponential_decay"])
    parser.add_argument("--normalize",        action="store_true", default=True,
                        help="Normalize wrapper entropy by wrapper length")

    # KL regularization (anchor — replaces causal LM anchor)
    parser.add_argument("--kl_weight",        type=float, default=1.0,
                        help="Weight for KL(adapter || base) regularization")
    parser.add_argument("--kl_temperature",   type=float, default=1.0,
                        help="Softmax temperature for KL")
    parser.add_argument("--kl_samples",       type=int,   default=None,
                        help="UltraChat KL prompts (default: match AttCT dataset size)")
    parser.add_argument("--kl_dataset",       default="ultrachat",
                        choices=["ultrachat", "alpaca"])
    parser.add_argument("--kl_ratio",         type=float, default=1.0,
                        help="Fraction of AttCT steps that also fire a KL step (1.0 = always)")

    # Training
    parser.add_argument("--num_epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",       type=int,   default=1)
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
    parser.add_argument("--csv_path",         default=None,
                        help="Path to clearharm_prefills.csv (default: datasets/clearharm_prefills.csv)")
    parser.add_argument("--limit",            type=int,   default=None,
                        help="Max (prompt, prefill) rows to load")

    # W&B
    parser.add_argument("--wandb_project",    default="AttCT")
    parser.add_argument("--wandb_name",       default=None)

    args = parser.parse_args()
    config = build_config(args)

    model_short = args.model.split("/")[-1]
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{model_short}_prefill_attct_klw{args.kl_weight}",
        config={**config,
                "kl_weight": args.kl_weight,
                "kl_temperature": args.kl_temperature,
                "kl_dataset": args.kl_dataset,
                "kl_ratio": args.kl_ratio},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # device_map="auto" + low_cpu_mem_usage stream weights directly to GPU
    # shard-by-shard — necessary for 27B+ models on hosts with limited RAM.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        low_cpu_mem_usage=True,
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
    # device_map="auto" already placed weights — no .to(device) needed.

    # ── AttCT data ───────────────────────────────────────────────────────────
    print("Loading clearharm_prefills.csv...")
    train_prompts, val_prompts, train_prefills, val_prefills = load_clearharm_prefills(
        csv_path=args.csv_path,
        limit=args.limit,
    )
    print(f"Train: {len(train_prompts)} (prompt, prefill) pairs")
    print(f"Val:   {len(val_prompts)} (prompt, prefill) pairs")

    train_dl = make_dataloader(train_prompts, train_prefills, tokenizer, args, shuffle=True)
    val_dl   = make_dataloader(val_prompts,   val_prefills,   tokenizer, args, shuffle=False)
    print(f"AttCT train batches: {len(train_dl)} | val batches: {len(val_dl)}")

    # ── KL regularization data + loss (anchor) ───────────────────────────────
    n_kl = args.kl_samples if args.kl_samples is not None else len(train_dl.dataset)
    kl_dl = get_kl_dataloader(
        config, tokenizer, n_samples=n_kl, kl_dataset=args.kl_dataset,
    )
    print(f"KL anchor: {len(kl_dl.dataset)} {args.kl_dataset} samples "
          f"(weight={args.kl_weight}, T={args.kl_temperature}, ratio={args.kl_ratio})")

    # ── Losses ──────────────────────────────────────────────────────────────
    loss_fn = WrapperEntropyRegularizationLoss(
        weight=args.loss_weight,
        normalize=args.normalize,
        layer_weights=args.layer_weights,
    )
    kl_loss_fn = KLRegularizationLoss(
        weight=args.kl_weight,
        temperature=args.kl_temperature,
    )
    print(f"Loss: WrapperEntropy(w={args.loss_weight}) + KL(w={args.kl_weight})")

    # ── Train (AttCT + KL interleaved) ───────────────────────────────────────
    # InterleavedTrainer:
    #   • per optimizer step backwards both AttCT loss and KL loss before stepping
    #   • KL clean pass uses model.disable_adapter_layers() to recover frozen base
    #   • kl_ratio < 1.0 fires KL only on a fraction of AttCT steps
    trainer = InterleavedTrainer(
        model=model,
        attct_dataloader=train_dl,
        kl_dataloader=kl_dl,
        loss_fn=loss_fn,
        kl_loss_fn=kl_loss_fn,
        config=config,
        device=device,
        ref_model=None,
        tokenizer=tokenizer,
        kl_ratio=args.kl_ratio,
    )
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
