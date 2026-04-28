"""
Prefill-AttCT: Attention Consistency Training for prefill attacks.

Uses WrapperEntropyRegularizationLoss to suppress attention flowing to
prefill token positions, combined with a causal LM loss on the clean
(no-prefill) pass to preserve generation quality.

    total_loss = attct_loss + lm_weight * lm_loss

Trained with k-fold cross validation over harmful_behaviors_pair.csv.
Each fold trains a fresh model on (k-1)/k of the data and reports val loss
on the held-out 1/k. Final metric is mean ± std across folds.

Usage:
    uv run python prefill_attct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir checkpoints/prefill_attct \
        --n_folds 5
"""

import argparse
from statistics import mean, stdev

import torch
import torch.nn.functional as F
import wandb
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    Extends PrefillAttackDataset with two changes:
      1. 1-to-1 pairing: each prompt is paired with exactly its corresponding
         prefill (no Cartesian product).
      2. Emits a wrapper_mask marking prefill positions [Lc, Lw).

    Args:
        prompts:    List of harmful prompts.
        tokenizer:  HuggingFace tokenizer.
        prefills:   List of prefills, len(prefills) == len(prompts), where
                    prefills[i] is the tailored prefix for prompts[i].
        max_length: Tokenizer truncation length.
    """

    def __init__(self, prompts, tokenizer, prefills, max_length=512):
        assert len(prompts) == len(prefills), (
            f"1-to-1 pairing requires len(prompts)={len(prompts)} == "
            f"len(prefills)={len(prefills)}"
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []
        for prompt, prefill in zip(prompts, prefills):
            clean_text = self._build_prompt(prompt)
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

    def eval_loss(self, dataloader=None) -> float:
        """
        Override Trainer.eval_loss to return the mean. Used by the k-fold
        loop to aggregate val loss across folds.
        """
        dl = dataloader if dataloader is not None else self.dataloader
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(dl, desc="eval", leave=False):
                loss_dict = self._step(batch)
                total += loss_dict["loss"].item()
                n += 1
        self.model.train()
        m = total / max(n, 1)
        wandb.log({"eval/mean_loss": m})
        print(f"\n--- Prefill-AttCT Eval --- mean_loss: {m:.4f}\n")
        return m


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
# K-fold helpers
# ---------------------------------------------------------------------------

def kfold_indices(n: int, k: int) -> list:
    """
    Return a list of (train_indices, val_indices) tuples for k folds.
    Last fold absorbs the remainder when n is not divisible by k.
    """
    fold_size = n // k
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else n
        val_idx = list(range(val_start, val_end))
        train_idx = list(range(0, val_start)) + list(range(val_end, n))
        folds.append((train_idx, val_idx))
    return folds


def setup_model(args, device):
    """
    Build a fresh (base + LoRA) model. Called once per fold so each fold
    starts from the same θ_init rather than carrying over weights.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    if args.lora_path:
        print(f"Loading LoRA adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(base_model, args.lora_path, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(base_model, lora_cfg)
        model.print_trainable_parameters()
    return model.to(device)


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
    parser.add_argument("--limit",            type=int,   default=None)
    parser.add_argument("--prefill_variants", nargs="+",  default=None)
    parser.add_argument("--n_folds",          type=int,   default=5,
                        help="Number of CV folds (5 → 80/20 train/val per fold)")

    # W&B
    parser.add_argument("--wandb_project",    default="AttCT")
    parser.add_argument("--wandb_name",       default=None)

    args = parser.parse_args()
    config = build_config(args)

    model_short = args.model.split("/")[-1]
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{model_short}_prefill_attct_kfold{args.n_folds}",
        config={**config, "lm_weight": args.lm_weight, "n_folds": args.n_folds},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Load all pairs (k-fold splits the full dataset internally) ────────
    print("Loading harmful_behaviors_pair.csv (full set for k-fold)...")
    all_prompts, _, all_prefills, _ = load_harmful_behaviors_pair(
        limit=args.limit, train_ratio=1.0,
    )
    if args.prefill_variants:
        # CLI override only valid if length matches (1-to-1 pairing requirement)
        assert len(args.prefill_variants) == len(all_prompts), (
            f"--prefill_variants ({len(args.prefill_variants)}) must match "
            f"prompts ({len(all_prompts)}) for 1-to-1 pairing"
        )
        all_prefills = args.prefill_variants
    print(f"Loaded {len(all_prompts)} prompt-prefill pairs total")

    # ── K-fold CV loop ─────────────────────────────────────────────────────
    folds = kfold_indices(len(all_prompts), args.n_folds)
    fold_val_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx + 1}/{args.n_folds}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        train_prompts  = [all_prompts[i]  for i in train_idx]
        train_prefills = [all_prefills[i] for i in train_idx]
        val_prompts    = [all_prompts[i]  for i in val_idx]
        val_prefills   = [all_prefills[i] for i in val_idx]

        train_dl = make_dataloader(train_prompts, train_prefills, tokenizer, args, shuffle=True)
        val_dl   = make_dataloader(val_prompts,   val_prefills,   tokenizer, args, shuffle=False)

        # Fresh model + LoRA per fold (start from θ_init each time)
        model = setup_model(args, device)

        loss_fn = WrapperEntropyRegularizationLoss(
            weight=args.loss_weight,
            normalize=args.normalize,
            layer_weights=args.layer_weights,
        )

        # Per-fold checkpoint directory
        fold_config = {**config}
        fold_config["training"] = {
            **config["training"],
            "save_dir": f"{args.output_dir}/fold_{fold_idx}",
        }

        trainer = PrefillAttCTTrainer(
            model=model,
            dataloader=train_dl,
            loss_fn=loss_fn,
            config=fold_config,
            device=device,
            ref_model=None,
            tokenizer=tokenizer,
            lm_weight=args.lm_weight,
        )
        trainer.train()

        print(f"\n[Fold {fold_idx + 1}] Eval on held-out val...")
        val_loss = trainer.eval_loss(val_dl)
        fold_val_losses.append(val_loss)
        wandb.log({f"fold_{fold_idx}/val_loss": val_loss})

        # Free GPU before the next fold
        del model, trainer
        torch.cuda.empty_cache()

    # ── Aggregate ──────────────────────────────────────────────────────────
    m = mean(fold_val_losses)
    s = stdev(fold_val_losses) if len(fold_val_losses) > 1 else 0.0
    print(f"\n{'='*60}")
    print(f"  K-Fold Summary ({args.n_folds} folds)")
    print(f"{'='*60}")
    for i, vl in enumerate(fold_val_losses):
        print(f"  Fold {i + 1}: val_loss = {vl:.4f}")
    print(f"\n  Mean val loss: {m:.4f}  (std: {s:.4f})")

    wandb.summary["kfold/mean_val_loss"] = m
    wandb.summary["kfold/std_val_loss"]  = s
    wandb.finish()


if __name__ == "__main__":
    main()
