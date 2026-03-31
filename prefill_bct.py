"""
Prefill-BCT: Bias-Augmented Consistency Training for prefill attacks.

Adapts BCT (from evaluate_bct.py / BCT paper) to the prefill-attack setting.

BCT insight: train the model so that its output distribution is consistent
between a "biased" prompt and an "unbiased" prompt. Here:
    - "biased"   = clean prompt + prefill attack appended (wrapped)
    - "unbiased" = clean prompt with no prefill (clean)

Loss = KL(p_clean || p_wrapped) averaged over the assistant-turn tokens,
       i.e. we pull the wrapped distribution toward the clean one.

This is identical in spirit to BCT's consistency loss, but operating on
free-form token distributions rather than MCQ answer logits.

Usage:
    uv run python prefill_bct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir checkpoints/prefill_bct

    # With a LoRA base checkpoint:
    uv run python prefill_bct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --lora_path checkpoints/bct_sft/epoch_1 \
        --output_dir checkpoints/prefill_bct
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prefill_dataset import (
    PREFILL_VARIANTS,
    get_prefill_dataloader,
    load_wildjailbreak_prompts,
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def prefill_consistency_loss(
    model,
    batch: dict,
    device: torch.device,
    kl_temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute KL(p_clean || p_wrapped) averaged over the *new* tokens that
    follow the prefill insertion point.

    BCT analogy
    -----------
    BCT computes: for each (biased_prompt, unbiased_prompt) pair, run both
    through the model and penalise the KL between the two output distributions
    over the answer token. We do the same but token-by-token over the full
    generation, using the clean sequence as the reference.

    Why KL(clean || wrapped)?
    -------------------------
    We treat the clean distribution as the "ground truth" behaviour we want
    to preserve and pull the attacked (wrapped) distribution toward it.
    This mirrors BCT where the unbiased prompt defines the reference.

    Masking
    -------
    Only tokens *after* the prefill start index in the wrapped sequence are
    penalised — tokens in the shared prefix are identical by construction so
    there is nothing to correct there.

    Because clean and wrapped sequences may differ in length (wrapped has the
    extra prefill tokens at the end of the prompt), we align them via the
    shared prompt prefix length (clean_len).

    Args:
        model:            The causal LM (with or without LoRA adapter).
        batch:            Collated batch from prefill_collate_fn.
        device:           Target device.
        kl_temperature:   Softmax temperature; > 1 smooths the distribution.

    Returns:
        Scalar loss tensor (mean KL across unmasked positions).
    """
    clean_ids   = batch["clean_input_ids"].to(device)        # (B, Lc)
    clean_mask  = batch["clean_attention_mask"].to(device)
    wrapped_ids = batch["wrapped_input_ids"].to(device)      # (B, Lw)
    wrapped_mask= batch["wrapped_attention_mask"].to(device)


    # Token position in the wrapped sequence where the prefill begins.
    # Tokens before this index are shared with the clean sequence and
    # should not be penalised (they would be identical anyway).
    # clean_len == len(clean tokens) == first position of prefill in wrapped.
    clean_len   = batch["clean_len"].to(device)              # (B,)

    # -----------------------------------------------------------------
    # Forward passes
    # -----------------------------------------------------------------
    # We need logits for *all* positions so we can align clean ↔ wrapped.
    # gradient only flows through the wrapped forward pass; the clean
    # forward is used as a fixed reference (no_grad).
    with torch.no_grad():
        clean_logits = model(
            input_ids=clean_ids,
            attention_mask=clean_mask,
        ).logits                                             # (B, Lc, V)

    wrapped_logits = model(
        input_ids=wrapped_ids,
        attention_mask=wrapped_mask,
    ).logits                                                 # (B, Lw, V)

    # -----------------------------------------------------------------
    # Align sequences
    # -----------------------------------------------------------------
    # After the generation prompt the clean sequence predicts token t+1
    # at position t. The wrapped sequence has the same prompt prefix
    # followed by prefill tokens. We align by taking, for each sequence,
    # the logits starting at the *last* position of the shared prefix
    # (clean_len - 1) onward — these are the first "new" prediction
    # positions where the model's output may diverge.
    #
    # clean   : [prompt ... <gen_prompt_end>]          length Lc
    # wrapped : [prompt ... <gen_prompt_end> <prefill>] length Lw
    #
    # Logits at position i predict token i+1.
    # The first position that matters is clean_len - 1 (last prompt token),
    # which predicts the first real generation token.

    B = clean_ids.shape[0]
    Lw = wrapped_ids.shape[1]

    total_kl = torch.zeros(1, dtype=torch.float32, device=device)
    n_tokens  = 0

    for b in range(B):
        cl = int(clean_len[b].item())   # shared prefix length
        Lc = int(clean_mask[b].sum().item())   # actual (unpadded) clean length

        pad_offset_c = clean_ids.shape[1] - Lc   # how many pad tokens on left
        div_idx_c    = pad_offset_c + cl - 1      # absolute position of last real clean token
        Lw_real      = int(wrapped_mask[b].sum().item())
        pad_offset_w = Lw - Lw_real
        div_idx_w    = pad_offset_w + cl - 1
        if div_idx_c < 0 or div_idx_c >= clean_ids.shape[1]:
            continue
        if div_idx_w < 0 or div_idx_w >= Lw - 1:
            continue

        # Reference: single distribution at divergence point in clean sequence
        # Shape: (1, V) — broadcast over all prefill positions
        ref_logit = clean_logits[b, div_idx_c, :].unsqueeze(0)         # (1, V)
        p_clean   = F.softmax(ref_logit / kl_temperature, dim=-1)      # (1, V)
        # Attacked positions: [div_idx_w ... Lw-2] in wrapped
        # (Lw-1 excluded: no target token follows it)
        w_logits  = wrapped_logits[b, div_idx_w : Lw - 1, :]           # (n, V)
        w_mask_v  = wrapped_mask[b, div_idx_w : Lw - 1]                # (n,)
        valid     = w_mask_v.bool()
        if valid.sum() == 0:
            continue
        w_logits_valid = w_logits[valid]                                # (n_valid, V)
        log_q_wrap     = F.log_softmax(w_logits_valid / kl_temperature, dim=-1)
        p_ref = p_clean.expand(log_q_wrap.shape[0], -1)                # (n_valid, V)
        kl = F.kl_div(log_q_wrap, p_ref, reduction="sum")
        total_kl = total_kl + kl
        n_tokens += log_q_wrap.shape[0]


    if n_tokens == 0:
        return torch.zeros(1, dtype=torch.float32, device=device).squeeze()

    return (total_kl / n_tokens).squeeze()



# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PrefillBCTTrainer:
    """
    Trains a model to be consistent under prefill attacks using BCT-style
    KL consistency loss.

    The training loop mirrors the BCT paper:
      1. For each (clean, wrapped) pair, compute KL(p_clean || p_wrapped).
      2. Back-prop through the wrapped forward pass only.
      3. Optionally add a supervised fine-tuning (SFT) loss on clean
         sequences to prevent the model from collapsing its distribution.

    LoRA
    ----
    Full fine-tuning on large models is expensive. By default we add a LoRA
    adapter so only ~0.1% of parameters are updated, which also reduces the
    risk of catastrophic forgetting.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_loader,
        val_loader,
        device: torch.device,
        output_dir: str,
        lr: float = 2e-5,
        num_epochs: int = 3,
        kl_temperature: float = 1.0,
        sft_coeff: float = 0.1,
        grad_clip: float = 1.0,
        log_every: int = 10,
    ):
        self.model         = model
        self.tokenizer     = tokenizer
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.device        = device
        self.output_dir    = Path(output_dir)
        self.num_epochs    = num_epochs
        self.kl_temperature= kl_temperature
        self.sft_coeff     = sft_coeff
        self.grad_clip     = grad_clip
        self.log_every     = log_every

        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )
        total_steps = num_epochs * len(train_loader)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _sft_loss(self, batch: dict) -> torch.Tensor:
        """
        Cross-entropy on clean sequences from clean_start_index onward.
        Acts as a regulariser to prevent distribution collapse, analogous
        to the SFT component in BCT's combined objective.
        """
        ids  = batch["clean_input_ids"].to(self.device)
        mask = batch["clean_attention_mask"].to(self.device)

        logits = self.model(input_ids=ids, attention_mask=mask).logits.float()  # (B, L, V)

        B, L, V = logits.shape
        loss = torch.zeros(1, dtype=torch.float32, device=self.device)
        n = 0

        for b in range(B):
            pred  = logits[b, :-1, :]                          # (L-1, V)
            tgt   = ids[b, 1:]                                 # (L-1,)
            valid = (mask[b, :-1] & mask[b, 1:]).bool()
            if valid.sum() == 0:
                continue
            loss  = loss + F.cross_entropy(pred[valid], tgt[valid], reduction="sum")
            n    += valid.sum().item()

        return (loss / max(n, 1)).squeeze()

    def train(self):
        global_step = 0
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_kl  = 0.0
            epoch_sft = 0.0
            epoch_tot = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
            for step, batch in enumerate(pbar):
                self.optimizer.zero_grad()

                kl_loss  = prefill_consistency_loss(
                    self.model, batch, self.device, self.kl_temperature
                )
                sft_loss = self._sft_loss(batch) if self.sft_coeff > 0 else torch.zeros(1)
                loss     = kl_loss + self.sft_coeff * sft_loss

                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip,
                    )
                self.optimizer.step()
                self.scheduler.step()
                global_step += 1

                kl_val  = kl_loss.item()
                sft_val = sft_loss.item() if self.sft_coeff > 0 else 0.0
                tot_val = loss.item()
                epoch_kl  += kl_val
                epoch_sft += sft_val
                epoch_tot += tot_val

                pbar.set_postfix(kl=f"{kl_val:.4f}", sft=f"{sft_val:.4f}")

                if global_step % self.log_every == 0:
                    wandb.log({
                        "train/kl_loss":    kl_val,
                        "train/sft_loss":   sft_val,
                        "train/total_loss": tot_val,
                        "train/lr":         self.scheduler.get_last_lr()[0],
                        "step":             global_step,
                    })

            n = len(self.train_loader)
            wandb.log({
                "epoch/kl_loss":    epoch_kl  / n,
                "epoch/sft_loss":   epoch_sft / n,
                "epoch/total_loss": epoch_tot / n,
                "epoch":            epoch,
            })

            val_loss = self._validate()
            wandb.log({"val/kl_loss": val_loss, "epoch": epoch})
            print(f"Epoch {epoch} | train_kl={epoch_kl/n:.4f} | val_kl={val_loss:.4f}")

            ckpt = self.output_dir / f"epoch_{epoch}"
            self.model.save_pretrained(str(ckpt))
            self.tokenizer.save_pretrained(str(ckpt))
            print(f"  Saved checkpoint → {ckpt}")

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in tqdm(self.val_loader, desc="  val", leave=False):
            loss = prefill_consistency_loss(
                self.model, batch, self.device, self.kl_temperature
            )
            total += loss.item()
            n     += 1
        self.model.train()
        return total / max(n, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          required=True,  help="HF model name or path")
    parser.add_argument("--lora_path",      default=None,   help="Load existing LoRA adapter before training")
    parser.add_argument("--output_dir",     default="checkpoints/prefill_bct")
    parser.add_argument("--num_epochs",     type=int,   default=3)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--kl_temperature", type=float, default=1.0,
                        help="Softmax temperature for KL loss (>1 = softer targets)")
    parser.add_argument("--sft_coeff",      type=float, default=0.1,
                        help="Weight of SFT regularisation loss (0 to disable)")
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--lora_r",         type=int,   default=16)
    parser.add_argument("--lora_alpha",     type=int,   default=32)
    parser.add_argument("--lora_dropout",   type=float, default=0.05)
    parser.add_argument("--limit",          type=int,   default=None,
                        help="Max harmful prompts to load (for quick runs)")
    parser.add_argument("--prefill_variants", nargs="+", default=None,
                        help="Override prefill strings (default: PREFILL_VARIANTS)")
    parser.add_argument("--wandb_project",  default="AttCT")
    parser.add_argument("--wandb_name",     default=None)
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{model_short}_prefill_bct",
        config=vars(args),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    if args.lora_path:
        print(f"Loading LoRA adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
    else:
        # Add a fresh LoRA adapter for prefill-BCT training
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            # Target the attention projections — standard for Llama-style models
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model = model.to(device)

    print("Loading WildJailbreak prompts...")
    prefill_variants = args.prefill_variants or PREFILL_VARIANTS
    train_prompts, val_prompts = load_wildjailbreak_prompts(limit=args.limit)

    train_loader = get_prefill_dataloader(
        train_prompts, tokenizer,
        prefill_variants=prefill_variants,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = get_prefill_dataloader(
        val_prompts, tokenizer,
        prefill_variants=prefill_variants,
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    trainer = PrefillBCTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        lr=args.lr,
        num_epochs=args.num_epochs,
        kl_temperature=args.kl_temperature,
        sft_coeff=args.sft_coeff,
        grad_clip=args.grad_clip,
    )
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()