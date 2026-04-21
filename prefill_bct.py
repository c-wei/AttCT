"""
Prefill-BCT: Bias-Augmented Consistency Training for prefill attacks.

Adapts BCT (from evaluate_bct.py / BCT paper) to the prefill-attack setting
using WildJailbreak vanilla_harmful prompts.

BCT insight: train the model so that its output distribution is consistent
between a "biased" prompt and an "unbiased" prompt. Here:
    - "biased"   = clean prompt + prefill attack appended (wrapped)
    - "unbiased" = clean prompt with no prefill (clean)

Loss = KL(p_clean || p_wrapped) averaged over the prefill token positions.

The trainer mirrors train.py's Trainer in structure and BCT fidelity:
  - _step():             wrapped pass (grad) + frozen clean pass (disable_adapter_layers)
  - train():             epoch loop with grad accumulation and checkpoint callbacks
  - _validate():         eval KL loss on val set
  - _save_checkpoint():  save LoRA weights
  - _log():              W&B + stdout logging

Usage:
    uv run python prefill_bct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir checkpoints/prefill_bct_advbench

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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prefill_dataset import (
    PREFILL_VARIANTS_TRAIN,
    PREFILL_VARIANTS_TEST,
    get_prefill_dataloader,
    load_harmful_behaviors_pair,
)


# ---------------------------------------------------------------------------
# KL consistency loss (pure tensor computation, decoupled from forward passes)
# ---------------------------------------------------------------------------

def _kl_loss(
    clean_logits: torch.Tensor,    # (B, Lc, V)  no-grad, frozen reference
    wrapped_logits: torch.Tensor,  # (B, Lw, V)  grad flows through this
    clean_len: torch.Tensor,       # (B,)         unpadded clean token count
    clean_mask: torch.Tensor,      # (B, Lc)
    wrapped_mask: torch.Tensor,    # (B, Lw)
    kl_temperature: float = 1.0,
) -> torch.Tensor:
    """
    KL(p_clean || p_wrapped) at and after the divergence point.

    BCT analogy
    -----------
    BCT penalises the model for changing its answer distribution when a bias
    is added to the prompt.  Here the "bias" is a prefill string appended
    after the assistant turn marker.

    clean and wrapped share the same token prefix up to (and including) the
    last clean token (position cl-1 in real-token space).  At that position
    the clean model would generate its first response token; the wrapped model
    has already been nudged by the prefill.  We compare distributions there
    and at every subsequent prefill position.

    Sequence layout
    ---------------
    clean  : [<sys> <user> ... <gen_prompt_end>]               length Lc
    wrapped: [<sys> <user> ... <gen_prompt_end> <prefill...>]  length Lw

    Both are left-padded in the batch, so real tokens sit at the right end.
    Divergence point in padded tensor:
        clean  : pad_offset_c + cl - 1  (= last position, always)
        wrapped: pad_offset_w + cl - 1  (same relative position)
    """
    B  = clean_logits.shape[0]
    Lw = wrapped_logits.shape[1]

    # Initialise accumulator on the same device/dtype as wrapped_logits so
    # the computation graph stays connected for backward().
    total_kl = wrapped_logits.new_zeros(1)
    n_tokens = 0

    for b in range(B):
        cl   = int(clean_len[b].item())
        Lc   = int(clean_mask[b].sum().item())

        pad_offset_c = clean_logits.shape[1] - Lc
        div_idx_c    = pad_offset_c + cl - 1

        Lw_real      = int(wrapped_mask[b].sum().item())
        pad_offset_w = Lw - Lw_real
        div_idx_w    = pad_offset_w + cl - 1

        if div_idx_c < 0 or div_idx_c >= clean_logits.shape[1]:
            continue
        if div_idx_w < 0 or div_idx_w >= Lw - 1:
            continue

        # Reference: frozen first-token distribution from base model (no grad)
        ref_logit = clean_logits[b, div_idx_c, :].unsqueeze(0)       # (1, V)
        p_clean   = F.softmax(ref_logit / kl_temperature, dim=-1)    # (1, V)

        # Attacked positions: [div_idx_w ... Lw-2] (Lw-1 has no next token)
        w_logits = wrapped_logits[b, div_idx_w : Lw - 1, :]          # (n, V)
        w_mask_v = wrapped_mask[b, div_idx_w : Lw - 1]               # (n,)
        valid    = w_mask_v.bool()

        if valid.sum() == 0:
            continue

        w_logits_valid = w_logits[valid]                              # (n_valid, V)
        log_q_wrap     = F.log_softmax(w_logits_valid / kl_temperature, dim=-1)
        p_ref          = p_clean.expand(log_q_wrap.shape[0], -1)     # (n_valid, V)

        total_kl = total_kl + F.kl_div(log_q_wrap, p_ref, reduction="sum")
        n_tokens += log_q_wrap.shape[0]

    if n_tokens == 0:
        return total_kl.squeeze()   # zero, still on computation graph

    return (total_kl / n_tokens).squeeze()


# ---------------------------------------------------------------------------
# Trainer — mirrors train.py Trainer in structure
# ---------------------------------------------------------------------------

class PrefillBCTTrainer:
    """
    Trains a model to be consistent under prefill attacks using BCT-style
    KL consistency loss.

    Mirrors train.py's Trainer:
      _step()             forward passes + loss (wrapped grad, clean frozen)
      train()             epoch loop, grad accumulation, checkpoint callbacks
      _validate()         eval KL on val set  (cf. Trainer.eval_loss)
      _save_checkpoint()  save LoRA / full weights
      _log()              W&B + stdout

    BCT alignment with train.py
    ---------------------------
    train.py._step() uses disable_adapter_layers() so the clean pass runs the
    frozen base weights (θ_init), not the currently adapted weights.  This
    anchors the reference distribution and prevents the KL loss from chasing
    a moving target as the LoRA adapters update.  This trainer does the same.
    For full fine-tuning pass ref_model (frozen copy of the initial weights).
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_loader,
        val_loader,
        device: torch.device,
        output_dir: str,
        ref_model=None,           # frozen θ_init for full FT; None → use LoRA disable
        lr: float = 2e-5,
        num_epochs: int = 3,
        kl_temperature: float = 1.0,
        sft_coeff: float = 0.1,
        grad_clip: float = 1.0,
        grad_accumulation: int = 1,
        log_every: int = 10,
        sft_pairs: list = None,   # list of (prompt, response) tuples
        checkpoint_fn=None,       # callable(global_step) → None, matches train.py
        max_steps: int = None,
    ):
        self.model            = model
        self.ref_model        = ref_model
        self.tokenizer        = tokenizer
        self.train_loader     = train_loader
        self.val_loader       = val_loader
        self.device           = device
        self.output_dir       = Path(output_dir)
        self.num_epochs       = num_epochs
        self.kl_temperature   = kl_temperature
        self.sft_coeff        = sft_coeff
        self.grad_clip        = grad_clip
        self.grad_accumulation = grad_accumulation
        self.log_every        = log_every
        self.checkpoint_fn    = checkpoint_fn
        self.max_steps        = max_steps
        self._sft_batch_size  = 4

        if sft_pairs:
            import random
            random.shuffle(sft_pairs)
            self.sft_pairs = sft_pairs
            self._sft_iter = iter(self._batch_pairs(sft_pairs))
        else:
            self.sft_pairs = None
            self._sft_iter = None

        # Checkpoint steps at ~33%, ~66%, 100% of total optimizer steps.
        # Mirrors train.py Trainer.__init__ checkpoint_steps logic exactly.
        if max_steps is not None:
            total_steps = max_steps
        else:
            total_batches = num_epochs * len(train_loader)
            total_steps   = max(1, total_batches // grad_accumulation)
        self.checkpoint_steps = {
            total_steps // 3,
            (2 * total_steps) // 3,
            total_steps,
        }
        self.checkpoint_steps.discard(0)
        print(f"Behavioral eval checkpoints at optimizer steps: {sorted(self.checkpoint_steps)}")

        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Core step ─────────────────────────────────────────────────────────────

    def _step(self, batch: dict) -> dict:
        """
        Mirrors train.py Trainer._step():
          - wrapped pass: gradient flows (BCT "biased" / attacked side)
          - clean pass:   torch.no_grad + frozen base via disable_adapter_layers()
                          (or ref_model for full FT) — identical to train.py L144-157

        Unlike the previous version of this file, the clean pass uses the
        *frozen* base model weights (θ_init), not the current adapted weights.
        This matches train.py and ensures the KL reference is stable.
        """
        clean_ids    = batch["clean_input_ids"].to(self.device)
        clean_mask   = batch["clean_attention_mask"].to(self.device)
        wrapped_ids  = batch["wrapped_input_ids"].to(self.device)
        wrapped_mask = batch["wrapped_attention_mask"].to(self.device)
        clean_len    = batch["clean_len"].to(self.device)

        # Wrapped (attacked) forward pass — gradients flow
        wrapped_logits = self.model(
            input_ids=wrapped_ids,
            attention_mask=wrapped_mask,
        ).logits.float()                                          # (B, Lw, V)

        # Clean (unbiased) forward pass — frozen reference, no gradients.
        # Mirrors train.py lines 144-157 exactly.
        with torch.no_grad():
            if self.ref_model is not None:
                # Full fine-tuning: use separately frozen copy of initial weights
                clean_logits = self.ref_model(
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                ).logits.float()
            else:
                # LoRA: disable adapters to recover θ_init for this pass
                self.model.disable_adapter_layers()
                clean_logits = self.model(
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                ).logits.float()
                self.model.enable_adapter_layers()               # (B, Lc, V)

        kl = _kl_loss(
            clean_logits, wrapped_logits,
            clean_len, clean_mask, wrapped_mask,
            self.kl_temperature,
        )
        loss_dict = {"kl_loss": kl.item(), "loss": kl}

        if self.sft_coeff > 0 and self.sft_pairs is not None:
            sft      = self._sft_loss(self._next_sft_batch())
            total    = kl + self.sft_coeff * sft
            loss_dict.update({"sft_loss": sft.item(), "loss": total})

        return loss_dict

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        """
        Mirrors train.py Trainer.train() with grad accumulation and checkpoint
        callbacks at ~33%, ~66%, and 100% of total optimizer steps.
        """
        global_step = 0
        batch_count = 0
        self.optimizer.zero_grad()

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_kl  = 0.0
            epoch_tot = 0.0
            n_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}", leave=False)
            for batch in pbar:
                if self.max_steps is not None and global_step >= self.max_steps:
                    break

                loss_dict = self._step(batch)
                (loss_dict["loss"] / self.grad_accumulation).backward()
                batch_count += 1

                epoch_kl  += loss_dict["kl_loss"]
                epoch_tot += loss_dict["loss"].item()
                n_batches += 1

                if batch_count % self.grad_accumulation == 0:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.grad_clip,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Fire checkpoint callback at the three designated steps.
                    # Model is saved first so checkpoint corresponds to the
                    # weights that behavioral eval will measure.
                    # Mirrors train.py Trainer.train() lines 206-209.
                    if self.checkpoint_fn is not None and global_step in self.checkpoint_steps:
                        self._save_checkpoint(tag=f"step_{global_step}")
                        print(f"\n[Checkpoint] Step {global_step} — running behavioral eval...")
                        self.checkpoint_fn(global_step)
                        self.model.train()

                    if global_step % self.log_every == 0:
                        self._log(epoch, global_step, loss_dict)

                pbar.set_postfix(kl=f"{loss_dict['kl_loss']:.4f}")

                if self.max_steps is not None and global_step >= self.max_steps:
                    break

            avg_kl  = epoch_kl  / max(n_batches, 1)
            avg_tot = epoch_tot / max(n_batches, 1)
            val_kl  = self._validate()

            wandb.log({
                "epoch/kl_loss":    avg_kl,
                "epoch/total_loss": avg_tot,
                "val/kl_loss":      val_kl,
                "epoch":            epoch,
            })
            print(f"Epoch {epoch} | train_kl={avg_kl:.4f} | val_kl={val_kl:.4f}")
            self._save_checkpoint(tag=f"epoch_{epoch}")

    def _validate(self) -> float:
        """
        Compute mean KL loss on the val set.
        Mirrors train.py Trainer.eval_loss().
        Only measures the KL component (no SFT) — consistent with how
        train.py eval_loss uses the consistency loss_fn directly.
        """
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="  val", leave=False):
                clean_ids    = batch["clean_input_ids"].to(self.device)
                clean_mask   = batch["clean_attention_mask"].to(self.device)
                wrapped_ids  = batch["wrapped_input_ids"].to(self.device)
                wrapped_mask = batch["wrapped_attention_mask"].to(self.device)
                clean_len    = batch["clean_len"].to(self.device)

                # Both passes are no-grad here; we just want the numeric loss.
                # No need to disable adapters — we compare base vs adapted, both frozen.
                if self.ref_model is not None:
                    clean_logits = self.ref_model(
                        input_ids=clean_ids, attention_mask=clean_mask
                    ).logits.float()
                else:
                    self.model.disable_adapter_layers()
                    clean_logits = self.model(
                        input_ids=clean_ids, attention_mask=clean_mask
                    ).logits.float()
                    self.model.enable_adapter_layers()

                wrapped_logits = self.model(
                    input_ids=wrapped_ids, attention_mask=wrapped_mask
                ).logits.float()

                kl = _kl_loss(
                    clean_logits, wrapped_logits,
                    clean_len, clean_mask, wrapped_mask,
                    self.kl_temperature,
                )
                total += kl.item()
                n     += 1

        self.model.train()
        mean = total / max(n, 1)
        wandb.log({"eval/mean_loss": mean})
        print(f"\n--- Prefill-BCT Eval --- mean_kl: {mean:.4f}\n")
        return mean

    def _save_checkpoint(self, tag: str):
        """Mirrors train.py Trainer._save_checkpoint()."""
        path = self.output_dir / tag
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        print(f"  Checkpoint saved → {path}")

    def _log(self, epoch: int, step: int, loss_dict: dict):
        """Mirrors train.py Trainer._log()."""
        metrics = {
            "train/kl_loss":  loss_dict["kl_loss"],
            "train/loss":     loss_dict["loss"].item(),
            "train/epoch":    epoch,
        }
        if "sft_loss" in loss_dict:
            metrics["train/sft_loss"] = loss_dict["sft_loss"]
        wandb.log(metrics, step=step)

        line = f"[epoch {epoch} | step {step}] kl={loss_dict['kl_loss']:.4f}"
        if "sft_loss" in loss_dict:
            line += f"  sft={loss_dict['sft_loss']:.4f}"
        print(line)

    # ── SFT helpers ───────────────────────────────────────────────────────────

    def _next_sft_batch(self) -> list:
        try:
            return next(self._sft_iter)
        except StopIteration:
            import random
            random.shuffle(self.sft_pairs)
            self._sft_iter = iter(self._batch_pairs(self.sft_pairs))
            return next(self._sft_iter)

    def _batch_pairs(self, pairs: list):
        for i in range(0, len(pairs), self._sft_batch_size):
            yield pairs[i : i + self._sft_batch_size]

    def _sft_loss(self, pairs: list) -> torch.Tensor:
        """
        CE loss on (prompt, refusal_response) pairs — response tokens only.
        Prevents the KL loss from collapsing both distributions to compliance.
        """
        full_texts, prompt_lens = [], []
        for prompt, response in pairs:
            full_texts.append(self.tokenizer.apply_chat_template(
                [
                    {"role": "system",    "content": "You are a helpful assistant."},
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False, add_generation_prompt=False,
            ))
            # Unpadded prompt length for response-start offset
            prompt_only = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": prompt},
                ],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_lens.append(len(
                self.tokenizer(prompt_only, return_tensors="pt",
                               padding=False, truncation=True,
                               max_length=512)["input_ids"][0]
            ))

        enc = self.tokenizer(
            full_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(self.device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        logits = self.model(input_ids=ids, attention_mask=mask).logits.float()

        B, L = ids.shape[:2]
        loss, n = logits.new_zeros(1), 0
        for b in range(B):
            pad_offset = L - int(mask[b].sum().item())
            resp_start = pad_offset + prompt_lens[b]
            if resp_start >= L - 1:
                continue
            pred  = logits[b, resp_start:-1, :]
            tgt   = ids[b, resp_start + 1:]
            valid = mask[b, resp_start + 1:].bool()
            if valid.sum() == 0:
                continue
            loss = loss + F.cross_entropy(pred[valid], tgt[valid], reduction="sum")
            n   += valid.sum().item()
        return (loss / max(n, 1)).squeeze()


# ---------------------------------------------------------------------------
# Refusal SFT data loader
# ---------------------------------------------------------------------------

def load_refusal_pairs(limit: int = None) -> list[tuple[str, str]]:
    """
    Load (prompt, refusal_response) pairs from mrfakename/refusal for SFT.
    Explicitly reinforces refusal behaviour so the KL loss cannot collapse
    both clean and attacked distributions toward compliance.
    """
    from datasets import load_dataset as _hf_load
    ds = _hf_load("mrfakename/refusal", split="train", streaming=True)
    pairs = []
    for item in ds:
        prompt   = item.get("input",  "").strip()
        response = item.get("output", "").strip()
        if prompt and response:
            pairs.append((prompt, response))
        if limit is not None and len(pairs) >= limit:
            break
    print(f"Loaded {len(pairs)} refusal pairs for SFT regularisation")
    return pairs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",            required=True,  help="HF model name or path")
    parser.add_argument("--lora_path",        default=None,   help="Load existing LoRA adapter before training")
    parser.add_argument("--output_dir",       default="checkpoints/prefill_bct")
    parser.add_argument("--num_epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",       type=int,   default=4)
    parser.add_argument("--grad_accumulation",type=int,   default=1)
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--kl_temperature",   type=float, default=1.0,
                        help="Softmax temperature for KL loss (>1 = softer targets)")
    parser.add_argument("--sft_coeff",        type=float, default=0.1,
                        help="Weight of SFT regularisation loss (0 to disable)")
    parser.add_argument("--grad_clip",        type=float, default=1.0)
    parser.add_argument("--lora_r",           type=int,   default=16)
    parser.add_argument("--lora_alpha",       type=int,   default=12)
    parser.add_argument("--lora_dropout",     type=float, default=0.05)
    parser.add_argument("--limit",            type=int,   default=None,
                        help="Max harmful prompts to load (for quick runs)")
    parser.add_argument("--max_steps",        type=int,   default=None,
                        help="Stop after this many optimizer steps")
    parser.add_argument("--prefill_variants", nargs="+",  default=None,
                        help="Override prefill strings (default: PREFILL_VARIANTS)")
    parser.add_argument("--log_every",        type=int,   default=10)
    parser.add_argument("--wandb_project",    default="AttCT")
    parser.add_argument("--wandb_name",       default=None)
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
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = model.to(device)

    print("Loading harmful_behaviors_pair.csv...")
    train_prompts, val_prompts, prefills = load_harmful_behaviors_pair(limit=args.limit)

    train_loader = get_prefill_dataloader(
        train_prompts, tokenizer,
        prefill_variants=args.prefill_variants or prefills,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = get_prefill_dataloader(
        val_prompts, tokenizer,
        prefill_variants=args.prefill_variants or prefills,
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    sft_pairs = None
    if args.sft_coeff > 0:
        sft_pairs = load_refusal_pairs(limit=None)
        print(f"SFT pairs: {len(sft_pairs)}")

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
        grad_accumulation=args.grad_accumulation,
        log_every=args.log_every,
        sft_pairs=sft_pairs,
        max_steps=args.max_steps,
    )
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
