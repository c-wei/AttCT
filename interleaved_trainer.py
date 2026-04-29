import os
import time
import json
import concurrent.futures
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import _hf_upload_pool


class InterleavedTrainer:
    """
    Training loop that alternates AttCT consistency and KL regularization steps.

    Each optimizer step accumulates gradients from one AttCT batch and one KL
    batch before updating weights.  The KL dataloader cycles if it is shorter
    than the AttCT dataloader.

    Args:
        model:             PEFT-wrapped HuggingFace model (on device).
        attct_dataloader:  DataLoader for AttCT consistency training.
        kl_dataloader:     DataLoader for KL regularization (UltraChat prompts).
        loss_fn:           Instantiated ConsistencyLoss subclass (e.g. JSD).
        kl_loss_fn:        Instantiated KLRegularizationLoss.
        config:            Full config dict.
        device:            torch.device.
        ref_model:         Optional frozen reference model for full FT mode.
        checkpoint_fn:     Optional callable(global_step) for behavioral eval.
        log_io_path:       Optional path for IO logging JSONL.
        tokenizer:         Optional tokenizer for logging.
    """

    def __init__(
        self,
        model,
        attct_dataloader: DataLoader,
        kl_dataloader: DataLoader,
        loss_fn,
        kl_loss_fn,
        config: dict,
        device: torch.device,
        ref_model=None,
        checkpoint_fn=None,
        log_io_path=None,
        tokenizer=None,
        hf_repo=None,
        run_name=None,
        kl_ratio: float = 1.0,
    ):
        self.model = model
        self.ref_model = ref_model
        self.attct_dataloader = attct_dataloader
        self.kl_dataloader = kl_dataloader
        self.loss_fn = loss_fn
        self.kl_loss_fn = kl_loss_fn
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.checkpoint_fn = checkpoint_fn

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.max_steps = train_cfg.get("max_steps", None)
        self.grad_clip = train_cfg.get("grad_clip")
        self.log_every = train_cfg.get("log_every_n_steps", 10)
        self.grad_accumulation = train_cfg.get("grad_accumulation_steps", 1)
        self.kl_ratio = kl_ratio  # prob of firing a KL step alongside each AttCT step

        model_cfg = config["model"]
        self.output_attentions = model_cfg.get("output_attentions", True)
        self.output_hidden_states = model_cfg.get("output_hidden_states", False)
        self.needs_clean_pass = loss_fn.needs_clean_pass
        self.save_dir = train_cfg.get("save_dir")
        self.hf_repo = hf_repo
        self.run_name = run_name

        # IO logging
        self._log_io_file = open(log_io_path, "w") if log_io_path else None

        # Training data log
        self._train_log_file = None
        if self.tokenizer is not None:
            log_dir = config.get("logging", {}).get("log_dir", "logs")
            os.makedirs(log_dir, exist_ok=True)
            train_log_path = os.path.join(log_dir, "training_data.jsonl")
            self._train_log_file = open(train_log_path, "w")
            print(f"Training data log: {train_log_path}")

        # Per-layer delta tracking
        self._first_layer_losses: list = []
        self._last_layer_losses: list = []

        # Checkpoint scheduling — based on the actual number of OPTIMIZER steps
        # the loop will take (i.e. micro-batches divided by grad_accumulation).
        dataset_micro_batches = max(1, len(attct_dataloader) * self.epochs)
        dataset_steps = max(1, dataset_micro_batches // self.grad_accumulation)
        if self.max_steps is not None:
            total_optimizer_steps = min(self.max_steps, dataset_steps)
        else:
            total_optimizer_steps = dataset_steps
        self.checkpoint_steps = {
            total_optimizer_steps // 3,
            (2 * total_optimizer_steps) // 3,
            total_optimizer_steps,
        }
        self.checkpoint_steps.discard(0)
        print(f"Behavioral eval checkpoints at optimizer steps: {sorted(self.checkpoint_steps)}")

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg["learning_rate"],
        )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward(self, input_ids, attention_mask):
        """Standard forward pass through the model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

    # ------------------------------------------------------------------
    # AttCT step (identical to Trainer._step)
    # ------------------------------------------------------------------

    def _attct_step(self, batch: dict) -> dict:
        """
        One AttCT consistency loss forward pass.

        Produces the paired clean/wrapped forward passes and computes the
        consistency loss.  Returns the loss dict (loss not yet backwarded).
        """
        wrapped_input_ids = batch["wrapped_input_ids"].to(self.device)
        wrapped_attention_mask = batch["wrapped_attention_mask"].to(self.device)

        assert batch["start_index"].unique().numel() == 1, \
            "All items in a batch must have the same start_index."
        assert batch["clean_start_index"].unique().numel() == 1, \
            "All items in a batch must have the same clean_start_index."
        assert batch["clean_len"].unique().numel() == 1, \
            "All items in a batch must have the same clean_len."

        start_index = int(batch["start_index"][0].item())
        clean_start_index = int(batch["clean_start_index"][0].item())
        clean_len = int(batch["clean_len"][0].item())
        match_len = int(batch["match_len"][0].item()) if "match_len" in batch else clean_len

        adv_outputs = self._forward(wrapped_input_ids, wrapped_attention_mask)

        if self.needs_clean_pass:
            clean_input_ids = batch["clean_input_ids"].to(self.device)
            clean_attention_mask = batch["clean_attention_mask"].to(self.device)
            with torch.no_grad():
                if self.ref_model is not None:
                    clean_outputs = self.ref_model(
                        input_ids=clean_input_ids,
                        attention_mask=clean_attention_mask,
                        output_attentions=self.output_attentions,
                        output_hidden_states=self.output_hidden_states,
                    )
                else:
                    self.model.disable_adapter_layers()
                    clean_outputs = self._forward(clean_input_ids, clean_attention_mask)
                    self.model.enable_adapter_layers()
        else:
            clean_outputs = None

        wrapper_mask = batch.get("wrapper_mask")
        if wrapper_mask is not None:
            wrapper_mask = wrapper_mask.to(self.device)

        return self.loss_fn(
            clean_outputs=clean_outputs,
            adv_outputs=adv_outputs,
            start_index=start_index,
            clean_start_index=clean_start_index,
            clean_len=clean_len,
            match_len=match_len,
            wrapper_mask=wrapper_mask,
        )

    # ------------------------------------------------------------------
    # KL regularization step
    # ------------------------------------------------------------------

    def _kl_step(self, batch: dict) -> dict:
        """
        One KL regularization forward pass.

        Computes KL(π_current || π_base) over the full prompt.  For LoRA,
        base logits are obtained by disabling adapter layers under no_grad.
        Base logits are computed first (smaller memory footprint since no
        computation graph is retained), then current logits with grad.

        Returns the loss dict (loss not yet backwarded).
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # 1) Base model logits (no gradient, no computation graph).
        with torch.no_grad():
            if self.ref_model is not None:
                base_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            else:
                self.model.disable_adapter_layers()
                base_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                self.model.enable_adapter_layers()
        base_logits = base_outputs.logits.detach()

        # 2) Current model logits (with adapters, in computation graph).
        current_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )

        return self.kl_loss_fn(
            current_logits=current_outputs.logits,
            base_logits=base_logits,
            attention_mask=attention_mask,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        self.model.train()
        global_step = 0     # optimizer steps (post grad-accum)
        batch_count = 0     # micro-batches accumulated since last optimizer step
        self.optimizer.zero_grad()

        # KL dataloader cycles to match the AttCT dataloader.
        kl_iter = iter(self.kl_dataloader)

        for epoch in range(1, self.epochs + 1):
            epoch_attct_loss = 0.0
            epoch_kl_loss = 0.0
            n_steps = 0

            pbar = tqdm(
                self.attct_dataloader,
                desc=f"Epoch {epoch}",
                leave=False,
            )

            for step_idx, attct_batch in enumerate(pbar):
                if self.max_steps is not None and global_step >= self.max_steps:
                    break

                # ── AttCT backward (scaled for grad accumulation) ─────
                attct_loss_dict = self._attct_step(attct_batch)
                self._write_train_record(epoch, step_idx + 1, attct_batch)
                (attct_loss_dict["loss"] / self.grad_accumulation).backward()

                # ── KL backward (controlled by kl_ratio) ─────────────
                # kl_ratio=1.0 → always; 0.5 → every other step; 0.1 → ~1 in 10
                import random as _random
                run_kl = _random.random() < self.kl_ratio
                kl_loss_dict = None
                if run_kl:
                    try:
                        kl_batch = next(kl_iter)
                    except StopIteration:
                        kl_iter = iter(self.kl_dataloader)
                        kl_batch = next(kl_iter)
                    kl_loss_dict = self._kl_step(kl_batch)
                    (kl_loss_dict["loss"] / self.grad_accumulation).backward()

                batch_count += 1

                # ── Optimizer step every grad_accumulation micro-batches ──
                if batch_count % self.grad_accumulation == 0:
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    n_steps += 1

                # Bookkeeping (per micro-batch, undo the /grad_accum scaling)
                attct_val = attct_loss_dict["loss"].item()
                kl_val = kl_loss_dict["loss"].item() if kl_loss_dict is not None else 0.0
                epoch_attct_loss += attct_val
                epoch_kl_loss += kl_val

                postfix = {"attct": f"{attct_val:.4f}"}
                if run_kl:
                    postfix["kl"] = f"{kl_val:.4f}"
                pbar.set_postfix(**postfix)

                # ── Checkpoint ────────────────────────────────────────
                if global_step in self.checkpoint_steps and batch_count % self.grad_accumulation == 0:
                    self._save_checkpoint(tag=f"step_{global_step}")
                    if self.checkpoint_fn is not None:
                        print(f"\n[Checkpoint] Step {global_step} — running behavioral eval...")
                        self.checkpoint_fn(global_step)
                        self.model.train()

                # ── Logging (on optimizer steps only) ─────────────────
                if global_step > 0 and global_step % self.log_every == 0 and batch_count % self.grad_accumulation == 0:
                    self._log(epoch, global_step, attct_loss_dict, kl_loss_dict)

            # End of epoch summary
            n_steps = max(1, n_steps)
            avg_attct = epoch_attct_loss / n_steps
            avg_kl = epoch_kl_loss / n_steps
            print(
                f"Epoch {epoch} complete — "
                f"avg attct: {avg_attct:.4f}, avg kl_reg: {avg_kl:.4f}"
            )
            # Save epoch checkpoint so run_act.sh's *epoch_${EPOCHS}* glob finds
            # it. Mirrors train.py:280 — without this, post-eval phase fails to
            # locate the trained adapter and the chained pipeline aborts.
            self._save_checkpoint(tag=f"epoch_{epoch}")

            if self.max_steps is not None and global_step >= self.max_steps:
                break

        # ── End of training ───────────────────────────────────────────
        print("Training complete.")

        if self._first_layer_losses and self._last_layer_losses:
            print("\n── Per-layer loss change (first log → last log) ──")
            for i, (first, last) in enumerate(
                zip(self._first_layer_losses, self._last_layer_losses)
            ):
                delta = last - first
                arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "→")
                print(f"  Layer {i:02d}: {first:.4f} → {last:.4f}  ({arrow} {abs(delta):.4f})")
            total_first = sum(self._first_layer_losses)
            total_last = sum(self._last_layer_losses)
            total_arrow = "↓" if total_last < total_first else "↑"
            print(
                f"  {'Total':>8}: {total_first:.4f} → {total_last:.4f}  "
                f"({total_arrow} {abs(total_last - total_first):.4f})"
            )
            print("──────────────────────────────────────────────────")

        if self._log_io_file is not None:
            self._log_io_file.close()
        if self._train_log_file is not None:
            self._train_log_file.close()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, epoch: int, step: int, attct_dict: dict, kl_dict: dict):
        """Log both AttCT and KL metrics to W&B and stdout. kl_dict may be None
        when this iteration skipped the KL step (kl_ratio < 1.0)."""
        attct_val = attct_dict["loss"].item()
        kl_val = kl_dict["loss"].item() if kl_dict is not None else 0.0

        metrics = {
            "attct/loss": attct_val,
            "train/epoch": epoch,
        }
        if kl_dict is not None:
            metrics["kl/loss"] = kl_val
            metrics["train/total_loss"] = attct_val + kl_val

        # AttCT-specific metrics
        if "mean_layer_loss" in attct_dict:
            metrics["attct/mean_layer_loss"] = attct_dict["mean_layer_loss"]
        if "mean_wrapper_attention" in attct_dict:
            metrics["attct/mean_wrapper_attention"] = attct_dict["mean_wrapper_attention"]
        if "jsd_loss" in attct_dict:
            metrics["attct/jsd_loss"] = attct_dict["jsd_loss"]
        if "wrapper_loss" in attct_dict:
            metrics["attct/wrapper_loss"] = attct_dict["wrapper_loss"]

        # Per-layer breakdown
        if "layer_losses" in attct_dict:
            for i, ll in enumerate(attct_dict["layer_losses"]):
                metrics[f"attct/layer_{i:02d}_loss"] = ll
            if not self._first_layer_losses:
                self._first_layer_losses = list(attct_dict["layer_losses"])
            self._last_layer_losses = list(attct_dict["layer_losses"])

        # KL-specific diagnostics
        if kl_dict is not None:
            if "kl_div" in kl_dict:
                metrics["kl/div"] = kl_dict["kl_div"]
            if "mean_per_token_kl" in kl_dict:
                metrics["kl/per_token"] = kl_dict["mean_per_token_kl"]

        wandb.log(metrics, step=step)

        # Compact stdout line
        kl_str = f"{kl_val:.4f}" if kl_dict is not None else "skipped"
        line = (
            f"[epoch {epoch} | step {step}] "
            f"attct: {attct_val:.4f}  kl_reg: {kl_str}"
        )
        if "mean_layer_loss" in attct_dict:
            line += f"  mean_layer: {attct_dict['mean_layer_loss']:.4f}"
        if "mean_wrapper_attention" in attct_dict:
            line += f"  wrapper_attn: {attct_dict['mean_wrapper_attention']:.4f}"
        print(line)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str):
        if self.save_dir is None:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_prefix = f"{self.run_name}__" if self.run_name else ""
        folder_name = f"{run_prefix}{tag}__{timestamp}"
        path = os.path.join(self.save_dir, folder_name)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        print(f"Checkpoint saved to {path}")

        if self.hf_repo:
            local_path = path
            subfolder = folder_name

            def _push():
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    api.create_repo(repo_id=self.hf_repo, repo_type="model", exist_ok=True, private=True)
                    api.upload_folder(
                        folder_path=local_path,
                        repo_id=self.hf_repo,
                        path_in_repo=subfolder,
                        repo_type="model",
                    )
                    print(f"[HF] Uploaded checkpoint → {self.hf_repo}/{subfolder}")
                except Exception as exc:
                    print(f"[HF] Upload failed for {subfolder}: {exc}")

            _hf_upload_pool.submit(_push)
            print(f"[HF] Upload queued → {self.hf_repo}/{subfolder}")

    def _write_train_record(self, epoch: int, batch_idx: int, batch: dict):
        """Write one JSONL record per AttCT batch (same format as Trainer)."""
        if self._train_log_file is None or self.tokenizer is None:
            return
        clean_ids = batch["clean_input_ids"][0].tolist()
        wrapped_ids = batch["wrapped_input_ids"][0].tolist()
        record = {
            "epoch": epoch,
            "batch": batch_idx,
            "clean_text": self.tokenizer.decode(clean_ids, skip_special_tokens=False),
            "wrapped_text": self.tokenizer.decode(wrapped_ids, skip_special_tokens=False),
            "start_index": int(batch["start_index"][0].item()),
            "clean_start_index": int(batch["clean_start_index"][0].item()),
            "clean_len": int(batch["clean_len"][0].item()),
        }
        self._train_log_file.write(json.dumps(record) + "\n")
        self._train_log_file.flush()


class IntelligenceTrainer:
    """
    Training loop for the intelligence (control) condition.

    Trains exclusively on UltraChat via KL regularization — no AttCT consistency
    loss, no adversarial wrapping.  Used to check whether neutral fine-tuning
    incidentally affects sycophancy or jailbreak behavior.

    Args:
        model:           PEFT-wrapped HuggingFace model (on device).
        kl_dataloader:   DataLoader of UltraChat prompts (from get_kl_dataloader).
        kl_loss_fn:      Instantiated KLRegularizationLoss.
        config:          Full config dict.
        device:          torch.device.
        ref_model:       Optional frozen reference model (full FT mode).
        checkpoint_fn:   Optional callable(global_step) for behavioral eval checkpoints.
    """

    def __init__(
        self,
        model,
        kl_dataloader,
        kl_loss_fn,
        config: dict,
        device: torch.device,
        ref_model=None,
        checkpoint_fn=None,
        hf_repo=None,
        run_name=None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.kl_dataloader = kl_dataloader
        self.kl_loss_fn = kl_loss_fn
        self.config = config
        self.device = device
        self.checkpoint_fn = checkpoint_fn
        self.hf_repo = hf_repo
        self.run_name = run_name

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.max_steps = train_cfg.get("max_steps", None)
        self.grad_clip = train_cfg.get("grad_clip")
        self.log_every = train_cfg.get("log_every_n_steps", 10)
        self.save_dir = train_cfg.get("save_dir")

        dataset_steps = max(1, len(kl_dataloader) * self.epochs)
        if self.max_steps is not None:
            total_optimizer_steps = min(self.max_steps, dataset_steps)
        else:
            total_optimizer_steps = dataset_steps
        self.checkpoint_steps = {
            total_optimizer_steps // 3,
            (2 * total_optimizer_steps) // 3,
            total_optimizer_steps,
        }
        self.checkpoint_steps.discard(0)
        print(f"Behavioral eval checkpoints at optimizer steps: {sorted(self.checkpoint_steps)}")

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg["learning_rate"],
        )

    def _kl_step(self, batch: dict) -> dict:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            if self.ref_model is not None:
                base_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                self.model.disable_adapter_layers()
                base_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                self.model.enable_adapter_layers()
        base_logits = base_outputs.logits.detach()

        current_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        return self.kl_loss_fn(
            current_logits=current_outputs.logits,
            base_logits=base_logits,
            attention_mask=attention_mask,
        )

    def _save_checkpoint(self, tag: str):
        if self.save_dir is None:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_prefix = f"{self.run_name}__" if self.run_name else ""
        folder_name = f"{run_prefix}{tag}__{timestamp}"
        path = os.path.join(self.save_dir, folder_name)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        print(f"Checkpoint saved to {path}")

        if self.hf_repo:
            local_path = path
            subfolder = folder_name

            def _push():
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    api.create_repo(repo_id=self.hf_repo, repo_type="model", exist_ok=True, private=True)
                    api.upload_folder(
                        folder_path=local_path,
                        repo_id=self.hf_repo,
                        path_in_repo=subfolder,
                        repo_type="model",
                    )
                    print(f"[HF] Uploaded checkpoint → {self.hf_repo}/{subfolder}")
                except Exception as exc:
                    print(f"[HF] Upload failed for {subfolder}: {exc}")

            _hf_upload_pool.submit(_push)
            print(f"[HF] Upload queued → {self.hf_repo}/{subfolder}")

    def train(self):
        self.model.train()
        global_step = 0
        self.optimizer.zero_grad()

        for epoch in range(1, self.epochs + 1):
            epoch_kl_loss = 0.0
            n_steps = 0

            pbar = tqdm(self.kl_dataloader, desc=f"Epoch {epoch}", leave=False)

            for batch in pbar:
                if self.max_steps is not None and global_step >= self.max_steps:
                    break

                kl_loss_dict = self._kl_step(batch)
                kl_loss_dict["loss"].backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1
                n_steps += 1

                kl_val = kl_loss_dict["loss"].item()
                epoch_kl_loss += kl_val
                pbar.set_postfix(kl=f"{kl_val:.4f}")

                if global_step in self.checkpoint_steps:
                    self._save_checkpoint(tag=f"step_{global_step}")
                    if self.checkpoint_fn is not None:
                        print(f"\n[Checkpoint] Step {global_step} — running behavioral eval...")
                        self.checkpoint_fn(global_step)
                        self.model.train()

                if global_step % self.log_every == 0:
                    metrics = {
                        "kl/loss": kl_val,
                        "train/total_loss": kl_val,
                        "train/epoch": epoch,
                    }
                    if "kl_div" in kl_loss_dict:
                        metrics["kl/div"] = kl_loss_dict["kl_div"]
                    if "mean_per_token_kl" in kl_loss_dict:
                        metrics["kl/per_token"] = kl_loss_dict["mean_per_token_kl"]
                    wandb.log(metrics, step=global_step)
                    print(f"[epoch {epoch} | step {global_step}] kl: {kl_val:.4f}")

            n_steps = max(1, n_steps)
            print(f"Epoch {epoch} complete — avg kl: {epoch_kl_loss / n_steps:.4f}")

            if self.max_steps is not None and global_step >= self.max_steps:
                break

        print("Training complete.")