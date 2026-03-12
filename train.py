import os
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import json


class Trainer:
    """
    Minimal training loop for AttCT consistency losses.

    Expects batches with the following keys (produced by the data pipeline):
        clean_input_ids       : LongTensor [batch, clean_seq_len]
        clean_attention_mask  : LongTensor [batch, clean_seq_len]
        wrapped_input_ids     : LongTensor [batch, wrapped_seq_len]
        wrapped_attention_mask: LongTensor [batch, wrapped_seq_len]
        start_index           : LongTensor [batch]
        clean_len             : LongTensor [batch]

    Optional key (only needed for WrapperEntropyRegularizationLoss):
        wrapper_mask          : BoolTensor [batch, wrapped_seq_len]

    Args:
        model:           A PEFT-wrapped HuggingFace model (already on device).
        dataloader:      DataLoader yielding batches in the format above.
        loss_fn:         An instantiated ConsistencyLoss subclass.
        config:          The full config dict (from config.yaml).
        device:          torch.device to run on.
        checkpoint_fn:   Optional callable(global_step: int) -> None.
                         Called at three evenly-spaced steps during training:
                         ~33%, ~66%, and 100% of total optimizer steps.
    """

    def __init__(
        self,
        model,
        dataloader: DataLoader,
        loss_fn,
        config: dict,
        device: torch.device,
        checkpoint_fn=None,
        log_io_path=None,
        tokenizer=None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.tokenizer = tokenizer

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.grad_clip = train_cfg.get("grad_clip")
        self.log_every = train_cfg.get("log_every_n_steps", 10)
        self.grad_accumulation = train_cfg.get("grad_accumulation_steps", 1)

        model_cfg = config["model"]
        self.output_attentions = model_cfg.get("output_attentions", True)
        self.output_hidden_states = model_cfg.get("output_hidden_states", False)

        self.needs_clean_pass = loss_fn.needs_clean_pass
        self.save_dir = train_cfg.get("save_dir")
        self.checkpoint_fn = checkpoint_fn

        # IO logging — open file handle if path provided, else None
        self._log_io_file = open(log_io_path, "w") if log_io_path is not None else None

        # Training data log — one JSONL record per batch.
        # Written to logs/training_data.jsonl so every example fed to the
        # model (clean text, wrapped text, boundary indices, loss) is auditable.
        self._train_log_file = None
        if self.tokenizer is not None:
            log_dir = config.get("logging", {}).get("log_dir", "logs")
            os.makedirs(log_dir, exist_ok=True)
            train_log_path = os.path.join(log_dir, "training_data.jsonl")
            self._train_log_file = open(train_log_path, "w")
            print(f"Training data log: {train_log_path}")

        # For end-of-training per-layer delta summary.
        # Populated on first and last log step that contains layer_losses.
        self._first_layer_losses: list = []
        self._last_layer_losses:  list = []

        # Compute the three step indices at which behavioral eval fires.
        # Total optimizer steps = (batches_per_epoch * epochs) / grad_accumulation.
        # We use integer division throughout; the final checkpoint is always the
        # last optimizer step, so behavioral eval always runs at end-of-training.
        batches_per_epoch = len(dataloader)
        total_batches = batches_per_epoch * self.epochs
        total_optimizer_steps = max(1, total_batches // self.grad_accumulation)
        self.checkpoint_steps = {
            total_optimizer_steps // 3,
            (2 * total_optimizer_steps) // 3,
            total_optimizer_steps,
        }
        # Remove zero in case total_optimizer_steps < 3
        self.checkpoint_steps.discard(0)
        print(f"Behavioral eval checkpoints at optimizer steps: {sorted(self.checkpoint_steps)}")

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg["learning_rate"],
        )

    def _forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

    def _step(self, batch):
        wrapped_input_ids      = batch["wrapped_input_ids"].to(self.device)
        wrapped_attention_mask = batch["wrapped_attention_mask"].to(self.device)
        assert batch["start_index"].unique().numel() == 1, \
            "All items in a batch must have the same start_index. Group by wrapper length."
        assert batch["clean_start_index"].unique().numel() == 1, \
            "All items in a batch must have the same clean_start_index. Group by wrapper length."
        assert batch["clean_len"].unique().numel() == 1, \
            "All items in a batch must have the same clean_len. Group by wrapper length."
        start_index       = int(batch["start_index"][0].item())
        clean_start_index = int(batch["clean_start_index"][0].item())
        clean_len         = int(batch["clean_len"][0].item())

        adv_outputs = self._forward(wrapped_input_ids, wrapped_attention_mask)
        self._write_io_record("wrapped", wrapped_input_ids, adv_outputs.logits)

        if self.needs_clean_pass:
            clean_input_ids      = batch["clean_input_ids"].to(self.device)
            clean_attention_mask = batch["clean_attention_mask"].to(self.device)
            clean_outputs = self._forward(clean_input_ids, clean_attention_mask)
            self._write_io_record("clean", clean_input_ids, clean_outputs.logits)
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
            wrapper_mask=wrapper_mask,
        )

    def train(self):
        self.model.train()
        global_step = 0       # counts optimizer (weight update) steps
        batch_count = 0       # counts individual batches (for grad accumulation)
        self.optimizer.zero_grad()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=False)

            for batch in pbar:
                loss_dict = self._step(batch)
                self._write_train_record(epoch, batch_count + 1, batch)
                loss = loss_dict["loss"] / self.grad_accumulation
                loss.backward()
                batch_count += 1

                if batch_count % self.grad_accumulation == 0:
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Fire checkpoint callback at the three designated steps.
                    # The model is saved first so the checkpoint corresponds to
                    # the weights that behavioral eval will measure.
                    if self.checkpoint_fn is not None and global_step in self.checkpoint_steps:
                        self._save_checkpoint(tag=f"step_{global_step}")
                        print(f"\n[Checkpoint] Step {global_step} — running behavioral eval...")
                        self.checkpoint_fn(global_step)
                        self.model.train()  # restore train mode after eval

                    if global_step % self.log_every == 0:
                        self._log(epoch, global_step, loss_dict)

                loss_val = loss_dict["loss"].item()
                epoch_loss += loss_val
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            avg = epoch_loss / len(self.dataloader)
            print(f"Epoch {epoch} complete — avg loss: {avg:.4f}")

        print("Training complete.")
        if self._first_layer_losses and self._last_layer_losses:
            print("\n── Per-layer loss change (first log → last log) ──")
            for i, (first, last) in enumerate(zip(self._first_layer_losses, self._last_layer_losses)):
                delta = last - first
                direction = "↓" if delta < 0 else ("↑" if delta > 0 else "→")
                print(f"  Layer {i:02d}: {first:.4f} → {last:.4f}  ({direction} {abs(delta):.4f})")
            total_first = sum(self._first_layer_losses)
            total_last  = sum(self._last_layer_losses)
            print(f"  {'Total':>8}: {total_first:.4f} → {total_last:.4f}  ({'↓' if total_last < total_first else '↑'} {abs(total_last - total_first):.4f})")
            print("──────────────────────────────────────────────────")
        if self._log_io_file is not None:
            self._log_io_file.close()
        if self._train_log_file is not None:
            self._train_log_file.close()

    def _write_io_record(self, label: str, input_ids: torch.Tensor, logits: torch.Tensor):
        """
        Decode one forward-pass input and its per-position greedy predictions,
        append as a JSONL record to the log file.

        Note: logits[0].argmax(dim=-1) gives next-token predictions at every
        input position simultaneously — not an autoregressive continuation.
        This is useful for debugging tokenization and attention slicing, but
        should not be interpreted as the model's response to the prompt.
        """
        if self._log_io_file is None or self.tokenizer is None:
            return
        import json
        ids = input_ids[0].tolist()
        input_text = self.tokenizer.decode(ids, skip_special_tokens=False)
        response_ids = logits[0].argmax(dim=-1).tolist()
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        record = {"pass": label, "input": input_text, "response": response_text}
        self._log_io_file.write(json.dumps(record) + "\n")
        self._log_io_file.flush()

    def _write_train_record(
        self,
        epoch: int,
        batch_idx: int,
        batch: dict,
    ):
        """
        Write one JSONL record to logs/training_data.jsonl for every batch.

        Fields:
          epoch, batch        — position in training
          clean_text          — decoded clean prompt (no special tokens)
          wrapped_text        — decoded wrapped prompt (no special tokens)
          start_index         — token position where content starts in wrapped seq
          clean_start_index   — token position where content starts in clean seq
          clean_len           — number of content tokens
        """
        if self._train_log_file is None or self.tokenizer is None:
            return
        clean_ids   = batch["clean_input_ids"][0].tolist()
        wrapped_ids = batch["wrapped_input_ids"][0].tolist()
        record = {
            "epoch":             epoch,
            "batch":             batch_idx,
            "clean_text":        self.tokenizer.decode(clean_ids,   skip_special_tokens=False),
            "wrapped_text":      self.tokenizer.decode(wrapped_ids, skip_special_tokens=False),
            "start_index":       int(batch["start_index"][0].item()),
            "clean_start_index": int(batch["clean_start_index"][0].item()),
            "clean_len":         int(batch["clean_len"][0].item()),
        }
        self._train_log_file.write(json.dumps(record) + "\n")
        self._train_log_file.flush()

    def _save_checkpoint(self, tag: str):

        """Save a LoRA checkpoint. Tag is used as the subdirectory name."""
        if self.save_dir is None:
            return
        path = os.path.join(self.save_dir, tag)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        print(f"Checkpoint saved to {path}")

    def _log(self, epoch: int, step: int, loss_dict: dict):
        loss_val = loss_dict["loss"].item()

        metrics = {
            "train/loss": loss_val,
            "train/epoch": epoch,
        }
        if "mean_layer_loss" in loss_dict:
            metrics["train/mean_layer_loss"] = loss_dict["mean_layer_loss"]
        if "mean_wrapper_attention" in loss_dict:
            metrics["train/mean_wrapper_attention"] = loss_dict["mean_wrapper_attention"]
        if "kl_loss" in loss_dict:
            metrics["train/kl_loss"] = loss_dict["kl_loss"]
        if "output_loss" in loss_dict:
            metrics["train/l2_loss"] = loss_dict["output_loss"]
        if "jsd_loss" in loss_dict:
            metrics["train/jsd_loss"] = loss_dict["jsd_loss"]
        if "wrapper_loss" in loss_dict:
            metrics["train/wrapper_loss"] = loss_dict["wrapper_loss"]
        if "top1_agreement" in loss_dict:
            metrics["train/top1_agreement"] = loss_dict["top1_agreement"]
        if "js_divergence" in loss_dict:
            metrics["train/js_divergence"] = loss_dict["js_divergence"]
        if "num_layers_used" in loss_dict:
            metrics["train/num_layers_used"] = loss_dict["num_layers_used"]

        # Per-layer breakdown — keys match eval/layer_XX_loss in the Evaluator
        # so they can be overlaid on the same W&B chart panel.
        if "layer_losses" in loss_dict:
            for i, layer_loss in enumerate(loss_dict["layer_losses"]):
                metrics[f"train/layer_{i:02d}_loss"] = layer_loss
            # Track first and last observation for end-of-training delta summary.
            if not self._first_layer_losses:
                self._first_layer_losses = list(loss_dict["layer_losses"])
            self._last_layer_losses = list(loss_dict["layer_losses"])

        wandb.log(metrics, step=step)

        line = f"[epoch {epoch} | step {step}] loss: {loss_val:.4f}"
        if "mean_layer_loss" in loss_dict:
            line += f"  mean_layer_loss: {loss_dict['mean_layer_loss']:.4f}"
        if "num_layers_used" in loss_dict:
            line += f"  layers: {loss_dict['num_layers_used']}"
        if "mean_wrapper_attention" in loss_dict:
            line += f"  mean_wrapper_attn: {loss_dict['mean_wrapper_attention']:.4f}"
        if "top1_agreement" in loss_dict:
            line += f"  top1_agree: {loss_dict['top1_agreement']:.4f}"
        if "js_divergence" in loss_dict:
            line += f"  js_div: {loss_dict['js_divergence']:.4f}"
        print(line)