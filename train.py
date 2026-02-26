import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


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
        model:      A PEFT-wrapped HuggingFace model (already on device).
        dataloader: DataLoader yielding batches in the format above.
        loss_fn:    An instantiated ConsistencyLoss subclass.
        config:     The full config dict (from config.yaml).
        device:     torch.device to run on.
    """

    def __init__(self, model, dataloader: DataLoader, loss_fn, config: dict, device: torch.device):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.config = config
        self.device = device

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.grad_clip = train_cfg.get("grad_clip")
        self.log_every = train_cfg.get("log_every_n_steps", 10)

        model_cfg = config["model"]
        self.output_attentions = model_cfg.get("output_attentions", True)
        self.output_hidden_states = model_cfg.get("output_hidden_states", False)

        # Losses that don't need a clean forward pass declare needs_clean_pass=False
        self.needs_clean_pass = loss_fn.needs_clean_pass

        self.save_dir = train_cfg.get("save_dir")

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg["learning_rate"]
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
        # Assert uniform within batch — if your data pipeline produces heterogeneous
        # start_index/clean_len values, batches must be grouped by wrapper length.
        assert batch["start_index"].unique().numel() == 1,             "All items in a batch must have the same start_index. Group by wrapper length."
        assert batch["clean_len"].unique().numel() == 1,             "All items in a batch must have the same clean_len. Group by wrapper length."
        start_index = int(batch["start_index"][0].item())
        clean_len   = int(batch["clean_len"][0].item())

        adv_outputs = self._forward(wrapped_input_ids, wrapped_attention_mask)

        if self.needs_clean_pass:
            clean_input_ids      = batch["clean_input_ids"].to(self.device)
            clean_attention_mask = batch["clean_attention_mask"].to(self.device)
            clean_outputs        = self._forward(clean_input_ids, clean_attention_mask)
        else:
            clean_outputs = None

        wrapper_mask = batch.get("wrapper_mask")
        if wrapper_mask is not None:
            wrapper_mask = wrapper_mask.to(self.device)

        return self.loss_fn(
            clean_outputs=clean_outputs,
            adv_outputs=adv_outputs,
            start_index=start_index,
            clean_len=clean_len,
            wrapper_mask=wrapper_mask,
        )

    def train(self):
        self.model.train()
        global_step = 0

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0

            for batch in self.dataloader:
                self.optimizer.zero_grad()

                loss_dict = self._step(batch)
                loss = loss_dict["loss"]

                loss.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()
                global_step += 1
                epoch_loss += loss.item()

                if global_step % self.log_every == 0:
                    self._log(epoch, global_step, loss_dict)

            avg = epoch_loss / len(self.dataloader)
            print(f"Epoch {epoch} complete — avg loss: {avg:.4f}")
            self._save_checkpoint(epoch)

        print("Training complete.")

    def _save_checkpoint(self, epoch: int):
        if self.save_dir is None:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"epoch_{epoch}")
        self.model.save_pretrained(path)
        print(f"Checkpoint saved to {path}")

    def _log(self, epoch: int, step: int, loss_dict: dict):
        loss_val = loss_dict["loss"].item()
        mean_layer = loss_dict.get("mean_layer_loss")
        wrapper_att = loss_dict.get("mean_wrapper_attention")

        line = f"[epoch {epoch} | step {step}] loss: {loss_val:.4f}"
        if mean_layer is not None:
            line += f"  mean_layer_loss: {mean_layer:.4f}"
        if wrapper_att is not None:
            line += f"  mean_wrapper_attn: {wrapper_att:.4f}"
        print(line)