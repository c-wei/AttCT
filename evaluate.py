import torch
from torch.utils.data import DataLoader

class Evaluator:
    """
    Runs the loss function over an eval DataLoader with no gradient updates
    and reports per-layer attention diagnostics averaged across all batches.

    Args:
        model:      A PEFT-wrapped HuggingFace model (already on device).
        dataloader: Eval DataLoader in the same format as the training DataLoader.
        loss_fn:    An instantiated ConsistencyLoss subclass.
        config:     The full config dict (from config.yaml).
        device:     torch.device to run on.
    """

    def __init__(self, model, dataloader: DataLoader, loss_fn, config: dict, device: torch.device):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device

        model_cfg = config["model"]
        self.output_attentions = model_cfg.get("output_attentions", True)
        self.output_hidden_states = model_cfg.get("output_hidden_states", False)
        self.needs_clean_pass = loss_fn.needs_clean_pass

    def _forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()

        total_loss = 0.0
        # Accumulate per-layer losses across batches
        all_layer_losses = []
        total_wrapper_attn = 0.0
        has_wrapper_attn = False

        for batch in self.dataloader:
            wrapped_input_ids      = batch["wrapped_input_ids"].to(self.device)
            wrapped_attention_mask = batch["wrapped_attention_mask"].to(self.device)
            assert batch["start_index"].unique().numel() == 1,                 "All items in a batch must have the same start_index. Group by wrapper length."
            assert batch["clean_len"].unique().numel() == 1,                 "All items in a batch must have the same clean_len. Group by wrapper length."
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

            loss_dict = self.loss_fn(
                clean_outputs=clean_outputs,
                adv_outputs=adv_outputs,
                start_index=start_index,
                clean_len=clean_len,
                wrapper_mask=wrapper_mask,
            )

            total_loss += loss_dict["loss"].item()

            if "layer_losses" in loss_dict:
                all_layer_losses.append(loss_dict["layer_losses"])

            if "mean_wrapper_attention" in loss_dict:
                total_wrapper_attn += loss_dict["mean_wrapper_attention"]
                has_wrapper_attn = True

        n_batches = len(self.dataloader)
        results = {"mean_loss": total_loss / n_batches}

        if all_layer_losses:
            n_layers = len(all_layer_losses[0])
            results["mean_layer_losses"] = [
                sum(batch_layers[i] for batch_layers in all_layer_losses) / n_batches
                for i in range(n_layers)
            ]

        if has_wrapper_attn:
            results["mean_wrapper_attention"] = total_wrapper_attn / n_batches

        self._report(results)
        return results

    def _report(self, results: dict):
        print(f"\n--- Eval Results [{self.loss_fn.__class__.__name__}] ---")
        print(f"  mean_loss: {results['mean_loss']:.4f}")

        if "mean_layer_losses" in results:
            print("  per_layer_losses:")
            for i, l in enumerate(results["mean_layer_losses"]):
                print(f"    layer {i:02d}: {l:.4f}")

        if "mean_wrapper_attention" in results:
            print(f"  mean_wrapper_attention: {results['mean_wrapper_attention']:.4f}")
        print()