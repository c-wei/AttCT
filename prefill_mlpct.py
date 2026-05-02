"""
Prefill-MLPCT: BCT defense + MLP Consistency regularisation for prefill attacks.

Combined objective:
    total = bct_kl_loss + mlpct_weight * mlpct_loss

  bct_kl_loss   — KL(p_clean(.|prompt) || p_wrapped(.|prompt+prefill)) at and
                  after the divergence index. Pulls the wrapped output
                  distribution at every prefill position toward the clean
                  first-response-token distribution. This is the actual
                  prefill-attack defense signal — non-trivial because the
                  wrapped query positions can attend back to the prefill.
  mlpct_loss    — MLPConsistencyLoss on prompt-region MLP states (variant
                  "hidden" = input to down_proj, "output" = output of
                  down_proj). Acts as a regulariser anchoring the adapter's
                  prompt-region MLP behaviour to the base model.

The stock Trainer in train.py handles both the clean pass (via
disable_adapter_layers) and MLP hook installation — no custom trainer
needed, since the combined loss declares
    needs_clean_pass = True, needs_mlp_hooks = True

Training data: datasets/clearharm_prefills.csv (output of
prefill_generation_clearharm.py) — each row pairs one prompt with one
prefill of one of 23 strategy types. 80/20 train/val split.

Usage:
    uv run python prefill_mlpct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir checkpoints/prefill_mlpct \
        --mlpct_weight 1.0
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prefill_dataset import (
    PrefillAttackDataset,
    prefill_collate_fn,
    load_clearharm_prefills,
)
from losses.losses import MLPConsistencyLoss
from train import Trainer
from prefill_train import _clean_checkpoint_names

assert torch.cuda.is_available(), "CUDA not available — refusing to run on CPU"


# ---------------------------------------------------------------------------
# Combined loss: BCT KL (defense) + MLPCT (regulariser)
# ---------------------------------------------------------------------------

def _bct_kl_at_prefill(
    clean_logits: torch.Tensor,    # (B, Lc, V) — frozen reference (no grad)
    wrapped_logits: torch.Tensor,  # (B, Lw, V) — adapted (grad flows)
    clean_len: int,                # batch-uniform Lc
    kl_temperature: float = 1.0,
) -> torch.Tensor:
    """
    BCT-style KL on prefill output positions.

    Reference distribution = clean's first-response-token distribution at
    position cl-1. Compared against wrapped's distribution at every position
    [cl-1, Lw-2] (i.e. all token-prediction positions covering the prefill
    region). Assumes batch-uniform clean_len and unpadded sequences — both
    guaranteed by Trainer._step's batch-uniformity assertions.
    """
    _, Lc_pad, V = clean_logits.shape
    Lw = wrapped_logits.shape[1]
    div_idx = clean_len - 1

    if div_idx < 0 or div_idx >= Lc_pad or div_idx >= Lw - 1:
        return wrapped_logits.new_zeros(())

    # Frozen first-token distribution from clean (detach: defensive — already no_grad)
    ref      = clean_logits[:, div_idx, :].detach()
    p_clean  = F.softmax(ref / kl_temperature, dim=-1)              # (B, V)

    # Wrapped predictions at positions [div_idx, Lw-1) — every prefill output
    w_logits = wrapped_logits[:, div_idx : Lw - 1, :]                # (B, n, V)
    n        = w_logits.shape[1]
    if n == 0:
        return wrapped_logits.new_zeros(())
    log_q    = F.log_softmax(w_logits / kl_temperature, dim=-1)      # (B, n, V)

    # Broadcast p_clean to every prefill position, then KL(adv || clean)
    p_ref = p_clean.unsqueeze(1).expand(-1, n, -1).reshape(-1, V)
    log_q_flat = log_q.reshape(-1, V)
    return F.kl_div(log_q_flat, p_ref, reduction="batchmean")


class BCTPlusMLPCTLoss(nn.Module):
    """
    BCT KL on prefill output positions (defense) + MLPCT on prompt-region
    MLP states (regulariser). Stock Trainer wires both clean pass and MLP
    hooks because of the two flags below.
    """

    needs_clean_pass: bool = True
    needs_mlp_hooks:  bool = True

    def __init__(
        self,
        weight: float = 1.0,
        mlpct_weight: float = 1.0,
        kl_temperature: float = 1.0,
        # MLPCT sub-loss
        variant: str = "hidden",
        layer_selection: str = "all",
        layer_weights: str = "uniform",
        distance_metric: str = "cosine",
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.weight         = weight
        self.mlpct_weight   = mlpct_weight
        self.kl_temperature = kl_temperature
        # variant attribute used by Trainer to install correct MLPHookManager
        self.variant        = variant

        self._mlpct = MLPConsistencyLoss(
            weight=1.0,                # mlpct_weight applied outside
            variant=variant,
            layer_selection=layer_selection,
            layer_weights=layer_weights,
            distance_metric=distance_metric,
            normalize=normalize,
        )

    def forward(
        self,
        clean_outputs,
        adv_outputs,
        start_index: int,
        clean_start_index: int,
        clean_len: int,
        clean_mlp_states=None,
        adv_mlp_states=None,
        **kwargs,
    ):
        # ── BCT defense ────────────────────────────────────────────────
        # Cast to float32 — softmax/kl in bf16 can be unstable (esp. on CPU).
        bct = _bct_kl_at_prefill(
            clean_logits=clean_outputs.logits.float(),
            wrapped_logits=adv_outputs.logits.float(),
            clean_len=clean_len,
            kl_temperature=self.kl_temperature,
        )

        # ── MLPCT regulariser ──────────────────────────────────────────
        mlp_dict = self._mlpct(
            clean_outputs=clean_outputs,
            adv_outputs=adv_outputs,
            start_index=start_index,
            clean_start_index=clean_start_index,
            clean_len=clean_len,
            clean_mlp_states=clean_mlp_states,
            adv_mlp_states=adv_mlp_states,
        )
        mlpct = mlp_dict["loss"]

        total = bct + self.mlpct_weight * mlpct

        return {
            "loss":            self.weight * total,
            "bct_loss":        bct.item(),
            "mlpct_loss":      mlpct.item(),
            "layer_losses":    mlp_dict.get("layer_losses", []),
            "mean_layer_loss": mlp_dict.get("mean_layer_loss", 0.0),
            "num_layers_used": mlp_dict.get("num_layers_used", 0),
        }


# ---------------------------------------------------------------------------
# Dataset: paired (prompt, prefill) items, indices set so MLPCT compares
# the shared prompt-region MLP states.
# ---------------------------------------------------------------------------

class PrefillMLPCTDataset(PrefillAttackDataset):
    """
    Pairs prompts with prefills 1-to-1 (no Cartesian product) and overrides
    __getitem__ to set:
        start_index       = 0   (prompt starts at pos 0 in wrapped)
        clean_start_index = 0   (prompt starts at pos 0 in clean)
        clean_len         = Lc  (entire shared prompt region — unchanged)

    This makes MLPConsistencyLoss slice [0:Lc] from both clean and wrapped
    MLP states.
    """

    def __init__(
        self,
        prompts: list[str],
        prefills: list[str],
        tokenizer,
        max_length: int = 512,
    ):
        assert len(prompts) == len(prefills), (
            f"PrefillMLPCTDataset is paired: len(prompts)={len(prompts)} "
            f"!= len(prefills)={len(prefills)}"
        )
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.items = []
        for prompt, prefill in zip(prompts, prefills):
            clean_text   = self._build_prompt(prompt)
            wrapped_text = clean_text + prefill
            self.items.append((prompt, clean_text, wrapped_text))

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["start_index"]       = torch.tensor(0, dtype=torch.long)
        item["clean_start_index"] = torch.tensor(0, dtype=torch.long)
        return item


def make_dataloader(prompts, prefills, tokenizer, args, shuffle=True):
    dataset = PrefillMLPCTDataset(
        prompts, prefills, tokenizer, max_length=args.max_length,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=prefill_collate_fn,
    )


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(args) -> dict:
    return {
        "model": {
            "name": args.model,
            # MLPCT uses MLPHookManager — no need for attentions or hidden_states.
            "output_attentions": False,
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
            "name": "BCTPlusMLPCTLoss",
            "weight": args.loss_weight,
            "kwargs": {
                "mlpct_weight":    args.mlpct_weight,
                "kl_temperature":  args.kl_temperature,
                "variant":         args.variant,
                "layer_selection": args.layer_selection,
                "layer_weights":   args.layer_weights,
                "distance_metric": args.distance_metric,
                "normalize":       args.normalize,
            },
        },
        "data": {
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prefill-MLPCT training")
    parser.add_argument("--model",            required=True,  help="HF model name or path")
    parser.add_argument("--lora_path",        default=None,   help="Load existing LoRA adapter")
    parser.add_argument("--output_dir",       default="checkpoints/prefill_mlpct")

    # Loss
    parser.add_argument("--loss_weight",      type=float, default=1.0,
                        help="Global weight on (bct + mlpct_weight*mlpct)")
    parser.add_argument("--mlpct_weight",     type=float, default=1.0,
                        help="Weight of MLPCT regulariser relative to BCT defense")
    parser.add_argument("--kl_temperature",   type=float, default=1.0,
                        help="Softmax temperature for BCT KL")
    # MLPCT sub-loss kwargs
    parser.add_argument("--variant",          default="hidden",
                        choices=["hidden", "output"],
                        help="'hidden' = input to down_proj, 'output' = down_proj output")
    parser.add_argument("--distance_metric",  default="cosine",
                        choices=["cosine", "mse", "smooth_l1", "normalized_mse"])
    parser.add_argument("--layer_selection",  default="all",
                        choices=["all", "last", "middle", "last_half", "last_quarter"])
    parser.add_argument("--layer_weights",    default="uniform",
                        choices=["uniform", "linear_decay", "exponential_decay"])
    parser.add_argument("--normalize",        action="store_true", default=False,
                        help="L2-normalize MLP states before distance")

    # Training
    parser.add_argument("--num_epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",       type=int,   default=1)
    parser.add_argument("--grad_accumulation",type=int,   default=1)
    parser.add_argument("--lr",               type=float, default=5e-6)
    parser.add_argument("--grad_clip",        type=float, default=1.0)
    parser.add_argument("--max_steps",        type=int,   default=None)
    parser.add_argument("--log_every",        type=int,   default=10)
    parser.add_argument("--max_length",       type=int,   default=512)

    # LoRA — default targets include MLP modules so MLPCT actually has knobs
    # to turn (q_proj/v_proj alone would give the loss no direct MLP control).
    parser.add_argument("--lora_r",           type=int,   default=8)
    parser.add_argument("--lora_alpha",       type=int,   default=16)
    parser.add_argument("--lora_dropout",     type=float, default=0.05)
    parser.add_argument("--lora_targets",     nargs="+",
                        default=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])

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
        name=args.wandb_name or f"{model_short}_prefill_mlpct_{args.variant}_{args.distance_metric}",
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # MLPCT only needs MLP hooks — sdpa is the fastest backend.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
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
            target_modules=args.lora_targets,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    model = model.to(device)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading clearharm_prefills.csv...")
    train_prompts, val_prompts, train_prefills, val_prefills = load_clearharm_prefills(
        csv_path=args.csv_path,
        limit=args.limit,
    )
    print(f"Train: {len(train_prompts)} (prompt, prefill) pairs")
    print(f"Val:   {len(val_prompts)} (prompt, prefill) pairs")

    train_dl = make_dataloader(train_prompts, train_prefills, tokenizer, args, shuffle=True)
    val_dl   = make_dataloader(val_prompts,   val_prefills,   tokenizer, args, shuffle=False)
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_fn = BCTPlusMLPCTLoss(
        weight=args.loss_weight,
        mlpct_weight=args.mlpct_weight,
        kl_temperature=args.kl_temperature,
        variant=args.variant,
        layer_selection=args.layer_selection,
        layer_weights=args.layer_weights,
        distance_metric=args.distance_metric,
        normalize=args.normalize,
    )
    print(f"Loss: BCT(T={args.kl_temperature}) + "
          f"{args.mlpct_weight}*MLPCT(variant={args.variant}, "
          f"distance={args.distance_metric}, layers={args.layer_selection}, "
          f"normalize={args.normalize})")

    # ── Train ────────────────────────────────────────────────────────────────
    # Stock Trainer auto-installs MLPHookManager because loss_fn declares
    # needs_mlp_hooks = True. It also handles the clean pass via
    # disable_adapter_layers(), grad accumulation, checkpointing, W&B logging.
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

    _clean_checkpoint_names(args.output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
