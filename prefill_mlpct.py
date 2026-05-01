"""
Prefill-MLPCT: MLP Consistency Training for prefill attacks.

Uses MLPConsistencyLoss (in losses/losses.py) to enforce consistent MLP
representations between the clean prompt (no prefill) and the wrapped
prompt (prompt + prefill). Compared to ACT (which compares residual stream
hidden states), MLPCT compares either:
    variant="hidden"  → input to down_proj (post-activation MLP neurons)
    variant="output"  → output of down_proj

The stock Trainer in train.py auto-installs an MLPHookManager because
MLPConsistencyLoss declares `needs_mlp_hooks = True` — no custom trainer
required, and forward passes do not need to materialise hidden_states or
attentions, saving memory.

Training data: datasets/clearharm_prefills.csv (output of
prefill_generation_clearharm.py) — each row pairs one prompt with one
prefill of one of 23 strategy types. 80/20 train/val split.

Note on prefill caveat
----------------------
The shared comparison region is the prompt itself (positions [0, Lc)).
Causal masking means tokens in this region cannot see the prefill at
[Lc, Lw), so MLP-state differences in this region come solely from LoRA
weight changes — the loss is a regulariser keeping the adapted model's
prompt processing close to the base model.

Usage:
    uv run python prefill_mlpct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir checkpoints/prefill_mlpct
"""

import argparse

import torch
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

assert torch.cuda.is_available(), "CUDA not available — refusing to run on CPU"


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
            "name": "MLPConsistencyLoss",
            "weight": args.loss_weight,
            "kwargs": {
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
                        help="Global weight for MLP consistency loss")
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
    loss_fn = MLPConsistencyLoss(
        weight=args.loss_weight,
        variant=args.variant,
        layer_selection=args.layer_selection,
        layer_weights=args.layer_weights,
        distance_metric=args.distance_metric,
        normalize=args.normalize,
    )
    print(f"Loss: MLPConsistencyLoss(w={args.loss_weight}, variant={args.variant}, "
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

    wandb.finish()


if __name__ == "__main__":
    main()
