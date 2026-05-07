"""
prefill_train.py — unified training entry point for prefill-attack defenses.

Dispatches to the correct (dataset, loss, trainer) tuple based on --mode.
Reuses dataset classes, loss functions, and trainers from the per-mode
scripts — no logic is duplicated here.

Modes
-----
  bct    Bias-Augmented Consistency Training. KL(p_clean || p_wrapped) at
         and after the divergence index, optionally regularised by SFT on
         mrfakename/refusal pairs. Custom PrefillBCTTrainer (in
         prefill_bct.py).
  act    Activation Consistency. MSE on prompt-region residual stream
         hidden states between clean (no prefill) and wrapped (with
         prefill) passes. Stock Trainer.
  attct  Attention Consistency via wrapper-entropy suppression. Penalises
         attention mass flowing to prefill positions, anchored by KL
         regularisation on UltraChat. InterleavedTrainer.
  mlpct  BCT KL defense at prefill positions + MLPConsistencyLoss
         regulariser on prompt-region MLP states. Stock Trainer.

All modes share the same data source: datasets/clearharm_prefills.csv,
80/20 train/val.

Usage
-----
  python prefill_train.py --mode bct \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --output_dir checkpoints/prefill_bct

  python prefill_train.py --mode act \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --output_dir checkpoints/prefill_act

  python prefill_train.py --mode attct \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --output_dir checkpoints/prefill_attct \\
      --kl_dataset ultrachat --kl_weight 1.0

  python prefill_train.py --mode mlpct \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --output_dir checkpoints/prefill_mlpct \\
      --mlpct_weight 1000
"""

import argparse
import os

import torch
import wandb
import yaml
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data.prefill_dataset import (
    prefill_collate_fn,
    load_clearharm_prefills,
)
from data.ultrachat_dataset import get_kl_dataloader

# Reuse per-mode dataset classes, losses, and trainers
from prefill_act   import PrefillACTDataset
from prefill_attct import PrefillAttCTDataset, prefill_attct_collate_fn
from prefill_bct   import PrefillPairedDataset, PrefillBCTTrainer, load_refusal_pairs
from prefill_mlpct import PrefillMLPCTDataset, BCTPlusMLPCTLoss

from losses.losses           import (
    ActivationConsistencyLoss,
    WrapperEntropyRegularizationLoss,
    JSDAttentionConsistencyLoss,
    CombinedJSDWrapperLoss,
)
from losses.kl_regularization import KLRegularizationLoss

from train               import Trainer
from interleaved_trainer  import InterleavedTrainer

assert torch.cuda.is_available(), "CUDA not available — refusing to run on CPU"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _default_config_path(mode: str) -> str:
    return os.path.join(os.path.dirname(__file__), "configs", "prefill", f"{mode}.yaml")


def _load_yaml_defaults(parser: argparse.ArgumentParser, mode: str, config_path: str | None):
    """Load per-mode YAML and update parser defaults so CLI > YAML > argparse defaults."""
    path = config_path or _default_config_path(mode)
    if not os.path.exists(path):
        print(f"[config] No config at {path} — using built-in argparse defaults")
        return path

    with open(path) as f:
        yaml_dict = yaml.safe_load(f) or {}

    # Filter to argparse arg names so unknown YAML keys don't raise
    valid = {a.dest for a in parser._actions}
    extra = set(yaml_dict) - valid
    if extra:
        print(f"[config] Ignoring unknown keys in {path}: {sorted(extra)}")
    yaml_dict = {k: v for k, v in yaml_dict.items() if k in valid}

    parser.set_defaults(**yaml_dict)
    print(f"[config] Loaded defaults from {path}")
    return path


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_dataloader(mode, prompts, prefills, tokenizer, args, shuffle):
    """Construct the mode-specific paired dataset + collate fn."""
    if mode == "act":
        ds = PrefillACTDataset(prompts, prefills, tokenizer, max_length=args.max_length)
        collate = prefill_collate_fn
    elif mode == "mlpct":
        ds = PrefillMLPCTDataset(prompts, prefills, tokenizer, max_length=args.max_length)
        collate = prefill_collate_fn
    elif mode == "attct":
        ds = PrefillAttCTDataset(prompts, prefills, tokenizer, max_length=args.max_length)
        collate = prefill_attct_collate_fn
    elif mode == "bct":
        ds = PrefillPairedDataset(prompts, prefills, tokenizer, max_length=args.max_length)
        collate = prefill_collate_fn
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate,
    )


def build_loss(mode, args):
    """Construct the loss function for ACT / AttCT / MLPCT.
    BCT mode returns None — PrefillBCTTrainer hard-codes its own KL loss."""
    if mode == "act":
        return ActivationConsistencyLoss(
            weight=args.loss_weight,
            layer_selection=args.layer_selection,
            normalize=args.normalize,
        )
    if mode == "mlpct":
        return BCTPlusMLPCTLoss(
            weight=args.loss_weight,
            mlpct_weight=args.mlpct_weight,
            kl_temperature=args.kl_temperature,
            variant=args.variant,
            layer_selection=args.layer_selection,
            layer_weights=args.layer_weights,
            distance_metric=args.distance_metric,
            normalize=args.normalize,
        )
    if mode == "attct":
        # --attct_loss_type picks which attention-consistency loss to use
        if args.attct_loss_type == "wrapper":
            return WrapperEntropyRegularizationLoss(
                weight=args.loss_weight,
                normalize=args.normalize,
                layer_weights=args.layer_weights,
            )
        if args.attct_loss_type == "jsd":
            return JSDAttentionConsistencyLoss(
                weight=args.loss_weight,
                layer_weights=args.layer_weights,
                layer_selection=args.layer_selection,
            )
        if args.attct_loss_type == "combined":
            return CombinedJSDWrapperLoss(
                weight=args.loss_weight,
                jsd_weight=args.jsd_weight,
                wrapper_weight=args.wrapper_weight,
            )
        raise ValueError(f"Unknown attct_loss_type: {args.attct_loss_type!r}")
    if mode == "bct":
        return None
    raise ValueError(f"Unknown mode: {mode}")


def build_config(mode, args):
    return {
        "model": {
            "name": args.model,
            "output_attentions":   mode == "attct",
            "output_hidden_states": mode == "act",  # mlpct uses hooks, not hidden_states
        },
        "training": {
            "epochs":                  args.num_epochs,
            "max_steps":               args.max_steps,
            "learning_rate":           args.lr,
            "grad_clip":               args.grad_clip,
            "log_every_n_steps":       args.log_every,
            "grad_accumulation_steps": args.grad_accumulation,
            "save_dir":                args.output_dir,
        },
        "loss": {"name": f"prefill_{mode}", "weight": args.loss_weight},
        "data": {"batch_size": args.batch_size, "max_length": args.max_length},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified prefill-defense training")
    parser.add_argument("--mode",              required=True,
                        choices=["bct", "act", "attct", "mlpct"])
    parser.add_argument("--config",            default=None,
                        help="Path to YAML config (default: configs/prefill/{mode}.yaml). "
                             "CLI flags override YAML; YAML overrides argparse defaults.")
    parser.add_argument("--model",             default=None,  help="HF model name or path (required; YAML or CLI)")
    parser.add_argument("--lora_path",         default=None,  help="Resume from LoRA adapter")
    parser.add_argument("--output_dir",        default=None,  help="Checkpoint dir (required; YAML or CLI)")
    parser.add_argument("--attn_impl",         default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"],
                        help="HF attn_implementation (YAML usually sets this per-mode)")

    # Loss weights (interpreted per mode)
    parser.add_argument("--loss_weight",       type=float, default=1.0,
                        help="Global loss multiplier for the chosen mode")
    parser.add_argument("--mlpct_weight",      type=float, default=1.0,
                        help="(mlpct) MLP regulariser weight relative to BCT KL")
    parser.add_argument("--kl_temperature",    type=float, default=1.0,
                        help="(bct, mlpct) softmax temperature for KL")
    parser.add_argument("--sft_coeff",         type=float, default=0.1,
                        help="(bct) SFT regularisation coefficient (0 disables refusal SFT)")

    # ACT / MLPCT layer + distance knobs
    parser.add_argument("--layer_selection",   default="all",
                        choices=["all", "last", "middle", "last_half", "last_quarter"])
    parser.add_argument("--layer_weights",     default="uniform",
                        choices=["uniform", "linear_decay", "exponential_decay"])
    parser.add_argument("--variant",           default="hidden",
                        choices=["hidden", "output"],
                        help="(mlpct) which MLP tensor to compare")
    parser.add_argument("--distance_metric",   default="cosine",
                        choices=["cosine", "mse", "smooth_l1", "normalized_mse"],
                        help="(mlpct) distance between aligned MLP states")
    parser.add_argument("--normalize",         action="store_true", default=False,
                        help="L2-normalize states before distance (act, mlpct)")

    # AttCT: choose attention-consistency loss flavour
    parser.add_argument("--attct_loss_type",   default="wrapper",
                        choices=["wrapper", "jsd", "combined"],
                        help="(attct) wrapper = WrapperEntropyRegularization, "
                             "jsd = JSDAttentionConsistency, combined = both")
    parser.add_argument("--jsd_weight",        type=float, default=0.5,
                        help="(attct combined) JSD component weight")
    parser.add_argument("--wrapper_weight",    type=float, default=0.5,
                        help="(attct combined) Wrapper component weight")

    # AttCT: KL anchor on UltraChat
    parser.add_argument("--kl_weight",         type=float, default=1.0,
                        help="(attct) UltraChat KL regularisation weight")
    parser.add_argument("--kl_samples",        type=int,   default=None,
                        help="(attct) UltraChat sample count (default: match AttCT dataset size)")
    parser.add_argument("--kl_dataset",        default="ultrachat",
                        choices=["ultrachat", "alpaca"],
                        help="(attct) Source for KL anchor")
    parser.add_argument("--kl_ratio",          type=float, default=1.0,
                        help="(attct) Fraction of AttCT steps that fire a KL step")

    # Training
    parser.add_argument("--num_epochs",        type=int,   default=3)
    parser.add_argument("--batch_size",        type=int,   default=1)
    parser.add_argument("--grad_accumulation", type=int,   default=1)
    parser.add_argument("--lr",                type=float, default=5e-6)
    parser.add_argument("--grad_clip",         type=float, default=1.0)
    parser.add_argument("--max_steps",         type=int,   default=None)
    parser.add_argument("--log_every",         type=int,   default=10)
    parser.add_argument("--max_length",        type=int,   default=512)

    # Quantization (QLoRA — base model in 4/8-bit, LoRA adapters in fp32 on top)
    parser.add_argument("--quantize",          default="none",
                        choices=["none", "4bit", "8bit"],
                        help="Load base model in 4-bit (nf4) or 8-bit via bitsandbytes. "
                             "Required to fit 27B+ on smaller GPUs. Drops weights ~4× (4bit) or ~2× (8bit).")

    # LoRA
    parser.add_argument("--lora_r",            type=int,   default=8)
    parser.add_argument("--lora_alpha",        type=int,   default=16)
    parser.add_argument("--lora_dropout",      type=float, default=0.05)
    parser.add_argument("--lora_targets",      nargs="+",  default=["q_proj", "v_proj"],
                        help="LoRA target_modules (per-mode YAML usually overrides this)")

    # Data
    parser.add_argument("--hf_dataset",        default="carolinewei/ClearHarm_prefills",
                        help="HuggingFace dataset id for prefill pairs (used unless --csv_path set)")
    parser.add_argument("--hf_split",          default="train",
                        help="Split of the HF dataset to load")
    parser.add_argument("--csv_path",          default=None,
                        help="Local CSV override — if set, ignores --hf_dataset")
    parser.add_argument("--train_ratio",       type=float, default=0.8,
                        help="Fraction of pairs used for training; remainder is val")
    parser.add_argument("--limit",             type=int,   default=None,
                        help="Max (prompt, prefill) rows to load")

    # W&B
    parser.add_argument("--wandb_project",     default="AttCT")
    parser.add_argument("--wandb_name",        default=None)

    # ── YAML defaults: parse twice so CLI > YAML > argparse defaults ─────
    # First parse only --mode and --config (both required to find the YAML);
    # parse_known_args lets unknown flags through without raising.
    prelim, _ = parser.parse_known_args()
    _load_yaml_defaults(parser, prelim.mode, prelim.config)
    args = parser.parse_args()
    mode = args.mode

    # `model` and `output_dir` may come from YAML or CLI, but must be set somewhere.
    if not args.model:
        parser.error("--model must be set via CLI or YAML")
    if not args.output_dir:
        parser.error("--output_dir must be set via CLI or YAML")

    attn_impl = args.attn_impl
    config = build_config(mode, args)

    model_short = args.model.split("/")[-1]
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{model_short}_prefill_{mode}",
        config={**config, "mode": mode, "lora_targets": args.lora_targets},
    )

    device = torch.device("cuda")

    # ── Model ────────────────────────────────────────────────────────────
    print(f"[{mode}] Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # device_map="auto" + low_cpu_mem_usage stream weights directly to GPU
    # shard-by-shard instead of materialising the full model in CPU RAM
    # first. Critical for 27B+ models in containers with limited host RAM
    # (a 27B bf16 model is ~54GB of weights — default loading OOMs CPU).
    # --quantize 4bit / 8bit further drops weight footprint via bitsandbytes
    # (QLoRA pattern). When set, bnb_4bit_compute_dtype controls math dtype,
    # so torch_dtype is dropped to avoid conflicting signals.
    load_kwargs = dict(
        attn_implementation=attn_impl,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if args.quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print(f"[{mode}] Loading base in 4-bit (nf4) via bitsandbytes")
    elif args.quantize == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print(f"[{mode}] Loading base in 8-bit via bitsandbytes")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    # QLoRA bookkeeping: cast layer norms / LM head to fp32 and enable
    # gradient checkpointing so backward through frozen 4-bit weights works.
    if args.quantize != "none":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

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
    # model = model.to(device)  # device_map="auto" already placed weights

    # ── Data ─────────────────────────────────────────────────────────────
    src = f"CSV {args.csv_path}" if args.csv_path else f"HF {args.hf_dataset}:{args.hf_split}"
    print(f"Loading prefill pairs from {src} (train_ratio={args.train_ratio})")
    train_prompts, val_prompts, train_prefills, val_prefills = load_clearharm_prefills(
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        csv_path=args.csv_path,
        limit=args.limit,
        train_ratio=args.train_ratio,
    )
    print(f"Train: {len(train_prompts)} pairs | Val: {len(val_prompts)} pairs")

    train_dl = build_dataloader(mode, train_prompts, train_prefills, tokenizer, args, shuffle=True)
    val_dl   = build_dataloader(mode, val_prompts,   val_prefills,   tokenizer, args, shuffle=False)
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

    # ── Trainer dispatch ─────────────────────────────────────────────────
    if mode == "bct":
        # BCT has its own KL loss + (optional) refusal SFT regulariser baked
        # into PrefillBCTTrainer; no separate loss_fn needed.
        sft_pairs = load_refusal_pairs(limit=None) if args.sft_coeff > 0 else None
        if sft_pairs is not None:
            print(f"SFT pairs: {len(sft_pairs)}")

        trainer = PrefillBCTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_dl,
            val_loader=val_dl,
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

    elif mode == "attct":
        # AttCT pairs wrapper-entropy suppression (on AttCT data) with KL
        # regularisation on neutral UltraChat, interleaved per optimizer step.
        loss_fn    = build_loss(mode, args)
        kl_loss_fn = KLRegularizationLoss(
            weight=args.kl_weight,
            temperature=args.kl_temperature,
        )
        n_kl = args.kl_samples if args.kl_samples is not None else len(train_dl.dataset)
        kl_dl = get_kl_dataloader(
            config, tokenizer, n_samples=n_kl, kl_dataset=args.kl_dataset,
        )
        print(f"KL anchor: {len(kl_dl.dataset)} {args.kl_dataset} samples "
              f"(weight={args.kl_weight}, T={args.kl_temperature}, ratio={args.kl_ratio})")

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

    else:
        # ACT and MLPCT — stock Trainer auto-handles clean pass and (for
        # MLPCT) MLPHookManager based on loss_fn flags.
        loss_fn = build_loss(mode, args)
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

        print("\nRunning eval on val set...")
        trainer.eval_loss(val_dl)

    wandb.finish()


if __name__ == "__main__":
    main()
