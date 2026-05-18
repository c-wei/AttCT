"""Entry point for the triple-sum (ACT + AttCT + MLPCT) sycophancy experiment.

Self-contained launcher: imports the model/LoRA/dataloader/evaluator
infrastructure from the main AttCT package, but defines its own
SummedTripleLoss (./triple_loss.py) and a tiny Trainer subclass that
forwards per-term sub-metrics to W&B.

Usage:
    python experiments/triple_sum/run_triple.py \\
        --config experiments/triple_sum/configs/llama31_8b.yaml \\
        --run-name llama31_8b_triple_sum \\
        --wandb-group triple_sum_v1
"""

from __future__ import annotations

# Run from the AttCT repo root: `python experiments/triple_sum/run_triple.py ...`
# Add the repo root to sys.path so `from losses.losses ...` works regardless of cwd.
import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import json
from typing import Optional

import torch
import wandb
import yaml
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.attct_datasets import get_dataloader
from evaluate_sycophancy import SycophancyEvaluator
from train import Trainer

from experiments.triple_sum.triple_loss import SummedTripleLoss


# Keys we expect from SummedTripleLoss.forward() — used to extend Trainer._log
_TRIPLE_SUB_KEYS = (
    "act/loss", "attct/loss", "mlpct/loss",
    "act/weighted_loss", "attct/weighted_loss", "mlpct/weighted_loss",
    "act/mean_layer_loss", "attct/mean_layer_loss", "mlpct/mean_layer_loss",
)


class TripleSumTrainer(Trainer):
    """Trainer that also forwards triple_loss sub-metrics to W&B + stdout."""

    def _log(self, epoch: int, step: int, loss_dict: dict):
        super()._log(epoch, step, loss_dict)
        p = self._phase_label_for(step)
        w_step = self.global_step_offset + step
        extra = {f"{p}/{k}": loss_dict[k] for k in _TRIPLE_SUB_KEYS if k in loss_dict}
        if extra:
            wandb.log(extra, step=w_step)
            # Compact stdout breakdown so the per-term contributions are visible
            print(
                f"  └ act={extra[f'{p}/act/weighted_loss']:.4g}  "
                f"attct={extra[f'{p}/attct/weighted_loss']:.4g}  "
                f"mlpct={extra[f'{p}/mlpct/weighted_loss']:.4g}"
            )


def _build_model_and_tokenizer(model_name: str, lora_cfg: dict, device: torch.device):
    """Load model (eager attention, bf16), wrap with LoRA, load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # eager is required because JSD AttCT needs attention weights;
        # SDPA does not expose them.
        attn_implementation="eager",
    )
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
    ))
    model.print_trainable_parameters()
    return model.to(device), tokenizer


def _run_sycophancy_eval(model, tokenizer, device, prefix: str, run_label: str,
                        held_out_path: Optional[str], max_samples: int,
                        disable_adapters: bool):
    csv_path = os.path.join("results", f"{run_label}_syco_results.csv")
    print(f"\n=== Sycophancy eval — prefix='{prefix}' (adapters {'OFF' if disable_adapters else 'ON'}) ===")
    if disable_adapters:
        model.disable_adapter_layers()
    model.eval()
    SycophancyEvaluator(
        model, tokenizer, device,
        prefix=prefix,
        results_csv=csv_path,
        max_samples=max_samples,
        held_out_path=held_out_path,
        anthropic_eval=True,
    ).evaluate()
    if disable_adapters:
        model.enable_adapter_layers()
    model.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-group", default="triple_sum_v1")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override training.max_steps from config.")
    p.add_argument("--eval-limit", type=int, default=1000,
                   help="Max MMLU / held-out / Anthropic samples per eval.")
    p.add_argument("--skip-pre-eval",  action="store_true")
    p.add_argument("--skip-post-eval", action="store_true")
    p.add_argument("--held-out", default="datasets/sycophancy_bct/control_cot_eval.jsonl",
                   help="Path to held-out OOD eval JSONL. Set to '' to skip.")
    p.add_argument("--w-act",   type=float, default=None, help="Override w_act from config.")
    p.add_argument("--w-attct", type=float, default=None, help="Override w_attct from config.")
    p.add_argument("--w-mlp",   type=float, default=None, help="Override w_mlp from config.")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps

    tl_cfg = config.get("triple_loss", {})
    if args.w_act   is not None: tl_cfg["w_act"]   = args.w_act
    if args.w_attct is not None: tl_cfg["w_attct"] = args.w_attct
    if args.w_mlp   is not None: tl_cfg["w_mlp"]   = args.w_mlp

    # Force the model forward to emit everything ACT + AttCT + MLPCT need.
    config.setdefault("model", {})
    config["model"]["output_attentions"]   = True
    config["model"]["output_hidden_states"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_label = os.path.splitext(os.path.basename(args.config))[0]
    run_name = args.run_name or f"triple_sum_{run_label}"

    wandb.init(project="AttCT", name=run_name, group=args.wandb_group, config=config)

    model, tokenizer = _build_model_and_tokenizer(
        config["model"]["name"], config["lora"], device,
    )

    # Pre-train baseline (base model, adapters disabled).
    held_out_path = args.held_out if args.held_out else None
    if not args.skip_pre_eval:
        _run_sycophancy_eval(model, tokenizer, device,
                             prefix="pre_train", run_label=run_label,
                             held_out_path=held_out_path,
                             max_samples=args.eval_limit,
                             disable_adapters=True)

    loss_fn = SummedTripleLoss(
        w_act=tl_cfg.get("w_act", 1.0),
        w_attct=tl_cfg.get("w_attct", 1.0),
        w_mlp=tl_cfg.get("w_mlp", 1.0),
        act_normalize=tl_cfg.get("act_normalize", True),
        act_loss_formulation=tl_cfg.get("act_loss_formulation", "paper"),
        mlp_distance_metric=tl_cfg.get("mlp_distance_metric", "cosine"),
        mlp_normalize=tl_cfg.get("mlp_normalize", True),
    ).to(device)
    print(
        f"SummedTripleLoss weights:  w_act={loss_fn.w_act}  "
        f"w_attct={loss_fn.w_attct}  w_mlp={loss_fn.w_mlp}"
    )

    config.setdefault("data", {})
    config["data"].setdefault("source", "sycophancy_bct")
    config["data"].setdefault("mode", "sycophancy")
    dataloader = get_dataloader(config, split="train")

    trainer = TripleSumTrainer(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        config=config,
        device=device,
        ref_model=None,            # LoRA path: clean pass uses disable_adapter_layers()
        checkpoint_fn=None,         # no mid-training behavioural evals; only pre + post
        log_io_path=None,
        tokenizer=tokenizer,
        hf_repo=None,
        run_name=run_name,
        phase_label="triple_sum",
    )

    trainer.train()

    if not args.skip_post_eval:
        _run_sycophancy_eval(model, tokenizer, device,
                             prefix="post_train", run_label=run_label,
                             held_out_path=held_out_path,
                             max_samples=args.eval_limit,
                             disable_adapters=False)

    wandb.finish()


if __name__ == "__main__":
    main()
