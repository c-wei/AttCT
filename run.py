import argparse
import yaml
import torch
import wandb
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from losses.losses import (
    AttentionConsistencyLoss,
    AttentionConsistencyLossV2,
    JSDAttentionConsistencyLoss,
    AttentionOutputConsistencyLoss,
    CombinedAttentionConsistencyLoss,
    WrapperEntropyRegularizationLoss,
    CombinedJSDWrapperLoss,
    ActivationConsistencyLoss,
    BehavioralConsistencyLoss,
)
from data import get_dataloader
from train import Trainer
from evaluate import Evaluator

LOSS_REGISTRY = {
    "AttentionConsistencyLoss":         AttentionConsistencyLoss,
    "AttentionConsistencyLossV2":       AttentionConsistencyLossV2,
    "JSDAttentionConsistencyLoss":      JSDAttentionConsistencyLoss,
    "AttentionOutputConsistencyLoss":   AttentionOutputConsistencyLoss,
    "CombinedAttentionConsistencyLoss": CombinedAttentionConsistencyLoss,
    "WrapperEntropyRegularizationLoss": WrapperEntropyRegularizationLoss,
    "CombinedJSDWrapperLoss":           CombinedJSDWrapperLoss,
    "ActivationConsistencyLoss":        ActivationConsistencyLoss,
    "BehavioralConsistencyLoss":        BehavioralConsistencyLoss,
}

def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        merged[k] = _deep_merge(merged[k], v) if isinstance(merged.get(k), dict) and isinstance(v, dict) else v
    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to a saved LoRA checkpoint to load")
    parser.add_argument("--run-name", default=None, help="W&B run name (default: loss class name)")
    parser.add_argument("--wandb-group", default=None, help="W&B group for organising related runs")
    parser.add_argument("--wandb-run-id", default=None, help="W&B run ID to resume (share a run across multiple scripts)")
    parser.add_argument("--metric-prefix", default="eval/", help="Prefix for eval metric keys logged to W&B")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the post-training evaluation pass")
    parser.add_argument("--save-dir", default=None, help="Override training.save_dir for checkpoints")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    if args.config != "config.yaml":
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        config = _deep_merge(config, {k: v for k, v in overrides.items() if k != "defaults"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lora_cfg = config["lora"]
    base_model = AutoModelForCausalLM.from_pretrained(config["model"]["name"], torch_dtype=torch.bfloat16, attn_implementation="eager")

    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.checkpoint, is_trainable=True)
        print(f"Loaded LoRA checkpoint from {args.checkpoint}")
    else:
        model = get_peft_model(base_model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
        ))
    model.print_trainable_parameters()

    loss_cfg = config["loss"]
    loss_name = loss_cfg["name"]
    loss_kwargs = loss_cfg.get("kwargs", {})
    loss_kwargs["output_hidden_states"] = config["model"].get("output_hidden_states", False)
    loss_fn = LOSS_REGISTRY[loss_name](weight=loss_cfg.get("weight", 1.0), **loss_kwargs)

    if args.save_dir:
        config.setdefault("training", {})["save_dir"] = args.save_dir

    wandb.init(
        project="AttCT",
        name=args.run_name or loss_name,
        group=args.wandb_group,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        config=config,
    )

    print(f"Loss: {loss_cfg['name']} | Device: {device}")
    model = model.to(device)
    if config.get("training", {}).get("epochs", 1) > 0:
        Trainer(model, get_dataloader(config, split="train"), loss_fn, config, device).train()
    if not args.skip_eval:
        Evaluator(model, get_dataloader(config, split="eval"), loss_fn, config, device, metric_prefix=args.metric_prefix).evaluate()
    wandb.finish()

if __name__ == "__main__":
    main()