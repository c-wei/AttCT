import argparse
import yaml
import torch
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
}

def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        merged[k] = _deep_merge(merged[k], v) if isinstance(merged.get(k), dict) and isinstance(v, dict) else v
    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    if args.config != "config.yaml":
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        config = _deep_merge(config, {k: v for k, v in overrides.items() if k != "defaults"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lora_cfg = config["lora"]
    model = AutoModelForCausalLM.from_pretrained(config["model"]["name"], torch_dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
    ))
    model.print_trainable_parameters()

    loss_cfg = config["loss"]
    loss_fn = LOSS_REGISTRY[loss_cfg["name"]](weight=loss_cfg.get("weight", 1.0), **loss_cfg.get("kwargs", {}))

    print(f"Loss: {loss_cfg['name']} | Device: {device}")
    model = model.to(device)
    Trainer(model, get_dataloader(config, split="train"), loss_fn, config, device).train()
    Evaluator(model, get_dataloader(config, split="eval"), loss_fn, config, device).evaluate()

if __name__ == "__main__":
    main()