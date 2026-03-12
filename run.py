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
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    if args.config != "config.yaml":
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        config = _deep_merge(config, {k: v for k, v in overrides.items() if k != "defaults"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_lora = bool(config.get("lora"))
    ref_model = None
    model_name = config["model"]["name"]

    if is_lora:
        lora_cfg = config["lora"]
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager")
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
        ))
        model.print_trainable_parameters()
    else:
        # Full fine-tuning: all params trainable; load a separate frozen ref model for clean pass (= θ_init)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager")
        ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager")
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters())
        print(f"Full FT: {trainable:,} trainable parameters")

    loss_cfg = config["loss"]
    loss_name = loss_cfg["name"]
    loss_kwargs = loss_cfg.get("kwargs", {})
    loss_kwargs["output_hidden_states"] = config["model"].get("output_hidden_states", False)
    loss_fn = LOSS_REGISTRY[loss_name](weight=loss_cfg.get("weight", 1.0), **loss_kwargs)

    model_short = config["model"]["name"].split("/")[-1]
    data_mode   = config.get("data", {}).get("mode", "jailbreak")
    lr          = config["training"]["learning_rate"]
    weight      = loss_cfg.get("weight", 1.0)
    run_name    = f"{model_short}_{data_mode}_lr{lr}_w{weight}_{args.config.split('/')[-1].replace('.yaml','')}"
    wandb.init(project="AttCT", name=run_name, config=config)

    print(f"Loss: {loss_cfg['name']} | Device: {device}")
    model = model.to(device)
    if ref_model is not None:
        ref_model = ref_model.to(device)

    is_sycophancy = config.get("data", {}).get("mode") == "sycophancy"
    is_sanity = config.get("data", {}).get("limit") is not None

    # Pre-training baseline: evaluate θ_init (base model, no LoRA / no FT updates yet)
    if is_sycophancy and not is_sanity:
        from evaluate_sycophancy import SycophancyEvaluator
        from transformers import AutoTokenizer
        import os as _os
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        run_label = _os.path.splitext(_os.path.basename(args.config))[0]
        results_csv = f"/workspace/results/{run_label}_syco_results.csv"
        print("\n=== Pre-training baseline (base model) ===")
        if is_lora:
            model.disable_adapter_layers()
            model.eval()
            SycophancyEvaluator(model, tokenizer, device, prefix="pre_train",
                                results_csv=results_csv).evaluate()
            model.enable_adapter_layers()
        else:
            SycophancyEvaluator(ref_model, tokenizer, device, prefix="pre_train",
                                results_csv=results_csv).evaluate()

    Trainer(model, get_dataloader(config, split="train"), loss_fn, config, device,
            ref_model=ref_model).train()
    Evaluator(model, get_dataloader(config, split="eval"), loss_fn, config, device).evaluate()

    # Post-training evaluation
    if is_sycophancy and not is_sanity:
        print("\n=== Post-training evaluation (trained LoRA model) ===")
        model.eval()
        SycophancyEvaluator(model, tokenizer, device, prefix="post_train",
                            results_csv=results_csv).evaluate()

    wandb.finish()

if __name__ == "__main__":
    main()