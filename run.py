import argparse
import yaml
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
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
from behavioral_evaluate import BehavioralEvaluator

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
    parser.add_argument("--log-io", metavar="FILE", default=None,
                        help="Write every model input (clean + wrapped) and its greedy-decoded "
                             "response to FILE as JSONL. One JSON object per forward pass.")

    # Behavioral eval JSONL paths. All four must be provided together to enable
    # behavioral evaluation at the three training checkpoints. If any are omitted,
    # training runs normally without behavioral eval.
    beval = parser.add_argument_group("behavioral_eval")
    beval.add_argument("--bct-cot",           dest="bct_cot_path",        default=None,
                       help="Path to bct_cot.jsonl (wrapped, chain-of-thought).")
    beval.add_argument("--bct-noncot",        dest="bct_noncot_path",     default=None,
                       help="Path to bct_non_cot.jsonl (wrapped, direct answer).")
    beval.add_argument("--control-cot",       dest="control_cot_path",    default=None,
                       help="Path to control_cot.jsonl (clean, chain-of-thought).")
    beval.add_argument("--control-noncot",    dest="control_noncot_path", default=None,
                       help="Path to control_non_cot.jsonl (clean, direct answer).")
    beval.add_argument("--beval-max-samples", dest="beval_max_samples",   type=int, default=200,
                       help="Max examples per JSONL file during behavioral eval (default: 200).")

    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    if args.config != "config.yaml":
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        config = _deep_merge(config, {k: v for k, v in overrides.items() if k != "defaults"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lora_cfg = config["lora"]
    model = AutoModelForCausalLM.from_pretrained(config["model"]["name"], torch_dtype=torch.bfloat16, attn_implementation="eager")
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
    loss_name = loss_cfg["name"]
    loss_kwargs = loss_cfg.get("kwargs", {})
    loss_kwargs["output_hidden_states"] = config["model"].get("output_hidden_states", False)
    loss_fn = LOSS_REGISTRY[loss_name](weight=loss_cfg.get("weight", 1.0), **loss_kwargs)

    wandb.init(project="AttCT", name=loss_name, config=config)

    print(f"Loss: {loss_cfg['name']} | Device: {device}")
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build BehavioralEvaluator only if all four JSONL paths were provided.
    beval_paths = [args.bct_cot_path, args.bct_noncot_path, args.control_cot_path, args.control_noncot_path]
    behavioral_evaluator = None
    if all(p is not None for p in beval_paths):
        config["behavioral_eval"] = {
            "bct_cot_path":        args.bct_cot_path,
            "bct_noncot_path":     args.bct_noncot_path,
            "control_cot_path":    args.control_cot_path,
            "control_noncot_path": args.control_noncot_path,
            "max_samples":         args.beval_max_samples,
        }
        behavioral_evaluator = BehavioralEvaluator(model, tokenizer, config, device)
        print("Behavioral evaluator configured — will run at 3 checkpoints during training.")
    else:
        print("No behavioral eval paths provided. Pass all four --bct-cot/--bct-noncot/--control-cot/--control-noncot to enable.")

    Trainer(
        model,
        get_dataloader(config, split="train"),
        loss_fn,
        config,
        device,
        log_io_path=args.log_io,
        tokenizer=tokenizer,
        checkpoint_fn=(
            lambda step: behavioral_evaluator.evaluate(global_step=step)
            if behavioral_evaluator is not None else None
        ),
    ).train()

    Evaluator(model, get_dataloader(config, split="eval"), loss_fn, config, device).evaluate()
    wandb.finish()

if __name__ == "__main__":
    main()