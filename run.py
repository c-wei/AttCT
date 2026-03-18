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
    # ActivationConsistencyLoss,
    # BehavioralConsistencyLoss,
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
    beval.add_argument("--beval-max-samples", dest="beval_max_samples",   type=int, default=500,
                       help="Max examples per JSONL file during behavioral eval (default: 200).")
    beval.add_argument("--mmlu-max-samples",  dest="mmlu_max_samples",    type=int, default=200,
                       help="Number of MMLU test questions to evaluate (0 = disabled, default: 200).")
    beval.add_argument("--mmlu-subject",      dest="mmlu_subject",        default="all",
                       help="MMLU subject config (default: 'all'). E.g. 'high_school_mathematics'.")
    beval.add_argument("--gsm8k-max-samples", dest="gsm8k_max_samples",   type=int, default=200,
                       help="Number of GSM8K test questions to evaluate (0 = disabled, default: 200).")

    # Data source / mode overrides. These take precedence over the YAML.
    # For sycophancy runs, --control-cot already sets source+mode implicitly.
    # For clear-harm and hardcoded runs, pass --data-source explicitly.
    parser.add_argument("--data-source", dest="data_source", default=None,
                        help="Override config data.source (clear-harm | hardcoded | sycophancy_bct | <path>).")
    parser.add_argument("--data-mode",   dest="data_mode",   default=None,
                        choices=["jailbreak", "sycophancy"],
                        help="Override config data.mode (jailbreak | sycophancy).")
    parser.add_argument("--data-limit",  dest="data_limit",  default=None, type=int,
                        help="Cap the number of training prompts (overrides config data.limit).")

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

    data_source_tag = (
        args.data_source if args.data_source is not None
        else "sycophancy_bct" if args.control_cot_path is not None
        else config.get("data", {}).get("source", "unknown")
    )
    wandb.init(project="AttCT", name=loss_name, group="week2", tags=[data_source_tag], config=config)

    print(f"Loss: {loss_cfg['name']} | Device: {device}")
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wire data source/mode into config.
    # Priority: --data-source/--data-mode > --control-cot (sycophancy shorthand) > YAML.
    if args.data_source is not None:
        config.setdefault("data", {})["source"] = args.data_source
    elif args.control_cot_path is not None:
        config.setdefault("data", {})["source"] = args.control_cot_path
    if args.data_mode is not None:
        config.setdefault("data", {})["mode"] = args.data_mode
    elif args.control_cot_path is not None and args.data_source is None:
        config.setdefault("data", {})["mode"] = "sycophancy"
    if args.data_limit is not None:
        config.setdefault("data", {})["limit"] = args.data_limit

    # Log wrapper config now that data mode is finalised.
    wrapper_mode = config.get("data", {}).get("mode", "jailbreak")
    wrapper_templates = "sycophancy" if wrapper_mode == "sycophancy" else "jailbreak_strong"
    wandb.config.update({"wrapper": {"mode": wrapper_mode, "templates": wrapper_templates}},
                        allow_val_change=True)

    # Build BehavioralEvaluator if any eval is enabled.
    beval_paths = [args.bct_cot_path, args.bct_noncot_path, args.control_cot_path, args.control_noncot_path]
    has_sycophancy_paths = all(p is not None for p in beval_paths)
    has_mmlu  = args.mmlu_max_samples > 0
    has_gsm8k = args.gsm8k_max_samples > 0
    behavioral_evaluator = None
    if has_sycophancy_paths or has_mmlu or has_gsm8k:
        config["behavioral_eval"] = {
            "bct_cot_path":        args.bct_cot_path,
            "bct_noncot_path":     args.bct_noncot_path,
            "control_cot_path":    args.control_cot_path,
            "control_noncot_path": args.control_noncot_path,
            "max_samples":         args.beval_max_samples,
            "mmlu_max_samples":    args.mmlu_max_samples,
            "mmlu_subject":        args.mmlu_subject,
            "gsm8k_max_samples":   args.gsm8k_max_samples,
        }
        behavioral_evaluator = BehavioralEvaluator(model, tokenizer, config, device)
        features = []
        if has_sycophancy_paths:
            features.append("sycophancy BCT")
        if has_mmlu:
            features.append(f"MMLU ({args.mmlu_max_samples} samples, subject={args.mmlu_subject})")
        if has_gsm8k:
            features.append(f"GSM8K ({args.gsm8k_max_samples} samples)")
        print(f"Behavioral evaluator configured [{', '.join(features)}] — will run at 3 checkpoints during training.")
    else:
        print("No behavioral eval configured. Pass --bct-cot/... for sycophancy, --mmlu-max-samples N for MMLU, or --gsm8k-max-samples N for GSM8K.")

    # Namespace log dir by loss + data source so sweep runs don't overwrite each other.
    import os
    _base_log_dir = config.get("logging", {}).get("log_dir", "logs")
    _data_source_tag = (
        args.data_source if args.data_source is not None
        else ("sycophancy_bct" if args.control_cot_path is not None else "unknown")
    )
    log_dir = os.path.join(_base_log_dir, f"{loss_name}__{_data_source_tag}")
    os.makedirs(log_dir, exist_ok=True)
    # Also update config so train.py writes training_data.jsonl into the same dir.
    config.setdefault("logging", {})["log_dir"] = log_dir

    def make_checkpoint_fn(evaluator):
        """Return a closure that passes a per-step log_path to evaluate()."""
        if evaluator is None:
            return None
        def _fn(step):
            log_path = os.path.join(log_dir, f"beval_step_{step}.jsonl")
            evaluator.evaluate(global_step=step, log_path=log_path)
        return _fn

    Trainer(
        model,
        get_dataloader(config, split="train"),
        loss_fn,
        config,
        device,
        log_io_path=args.log_io,
        tokenizer=tokenizer,
        checkpoint_fn=make_checkpoint_fn(behavioral_evaluator),
    ).train()

    Evaluator(model, get_dataloader(config, split="eval"), loss_fn, config, device).evaluate()
    wandb.finish()

if __name__ == "__main__":
    main()