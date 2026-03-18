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
    ActivationConsistencyLoss,
    SFTLoss,
)
from data import get_dataloader
from data.attct_datasets import get_bct_dataloader
from train import Trainer, BCTTrainer
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
    "SFTLoss":                          SFTLoss,
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
    needs_attn_weights = config["model"].get("output_attentions", True)
    if needs_attn_weights:
        attn_impl = "eager"          # FA2 cannot return attention weights
    else:
        try:
            import flash_attn        # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"       # torch built-in, no install required
    print(f"attn_implementation: {attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"], dtype=torch.bfloat16, attn_implementation=attn_impl
    )
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

    model_short = config["model"]["name"].split("/")[-1]
    if loss_name == "SFTLoss":
        data_mode = "sft"
    else:
        data_mode = config.get("data", {}).get("mode", "jailbreak")
    lr          = config["training"]["learning_rate"]
    weight      = loss_cfg.get("weight", 1.0)
    run_name    = f"{model_short}_{data_mode}_lr{lr}_w{weight}_{args.config.split('/')[-1].replace('.yaml', '')}"
    wandb.init(project="AttCT", name=run_name, config=config)

    print(f"Loss: {loss_cfg['name']} | Device: {device}")
    model = model.to(device)

    if isinstance(loss_fn, SFTLoss):
        train_dl    = get_bct_dataloader(config, split="train")
        eval_dl     = get_bct_dataloader(config, split="eval")
        bct_trainer = BCTTrainer(model, train_dl, loss_fn, config, device)
        bct_trainer.train()
        bct_trainer.eval_loss(eval_dl)
    else:
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

        # Namespace log dir by loss + data source so sweep runs don't overwrite each other.
        import os
        _base_log_dir = config.get("logging", {}).get("log_dir", "logs")
        _data_source_tag = (
            args.data_source if args.data_source is not None
            else ("sycophancy_bct" if args.control_cot_path is not None else "unknown")
        )
        log_dir = os.path.join(_base_log_dir, f"{loss_name}__{_data_source_tag}")
        os.makedirs(log_dir, exist_ok=True)
        config.setdefault("logging", {})["log_dir"] = log_dir

        def make_checkpoint_fn(evaluator):
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