import argparse
import os
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
    MLPConsistencyLoss,
    SFTLoss,
)
from data import get_dataloader
from data.attct_datasets import get_bct_dataloader
from train import Trainer, BCTTrainer
from evaluate import Evaluator
from behavioral_evaluate import BehavioralEvaluator
from evaluate_brr import BRREvaluator
from evaluate_asr import ASREvaluator

LOSS_REGISTRY = {
    "AttentionConsistencyLoss":         AttentionConsistencyLoss,
    "AttentionConsistencyLossV2":       AttentionConsistencyLossV2,
    "JSDAttentionConsistencyLoss":      JSDAttentionConsistencyLoss,
    "AttentionOutputConsistencyLoss":   AttentionOutputConsistencyLoss,
    "CombinedAttentionConsistencyLoss": CombinedAttentionConsistencyLoss,
    "WrapperEntropyRegularizationLoss": WrapperEntropyRegularizationLoss,
    "CombinedJSDWrapperLoss":           CombinedJSDWrapperLoss,
    "ActivationConsistencyLoss":        ActivationConsistencyLoss,
    "MLPConsistencyLoss":               MLPConsistencyLoss,
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
    beval.add_argument("--mmlu-max-samples",  dest="mmlu_max_samples",    type=int, default=200,
                       help="Number of MMLU test questions to evaluate (0 = disabled, default: 200).")
    beval.add_argument("--mmlu-subject",      dest="mmlu_subject",        default="all",
                       help="MMLU subject config (default: 'all'). E.g. 'high_school_mathematics'.")
    beval.add_argument("--gsm8k-max-samples", dest="gsm8k_max_samples",   type=int, default=200,
                       help="Number of GSM8K test questions to evaluate (0 = disabled, default: 200).")

    # BRR evaluation — uses held-out clean prompts, wraps on-the-fly.
    parser.add_argument("--brr-eval-path", dest="brr_eval_path", default=None,
                        help="Path to control_cot_eval.jsonl for BRR evaluation. "
                             "Enables BRR eval at pre-train, checkpoints, and post-train.")

    # ASR evaluation — for jailbreak experiments.
    parser.add_argument("--asr-eval", dest="asr_eval", action="store_true", default=False,
                        help="Enable ASR (Attack Success Rate) evaluation for jailbreak experiments. "
                             "Runs at pre-train, checkpoints, and post-train.")
    parser.add_argument("--asr-max-prompts", dest="asr_max_prompts", type=int, default=200,
                        help="Max harmful prompts for ASR eval (default: 200).")
    parser.add_argument("--asr-max-new-tokens", dest="asr_max_new_tokens", type=int, default=256,
                        help="Max tokens to generate per response in ASR eval (default: 256).")

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
    parser.add_argument("--eval-limit",  dest="eval_limit",  default=None, type=int,
                        help="Cap the number of eval prompts for the end-of-training Evaluator.")

    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    if args.config != "config.yaml":
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        config = _deep_merge(config, {k: v for k, v in overrides.items() if k != "defaults"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model loading ─────────────────────────────────────────────────────────
    is_lora = bool(config.get("lora"))
    ref_model = None
    model_name = config["model"]["name"]

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

    if is_lora:
        lora_cfg = config["lora"]
        load_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
        quant_cfg = config.get("quantization")
        if quant_cfg and quant_cfg.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
            )
            print("Loading model in 4-bit (QLoRA mode)")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
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
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
        ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
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

    # ── W&B init ──────────────────────────────────────────────────────────────
    model_short = os.path.basename(model_name)
    if isinstance(loss_fn, SFTLoss):
        data_source_tag = "sft"
    else:
        data_source_tag = (
            args.data_source if args.data_source is not None
            else "sycophancy_bct" if args.control_cot_path is not None
            else config.get("data", {}).get("source", "unknown")
        )
    lr     = config["training"]["learning_rate"]
    weight = loss_cfg.get("weight", 1.0)
    run_name = f"{model_short}_{data_source_tag}_lr{lr}_w{weight}_{os.path.basename(args.config).replace('.yaml', '')}"
    wandb.init(project="AttCT", name=run_name, group="week2", tags=[data_source_tag], config=config)

    print(f"Loss: {loss_cfg['name']} | Device: {device}")
    model = model.to(device)
    if ref_model is not None:
        ref_model = ref_model.to(device)
# ── Training path ─────────────────────────────────────────────────────────
    if isinstance(loss_fn, SFTLoss):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # BRR eval for BCT (if --brr-eval-path provided)
        brr_evaluator = None
        if args.brr_eval_path is not None:
            run_label = os.path.splitext(os.path.basename(args.config))[0]
            brr_csv = os.path.join("results", f"{run_label}_brr.csv")
            brr_evaluator = BRREvaluator(
                model, tokenizer, device,
                eval_path=args.brr_eval_path,
                results_csv=brr_csv,
                mmlu_max_samples=args.mmlu_max_samples,
            )
            print(f"BRR evaluator configured for BCT ({len(brr_evaluator.questions)} questions)")

            # Pre-training BRR eval
            print("\n=== Pre-training baseline (base model) ===")
            if is_lora:
                model.disable_adapter_layers()
                model.eval()
                brr_evaluator.evaluate(stage="pre_train", step=0)
                model.enable_adapter_layers()
                model.train()
            else:
                brr_evaluator.evaluate(stage="pre_train", step=0)

        train_dl    = get_bct_dataloader(config, split="train")
        eval_dl     = get_bct_dataloader(config, split="eval")
        bct_trainer = BCTTrainer(model, train_dl, loss_fn, config, device)
        bct_trainer.train()
        bct_trainer.eval_loss(eval_dl)

        # Post-training BRR eval
        if brr_evaluator is not None:
            print("\n=== Post-training evaluation (trained model) ===")
            model.eval()
            max_steps = config["training"].get("max_steps", len(train_dl))
            brr_evaluator.evaluate(stage="post_train", step=max_steps)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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

        # Build evaluator for checkpoints.
        # BRR evaluator (sycophancy), ASR evaluator (jailbreak), or legacy BehavioralEvaluator.
        brr_evaluator = None
        asr_evaluator = None
        behavioral_evaluator = None

        if args.asr_eval:
            run_label = os.path.splitext(os.path.basename(args.config))[0]
            asr_csv = os.path.join("results", f"{run_label}_asr.csv")
            asr_evaluator = ASREvaluator(
                model, tokenizer, device,
                results_csv=asr_csv,
                max_prompts=args.asr_max_prompts,
                max_new_tokens=args.asr_max_new_tokens,
                mmlu_max_samples=args.mmlu_max_samples,
            )
            print(f"ASR evaluator configured ({len(asr_evaluator.prompts)} prompts) — will run at pre-train, checkpoints, and post-train.")

        if args.brr_eval_path is not None:
            run_label = os.path.splitext(os.path.basename(args.config))[0]
            brr_csv = os.path.join("results", f"{run_label}_brr.csv")
            brr_evaluator = BRREvaluator(
                model, tokenizer, device,
                eval_path=args.brr_eval_path,
                results_csv=brr_csv,
                mmlu_max_samples=args.mmlu_max_samples,
            )
            print(f"BRR evaluator configured ({len(brr_evaluator.questions)} questions, MMLU={args.mmlu_max_samples}) — will run at pre-train, checkpoints, and post-train.")
        else:
            # Fall back to legacy BehavioralEvaluator if old-style paths provided.
            beval_paths = [args.bct_cot_path, args.bct_noncot_path, args.control_cot_path, args.control_noncot_path]
            has_sycophancy_paths = all(p is not None for p in beval_paths)
            has_mmlu  = args.mmlu_max_samples > 0
            has_gsm8k = args.gsm8k_max_samples > 0
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
                print(f"Legacy behavioral evaluator configured.")
            else:
                print("No eval configured. Pass --brr-eval-path for BRR eval, or --bct-cot/... for legacy eval.")

        # Namespace log dir by loss + data source so sweep runs don't overwrite each other.
        _base_log_dir = config.get("logging", {}).get("log_dir", "logs")
        _data_source_tag = (
            args.data_source if args.data_source is not None
            else ("sycophancy_bct" if args.control_cot_path is not None else "unknown")
        )
        log_dir = os.path.join(_base_log_dir, f"{loss_name}__{_data_source_tag}")
        os.makedirs(log_dir, exist_ok=True)
        config.setdefault("logging", {})["log_dir"] = log_dir

        # Build checkpoint callback.
        def make_checkpoint_fn(brr_eval, asr_eval, beval):
            evals = []
            if brr_eval is not None:
                evals.append(lambda step: brr_eval.evaluate(stage="checkpoint", step=step))
            if asr_eval is not None:
                evals.append(lambda step: asr_eval.evaluate(stage="checkpoint", step=step))
            if beval is not None:
                evals.append(lambda step: beval.evaluate(global_step=step, log_path=os.path.join(log_dir, f"beval_step_{step}.jsonl")))
            if evals:
                def _fn(step):
                    for ev in evals:
                        ev(step)
                return _fn
            return None

        is_sycophancy = config.get("data", {}).get("mode") == "sycophancy"
        is_sanity = config.get("data", {}).get("limit") is not None

        # Pre-training eval.
        has_pre_eval = (brr_evaluator is not None or asr_evaluator is not None) and not is_sanity
        if has_pre_eval:
            print("\n=== Pre-training baseline (base model) ===")
            if is_lora:
                model.disable_adapter_layers()
                model.eval()
                if brr_evaluator is not None:
                    brr_evaluator.evaluate(stage="pre_train", step=0)
                if asr_evaluator is not None:
                    asr_evaluator.evaluate(stage="pre_train", step=0)
                model.enable_adapter_layers()
                model.train()
            else:
                if brr_evaluator is not None:
                    brr_evaluator.evaluate(stage="pre_train", step=0)
                if asr_evaluator is not None:
                    asr_evaluator.evaluate(stage="pre_train", step=0)
        elif is_sycophancy and not is_sanity:
            from evaluate_sycophancy import SycophancyEvaluator
            run_label = os.path.splitext(os.path.basename(args.config))[0]
            results_csv = os.path.join("results", f"{run_label}_syco_results.csv")
            print("\n=== Pre-training baseline (base model) ===")
            if is_lora:
                model.disable_adapter_layers()
                model.eval()
                SycophancyEvaluator(model, tokenizer, device, prefix="pre_train",
                                    results_csv=results_csv).evaluate()
                model.enable_adapter_layers()
                model.train()
            else:
                SycophancyEvaluator(ref_model, tokenizer, device, prefix="pre_train",
                                    results_csv=results_csv).evaluate()

        Trainer(
            model,
            get_dataloader(config, split="train"),
            loss_fn,
            config,
            device,
            ref_model=ref_model,
            log_io_path=args.log_io,
            tokenizer=tokenizer,
            checkpoint_fn=make_checkpoint_fn(brr_evaluator, asr_evaluator, behavioral_evaluator),
        ).train()

        # Run MLP consistency loss eval only if BRR eval is not configured
        # (BRR eval already measures the model; this would redundantly re-run
        # the training data through the loss function).
        if brr_evaluator is None:
            eval_config = config.copy()
            if args.eval_limit is not None:
                eval_config.setdefault("data", {})["limit"] = args.eval_limit
            Evaluator(model, get_dataloader(eval_config, split="eval"), loss_fn, eval_config, device).evaluate()

        # Post-training eval.
        has_post_eval = (brr_evaluator is not None or asr_evaluator is not None) and not is_sanity
        if has_post_eval:
            print("\n=== Post-training evaluation (trained model) ===")
            model.eval()
            max_steps = config["training"].get("max_steps", 500)
            if brr_evaluator is not None:
                brr_evaluator.evaluate(stage="post_train", step=max_steps)
            if asr_evaluator is not None:
                asr_evaluator.evaluate(stage="post_train", step=max_steps)
        elif is_sycophancy and not is_sanity:
            print("\n=== Post-training evaluation (trained model) ===")
            model.eval()
            SycophancyEvaluator(model, tokenizer, device, prefix="post_train",
                                results_csv=results_csv).evaluate()

    wandb.finish()

if __name__ == "__main__":
    main()
