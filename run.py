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
    SFTLoss,
)
from losses.kl_regularization import KLRegularizationLoss
from data import get_dataloader
from data.attct_datasets import get_bct_dataloader
from data.ultrachat_dataset import get_kl_dataloader
from train import Trainer, BCTTrainer
from interleaved_trainer import InterleavedTrainer
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

    # W&B sharing / persona sweep args (used by sweep_persona_experiments.sh)
    parser.add_argument("--checkpoint", default=None, help="Path to a saved LoRA checkpoint to resume from")
    parser.add_argument("--run-name", default=None, help="W&B run name (default: auto-generated)")
    parser.add_argument("--wandb-group", default=None, help="W&B group for organising related runs")
    parser.add_argument("--wandb-run-id", default=None, help="W&B run ID to resume (share a run across scripts)")
    parser.add_argument("--metric-prefix", default="eval/", help="Prefix for eval metric keys logged to W&B")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the post-training evaluation pass")
    parser.add_argument("--save-dir", default=None, help="Override training.save_dir for checkpoints")

    parser.add_argument("--log-io", metavar="FILE", default=None,
                        help="Write every model input (clean + wrapped) and its greedy-decoded "
                             "response to FILE as JSONL. One JSON object per forward pass.")

    # Behavioral eval JSONL paths
    beval = parser.add_argument_group("behavioral_eval")
    beval.add_argument("--bct-cot",           dest="bct_cot_path",        default=None)
    beval.add_argument("--bct-noncot",        dest="bct_noncot_path",     default=None)
    beval.add_argument("--control-cot",       dest="control_cot_path",    default=None)
    beval.add_argument("--control-noncot",    dest="control_noncot_path", default=None)
    beval.add_argument("--beval-max-samples", dest="beval_max_samples",   type=int, default=500)
    beval.add_argument("--mmlu-max-samples",  dest="mmlu_max_samples",    type=int, default=200)
    beval.add_argument("--mmlu-subject",      dest="mmlu_subject",        default="all")
    beval.add_argument("--gsm8k-max-samples", dest="gsm8k_max_samples",   type=int, default=200)

    parser.add_argument(
        "--model",
        default=None,
        choices=["llama", "qwen"],
        help="Model to use: 'llama' (meta-llama/Llama-3.1-8B-Instruct) or 'qwen' (Qwen/Qwen3-8B). Overrides config.yaml model.name.",
    )

    parser.add_argument("--data-source", dest="data_source", default=None)
    parser.add_argument("--data-mode",   dest="data_mode",   required=True, choices=["jailbreak", "sycophancy"],
                        help="Training mode: 'jailbreak' or 'sycophancy'. Must be supplied explicitly.")
    parser.add_argument("--data-limit",  dest="data_limit",  default=None, type=int)
    parser.add_argument("--eval-limit",  dest="eval_limit",  default=None, type=int)
    parser.add_argument("--max-steps",     dest="max_steps",     default=None, type=int,
                        help="Override training.max_steps from config")
    parser.add_argument("--no-checkpoint", dest="no_checkpoint", action="store_true",
                        help="Skip mid-training behavioral eval checkpoints (pre/post baselines still run)")

    # Interleaved training (AttCT + KL regularization)
    interleave_group = parser.add_argument_group("interleaved_training")
    interleave_group.add_argument(
        "--interleave", action="store_true",
        help="Enable interleaved training: alternate AttCT and KL regularization steps",
    )
    interleave_group.add_argument(
        "--kl-weight", type=float, default=1.0,
        help="Weight for KL regularization loss (default: 1.0)",
    )
    interleave_group.add_argument(
        "--kl-samples", type=int, default=None,
        help="Number of UltraChat prompts for KL regularization (default: match AttCT dataset size)",
    )
    interleave_group.add_argument(
        "--kl-temperature", type=float, default=1.0,
        help="Softmax temperature for KL loss (default: 1.0)",
    )

    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    if args.config != "config.yaml":
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        config = _deep_merge(config, {k: v for k, v in overrides.items() if k != "defaults"})

    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model loading ─────────────────────────────────────────────────────────
    is_lora = bool(config.get("lora"))
    ref_model = None
    _MODEL_ALIASES = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen":  "Qwen/Qwen3-8B",
    }
    model_name = _MODEL_ALIASES.get(args.model) or config["model"]["name"]
    config["model"]["name"] = model_name

    needs_attn_weights = config["model"].get("output_attentions", True)
    if needs_attn_weights:
        attn_impl = "eager"
    else:
        try:
            import flash_attn        # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
    print(f"attn_implementation: {attn_impl}")

    if is_lora:
        lora_cfg = config["lora"]
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
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
    else:
        # Full fine-tuning: load a frozen ref model for clean pass (= θ_init)
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

    if args.save_dir:
        config.setdefault("training", {})["save_dir"] = args.save_dir

    # ── W&B init ──────────────────────────────────────────────────────────────
    if args.run_name:
        run_name = args.run_name
    else:
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

    wandb.init(
        project="AttCT",
        name=run_name,
        group=args.wandb_group,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        config=config,
    )

    print(f"Loss: {loss_cfg['name']} | Device: {device}")
    model = model.to(device)
    if ref_model is not None:
        ref_model = ref_model.to(device)

    # ── Training path ─────────────────────────────────────────────────────────
    if isinstance(loss_fn, SFTLoss):
        train_dl    = get_bct_dataloader(config, split="train")
        eval_dl     = get_bct_dataloader(config, split="eval")
        bct_trainer = BCTTrainer(model, train_dl, loss_fn, config, device)
        bct_trainer.train()
        bct_trainer.eval_loss(eval_dl)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if args.data_source is not None:
            config.setdefault("data", {})["source"] = args.data_source
        elif args.control_cot_path is not None:
            config.setdefault("data", {})["source"] = args.control_cot_path
        config.setdefault("data", {})["mode"] = args.data_mode
        if args.data_limit is not None:
            config.setdefault("data", {})["limit"] = args.data_limit

        wrapper_mode = config.get("data", {}).get("mode", "jailbreak")
        wrapper_templates = "sycophancy" if wrapper_mode == "sycophancy" else "jailbreak_strong"
        wandb.config.update({"wrapper": {"mode": wrapper_mode, "templates": wrapper_templates}},
                            allow_val_change=True)

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
            print("No behavioral eval configured.")

        _base_log_dir = config.get("logging", {}).get("log_dir", "logs")
        _data_source_tag = config.get("data", {}).get("source", "unknown")
        # Use a short label for known built-in sources; use the basename for file paths.
        if _data_source_tag not in ("clear-harm", "hardcoded", "sycophancy_bct"):
            _data_source_tag = os.path.splitext(os.path.basename(_data_source_tag))[0] or "unknown"
        log_dir = os.path.join(_base_log_dir, f"{loss_name}__{_data_source_tag}")
        os.makedirs(log_dir, exist_ok=True)
        config.setdefault("logging", {})["log_dir"] = log_dir

        is_sycophancy = args.data_mode == "sycophancy"
        is_jailbreak  = args.data_mode == "jailbreak"
        is_sanity = config.get("data", {}).get("limit") is not None

        run_label = os.path.splitext(os.path.basename(args.config))[0]

        if not is_sanity and is_sycophancy:
            from evaluate_sycophancy import SycophancyEvaluator
            results_csv = os.path.join("results", f"{run_label}_syco_results.csv")

        if not is_sanity and is_jailbreak:
            from evaluate_jailbreak import JailbreakEvaluator
            jailbreak_results_csv = os.path.join("results", f"{run_label}_jailbreak_results.csv")
            _jailbreak_data_source = config.get("data", {}).get("source", "clear-harm")

        _PROBE_QUESTIONS = [
            "How was your day?",
            "How do airplanes stay in the air?",
            "Why do people celebrate birthdays?",
        ]

        def _run_probe_questions(step):
            probe_log = os.path.join(log_dir, f"probe_step_{step}.jsonl")
            print(f"\n--- Probe questions (step {step}) ---")
            import json
            model.eval()
            with open(probe_log, "a") as f:
                for question in _PROBE_QUESTIONS:
                    input_ids = tokenizer.apply_chat_template(
                        [{"role": "user", "content": question}],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            pad_token_id=tokenizer.eos_token_id,
                            max_new_tokens=200,
                            do_sample=False,
                        )
                    response = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
                    print(f"Q: {question}\nA: {response}\n")
                    f.write(json.dumps({"step": step, "question": question, "response": response}) + "\n")
            print(f"[Probe responses saved to {probe_log}]")

        def make_checkpoint_fn():
            if is_sanity or args.no_checkpoint:
                return None
            def _fn(step):
                print(f"\n=== Checkpoint eval (step {step}) ===")
                if is_lora:
                    model.disable_adapter_layers()
                    model.eval()
                    if is_sycophancy:
                        SycophancyEvaluator(model, tokenizer, device, prefix=f"checkpoint_step_{step}",
                                            results_csv=results_csv).evaluate()
                    if is_jailbreak:
                        JailbreakEvaluator(model, tokenizer, device, data_source=_jailbreak_data_source,
                                           prefix=f"checkpoint_step_{step}",
                                           results_csv=jailbreak_results_csv).evaluate()
                    _run_probe_questions(step)
                    model.enable_adapter_layers()
                    model.train()
                else:
                    if is_sycophancy:
                        SycophancyEvaluator(model, tokenizer, device, prefix=f"checkpoint_step_{step}",
                                            results_csv=results_csv).evaluate()
                    if is_jailbreak:
                        JailbreakEvaluator(model, tokenizer, device, data_source=_jailbreak_data_source,
                                           prefix=f"checkpoint_step_{step}",
                                           results_csv=jailbreak_results_csv).evaluate()
                    _run_probe_questions(step)
                    model.train()
            return _fn

        if not is_sanity and is_sycophancy:
            print("\n=== Pre-training baseline (base model) — sycophancy eval ===")
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

        if not is_sanity and is_jailbreak:
            print("\n=== Pre-training baseline (base model) — jailbreak eval ===")
            if is_lora:
                model.disable_adapter_layers()
                model.eval()
                JailbreakEvaluator(model, tokenizer, device, data_source=_jailbreak_data_source,
                                   prefix="pre_train", results_csv=jailbreak_results_csv).evaluate()
                model.enable_adapter_layers()
                model.train()
            else:
                JailbreakEvaluator(ref_model, tokenizer, device, data_source=_jailbreak_data_source,
                                   prefix="pre_train", results_csv=jailbreak_results_csv).evaluate()

        attct_dl = get_dataloader(config, split="train")

        if args.interleave:
            kl_loss_fn = KLRegularizationLoss(
                weight=args.kl_weight,
                temperature=args.kl_temperature,
            )
            n_kl = args.kl_samples if args.kl_samples is not None else len(attct_dl.dataset)
            kl_dl = get_kl_dataloader(
                config, tokenizer, n_samples=n_kl,
            )
            print(
                f"Interleaved training: {len(attct_dl.dataset)} AttCT samples + "
                f"{len(kl_dl.dataset)} KL reg samples "
                f"(weight={args.kl_weight}, temp={args.kl_temperature})"
            )
            wandb.config.update({
                "interleave": True,
                "kl_weight": args.kl_weight,
                "kl_samples": n_kl,
                "kl_temperature": args.kl_temperature,
            }, allow_val_change=True)

            InterleavedTrainer(
                model=model,
                attct_dataloader=attct_dl,
                kl_dataloader=kl_dl,
                loss_fn=loss_fn,
                kl_loss_fn=kl_loss_fn,
                config=config,
                device=device,
                ref_model=ref_model,
                log_io_path=args.log_io,
                tokenizer=tokenizer,
                checkpoint_fn=make_checkpoint_fn(),
            ).train()
        else:
            Trainer(
                model,
                attct_dl,
                loss_fn,
                config,
                device,
                ref_model=ref_model,
                log_io_path=args.log_io,
                tokenizer=tokenizer,
                checkpoint_fn=make_checkpoint_fn(),
            ).train()

        if not args.skip_eval:
            eval_config = config.copy()
            if args.eval_limit is not None:
                eval_config.setdefault("data", {})["limit"] = args.eval_limit
            Evaluator(model, get_dataloader(eval_config, split="eval"), loss_fn, eval_config, device,
                      metric_prefix=args.metric_prefix).evaluate()

        if not is_sanity and is_sycophancy:
            print("\n=== Post-training evaluation (trained model) — sycophancy ===")
            model.eval()
            SycophancyEvaluator(model, tokenizer, device, prefix="post_train",
                                results_csv=results_csv).evaluate()

        if not is_sanity and is_jailbreak:
            print("\n=== Post-training evaluation (trained model) — jailbreak ===")
            model.eval()
            JailbreakEvaluator(model, tokenizer, device, data_source=_jailbreak_data_source,
                               prefix="post_train", results_csv=jailbreak_results_csv).evaluate()

    wandb.finish()

if __name__ == "__main__":
    main()