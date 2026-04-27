import argparse
import os
import random
import yaml
import numpy as np
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from losses.losses import (
    AttentionConsistencyLoss,
    AttentionConsistencyLossV2,
    JSDAttentionConsistencyLoss,
    DirectedAttentionConsistencyLoss,
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
from interleaved_trainer import InterleavedTrainer, IntelligenceTrainer
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
    parser.add_argument("--full-eval", dest="full_eval", action="store_true",
                        help="Run the full-dataset loss eval pass before post-training behavioral evals (slow, off by default)")
    parser.add_argument("--save-dir", default=None, help="Override training.save_dir for checkpoints")
    parser.add_argument("--lora-rank", dest="lora_rank", default=None, type=int,
                        help="Override lora.r from config (default: 8)")
    parser.add_argument("--lora-alpha", dest="lora_alpha", default=None, type=int,
                        help="Override lora.lora_alpha from config")
    parser.add_argument("--lora-targets", dest="lora_targets", nargs="+", default=None, metavar="MODULE",
                        help="Override lora.target_modules from config (e.g. --lora-targets q_proj k_proj v_proj o_proj)")
    parser.add_argument("--loss-layer-selection", dest="loss_layer_selection", default=None,
                        help="Override loss.kwargs.layer_selection (e.g. 'all', 'last_half', 'last_quarter')")

    parser.add_argument("--hf-repo", dest="hf_repo", default=None,
                        help="HuggingFace repo ID to push checkpoints to asynchronously (e.g. username/my-model). "
                             "Requires huggingface_hub and HF_TOKEN env var.")
    parser.add_argument("--fresh-data", action="store_true",
                        help="Generate fresh BCT targets via OpenRouter before training.")
    parser.add_argument("--bct-root", default=None,
                        help="Override data.bct_root in config.")

    parser.add_argument("--log-io", metavar="FILE", default=None,
                        help="Write every model input (clean + wrapped) and its greedy-decoded "
                             "response to FILE as JSONL. One JSON object per forward pass.")

    parser.add_argument("--control-cot", dest="control_cot_path", default=None,
                        help="Path to control CoT JSONL — used to select sycophancy_bct data source")

    parser.add_argument(
        "--model",
        default=None,
        choices=["llama", "qwen", "qwen3-4b", "gemma-4b", "gemma-27b"],
        help=(
            "Model shorthand (overrides config.yaml model.name): "
            "llama=meta-llama/Llama-3.1-8B-Instruct, "
            "qwen=Qwen/Qwen3-8B, "
            "qwen3-4b=Qwen/Qwen3-4B-Instruct-2507, "
            "gemma-4b=google/gemma-3-4b-it, "
            "gemma-27b=google/gemma-3-27b-it."
        ),
    )

    parser.add_argument("--data-source", dest="data_source", nargs="+", default=None, metavar="SOURCE",
                        help="Training data source(s). Pass multiple to combine (jailbreak mode only), e.g. --data-source clear-harm wildjailbreak harmbench.")
    parser.add_argument("--data-mode",   dest="data_mode",   required=True,
                        choices=["jailbreak", "sycophancy", "intelligence"],
                        help="Training mode: 'jailbreak', 'sycophancy', or 'intelligence' (UltraChat KL-only control).")
    parser.add_argument("--eval-sycophancy", dest="eval_sycophancy", action="store_true",
                        help="Run sycophancy evaluator post-training regardless of data-mode.")
    parser.add_argument("--eval-jailbreak", dest="eval_jailbreak_source", action="store_const", const=True, default=None,
                        help="Run jailbreak evaluator post-training (always uses jbb harmful + jbb-benign).")
    # Behavioral eval JSONL paths
    beval = parser.add_argument_group("behavioral_eval")
    beval.add_argument("--bct-cot",           dest="bct_cot_path",        default=None)
    beval.add_argument("--bct-noncot",        dest="bct_noncot_path",     default=None)
    beval.add_argument("--control-noncot",    dest="control_noncot_path", default=None)
    beval.add_argument("--beval-max-samples", dest="beval_max_samples",   type=int, default=500)
    beval.add_argument("--mmlu-max-samples",  dest="mmlu_max_samples",    type=int, default=200)
    beval.add_argument("--mmlu-subject",      dest="mmlu_subject",        default="all")
    beval.add_argument("--gsm8k-max-samples", dest="gsm8k_max_samples",   type=int, default=200)
    parser.add_argument("--data-limit",  dest="data_limit",  default=None, type=int,
                        help="Cap the number of training prompts (overrides config data.limit).")
    parser.add_argument("--eval-limit",  dest="eval_limit",  default=None, type=int)
    parser.add_argument("--max-steps",     dest="max_steps",     default=None, type=int,
                        help="Override training.max_steps from config")
    parser.add_argument("--no-checkpoint", dest="no_checkpoint", action="store_true",
                        help="Skip mid-training behavioral eval checkpoints (pre/post baselines still run)")
    parser.add_argument("--brr_test_root", default=None,
                        help="Path to cot-transparency test sets. If provided, runs BRR "
                             "evaluation after BCT training and logs metrics to W&B.")
    parser.add_argument("--brr_limit", type=int, default=150,
                        help="Max records per bias type for BRR eval (default: 150).")
    parser.add_argument("--brr_baseline_json", default=None,
                        help="Path to baseline BRR JSON for computing BRR ratio.")

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
    interleave_group.add_argument(
        "--kl-dataset", dest="kl_dataset", default="ultrachat",
        choices=["ultrachat", "alpaca"],
        help="Dataset for KL regularization / intelligence training (default: ultrachat).",
    )
    interleave_group.add_argument(
        "--kl-ratio", dest="kl_ratio", type=float, default=1.0,
        help="Fraction of AttCT steps that also fire a KL step (default: 1.0 = always). "
             "E.g. 0.1 means ~1 KL step per 10 AttCT steps.",
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
    elif args.data_mode == "intelligence":
        # Step count is determined by --kl-samples; don't let config.yaml cap it.
        config.setdefault("training", {})["max_steps"] = None
    else:
        # Apply per-source default max_steps if not explicitly overridden.
        # For multiple sources, use the largest default so all data is seen.
        source = (
            args.data_source
            or config.get("data", {}).get("source", "clear-harm")
        )
        source_max_steps = config.get("data", {}).get("source_max_steps", {})
        if isinstance(source, list):
            steps = [source_max_steps[s] for s in source if s in source_max_steps]
            if steps:
                config.setdefault("training", {})["max_steps"] = max(steps)
        elif source in source_max_steps:
            config.setdefault("training", {})["max_steps"] = source_max_steps[source]

    if args.lora_rank is not None:
        config.setdefault("lora", {})["r"] = args.lora_rank
    if args.lora_alpha is not None:
        config.setdefault("lora", {})["lora_alpha"] = args.lora_alpha
    if args.lora_targets is not None:
        config.setdefault("lora", {})["target_modules"] = args.lora_targets
    if args.loss_layer_selection is not None:
        config.setdefault("loss", {}).setdefault("kwargs", {})["layer_selection"] = args.loss_layer_selection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model loading ─────────────────────────────────────────────────────────
    is_lora = bool(config.get("lora"))
    ref_model = None

    _MODEL_ALIASES = {
        "llama":     "meta-llama/Llama-3.1-8B-Instruct",
        "qwen":      "Qwen/Qwen3-8B",
        "qwen3-4b":  "Qwen/Qwen3-4B-Instruct-2507",
        "gemma-4b":  "google/gemma-3-4b-it",
        "gemma-27b": "google/gemma-3-27b-it",
    }
    seed = config.get("training", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = _MODEL_ALIASES.get(args.model) or config["model"]["name"]
    loss_cfg  = config["loss"]
    loss_name = loss_cfg["name"]


    needs_attn_weights = config["model"].get("output_attentions", True)
    force_attn_impl    = config["model"].get("attn_implementation", None)
    if force_attn_impl:
        attn_impl = force_attn_impl
    elif needs_attn_weights:
        attn_impl = "eager"
    else:
        try:
            import flash_attn        # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
    print(f"attn_implementation: {attn_impl}")

    quant_cfg = config["model"].get("quantization", None)
    if quant_cfg == "bitsandbytes":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs = dict(quantization_config=bnb_config, attn_implementation=attn_impl, device_map="auto")
        print("Quantization: 4-bit NF4 (QLoRA)")
    elif quant_cfg is not None:
        raise ValueError(f"Unsupported quantization type: {quant_cfg!r}. Use 'bitsandbytes' or omit.")
    else:
        load_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation=attn_impl)

    if is_lora:
        lora_cfg = config["lora"]
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if quant_cfg == "4bit":
            from peft import prepare_model_for_kbit_training
            base_model = prepare_model_for_kbit_training(base_model)
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
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if loss_name != "SFTLoss":
            # AttCT full FT: needs frozen ref model for the clean pass (= θ_init)
            ref_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters())
        print(f"Full FT: {trainable:,} trainable parameters")

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
                "+".join(args.data_source) if args.data_source is not None
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

    # ── Sweep parameter injection ──────────────────────────────────────────────
    # When running under `wandb agent`, wandb.config contains the sweep-chosen
    # hyperparameters. We apply them back into our config dict using dotted keys
    # (e.g. "training.learning_rate" → config["training"]["learning_rate"]).
    # Supported sweep keys: training.learning_rate, training.weight_decay,
    # training.epochs, lora.r, lora.lora_alpha, lora.lora_dropout.
    _sweep_params = dict(wandb.config)
    if _sweep_params:
        for dotted_key, value in _sweep_params.items():
            parts = dotted_key.split(".")
            if len(parts) == 2:
                section, key = parts
                if section in config and key in config[section]:
                    config[section][key] = value
                    print(f"[sweep] {dotted_key} = {value}")
        # Re-read values that may have changed
        model_name = config["model"]["name"]
        loss_cfg   = config["loss"]
        loss_name  = loss_cfg["name"]

    if config.get("training", {}).get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("Gradient checkpointing enabled.")

    print(f"Loss: {loss_cfg['name']} | Device: {device}")
    if quant_cfg is None:
        model = model.to(device)
        if ref_model is not None:
            ref_model = ref_model.to(device)

    # ── Training path ─────────────────────────────────────────────────────────
    if isinstance(loss_fn, SFTLoss):
        if args.bct_root:
            config.setdefault("data", {})["bct_root"] = args.bct_root

        if args.fresh_data:
            import tempfile
            from generate_fresh_bct_data import run_generation
            fresh_dir = tempfile.mkdtemp(prefix="fresh_bct_")
            bct_root  = config.get("data", {}).get("bct_root", "datasets/sycophancy_bct")
            limit     = config.get("data", {}).get("limit", None)
            print(f"==> Generating fresh BCT data via OpenRouter → {fresh_dir}")
            run_generation(model_name=model_name, bct_root=bct_root,
                           output_dir=fresh_dir, limit=limit)
            config["data"]["bct_root"] = fresh_dir

        train_dl    = get_bct_dataloader(config, split="train")
        eval_dl     = get_bct_dataloader(config, split="eval")
        bct_trainer = BCTTrainer(model, train_dl, loss_fn, config, device)
        bct_trainer.train()
        bct_trainer.eval_loss(eval_dl)

        if args.brr_test_root:
            from evaluate_bct import run_brr_eval
            checkpoint_path = os.path.join(config["training"]["save_dir"], "epoch_1")
            print(f"\n==> BRR evaluation (limit={args.brr_limit} records/bias)...")
            # Free training model from GPU before vLLM loads the checkpoint
            del model
            if ref_model is not None:
                del ref_model
            torch.cuda.empty_cache()
            run_brr_eval(
                model_name=model_name,
                lora_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
                test_root=args.brr_test_root,
                limit=args.brr_limit,
                baseline_json=args.brr_baseline_json,
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if args.data_source is not None:
            # single string or list — get_dataloader handles both
            config.setdefault("data", {})["source"] = args.data_source if len(args.data_source) > 1 else args.data_source[0]
        elif args.control_cot_path is not None:
            config.setdefault("data", {})["source"] = args.control_cot_path
        config.setdefault("data", {})["mode"] = args.data_mode
        if args.data_limit is not None:
            config.setdefault("data", {})["limit"] = args.data_limit

        wrapper_mode = config.get("data", {}).get("mode", "jailbreak")
        wrapper_templates = "sycophancy" if wrapper_mode == "sycophancy" else "jailbreak_strong"
        wandb.config.update({"wrapper": {"mode": wrapper_mode, "templates": wrapper_templates}},
                            allow_val_change=True)


        _base_log_dir = config.get("logging", {}).get("log_dir", "logs")
        _data_source_tag = config.get("data", {}).get("source", "unknown")
        # Use a short label for known built-in sources; use the basename for file paths.
        if isinstance(_data_source_tag, list):
            _data_source_tag = "+".join(_data_source_tag)
        elif _data_source_tag not in ("clear-harm", "sycophancy_bct"):
            _data_source_tag = os.path.splitext(os.path.basename(_data_source_tag))[0] or "unknown"
        log_dir = os.path.join(_base_log_dir, f"{loss_name}__{_data_source_tag}")
        os.makedirs(log_dir, exist_ok=True)
        config.setdefault("logging", {})["log_dir"] = log_dir

        is_sycophancy = args.data_mode == "sycophancy" or args.eval_sycophancy
        is_jailbreak  = args.data_mode == "jailbreak"  or (args.eval_jailbreak_source is not None)
        is_intelligence = args.data_mode == "intelligence"
        is_sanity = config.get("data", {}).get("limit") is not None

        run_label = os.path.splitext(os.path.basename(args.config))[0]

        if not is_sanity and is_sycophancy:
            from evaluate_sycophancy import SycophancyEvaluator
            results_csv = os.path.join("results", f"{run_label}_syco_results.csv")

        _eval_limit = args.eval_limit  # max samples for all behavioral evaluators

        if not is_sanity and is_jailbreak:
            from evaluate_jailbreak import JailbreakEvaluator
            # Primary eval: ASR + over-refusal on JailbreakBench (jbb-harmful + jbb-benign).
            # Secondary eval: ASR-only on ClearHarm (no over-refusal — that's a model-level
            # property already captured by the jbb-benign run).
            jailbreak_jbb_csv        = os.path.join("results", f"{run_label}_jailbreak_jbb_results.csv")
            jailbreak_clearharm_csv  = os.path.join("results", f"{run_label}_jailbreak_clearharm_results.csv")

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
                    )["input_ids"].to(device)
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
                    model.eval()
                    if is_sycophancy:
                        SycophancyEvaluator(model, tokenizer, device, prefix=f"checkpoint_step_{step}",
                                            results_csv=results_csv, max_samples=_eval_limit).evaluate()
                    if is_jailbreak:
                        JailbreakEvaluator(model, tokenizer, device,
                                           harmful_source="jbb",
                                           prefix=f"checkpoint_step_{step}/jbb",
                                           results_csv=jailbreak_jbb_csv, max_samples=_eval_limit).evaluate()
                        JailbreakEvaluator(model, tokenizer, device,
                                           harmful_source="clear-harm", measure_overrefusal=False,
                                           prefix=f"checkpoint_step_{step}/clearharm",
                                           results_csv=jailbreak_clearharm_csv, max_samples=_eval_limit).evaluate()
                    _run_probe_questions(step)
                    model.train()
                else:
                    if is_sycophancy:
                        SycophancyEvaluator(model, tokenizer, device, prefix=f"checkpoint_step_{step}",
                                            results_csv=results_csv, max_samples=_eval_limit).evaluate()
                    if is_jailbreak:
                        JailbreakEvaluator(model, tokenizer, device,
                                           harmful_source="jbb",
                                           prefix=f"checkpoint_step_{step}/jbb",
                                           results_csv=jailbreak_jbb_csv, max_samples=_eval_limit).evaluate()
                        JailbreakEvaluator(model, tokenizer, device,
                                           harmful_source="clear-harm", measure_overrefusal=False,
                                           prefix=f"checkpoint_step_{step}/clearharm",
                                           results_csv=jailbreak_clearharm_csv, max_samples=_eval_limit).evaluate()
                    _run_probe_questions(step)
                    model.train()
            return _fn

        if not is_sanity and is_sycophancy:
            print("\n=== Pre-training baseline (base model) — sycophancy eval ===")
            if is_lora:
                model.disable_adapter_layers()
                model.eval()
                SycophancyEvaluator(model, tokenizer, device, prefix="pre_train",
                                    results_csv=results_csv, max_samples=_eval_limit).evaluate()
                model.enable_adapter_layers()
                model.train()
            else:
                SycophancyEvaluator(ref_model, tokenizer, device, prefix="pre_train",
                                    results_csv=results_csv, max_samples=_eval_limit).evaluate()

        if not is_sanity and is_jailbreak:
            print("\n=== Pre-training baseline (base model) — jailbreak eval ===")
            _eval_model = model if is_lora else ref_model
            if is_lora:
                model.disable_adapter_layers()
                model.eval()
            JailbreakEvaluator(_eval_model, tokenizer, device,
                               harmful_source="jbb",
                               prefix="pre_train/jbb", results_csv=jailbreak_jbb_csv,
                               max_samples=_eval_limit).evaluate()
            JailbreakEvaluator(_eval_model, tokenizer, device,
                               harmful_source="clear-harm", measure_overrefusal=False,
                               prefix="pre_train/clearharm", results_csv=jailbreak_clearharm_csv,
                               max_samples=_eval_limit).evaluate()
            if is_lora:
                model.enable_adapter_layers()
                model.train()

        if is_intelligence:
            kl_loss_fn = KLRegularizationLoss(
                weight=args.kl_weight,
                temperature=args.kl_temperature,
            )
            n_kl = args.kl_samples if args.kl_samples is not None else 2500
            if args.max_steps is not None:
                n_kl = max(n_kl, args.max_steps)
            kl_dl = get_kl_dataloader(config, tokenizer, n_samples=n_kl, kl_dataset=args.kl_dataset)
            print(f"Intelligence (control) training: {len(kl_dl.dataset)} {args.kl_dataset} samples (KL only, no AttCT)")
            wandb.config.update({
                "intelligence_control": True,
                "kl_weight": args.kl_weight,
                "kl_samples": n_kl,
                "kl_temperature": args.kl_temperature,
                "kl_dataset": args.kl_dataset,
            }, allow_val_change=True)
            IntelligenceTrainer(
                model=model,
                kl_dataloader=kl_dl,
                kl_loss_fn=kl_loss_fn,
                config=config,
                device=device,
                ref_model=ref_model,
                checkpoint_fn=make_checkpoint_fn(),
                hf_repo=args.hf_repo,
                run_name=run_name,
            ).train()

        else:
            attct_dl = get_dataloader(config, split="train")

        if not is_intelligence and args.interleave:
            kl_loss_fn = KLRegularizationLoss(
                weight=args.kl_weight,
                temperature=args.kl_temperature,
            )
            n_kl = args.kl_samples if args.kl_samples is not None else len(attct_dl.dataset)
            kl_dl = get_kl_dataloader(
                config, tokenizer, n_samples=n_kl, kl_dataset=args.kl_dataset,
            )
            print(
                f"Interleaved training: {len(attct_dl.dataset)} AttCT samples + "
                f"{len(kl_dl.dataset)} {args.kl_dataset} KL reg samples "
                f"(weight={args.kl_weight}, temp={args.kl_temperature})"
            )
            wandb.config.update({
                "interleave": True,
                "kl_weight": args.kl_weight,
                "kl_samples": n_kl,
                "kl_temperature": args.kl_temperature,
                "kl_dataset": args.kl_dataset,
                "kl_ratio": args.kl_ratio,
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
                hf_repo=args.hf_repo,
                run_name=run_name,
                kl_ratio=args.kl_ratio,
            ).train()
        elif not is_intelligence:
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
                hf_repo=args.hf_repo,
                run_name=run_name,
            ).train()

        if args.full_eval and not args.skip_eval and not is_intelligence:
            eval_config = config.copy()
            if args.eval_limit is not None:
                eval_config.setdefault("data", {})["limit"] = args.eval_limit
            Evaluator(model, get_dataloader(eval_config, split="eval"), loss_fn, eval_config, device,
                      metric_prefix=args.metric_prefix).evaluate()

        if not is_sanity and is_sycophancy:
            print("\n=== Post-training evaluation (trained model) — sycophancy ===")
            model.eval()
            SycophancyEvaluator(model, tokenizer, device, prefix="post_train",
                                results_csv=results_csv, max_samples=_eval_limit).evaluate()

        if not is_sanity and is_jailbreak:
            print("\n=== Post-training evaluation (trained model) — jailbreak ===")
            model.eval()
            JailbreakEvaluator(model, tokenizer, device,
                               harmful_source="jbb",
                               prefix="post_train/jbb", results_csv=jailbreak_jbb_csv,
                               max_samples=_eval_limit).evaluate()
            JailbreakEvaluator(model, tokenizer, device,
                               harmful_source="clear-harm", measure_overrefusal=False,
                               prefix="post_train/clearharm", results_csv=jailbreak_clearharm_csv,
                               max_samples=_eval_limit).evaluate()

    wandb.finish()

if __name__ == "__main__":
    main()
