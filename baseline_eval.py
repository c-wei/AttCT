"""
baseline_eval.py — Run behavioral evaluation on the base model (step 0).

Usage:
    python baseline_eval.py \
        --bct-cot    datasets/sycophancy_bct/bct_cot.jsonl \
        --bct-noncot datasets/sycophancy_bct/bct_non_cot.jsonl \
        --control-cot    datasets/sycophancy_bct/control_cot.jsonl \
        --control-noncot datasets/sycophancy_bct/control_non_cot.jsonl \
        [--max-samples 500] [--log-path logs/baseline/beval_step_0.jsonl]
"""
import argparse
import torch
import wandb
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from behavioral_evaluate import BehavioralEvaluator


def main():
    parser = argparse.ArgumentParser(description="Behavioral baseline eval on the untrained model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--bct-cot",         dest="bct_cot_path",        required=True)
    parser.add_argument("--bct-noncot",      dest="bct_noncot_path",     required=True)
    parser.add_argument("--control-cot",     dest="control_cot_path",    required=True)
    parser.add_argument("--control-noncot",  dest="control_noncot_path", required=True)
    parser.add_argument("--max-samples",     dest="max_samples", type=int, default=500)
    parser.add_argument("--log-path",        dest="log_path", default="logs/baseline/beval_step_0.jsonl")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["behavioral_eval"] = {
        "bct_cot_path":        args.bct_cot_path,
        "bct_noncot_path":     args.bct_noncot_path,
        "control_cot_path":    args.control_cot_path,
        "control_noncot_path": args.control_noncot_path,
        "max_samples":         args.max_samples,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config["model"]["name"]
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="eager")
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wandb.init(project="AttCT", name="baseline", group="week2", tags=["baseline"], config=config)

    evaluator = BehavioralEvaluator(model, tokenizer, config, device)
    evaluator.evaluate(global_step=0, log_path=args.log_path)

    wandb.finish()


if __name__ == "__main__":
    main()
