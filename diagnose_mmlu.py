"""
MMLU diagnostic: measures accuracy of base model vs LoRA-trained model.
Helps identify whether ACT training degrades general capabilities.

Usage:
    # Base model only (baseline)
    python diagnose_mmlu.py --model meta-llama/Llama-3.1-8B

    # Base model + compare against a trained LoRA checkpoint
    python diagnose_mmlu.py --model meta-llama/Llama-3.1-8B --lora_path /workspace/checkpoints/act_sycophancy/epoch_1

    # Full MMLU dataset
    python diagnose_mmlu.py --model meta-llama/Llama-3.1-8B --n 0

    # Quick spot-check (5 questions with verbose output)
    python diagnose_mmlu.py --model meta-llama/Llama-3.1-8B --n 5 --verbose
"""
import argparse
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm


def get_answer_scores(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    log_probs = torch.log_softmax(out.logits[0, -1, :], dim=-1)
    scores = []
    for letter in ["A", "B", "C", "D"]:
        candidates = [letter, f" {letter}", f"\n{letter}", f"({letter})"]
        best = max(
            log_probs[tokenizer.encode(c, add_special_tokens=False)[0]].item()
            for c in candidates
            if tokenizer.encode(c, add_special_tokens=False)
        )
        scores.append(best)
    return scores


def format_prompt(example):
    letters = ["A", "B", "C", "D"]
    choice_lines = "\n".join(f"{letters[i]}. {example['choices'][i]}" for i in range(len(example["choices"])))
    return f"{example['question']}\n{choice_lines}\nAnswer:", int(example["answer"])


def evaluate(model, tokenizer, questions, device, desc, verbose=False):
    letters = ["A", "B", "C", "D"]
    n_correct = 0
    by_subject = defaultdict(lambda: [0, 0])  # subject -> [correct, total]

    for ex in tqdm(questions, desc=desc):
        prompt, correct_idx = format_prompt(ex)
        scores = get_answer_scores(model, tokenizer, prompt, device)
        pred = scores.index(max(scores))
        subject = ex.get("subject", "unknown")
        by_subject[subject][1] += 1
        if pred == correct_idx:
            n_correct += 1
            by_subject[subject][0] += 1

        if verbose:
            print(f"\n  Q: {ex['question'][:70]}...")
            print(f"  Correct: {letters[correct_idx]} | Pred: {letters[pred]}")
            top5 = sorted(zip(letters, scores), key=lambda x: -x[1])[:5]
            print(f"  Scores: {[(l, f'{s:.2f}') for l, s in top5]}")

    n = len(questions)
    acc = n_correct / n
    print(f"\n{desc}: {n_correct}/{n} = {acc:.1%}")

    # Print per-subject breakdown (worst 10 and best 10)
    subject_accs = {s: v[0]/v[1] for s, v in by_subject.items() if v[1] > 0}
    if len(subject_accs) > 1:
        sorted_subjects = sorted(subject_accs.items(), key=lambda x: x[1])
        print("\n  Worst 10 subjects:")
        for s, a in sorted_subjects[:10]:
            c, t = by_subject[s]
            print(f"    {s:45s} {a:.1%}  ({c}/{t})")
        print("  Best 10 subjects:")
        for s, a in sorted_subjects[-10:]:
            c, t = by_subject[s]
            print(f"    {s:45s} {a:.1%}  ({c}/{t})")

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA checkpoint to compare against base")
    parser.add_argument("--n", type=int, default=0, help="Questions to evaluate (0 = full dataset)")
    parser.add_argument("--verbose", action="store_true", help="Print per-question output (use with small --n)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Token mapping sanity check ────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Token ID mapping for answer letters:")
    for letter in ["A", "B", "C", "D"]:
        for form in [letter, f" {letter}", f"\n{letter}"]:
            ids = tokenizer.encode(form, add_special_tokens=False)
            print(f"  {repr(form):8s} → {ids}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\nLoading MMLU test set...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    questions = list(ds) if args.n == 0 else list(ds.select(range(args.n)))
    print(f"Evaluating on {len(questions)} questions")

    # ── Base model ────────────────────────────────────────────────────────────
    print(f"\nLoading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    base_acc = evaluate(model, tokenizer, questions, device, desc="Base model", verbose=args.verbose)

    # ── LoRA model (optional) ─────────────────────────────────────────────────
    if args.lora_path:
        print(f"\nLoading LoRA checkpoint: {args.lora_path}")
        lora_model = PeftModel.from_pretrained(model, args.lora_path)
        lora_model.eval()
        lora_acc = evaluate(lora_model, tokenizer, questions, device, desc="LoRA model", verbose=args.verbose)
        print(f"\n{'='*50}")
        print(f"Base model accuracy:  {base_acc:.1%}")
        print(f"LoRA model accuracy:  {lora_acc:.1%}")
        print(f"Delta:                {lora_acc - base_acc:+.1%}")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()
