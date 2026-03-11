"""
Quick diagnostic for MMLU answer extraction.
Run this to see exactly what tokens and log-probs are being compared.

Usage:
    python diagnose_mmlu.py --model meta-llama/Llama-3.1-8B
    python diagnose_mmlu.py --model google/gemma-2-2b
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--n", type=int, default=5, help="Number of questions to diagnose")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # ── Show token IDs for all answer surface forms ───────────────────────────
    print("\n=== Token ID mapping for answer letters ===")
    for letter in ["A", "B", "C", "D"]:
        for form in [letter, f" {letter}", f"\n{letter}", f"({letter})"]:
            ids = tokenizer.encode(form, add_special_tokens=False)
            print(f"  {repr(form):10s} → ids={ids}  decoded={[tokenizer.decode([i]) for i in ids]}")

    # ── Run on a few MMLU questions ───────────────────────────────────────────
    print(f"\n=== MMLU diagnostic ({args.n} questions) ===")
    ds = load_dataset("cais/mmlu", "all", split="test")
    letters = ["A", "B", "C", "D"]
    n_correct = 0

    for i in range(args.n):
        ex = ds[i]
        q = ex["question"]
        choices = ex["choices"]
        correct_idx = int(ex["answer"])
        choice_lines = "\n".join(f"{letters[j]}. {choices[j]}" for j in range(len(choices)))
        prompt = f"{q}\n{choice_lines}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        last_logits = out.logits[0, -1, :]
        log_probs = torch.log_softmax(last_logits, dim=-1)

        # Show top-5 predicted tokens
        top5 = torch.topk(log_probs, 5)
        top5_tokens = [(tokenizer.decode([tid.item()]), lp.item()) for tid, lp in zip(top5.indices, top5.values)]

        # Score each answer using current method (" A" only)
        scores_old = []
        for letter in letters:
            ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
            scores_old.append(log_probs[ids[0]].item())

        # Score using all surface forms (proposed fix)
        scores_new = []
        for letter in letters:
            candidates = [letter, f" {letter}", f"\n{letter}", f"({letter})"]
            best = max(
                log_probs[tokenizer.encode(c, add_special_tokens=False)[0]].item()
                for c in candidates
                if tokenizer.encode(c, add_special_tokens=False)
            )
            scores_new.append(best)

        pred_old = letters[scores_old.index(max(scores_old))]
        pred_new = letters[scores_new.index(max(scores_new))]
        correct  = letters[correct_idx]
        if pred_new == correct:
            n_correct += 1

        print(f"\nQ{i+1}: {q[:60]}...")
        print(f"  Correct: {correct} | Old pred: {pred_old} | New pred: {pred_new}")
        print(f"  Top-5 next tokens: {top5_tokens}")
        print(f"  Old scores (space+letter): {dict(zip(letters, [f'{s:.2f}' for s in scores_old]))}")
        print(f"  New scores (best form):    {dict(zip(letters, [f'{s:.2f}' for s in scores_new]))}")

    print(f"\nNew method accuracy on {args.n} questions: {n_correct}/{args.n} = {n_correct/args.n:.0%}")

if __name__ == "__main__":
    main()
