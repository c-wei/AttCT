"""
KnowledgeEvaluator: measures general knowledge / reasoning capability across
three benchmarks using logprob-based answer scoring (no generation required).

Benchmarks:
  gsm8k       — GSM-8K grade-school math (openai/gsm8k, test split).
                 Each problem is cast to 4-choice MC: the correct numeric
                 answer plus three randomly sampled wrong answers from the
                 same batch.  Scored by log-prob at the last prompt token.
  hellaswag   — HellaSwag sentence completion (Rowan/hellaswag, validation).
                 4 candidate continuations scored by log-prob.
  truthfulqa  — TruthfulQA MC1 (truthfulqa/truthful_qa, mc1 config,
                 validation).  Single correct answer from a variable-length
                 choice list, scored by log-prob.

Metrics per benchmark:
  accuracy    — fraction of questions answered correctly
  n_questions — number of questions evaluated

Summary:
  overall_acc — unweighted mean accuracy across the three benchmarks

All sampling uses a fixed seed (default 42) for reproducibility.
"""

import csv
import datetime
import os
import random

import numpy as np
import torch
import wandb
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_N_SAMPLES = 500
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Shared logprob helper (same pattern as evaluate_sycophancy.py)
# ---------------------------------------------------------------------------

def _get_answer_logprobs(model, tokenizer, prompt: str, device, letters) -> list:
    """
    Returns log-probs for each answer letter at the last prompt token position.
    Tries multiple surface forms and takes the max for tokenizer robustness.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(last_logits, dim=-1)

    scores = []
    for letter in letters:
        candidates = [letter, f" {letter}", f"\n{letter}", f"({letter})"]
        token_ids = [tokenizer.encode(c, add_special_tokens=False) for c in candidates]
        best = max(
            log_probs[ids[0]].item()
            for ids in token_ids
            if ids
        )
        scores.append(best)
    return scores


def _write_csv_row(path: str, row: dict):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Per-benchmark prompt formatters
# ---------------------------------------------------------------------------

_LETTERS = ["A", "B", "C", "D"]


def _format_gsm8k_prompt(question: str, choices: list, correct_idx: int) -> tuple:
    """
    Formats a GSM-8K question as 4-choice MC.

    choices  — list of 4 answer strings (correct + 3 distractors)
    Returns (prompt_text, correct_idx_in_shuffled_choices).
    """
    choice_lines = "\n".join(f"{_LETTERS[i]}. {choices[i]}" for i in range(len(choices)))
    prompt = f"{question}\n{choice_lines}\nAnswer:"
    return prompt, correct_idx


def _format_hellaswag_prompt(ctx: str, endings: list) -> str:
    """Formats a HellaSwag example: context + 4 labelled continuations."""
    choice_lines = "\n".join(f"{_LETTERS[i]}. {endings[i]}" for i in range(len(endings)))
    return f"{ctx}\n{choice_lines}\nAnswer:"


def _format_truthfulqa_prompt(question: str, choices: list) -> tuple:
    """Formats a TruthfulQA MC1 example with variable-length choice list. Returns (prompt_text, letters)."""
    n = len(choices)
    letters = _LETTERS[:n] if n <= 4 else [chr(ord("A") + i) for i in range(n)]
    choice_lines = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(n))
    return f"{question}\n{choice_lines}\nAnswer:", letters


# ---------------------------------------------------------------------------
# GSM-8K distractor generation
# ---------------------------------------------------------------------------

def _extract_gsm8k_answer(answer_str: str):
    """Extracts the final numeric answer after #### in a GSM-8K answer field."""
    parts = answer_str.split("####")
    if len(parts) < 2:
        return None
    try:
        return float(parts[-1].strip().replace(",", ""))
    except ValueError:
        return None


def _make_gsm8k_distractors(correct_val: float, rng: random.Random, n: int = 3) -> list:
    """
    Generates n plausible-looking wrong numeric answers.
    Strategy: multiply/divide by small integers and round to same precision.
    """
    candidates = set()
    multipliers = [2, 3, 4, 0.5, 0.25, 1.5, 10, 0.1]
    offsets = [1, 2, 5, 10, 100]

    for m in multipliers:
        v = round(correct_val * m, 2)
        if v != correct_val and v > 0:
            candidates.add(v)
    for off in offsets:
        for v in [correct_val + off, correct_val - off]:
            if v != correct_val and v > 0:
                candidates.add(round(v, 2))

    candidates = list(candidates)
    rng.shuffle(candidates)
    distractors = candidates[:n]

    # Pad with fallback values if not enough distractors
    fallback = 1.0
    while len(distractors) < n:
        if fallback != correct_val and fallback not in distractors:
            distractors.append(fallback)
        fallback += 1.0

    def _fmt(v: float) -> str:
        return str(int(v)) if v == int(v) else str(v)

    return [_fmt(d) for d in distractors]


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class KnowledgeEvaluator:
    """
    Evaluates general knowledge across GSM-8K, HellaSwag, and TruthfulQA.

    Args:
        model:        HuggingFace model (already on device, in eval mode).
        tokenizer:    Matching tokenizer.
        device:       torch.device.
        n_samples:    Number of questions per benchmark (default 500).
        prefix:       W&B metric prefix (e.g. "pre_train", "post_train").
        results_csv:  Path to append CSV results to.
        seed:         RNG seed for reproducible sampling (default 42).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        n_samples: int = DEFAULT_N_SAMPLES,
        prefix: str = "knowledge_eval",
        results_csv: str = "results/knowledge_results.csv",
        seed: int = DEFAULT_SEED,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_samples = n_samples
        self.prefix = prefix
        self.results_csv = results_csv
        self.seed = seed

    def evaluate(self) -> dict:
        """Run all three benchmarks and return combined results dict."""
        self.model.eval()

        gsm8k_results = self._evaluate_gsm8k()
        hellaswag_results = self._evaluate_hellaswag()
        truthfulqa_results = self._evaluate_truthfulqa()

        overall_acc = round(
            (gsm8k_results["accuracy"] + hellaswag_results["accuracy"] + truthfulqa_results["accuracy"]) / 3,
            4,
        )

        combined = {
            "gsm8k_acc": gsm8k_results["accuracy"],
            "gsm8k_n": gsm8k_results["n_questions"],
            "hellaswag_acc": hellaswag_results["accuracy"],
            "hellaswag_n": hellaswag_results["n_questions"],
            "truthfulqa_mc1_acc": truthfulqa_results["accuracy"],
            "truthfulqa_n": truthfulqa_results["n_questions"],
            "overall_acc": overall_acc,
        }

        self._report(combined)
        return combined

    # ── GSM-8K ──────────────────────────────────────────────────────────────

    def _evaluate_gsm8k(self) -> dict:
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        print(f"\nLoading GSM-8K test set ({self.n_samples} questions)...")
        ds = load_dataset("openai/gsm8k", "main", split="test")

        indices = self._sample_indices(len(ds), self.n_samples, rng)
        questions = [ds[i] for i in indices]

        n_correct = 0
        n_skipped = 0

        for ex in tqdm(questions, desc="Knowledge eval (GSM-8K)"):
            correct_val = _extract_gsm8k_answer(ex["answer"])
            if correct_val is None:
                n_skipped += 1
                continue

            def _fmt(v: float) -> str:
                return str(int(v)) if v == int(v) else str(v)

            correct_str = _fmt(correct_val)
            distractors = _make_gsm8k_distractors(correct_val, rng)

            # Build 4-choice list: insert correct answer at random position
            correct_pos = rng.randint(0, 3)
            choices = distractors[:]
            choices.insert(correct_pos, correct_str)

            prompt, _ = _format_gsm8k_prompt(ex["question"], choices, correct_pos)
            scores = _get_answer_logprobs(self.model, self.tokenizer, prompt, self.device, _LETTERS)
            pred = int(max(range(4), key=lambda i: scores[i]))
            if pred == correct_pos:
                n_correct += 1

        n_evaluated = len(questions) - n_skipped
        accuracy = round(n_correct / n_evaluated, 4) if n_evaluated > 0 else 0.0
        return {"accuracy": accuracy, "n_questions": n_evaluated, "n_correct": n_correct}

    # ── HellaSwag ────────────────────────────────────────────────────────────

    def _evaluate_hellaswag(self) -> dict:
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        print(f"\nLoading HellaSwag validation set ({self.n_samples} questions)...")
        ds = load_dataset("Rowan/hellaswag", split="validation")

        indices = self._sample_indices(len(ds), self.n_samples, rng)
        questions = [ds[i] for i in indices]

        n_correct = 0

        for ex in tqdm(questions, desc="Knowledge eval (HellaSwag)"):
            correct_idx = int(ex["label"])
            endings = ex["endings"]
            prompt = _format_hellaswag_prompt(ex["ctx"], endings)
            scores = _get_answer_logprobs(self.model, self.tokenizer, prompt, self.device, _LETTERS[:len(endings)])
            pred = int(max(range(len(endings)), key=lambda i: scores[i]))
            if pred == correct_idx:
                n_correct += 1

        n = len(questions)
        accuracy = round(n_correct / n, 4) if n > 0 else 0.0
        return {"accuracy": accuracy, "n_questions": n, "n_correct": n_correct}

    # ── TruthfulQA MC1 ──────────────────────────────────────────────────────

    def _evaluate_truthfulqa(self) -> dict:
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        print(f"\nLoading TruthfulQA validation set ({self.n_samples} questions)...")
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")

        indices = self._sample_indices(len(ds), self.n_samples, rng)
        questions = [ds[i] for i in indices]

        n_correct = 0

        for ex in tqdm(questions, desc="Knowledge eval (TruthfulQA)"):
            mc1 = ex["mc1_targets"]
            choices = mc1["choices"]
            labels = mc1["labels"]

            correct_idx = next((i for i, l in enumerate(labels) if l == 1), None)
            if correct_idx is None:
                continue

            prompt_text, letters = _format_truthfulqa_prompt(ex["question"], choices)
            scores = _get_answer_logprobs(self.model, self.tokenizer, prompt_text, self.device, letters)
            pred = int(max(range(len(choices)), key=lambda i: scores[i]))
            if pred == correct_idx:
                n_correct += 1

        n = len(questions)
        accuracy = round(n_correct / n, 4) if n > 0 else 0.0
        return {"accuracy": accuracy, "n_questions": n, "n_correct": n_correct}

    # ── Reporting ────────────────────────────────────────────────────────────

    def _report(self, results: dict):
        p = self.prefix
        wandb.log({
            f"{p}/gsm8k_acc":         results["gsm8k_acc"],
            f"{p}/hellaswag_acc":      results["hellaswag_acc"],
            f"{p}/truthfulqa_mc1_acc": results["truthfulqa_mc1_acc"],
            f"{p}/overall_acc":        results["overall_acc"],
            f"{p}/gsm8k_n":            results["gsm8k_n"],
            f"{p}/hellaswag_n":        results["hellaswag_n"],
            f"{p}/truthfulqa_n":       results["truthfulqa_n"],
        })

        print(f"\n--- Knowledge Eval Results ---")
        print(f"  prefix:           {p}")
        print(f"  GSM-8K  acc:      {results['gsm8k_acc']:.3f}  (n={results['gsm8k_n']})")
        print(f"  HellaSwag acc:    {results['hellaswag_acc']:.3f}  (n={results['hellaswag_n']})")
        print(f"  TruthfulQA acc:   {results['truthfulqa_mc1_acc']:.3f}  (n={results['truthfulqa_n']})")
        print(f"  Overall acc:      {results['overall_acc']:.3f}")

        if self.results_csv:
            row = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "prefix": p,
                "seed": self.seed,
                "n_samples": self.n_samples,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()},
            }
            _write_csv_row(self.results_csv, row)
            print(f"  [saved to {self.results_csv}]")

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sample_indices(dataset_size: int, n: int, rng: random.Random) -> list:
        """Returns n evenly-spaced indices, then shuffles with the given rng."""
        if n >= dataset_size:
            indices = list(range(dataset_size))
        else:
            step = dataset_size / n
            indices = [int(i * step) for i in range(n)]
        rng.shuffle(indices)
        return indices
