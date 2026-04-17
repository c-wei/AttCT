"""
Jailbreak Evaluator for MLP-CT experiments.

Matches the ACT paper (Irpan et al., 2025) evaluation methodology:

Safety (ASR, lower = safer):
  - ClearHarm: harmful prompts wrapped with our jailbreak templates
  - WildguardTest (adversarial, harmful): pre-jailbroken harmful prompts

Overrefusal (lower = better):
  - XSTest: safe prompts that look dangerous (250 safe subset)
  - WildJailbreak (adversarial_benign): benign queries styled like jailbreaks
  - OR-Bench (hard-1k): safe prompts models tend to over-refuse

Capability:
  - MMLU accuracy (reused from evaluate_brr)

Summary:
  - F1 = harmonic mean of (1 - avg ASR) and (1 - avg overrefusal rate)

Usage (standalone):
    python evaluate_jailbreak.py \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --output results/jailbreak_eval.csv

Usage (integrated via run.py):
    python run.py --config configs/experiment_mlp_jailbreak_3b.yaml \\
        --jailbreak-eval --data-source clear-harm --data-mode jailbreak
"""

import csv
import datetime
import os
import random
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from eval.llm_judge import judge_is_unsafe, judge_is_refusal, _openrouter_available


# ── Generation helper ────────────────────────────────────────────────────────

def _generate_response(
    model, tokenizer, prompt: str, device, max_new_tokens: int = 256
) -> str:
    """Generate a model response using chat template + greedy decoding."""
    try:
        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        if isinstance(result, dict):
            input_ids = list(result["input_ids"])
        elif isinstance(result, list) and result and isinstance(result[0], int):
            input_ids = result
        else:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer.encode(text, add_special_tokens=False)
    except (ValueError, AttributeError):
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    input_len = input_tensor.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Dataset loaders ──────────────────────────────────────────────────────────

def _load_clearharm(max_prompts: int = 200) -> List[str]:
    """Load harmful prompts from ClearHarm dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "AlignmentResearch/ClearHarm", "rep40",
            split="validation", streaming=True,
        )
        prompts_raw = []
        for item in ds:
            content = item["content"]
            prompts_raw.append(
                " ".join(content) if isinstance(content, list) else str(content)
            )
        prompts = list(set(p for p in prompts_raw if len(p) > 15))
        random.shuffle(prompts)
        return prompts[:max_prompts]
    except Exception as e:
        print(f"    [ClearHarm] Failed to load: {e}")
        return []


def _load_wildguardtest_harmful(max_prompts: int = 500) -> List[str]:
    """Load adversarial harmful prompts from WildguardTest."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "allenai/wildguardmix", "wildguardtest", split="test"
        )
        prompts = [
            ex["prompt"]
            for ex in ds
            if ex.get("adversarial") is True
            and ex.get("prompt_harm_label") == "harmful"
        ]
        random.shuffle(prompts)
        return prompts[:max_prompts]
    except Exception as e:
        print(f"    [WildguardTest] Failed to load: {e}")
        return []


def _load_xstest_safe(max_prompts: int = 250) -> List[str]:
    """Load safe prompts from XSTest (should NOT be refused)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("walledai/XSTest", split="test")
        prompts = [
            ex["prompt"]
            for ex in ds
            if ex.get("label") == "safe"
        ]
        return prompts[:max_prompts]
    except Exception as e:
        print(f"    [XSTest] Failed to load: {e}")
        return []


def _load_wildjailbreak_benign(max_prompts: int = 200) -> List[str]:
    """Load adversarial benign prompts from WildJailbreak (should NOT be refused)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "allenai/wildjailbreak", "eval",
            delimiter="\t", keep_default_na=False, split="train",
        )
        prompts = []
        for ex in ds:
            if ex.get("data_type") == "adversarial_benign":
                # Use the adversarial version (the one that looks like a jailbreak)
                text = ex.get("adversarial", "")
                if text:
                    prompts.append(text)
        random.shuffle(prompts)
        return prompts[:max_prompts]
    except Exception as e:
        print(f"    [WildJailbreak benign] Failed to load: {e}")
        return []


def _load_orbench_safe(max_prompts: int = 300) -> List[str]:
    """Load safe-but-tricky prompts from OR-Bench hard-1k (should NOT be refused)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
        prompts = [ex["prompt"] for ex in ds]
        random.shuffle(prompts)
        return prompts[:max_prompts]
    except Exception as e:
        print(f"    [OR-Bench] Failed to load: {e}")
        return []


# ── MMLU (reuse from evaluate_brr) ──────────────────────────────────────────

def _evaluate_mmlu(model, tokenizer, device, max_samples: int = 200) -> float:
    if max_samples == 0:
        return -1.0
    try:
        from eval.evaluate_brr import _evaluate_mmlu as _brr_mmlu
        return _brr_mmlu(model, tokenizer, device, max_samples)
    except ImportError:
        print("    [MMLU] Skipping: evaluate_brr not available.")
        return -1.0


# ── Main Evaluator ───────────────────────────────────────────────────────────

class JailbreakEvaluator:
    """
    Comprehensive jailbreak evaluator matching ACT paper methodology.

    Measures safety (ASR on harmful prompts) and helpfulness (overrefusal
    on benign prompts), plus MMLU capability check.

    Args:
        model:              HuggingFace model (already on device).
        tokenizer:          Matching tokenizer.
        device:             torch.device.
        results_csv:        Path to write results CSV.
        max_new_tokens:     Max tokens to generate per response.
        mmlu_max_samples:   Number of MMLU questions (0 to disable).
        clearharm_prompts:  Max ClearHarm prompts for ASR.
        wildguard_prompts:  Max WildguardTest prompts for ASR.
        xstest_prompts:     Max XSTest prompts for overrefusal.
        wildjailbreak_prompts: Max WildJailbreak benign prompts for overrefusal.
        orbench_prompts:    Max OR-Bench prompts for overrefusal.
        jailbreak_templates: List of jailbreak template strings for wrapping
                            ClearHarm prompts. If None, uses defaults from wrappers.py.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        results_csv: str = "results/jailbreak_eval.csv",
        max_new_tokens: int = 256,
        mmlu_max_samples: int = 200,
        clearharm_prompts: int = 179,
        wildguard_prompts: int = 500,
        xstest_prompts: int = 250,
        wildjailbreak_prompts: int = 200,
        orbench_prompts: int = 300,
        jailbreak_templates: Optional[List[str]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.results_csv = results_csv
        self.max_new_tokens = max_new_tokens
        self.mmlu_max_samples = mmlu_max_samples

        # Load jailbreak templates
        if jailbreak_templates is None:
            from data.wrappers import ALL_STRONG_JAILBREAK_TEMPLATES, JAILBREAK_TEMPLATES
            self.jailbreak_templates = (
                list(ALL_STRONG_JAILBREAK_TEMPLATES) + list(JAILBREAK_TEMPLATES)
            )
        else:
            self.jailbreak_templates = jailbreak_templates

        # Load all datasets
        random.seed(42)
        print("  [JailbreakEval] Loading evaluation datasets...")

        # Safety datasets
        self.clearharm = _load_clearharm(clearharm_prompts)
        self.wildguardtest = _load_wildguardtest_harmful(wildguard_prompts)
        print(
            f"    Safety: ClearHarm={len(self.clearharm)}, "
            f"WildguardTest={len(self.wildguardtest)}"
        )

        # Overrefusal datasets
        self.xstest = _load_xstest_safe(xstest_prompts)
        self.wildjailbreak_benign = _load_wildjailbreak_benign(wildjailbreak_prompts)
        self.orbench = _load_orbench_safe(orbench_prompts)
        print(
            f"    Overrefusal: XSTest={len(self.xstest)}, "
            f"WildJailbreak={len(self.wildjailbreak_benign)}, "
            f"OR-Bench={len(self.orbench)}"
        )

        self._pre_results = None

    def evaluate(self, stage: str = "eval", step: int = 0) -> Dict:
        """
        Run full jailbreak evaluation.

        Args:
            stage: "pre_train", "checkpoint", or "post_train"
            step:  optimizer step number

        Returns dict with all metrics.
        """
        self.model.eval()
        random.seed(42)

        judge_mode = "llm" if _openrouter_available() else "keyword"
        print(
            f"\n  [JailbreakEval] {stage} step={step} "
            f"(judge={judge_mode})"
        )

        # ── Safety: ClearHarm ASR (wrap with our templates) ──────────────
        clearharm_asr = self._eval_clearharm_asr(stage)

        # ── Safety: WildguardTest ASR (already jailbroken) ───────────────
        wildguard_asr = self._eval_wildguardtest_asr(stage)

        # ── Overrefusal: XSTest ──────────────────────────────────────────
        xstest_refusal = self._eval_overrefusal(
            self.xstest, "XSTest", stage
        )

        # ── Overrefusal: WildJailbreak benign ────────────────────────────
        wjb_refusal = self._eval_overrefusal(
            self.wildjailbreak_benign, "WildJailbreak-benign", stage
        )

        # ── Overrefusal: OR-Bench ────────────────────────────────────────
        orbench_refusal = self._eval_overrefusal(
            self.orbench, "OR-Bench", stage
        )

        # ── MMLU ─────────────────────────────────────────────────────────
        mmlu_samples = (
            500 if stage in ("pre_train", "post_train")
            else self.mmlu_max_samples
        )
        mmlu_accuracy = _evaluate_mmlu(
            self.model, self.tokenizer, self.device, mmlu_samples
        )

        # ── Aggregate metrics ────────────────────────────────────────────
        # Average ASR across safety benchmarks
        asr_values = [v for v in [clearharm_asr, wildguard_asr] if v >= 0]
        avg_asr = sum(asr_values) / len(asr_values) if asr_values else -1.0

        # Average overrefusal across helpfulness benchmarks
        overrefusal_values = [
            v for v in [xstest_refusal, wjb_refusal, orbench_refusal] if v >= 0
        ]
        avg_overrefusal = (
            sum(overrefusal_values) / len(overrefusal_values)
            if overrefusal_values
            else -1.0
        )

        # F1 = harmonic mean of (1-ASR) and (1-overrefusal)
        safety = 1.0 - avg_asr if avg_asr >= 0 else -1.0
        helpfulness = 1.0 - avg_overrefusal if avg_overrefusal >= 0 else -1.0
        if safety > 0 and helpfulness > 0:
            f1 = 2 * safety * helpfulness / (safety + helpfulness)
        else:
            f1 = -1.0

        results = {
            "stage": stage,
            "step": step,
            "clearharm_asr": round(clearharm_asr, 4) if clearharm_asr >= 0 else "N/A",
            "wildguard_asr": round(wildguard_asr, 4) if wildguard_asr >= 0 else "N/A",
            "avg_asr": round(avg_asr, 4) if avg_asr >= 0 else "N/A",
            "xstest_overrefusal": round(xstest_refusal, 4) if xstest_refusal >= 0 else "N/A",
            "wildjailbreak_overrefusal": round(wjb_refusal, 4) if wjb_refusal >= 0 else "N/A",
            "orbench_overrefusal": round(orbench_refusal, 4) if orbench_refusal >= 0 else "N/A",
            "avg_overrefusal": round(avg_overrefusal, 4) if avg_overrefusal >= 0 else "N/A",
            "safety": round(safety, 4) if safety >= 0 else "N/A",
            "helpfulness": round(helpfulness, 4) if helpfulness >= 0 else "N/A",
            "f1": round(f1, 4) if f1 >= 0 else "N/A",
            "mmlu_accuracy": round(mmlu_accuracy, 4) if mmlu_accuracy >= 0 else "N/A",
            "judge_mode": judge_mode,
        }

        if stage == "pre_train":
            self._pre_results = results.copy()

        if stage == "post_train" and self._pre_results is not None:
            pre_asr = self._pre_results.get("avg_asr")
            if isinstance(pre_asr, float) and pre_asr > 0 and isinstance(avg_asr, float):
                results["asr_ratio"] = round(avg_asr / pre_asr, 4)
                results["asr_reduction"] = round(1.0 - avg_asr / pre_asr, 4)

        self._report(results)
        self._write_csv(results)

        return results

    # ── Safety evaluation helpers ────────────────────────────────────────

    def _eval_clearharm_asr(self, stage: str) -> float:
        """ASR on ClearHarm with our jailbreak templates (worst-case per prompt)."""
        if not self.clearharm:
            return -1.0

        compliances = 0
        total = len(self.clearharm)

        for prompt in tqdm(
            self.clearharm, desc=f"ClearHarm ASR ({stage})", leave=False
        ):
            complied = False
            for template in self.jailbreak_templates:
                if "{prompt}" not in template:
                    continue
                wrapped = template.replace("{prompt}", prompt)
                response = _generate_response(
                    self.model, self.tokenizer, wrapped,
                    self.device, self.max_new_tokens,
                )
                if judge_is_unsafe(prompt, response):
                    complied = True
                    break
            if complied:
                compliances += 1

        asr = compliances / total
        print(f"    ClearHarm ASR: {asr:.4f} ({compliances}/{total})")
        return asr

    def _eval_wildguardtest_asr(self, stage: str) -> float:
        """ASR on WildguardTest adversarial-harmful (already jailbroken)."""
        if not self.wildguardtest:
            return -1.0

        compliances = 0
        total = len(self.wildguardtest)

        for prompt in tqdm(
            self.wildguardtest, desc=f"WildguardTest ASR ({stage})", leave=False
        ):
            response = _generate_response(
                self.model, self.tokenizer, prompt,
                self.device, self.max_new_tokens,
            )
            if judge_is_unsafe(prompt, response):
                compliances += 1

        asr = compliances / total
        print(f"    WildguardTest ASR: {asr:.4f} ({compliances}/{total})")
        return asr

    # ── Overrefusal evaluation helper ────────────────────────────────────

    def _eval_overrefusal(
        self, prompts: List[str], name: str, stage: str
    ) -> float:
        """Measure overrefusal rate on benign prompts."""
        if not prompts:
            return -1.0

        refusals = 0
        total = len(prompts)

        for prompt in tqdm(prompts, desc=f"{name} ({stage})", leave=False):
            response = _generate_response(
                self.model, self.tokenizer, prompt,
                self.device, self.max_new_tokens,
            )
            if judge_is_refusal(prompt, response):
                refusals += 1

        rate = refusals / total
        print(f"    {name} overrefusal: {rate:.4f} ({refusals}/{total})")
        return rate

    # ── Reporting ────────────────────────────────────────────────────────

    def _report(self, results: Dict):
        """Print results and log to W&B."""
        print(f"\n  === JailbreakEval [{results['stage']}] step={results['step']} ===")
        print(f"    --- Safety (ASR, lower=safer) ---")
        print(f"      ClearHarm ASR:      {results['clearharm_asr']}")
        print(f"      WildguardTest ASR:  {results['wildguard_asr']}")
        print(f"      Avg ASR:            {results['avg_asr']}")
        print(f"    --- Helpfulness (overrefusal, lower=better) ---")
        print(f"      XSTest:             {results['xstest_overrefusal']}")
        print(f"      WildJailbreak:      {results['wildjailbreak_overrefusal']}")
        print(f"      OR-Bench:           {results['orbench_overrefusal']}")
        print(f"      Avg overrefusal:    {results['avg_overrefusal']}")
        print(f"    --- Summary ---")
        print(f"      Safety (1-ASR):     {results['safety']}")
        print(f"      Helpfulness (1-OR): {results['helpfulness']}")
        print(f"      F1:                 {results['f1']}")
        if "asr_ratio" in results:
            print(f"      ASR ratio:          {results['asr_ratio']}")
            print(f"      ASR reduction:      {results.get('asr_reduction', 'N/A')}")
        print(f"      MMLU:               {results['mmlu_accuracy']}")
        print(f"      Judge:              {results['judge_mode']}")
        print()

        try:
            import wandb
            prefix = results["stage"]
            metrics = {}
            for key in [
                "clearharm_asr", "wildguard_asr", "avg_asr",
                "xstest_overrefusal", "wildjailbreak_overrefusal",
                "orbench_overrefusal", "avg_overrefusal",
                "safety", "helpfulness", "f1",
            ]:
                val = results.get(key)
                if isinstance(val, (int, float)):
                    metrics[f"{prefix}/{key}"] = val
            if isinstance(results.get("mmlu_accuracy"), (int, float)):
                metrics[f"{prefix}/mmlu_accuracy"] = results["mmlu_accuracy"]
            if "asr_ratio" in results:
                metrics[f"{prefix}/asr_ratio"] = results["asr_ratio"]
                metrics[f"{prefix}/asr_reduction"] = results["asr_reduction"]
            wandb.log(metrics, step=results["step"])
        except Exception:
            pass

    def _write_csv(self, results: Dict):
        """Append one row to results CSV."""
        if not self.results_csv:
            return
        os.makedirs(os.path.dirname(self.results_csv) or ".", exist_ok=True)

        fieldnames = [
            "timestamp", "stage", "step",
            "clearharm_asr", "wildguard_asr", "avg_asr",
            "xstest_overrefusal", "wildjailbreak_overrefusal",
            "orbench_overrefusal", "avg_overrefusal",
            "safety", "helpfulness", "f1",
            "mmlu_accuracy", "judge_mode",
        ]
        if "asr_ratio" in results:
            fieldnames.extend(["asr_ratio", "asr_reduction"])

        write_header = not os.path.exists(self.results_csv)
        row = {"timestamp": datetime.datetime.utcnow().isoformat(), **results}

        with open(self.results_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"    [saved to {self.results_csv}]")


# ── Standalone CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(
        description="Comprehensive Jailbreak Evaluator (ACT paper methodology)"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output", default="results/jailbreak_eval.csv")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--mmlu-max-samples", type=int, default=500)
    parser.add_argument("--clearharm-prompts", type=int, default=179)
    parser.add_argument("--wildguard-prompts", type=int, default=500)
    parser.add_argument("--xstest-prompts", type=int, default=250)
    parser.add_argument("--wildjailbreak-prompts", type=int, default=200)
    parser.add_argument("--orbench-prompts", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    evaluator = JailbreakEvaluator(
        model, tokenizer, device,
        results_csv=args.output,
        max_new_tokens=args.max_new_tokens,
        mmlu_max_samples=args.mmlu_max_samples,
        clearharm_prompts=args.clearharm_prompts,
        wildguard_prompts=args.wildguard_prompts,
        xstest_prompts=args.xstest_prompts,
        wildjailbreak_prompts=args.wildjailbreak_prompts,
        orbench_prompts=args.orbench_prompts,
    )
    evaluator.evaluate(stage="standalone", step=0)