import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.datasets_ClearHarm import AttCTDataset, collate_fn_batch1, get_prompts
from losses.losses import _jsd, _get_layer_weight
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

"""
Evaluation utilities for Attention Consistency Training (AttCT).

Measures:
1. Attention consistency metrics (JSD, KL, L2 between clean and adversarial attention)
2. Wrapper attention mass (how much attention flows to wrapper tokens)
3. Refusal rate under adversarial wrapping (safety metric)
4. Generation quality / output similarity (clean vs wrapped generation)
"""


import torch.nn.functional as F

from data.wrappers_jailbreak import (
    REFUSAL_KEYWORDS,
    STRONG_JAILBREAK_TEMPLATES,
    JAILBREAK_TEMPLATES,
    AdversarialWrapper,
    IdentityWrapper,
)


# ---------------------------------------------------------------------------
# Core attention metrics
# ---------------------------------------------------------------------------


def compute_attention_metrics(
    clean_attentions: Tuple[torch.Tensor, ...],
    adv_attentions: Tuple[torch.Tensor, ...],
    start_index: int,
    clean_len: int,
) -> Dict[str, float]:
    """
    Compute per-layer and aggregate attention consistency metrics.

    Args:
        clean_attentions: Tuple of [batch, heads, clean_seq, clean_seq] per layer.
        adv_attentions:   Tuple of [batch, heads, adv_seq, adv_seq] per layer.
        start_index:      Token offset where the clean prompt starts in the adversarial sequence.
        clean_len:        Number of tokens in the clean prompt.

    Returns:
        Dictionary of scalar metrics.
    """
    num_layers = len(clean_attentions)
    end_index = start_index + clean_len

    layer_jsd = []
    layer_kl = []
    layer_l2 = []

    for layer_idx in range(num_layers):
        clean_att = clean_attentions[layer_idx]
        adv_att = adv_attentions[layer_idx]

        sliced_adv = adv_att[:, :, start_index:end_index, start_index:end_index]

        min_q = min(clean_att.shape[-2], sliced_adv.shape[-2])
        min_k = min(clean_att.shape[-1], sliced_adv.shape[-1])

        p = clean_att[..., :min_q, :min_k].detach()
        q = sliced_adv[..., :min_q, :min_k].detach()

        # JSD
        jsd_val = _jsd(p, q).item()
        layer_jsd.append(jsd_val)

        # KL (clean || adv)
        log_q = torch.clamp(q, min=1e-9).log()
        kl_val = F.kl_div(log_q, p, reduction="batchmean").item()
        layer_kl.append(kl_val)

        # L2 (MSE)
        l2_val = F.mse_loss(q, p).item()
        layer_l2.append(l2_val)

    return {
        "jsd_mean": sum(layer_jsd) / num_layers,
        "kl_mean": sum(layer_kl) / num_layers,
        "l2_mean": sum(layer_l2) / num_layers,
        "jsd_per_layer": layer_jsd,
        "kl_per_layer": layer_kl,
        "l2_per_layer": layer_l2,
        "jsd_last_layer": layer_jsd[-1],
        "kl_last_layer": layer_kl[-1],
        "l2_last_layer": layer_l2[-1],
    }


def compute_wrapper_attention_mass(
    adv_attentions: Tuple[torch.Tensor, ...],
    start_index: int,
    clean_len: int,
    wrapper_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Measure how much attention mass flows to wrapper (non-clean) token positions.

    A well-trained AttCT model should show low wrapper attention mass,
    meaning the adversarial wrapper tokens are effectively ignored.

    Returns:
        Dictionary with per-layer and aggregate wrapper attention statistics.
    """
    num_layers = len(adv_attentions)
    layer_wrapper_mass = []

    for adv_att in adv_attentions:
        batch_size, num_heads, adv_seq_q, adv_seq_k = adv_att.shape

        if wrapper_mask is not None:
            mask = wrapper_mask.float().unsqueeze(1).unsqueeze(2)
            mask = mask.expand(batch_size, num_heads, adv_seq_q, adv_seq_k)
        else:
            # Default: prefix tokens [0, start_index) are wrapper
            mask = torch.zeros(
                batch_size, num_heads, adv_seq_q, adv_seq_k, device=adv_att.device
            )
            mask[:, :, :, :start_index] = 1.0
            # Also mark suffix tokens as wrapper
            end_index = start_index + clean_len
            if end_index < adv_seq_k:
                mask[:, :, :, end_index:] = 1.0

        wrapper_mass = (adv_att.detach() * mask).sum(dim=-1).mean().item()
        layer_wrapper_mass.append(wrapper_mass)

    return {
        "wrapper_attention_mean": sum(layer_wrapper_mass) / num_layers,
        "wrapper_attention_per_layer": layer_wrapper_mass,
        "wrapper_attention_last_layer": layer_wrapper_mass[-1],
    }


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------


def detect_refusal(text: str, keywords: List[str] = None) -> bool:
    """Check if a generation contains refusal indicators."""
    if keywords is None:
        keywords = REFUSAL_KEYWORDS
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def compute_refusal_rate(
    model,
    tokenizer,
    prompts: List[str],
    wrapper: AdversarialWrapper,
    device: torch.device,
    max_new_tokens: int = 128,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Measure refusal rate: fraction of adversarially-wrapped harmful prompts
    where the model still refuses.

    Higher refusal rate = safer model = better AttCT training.

    Returns:
        Dictionary with refusal statistics and per-prompt details.
    """
    model.eval()
    refusals = 0
    total = 0
    results = []

    for prompt in tqdm(prompts, desc="Evaluating refusal rate"):
        wrapped_text, _, _ = wrapper.wrap(prompt)

        inputs = tokenizer(
            wrapped_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        refused = detect_refusal(generated_text)
        refusals += int(refused)
        total += 1

        results.append(
            {
                "prompt": prompt,
                "wrapped_prompt": wrapped_text[:200] + "...",
                "generation": generated_text[:300],
                "refused": refused,
            }
        )

    refusal_rate = refusals / total if total > 0 else 0.0

    return {
        "refusal_rate": refusal_rate,
        "total_prompts": total,
        "refusals": refusals,
        "compliances": total - refusals,
        "per_prompt_results": results,
    }


# ---------------------------------------------------------------------------
# Generation similarity (clean vs adversarial)
# ---------------------------------------------------------------------------


def compute_generation_similarity(
    model,
    tokenizer,
    prompts: List[str],
    wrapper: AdversarialWrapper,
    device: torch.device,
    max_new_tokens: int = 64,
) -> Dict[str, float]:
    """
    Compare generations from clean vs adversarially-wrapped prompts.

    Measures whether the model produces similar outputs regardless of wrapping.
    Uses simple token-level overlap (Jaccard) as a lightweight proxy.

    Returns:
        Dictionary with similarity statistics.
    """
    model.eval()
    similarities = []

    for prompt in tqdm(prompts, desc="Evaluating generation similarity"):
        wrapped_text, _, _ = wrapper.wrap(prompt)

        # Generate from clean prompt
        clean_inputs = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(device)
        wrapped_inputs = tokenizer(
            wrapped_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            clean_out = model.generate(
                **clean_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            wrapped_out = model.generate(
                **wrapped_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        clean_gen = tokenizer.decode(
            clean_out[0, clean_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        wrapped_gen = tokenizer.decode(
            wrapped_out[0, wrapped_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Jaccard similarity on token sets
        clean_tokens = set(clean_gen.lower().split())
        wrapped_tokens = set(wrapped_gen.lower().split())

        if len(clean_tokens | wrapped_tokens) > 0:
            jaccard = len(clean_tokens & wrapped_tokens) / len(
                clean_tokens | wrapped_tokens
            )
        else:
            jaccard = 1.0

        similarities.append(jaccard)

    return {
        "mean_jaccard_similarity": sum(similarities) / len(similarities)
        if similarities
        else 0.0,
        "min_jaccard_similarity": min(similarities) if similarities else 0.0,
        "max_jaccard_similarity": max(similarities) if similarities else 0.0,
        "num_prompts": len(similarities),
    }


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------


class AttCTEvaluator:
    """
    End-to-end evaluator for AttCT models.

    Runs all metrics:
        1. Attention consistency (JSD, KL, L2) on clean vs adversarial
        2. Wrapper attention mass
        3. Refusal rate under adversarial wrapping
        4. Generation similarity (clean vs wrapped)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        eval_prompts: Optional[List[str]] = None,
        wrapper: Optional[AdversarialWrapper] = None,
        use_strong_templates: bool = True,
        max_prompts: int = 100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.wrapper = wrapper or AdversarialWrapper(
            templates=STRONG_JAILBREAK_TEMPLATES if use_strong_templates else JAILBREAK_TEMPLATES,
            strategy="sequential",
            use_strong_templates=use_strong_templates,
        )

        if eval_prompts is None:
            eval_prompts = get_prompts("hardcoded", limit=max_prompts)
        self.eval_prompts = eval_prompts[:max_prompts]

    def evaluate_attention_consistency(
        self, num_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute attention consistency metrics across evaluation prompts.

        Returns aggregated JSD, KL, L2 metrics and wrapper attention mass.
        """
        self.model.eval()
        prompts = self.eval_prompts[:num_samples] if num_samples else self.eval_prompts

        dataset = AttCTDataset(
            prompts=prompts,
            tokenizer=self.tokenizer,
            wrapper=self.wrapper,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_batch1)

        all_metrics = defaultdict(list)

        for batch in tqdm(dataloader, desc="Evaluating attention consistency"):
            clean_ids = batch["clean_input_ids"].to(self.device)
            clean_mask = batch["clean_attention_mask"].to(self.device)
            wrapped_ids = batch["wrapped_input_ids"].to(self.device)
            wrapped_mask = batch["wrapped_attention_mask"].to(self.device)
            start_index = int(batch["start_index"][0].item())
            clean_len = int(batch["clean_len"][0].item())

            with torch.no_grad():
                clean_out = self.model(
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                    output_attentions=True,
                    output_hidden_states=False,
                )
                adv_out = self.model(
                    input_ids=wrapped_ids,
                    attention_mask=wrapped_mask,
                    output_attentions=True,
                    output_hidden_states=False,
                )

            attn_metrics = compute_attention_metrics(
                clean_out.attentions, adv_out.attentions, start_index, clean_len
            )
            wrapper_metrics = compute_wrapper_attention_mass(
                adv_out.attentions, start_index, clean_len
            )

            for k, v in attn_metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[k].append(v)
            for k, v in wrapper_metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[k].append(v)

        # Aggregate
        aggregated = {}
        for k, vals in all_metrics.items():
            aggregated[f"{k}_mean"] = sum(vals) / len(vals)
            aggregated[f"{k}_std"] = (
                sum((v - aggregated[f"{k}_mean"]) ** 2 for v in vals) / len(vals)
            ) ** 0.5

        aggregated["num_samples"] = len(prompts)
        return aggregated

    def evaluate_refusal_rate(
        self,
        num_samples: Optional[int] = None,
        max_new_tokens: int = 128,
    ) -> Dict[str, float]:
        """Measure refusal rate under adversarial wrapping."""
        prompts = self.eval_prompts[:num_samples] if num_samples else self.eval_prompts
        return compute_refusal_rate(
            self.model,
            self.tokenizer,
            prompts,
            self.wrapper,
            self.device,
            max_new_tokens=max_new_tokens,
        )

    def evaluate_generation_similarity(
        self,
        num_samples: Optional[int] = None,
        max_new_tokens: int = 64,
    ) -> Dict[str, float]:
        """Compare clean vs adversarial generation similarity."""
        prompts = self.eval_prompts[:num_samples] if num_samples else self.eval_prompts
        return compute_generation_similarity(
            self.model,
            self.tokenizer,
            prompts,
            self.wrapper,
            self.device,
            max_new_tokens=max_new_tokens,
        )

    def run_full_evaluation(
        self,
        num_attention_samples: int = 50,
        num_refusal_samples: int = 50,
        num_similarity_samples: int = 20,
        max_new_tokens: int = 128,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Run the complete evaluation suite.

        Args:
            num_attention_samples:  Number of prompts for attention metrics.
            num_refusal_samples:    Number of prompts for refusal rate.
            num_similarity_samples: Number of prompts for generation similarity.
            max_new_tokens:         Max tokens to generate for refusal/similarity.
            output_path:            If provided, save results as JSON.

        Returns:
            Combined results dictionary.
        """
        print("=" * 60)
        print("AttCT Evaluation Suite")
        print("=" * 60)

        results = {}

        # 1. Attention consistency
        print("\n[1/3] Attention Consistency Metrics...")
        attn_results = self.evaluate_attention_consistency(
            num_samples=num_attention_samples
        )
        results["attention_consistency"] = attn_results
        print(f"  JSD (mean):              {attn_results.get('jsd_mean_mean', 'N/A'):.6f}")
        print(f"  KL (mean):               {attn_results.get('kl_mean_mean', 'N/A'):.6f}")
        print(f"  L2 (mean):               {attn_results.get('l2_mean_mean', 'N/A'):.6f}")
        print(
            f"  Wrapper attn mass (mean): {attn_results.get('wrapper_attention_mean_mean', 'N/A'):.6f}"
        )

        # 2. Refusal rate
        print("\n[2/3] Refusal Rate...")
        refusal_results = compute_refusal_rate(
            self.model,
            self.tokenizer,
            self.eval_prompts[:num_refusal_samples],
            self.wrapper,
            self.device,
            max_new_tokens=max_new_tokens,
        )
        # Don't include per-prompt details in top-level summary
        results["refusal"] = {
            k: v for k, v in refusal_results.items() if k != "per_prompt_results"
        }
        results["refusal_details"] = refusal_results.get("per_prompt_results", [])
        print(f"  Refusal rate: {refusal_results['refusal_rate']:.2%}")
        print(
            f"  ({refusal_results['refusals']}/{refusal_results['total_prompts']} refused)"
        )

        # 3. Generation similarity
        print("\n[3/3] Generation Similarity (clean vs wrapped)...")
        sim_results = self.evaluate_generation_similarity(
            num_samples=num_similarity_samples,
            max_new_tokens=max_new_tokens,
        )
        results["generation_similarity"] = sim_results
        print(
            f"  Mean Jaccard similarity: {sim_results['mean_jaccard_similarity']:.4f}"
        )

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Attention JSD:           {attn_results.get('jsd_mean_mean', 0):.6f}  (lower = more consistent)")
        print(f"  Wrapper Attention Mass:  {attn_results.get('wrapper_attention_mean_mean', 0):.6f}  (lower = better)")
        print(f"  Refusal Rate:            {refusal_results['refusal_rate']:.2%}  (higher = safer)")
        print(f"  Generation Similarity:   {sim_results['mean_jaccard_similarity']:.4f}  (higher = more robust)")

        # Save results
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            # Filter out non-serializable values
            serializable = _make_serializable(results)
            with open(output_path, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"\nResults saved to {output_path}")

        return results


def _make_serializable(obj):
    """Recursively convert tensors and other non-JSON types to serializable formats."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        return obj
    return obj


# ---------------------------------------------------------------------------
# Comparison utility: before vs after training
# ---------------------------------------------------------------------------


def compare_models(
    base_model,
    trained_model,
    tokenizer,
    device: torch.device,
    prompts: Optional[List[str]] = None,
    num_samples: int = 30,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Compare a base model vs an AttCT-trained model on all metrics.

    Useful for measuring the impact of consistency training.
    """
    if prompts is None:
        prompts = get_prompts("hardcoded", limit=num_samples)

    results = {}

    for label, model in [("base", base_model), ("trained", trained_model)]:
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        evaluator = AttCTEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            eval_prompts=prompts,
            use_strong_templates=True,
        )

        results[label] = evaluator.run_full_evaluation(
            num_attention_samples=num_samples,
            num_refusal_samples=num_samples,
            num_similarity_samples=min(num_samples, 10),
        )

    # Compute deltas
    deltas = {}
    base_attn = results["base"].get("attention_consistency", {})
    trained_attn = results["trained"].get("attention_consistency", {})

    for metric in ["jsd_mean_mean", "kl_mean_mean", "l2_mean_mean", "wrapper_attention_mean_mean"]:
        base_val = base_attn.get(metric, 0)
        trained_val = trained_attn.get(metric, 0)
        deltas[metric] = trained_val - base_val

    base_refusal = results["base"].get("refusal", {}).get("refusal_rate", 0)
    trained_refusal = results["trained"].get("refusal", {}).get("refusal_rate", 0)
    deltas["refusal_rate"] = trained_refusal - base_refusal

    results["deltas"] = deltas

    print(f"\n{'='*60}")
    print("Comparison (trained - base):")
    print(f"{'='*60}")
    for k, v in deltas.items():
        direction = "↓ better" if (("jsd" in k or "kl" in k or "l2" in k or "wrapper" in k) and v < 0) else ""
        if "refusal" in k and v > 0:
            direction = "↑ better"
        print(f"  Δ {k}: {v:+.6f}  {direction}")

    if output_path:
        serializable = _make_serializable(results)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nComparison results saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate an AttCT-trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to PEFT/LoRA adapter (if applicable)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=50, help="Number of evaluation samples"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Max tokens to generate"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results.json", help="Output JSON path"
    )
    parser.add_argument(
        "--prompt_source",
        type=str,
        default="hardcoded",
        help="Prompt source: 'hardcoded', 'clear-harm', or file path",
    )
    parser.add_argument(
        "--use_strong_templates",
        action="store_true",
        default=True,
        help="Use strong jailbreak templates",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model & tokenizer

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    if args.adapter_path:

        print(f"Loading adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    if device.type != "cuda":
        model = model.to(device)

    model.eval()

    # Load prompts
    prompts = get_prompts(args.prompt_source, limit=args.num_samples)
    print(f"Loaded {len(prompts)} evaluation prompts")

    # Run evaluation
    evaluator = AttCTEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_prompts=prompts,
        use_strong_templates=args.use_strong_templates,
    )

    results = evaluator.run_full_evaluation(
        num_attention_samples=min(args.num_samples, 50),
        num_refusal_samples=args.num_samples,
        num_similarity_samples=min(args.num_samples, 20),
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
    )