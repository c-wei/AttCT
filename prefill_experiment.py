from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import json
from datasets import load_dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

def visualize_attention_layers(
    attn_baseline: torch.Tensor, #(num_layers, num_heads, seq_base, seq_base)
    attn_attack: torch.Tensor, #(num_layers, num_heads, seq_attack, seq_attack)
    kl_per_layer: list[float],
    tokens_baseline: list[str],
    prefill: str,
    harmful_prompt: str,
    output_path: str,
    top_k_layers: int = 6,
):
    """
    For each of the top-k highest-KL layers, plot:
      - Left:  baseline last-token attention (over baseline tokens)
      - Middle: attack last-token attention (over baseline tokens)
      - Right:  difference (attack - baseline)
    Saves as a single JPEG file.
    """
    num_layers = attn_baseline.shape[0]
    seq_base = attn_baseline.shape[2]

    kl_tensor = torch.tensor(kl_per_layer)
    top_layers = kl_tensor.topk(min(top_k_layers, num_layers)).indices.tolist()
    top_layers_sorted = sorted(top_layers, key=lambda i: -kl_per_layer[i])

    tokens = tokens_baseline[:seq_base]

    n_rows = len(top_layers_sorted)
    fig = plt.figure(figsize=(24, 5 * n_rows))
    fig.suptitle(
        f"Attention Analysis — Prefill: {prefill!r}\n"
        f"Prompt: {str(harmful_prompt)[:80]}",
        fontsize=11, y=1.01, wrap=True
    )

    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.5, wspace=0.35)

    for row_idx, layer_idx in enumerate(top_layers_sorted):
        base_last = attn_baseline[layer_idx, :, -1, :].mean(dim=0).cpu().numpy()

        attack_last_raw = attn_attack[layer_idx, :, -1, :seq_base]
        attack_last_raw = attack_last_raw / attack_last_raw.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        attack_last = attack_last_raw.mean(dim=0).cpu().numpy()

        diff = attack_last - base_last
        kl_val = kl_per_layer[layer_idx]

        short_tokens = [t.replace("Ġ", " ").replace("Ċ", "\\n")[:8] for t in tokens]

        def make_heatmap(ax, data, title, cmap, vmin=None, vmax=None):
            """Plot 1D attention as a (1, seq_len) heatmap with token labels."""
            im = ax.imshow(
                data.reshape(1, -1),
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks(range(len(short_tokens)))
            ax.set_xticklabels(short_tokens, rotation=45, ha="right", fontsize=7)
            ax.set_yticks([])
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax_base   = fig.add_subplot(gs[row_idx, 0])
        ax_attack = fig.add_subplot(gs[row_idx, 1])
        ax_diff   = fig.add_subplot(gs[row_idx, 2])

        vmax_shared = max(base_last.max(), attack_last.max())

        make_heatmap(ax_base,   base_last,   f"Layer {layer_idx} — Baseline",        "Blues", 0, vmax_shared)
        make_heatmap(ax_attack, attack_last, f"Layer {layer_idx} — Attack (KL={kl_val:.4f})", "Reds",  0, vmax_shared)
        make_heatmap(ax_diff,   diff,        f"Layer {layer_idx} — Δ (Attack−Baseline)", "RdBu_r",
                     vmin=-abs(diff).max(), vmax=abs(diff).max())

    plt.tight_layout()
    fig.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved attention visualization → {output_path}")


def get_tokens(prompt_text: str) -> list[str]:
    """Return the string representation of each token in the prompt."""
    ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
    return [tokenizer.convert_ids_to_tokens(i.item()) for i in ids]

    
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    output_attentions=True,
)
model.eval()

# dataset = load_dataset("AlignmentResearch/ClearHarm", split="train", streaming=True)
dataset = load_dataset("allenai/wildjailbreak", split="train", streaming=True)


def build_prompt(harmful_prompt: str, prefill_text: str = "") -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": harmful_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + prefill_text


def get_attentions(prompt_text: str) -> tuple[torch.Tensor, int]:
    """
    Returns:
        attentions: tensor of shape (num_layers, num_heads, seq_len, seq_len)
        seq_len: int
    """
    inputs = tokenizer(prompt_text, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = torch.stack([layer[0].float() for layer in outputs.attentions])

    print(f"  [DEBUG] attentions shape: {attentions.shape}")
    print(f"  [DEBUG] attentions dtype: {attentions.dtype}")
    print(f"  [DEBUG] attentions min: {attentions.min():.6f}, max: {attentions.max():.6f}")
    print(f"  [DEBUG] attentions mean: {attentions.mean():.6f}")
    print(f"  [DEBUG] any NaN: {attentions.isnan().any()}, any Inf: {attentions.isinf().any()}")
    row_sums = attentions[0, 0].sum(dim=-1)
    print(f"  [DEBUG] layer0 head0 row sums (should be ~1.0): {row_sums}")
    
    return attentions, seq_len


def kl_divergence_per_layer(
    attn_baseline: torch.Tensor,
    attn_attack: torch.Tensor,
) -> list[float]:
    """
    Compute mean KL divergence per layer between two attention tensors.
    Compares only over the shared (baseline) sequence length to handle
    the extra prefill tokens in the attacked prompt.

    Args:
        attn_baseline: (num_layers, num_heads, seq_len_base, seq_len_base)
        attn_attack:   (num_layers, num_heads, seq_len_attack, seq_len_attack)

    Returns:
        List of per-layer mean KL divergences (float), length = num_layers
    """
    num_layers = attn_baseline.shape[0]
    seq_len = attn_baseline.shape[2]

    kl_per_layer = []

    for layer_idx in range(num_layers):
        base_last = attn_baseline[layer_idx, :, -1, :]
        attack_last = attn_attack[layer_idx, :, -1, :seq_len]

        attack_last = attack_last / attack_last.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        base_last = base_last.clamp(min=1e-10)
        attack_last = attack_last.clamp(min=1e-10)

        kl = (base_last * (base_last.log() - attack_last.log())).sum(dim=-1)
        kl_per_layer.append(kl.mean().item())

    return kl_per_layer


prefill_variants = [
    "Sure! ",
    "Of course, here's how:",
    "Absolutely. Step 1:",
    "Here is the information you requested:",
]

results = []
os.makedirs("attention_viz", exist_ok=True)

for item_idx, item in enumerate(dataset.take(5)):
    if item["data_type"] != "adversarial_harmful":
        continue
    harmful_prompt = item["data_type"]
    # harmful_prompt = item["content"]
    print(f"\nHarmful Prompt: {harmful_prompt}\n{'-'*80}")

    entry = {
        "harmful_prompt": harmful_prompt,
        "num_layers": None,
        "baseline_seq_len": None,
        "prefill_attacks": [],
    }

    baseline_prompt = build_prompt(harmful_prompt)
    attn_baseline, baseline_seq_len = get_attentions(baseline_prompt)
    baseline_tokens = get_tokens(baseline_prompt)

    entry["num_layers"] = attn_baseline.shape[0]
    entry["baseline_seq_len"] = baseline_seq_len
    print(f"Baseline seq len: {baseline_seq_len}, Layers: {attn_baseline.shape[0]}")

    for prefill_idx, prefill in enumerate(prefill_variants):
        attack_prompt = build_prompt(harmful_prompt, prefill)
        attn_attack, attack_seq_len = get_attentions(attack_prompt)
        kl_per_layer = kl_divergence_per_layer(attn_baseline, attn_attack)

        prefill_slug = prefill.strip().replace(" ", "_").replace(":", "").replace("\n", "NL")[:20]
        output_path = f"attention_viz/prompt{item_idx}_prefill{prefill_idx}_{prefill_slug}.jpg"

        visualize_attention_layers(
            attn_baseline=attn_baseline,
            attn_attack=attn_attack,
            kl_per_layer=kl_per_layer,
            tokens_baseline=baseline_tokens,
            prefill=prefill,
            harmful_prompt=harmful_prompt,
            output_path=output_path,
        )

        attack_entry = {
            "prefill": prefill,
            "attack_seq_len": attack_seq_len,
            "kl_per_layer": kl_per_layer,
            "mean_kl_all_layers": sum(kl_per_layer) / len(kl_per_layer),
            "max_kl_layer": int(torch.tensor(kl_per_layer).argmax().item()),
            "max_kl_value": max(kl_per_layer),
            "viz_path": output_path,
        }

        print(
            f"  Prefill: {prefill!r:40s} | "
            f"Mean KL: {attack_entry['mean_kl_all_layers']:.4f} | "
            f"Max KL layer: {attack_entry['max_kl_layer']} "
            f"({attack_entry['max_kl_value']:.4f})"
        )
        entry["prefill_attacks"].append(attack_entry)

    results.append(entry)

# --- Save ---
output_path = "prefill_attention_kl_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {output_path}")