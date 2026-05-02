import csv
import os
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import load_dataset
import torch


def _load_prefill_variants() -> list[str]:
    """Load prefill variants from attacks.csv."""
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "attacks.csv")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row["prefill"] for row in reader if row["prefill"].strip()]


PREFILL_VARIANTS = _load_prefill_variants()

# 80/20 train/test split on prefill variants
_SPLIT = int(0.9 * len(PREFILL_VARIANTS))
PREFILL_VARIANTS_TRAIN = PREFILL_VARIANTS[:_SPLIT]
PREFILL_VARIANTS_TEST  = PREFILL_VARIANTS[_SPLIT:]


class PrefillAttackDataset(Dataset):
    """
    Each item is one (clean_prompt, prefill_attack_prompt) pair.
    Produces batches compatible with Trainer._step().

    clean  = baseline prompt with add_generation_prompt=True, NO prefill
    wrapped = same prompt + prefill text appended after the assistant turn marker

    start_index       = token position in wrapped where prefill begins
                        (= len(clean_tokens), since prefill is appended at end)
    clean_start_index = token position in clean where assistant turn begins
                        (same value — both sequences share the same prefix)
    clean_len         = number of tokens in the clean sequence
    """

    def __init__(
        self,
        prompts: list[str],
        tokenizer: PreTrainedTokenizer,
        prefill_variants: list[str] = PREFILL_VARIANTS,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []

        for prompt in prompts:
            clean_text = self._build_prompt(prompt)
            for prefill in prefill_variants:
                wrapped_text = clean_text + prefill
                self.items.append((prompt, clean_text, wrapped_text))

    def _build_prompt(self, harmful_prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": harmful_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        harmful_prompt, clean_text, wrapped_text = self.items[idx]

        clean_enc = self.tokenizer(
            clean_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        wrapped_enc = self.tokenizer(
            wrapped_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        clean_ids   = clean_enc["input_ids"][0]
        wrapped_ids = wrapped_enc["input_ids"][0]

        clean_len = clean_ids.shape[0]
        wrapped_len = wrapped_ids.shape[0]

        start_index=clean_len # where prefill begins in wrapped seq
        clean_start_index =clean_len # where assistant turn starts in clean seq


        return {
            "clean_input_ids":        clean_ids,
            "clean_attention_mask":   clean_enc["attention_mask"][0],
            "wrapped_input_ids":      wrapped_ids,
            "wrapped_attention_mask": wrapped_enc["attention_mask"][0],
            "start_index":            torch.tensor(start_index,       dtype=torch.long),
            "clean_start_index":      torch.tensor(clean_start_index, dtype=torch.long),
            "clean_len":              torch.tensor(clean_len,         dtype=torch.long),
        }


def prefill_collate_fn(batch: list[dict]) -> dict:
    """
    Pad all sequences in the batch to the same length.
    Trainer._step asserts start_index, clean_start_index, clean_len are
    identical within a batch — so we group by (clean_len, wrapped_len) first.
    """
    tokenizer = None

    def pad_seq(seqs, pad_val=0):
        max_len = max(s.shape[0] for s in seqs)
        return torch.stack([
            torch.cat([s, torch.full((max_len - s.shape[0],), pad_val, dtype=s.dtype)])
            for s in seqs
        ])

    return {
        "clean_input_ids":        pad_seq([b["clean_input_ids"]        for b in batch]),
        "clean_attention_mask":   pad_seq([b["clean_attention_mask"]   for b in batch]),
        "wrapped_input_ids":      pad_seq([b["wrapped_input_ids"]      for b in batch]),
        "wrapped_attention_mask": pad_seq([b["wrapped_attention_mask"] for b in batch]),
        "start_index":            torch.stack([b["start_index"]        for b in batch]),
        "clean_start_index":      torch.stack([b["clean_start_index"]  for b in batch]),
        "clean_len":              torch.stack([b["clean_len"]          for b in batch]),
    }


def get_prefill_dataloader(
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    prefill_variants: list[str] = PREFILL_VARIANTS,
    batch_size: int = 1, 
    max_length: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    dataset = PrefillAttackDataset(prompts, tokenizer, prefill_variants, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=prefill_collate_fn,
    )

def load_wildjailbreak_prompts(
    content_column: str = "vanilla",
    data_type_column: str = "data_type",
    harmful_label: str = "vanilla_harmful",
    limit: int = None,
    train_ratio: float = 0.9,
) -> tuple[list[str], list[str]]:
    """
    Loads harmful prompts from the WildJailbreak eval subset.
    Filters to rows where data_type == 'vanilla' (direct harmful prompts,
    no jailbreak wrapper) as the clean harmful baseline.
    """
    hf_dataset = load_dataset(
        "allenai/wildjailbreak",
        "train", 
        split="train",
        streaming=True,
        delimiter="\t",

    )

    prompts = []
    for item in hf_dataset:
        if item[data_type_column] != harmful_label:
            continue
        text = item[content_column]
        if text and text.strip():
            prompts.append(text.strip())
        if limit is not None and len(prompts) >= limit:
            break

    print(f"Loaded {len(prompts)} '{harmful_label}' prompts from WildJailbreak eval subset")

    cut = int(train_ratio * len(prompts))
    return prompts[:cut], prompts[cut:]


def load_harmful_behaviors_pair(
    csv_path: str = None,
    limit: int = None,
    train_ratio: float = 0.8,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Loads harmful prompts and their paired prefills from harmful_behaviors_pair.csv.

    Args:
        csv_path:    Path to the CSV file. Defaults to datasets/harmful_behaviors_pair.csv
                     relative to the repo root.
        limit:       Max rows to load (None = all).
        train_ratio: Fraction used for training; remainder is val.
                     Pass 1.0 to put everything in the train slot (e.g. for k-fold CV).

    Returns:
        (train_prompts, val_prompts, train_prefills, val_prefills)
        prefills[i] corresponds to prompts[i] (the target compliant prefix).
    """
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "datasets", "harmful_behaviors_pair.csv"
        )

    prompts = []
    prefills = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            goal = row["goal"].strip() if row.get("goal") else ""
            target = row["target"].strip() if row.get("target") else ""
            if goal and target:
                prompts.append(goal)
                prefills.append(target)
            if limit is not None and len(prompts) >= limit:
                break

    print(f"Loaded {len(prompts)} prompt-prefill pairs from {os.path.basename(csv_path)}")

    cut = int(train_ratio * len(prompts))
    return prompts[:cut], prompts[cut:], prefills[:cut], prefills[cut:]


def load_clearharm_prefills(
    hf_dataset: str = "carolinewei/ClearHarm_prefills",
    hf_split: str = "train",
    csv_path: str = None,
    limit: int = None,
    train_ratio: float = 0.8,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Loads (prompt, prefill) pairs from either a HuggingFace dataset (default)
    or a local CSV.

    Source selection:
        csv_path is not None  → load from local CSV at that path
        csv_path is None       → load from `hf_dataset` at split `hf_split`

    Both sources expect columns `prompt` and `prefill` (a `prefill_type`
    column is read but not returned). Each row pairs one prompt with one
    prefill (the abliterated model's generated attack of one of 23 strategy
    types). Multiple rows per prompt = multiple prefill variants per prompt.

    Returns:
        (train_prompts, val_prompts, train_prefills, val_prefills)
        prefills[i] corresponds 1-to-1 with prompts[i] (no Cartesian product).
    """
    prompts:  list[str] = []
    prefills: list[str] = []

    if csv_path is not None:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt  = (row.get("prompt")  or "").strip()
                prefill = (row.get("prefill") or "").strip()
                if prompt and prefill:
                    prompts.append(prompt)
                    prefills.append(prefill)
                if limit is not None and len(prompts) >= limit:
                    break
        source_label = os.path.basename(csv_path)
    else:
        ds = load_dataset(hf_dataset, split=hf_split)
        for item in ds:
            prompt  = str(item.get("prompt")  or "").strip()
            prefill = str(item.get("prefill") or "").strip()
            if prompt and prefill:
                prompts.append(prompt)
                prefills.append(prefill)
            if limit is not None and len(prompts) >= limit:
                break
        source_label = f"{hf_dataset}:{hf_split}"

    print(f"Loaded {len(prompts)} (prompt, prefill) rows from {source_label}")

    cut = int(train_ratio * len(prompts))
    return prompts[:cut], prompts[cut:], prefills[:cut], prefills[cut:]


def load_advbench_prompts(
    prompt_column: str = "prompt",
    limit: int = None,
    train_ratio: float = 0.9,
) -> tuple[list[str], list[str]]:
    """
    Loads harmful prompts from the AdvBench dataset (walledai/AdvBench).
    500 harmful behavior instructions sourced from llm-attacks.

    Args:
        prompt_column: Column containing the harmful instruction (default: "goal").
        limit:         Max prompts to load (None = all 500).
        train_ratio:   Fraction used for training; remainder is val.

    Returns:
        (train_prompts, val_prompts)
    """
    hf_dataset = load_dataset("walledai/AdvBench", split="train")

    prompts = []
    for item in hf_dataset:
        text = item[prompt_column]
        if text and text.strip():
            prompts.append(text.strip())
        if limit is not None and len(prompts) >= limit:
            break

    print(f"Loaded {len(prompts)} prompts from AdvBench")

    cut = int(train_ratio * len(prompts))
    return prompts[:cut], prompts[cut:]