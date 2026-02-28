"""Dataset classes for consistency training (cot-transparency sycophancy)"""

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import List, Literal, Optional

from wrappers_sycophancy import AdversarialWrapper, SYCOPHANCY_TEMPLATES


def _read_jsonl_user_messages(path: str | Path) -> List[str]:
    p = Path(path)
    prompts: List[str] = []
    with p.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {p} line {line_num}: {e}") from e

            if not isinstance(obj, dict) or "messages" not in obj:
                raise ValueError(f"Expected object with 'messages' at {p} line {line_num}")
            messages = obj["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Expected non-empty 'messages' list at {p} line {line_num}")
            first = messages[0]
            if not isinstance(first, dict) or first.get("role") != "user" or "content" not in first:
                raise ValueError(f"Expected first message to be user w/ content at {p} line {line_num}")
            prompts.append(str(first["content"]))
    return prompts


def _load_sycophancy_bct_clean_prompts(
    *,
    style: Literal["cot", "non_cot"] = "cot",
    local_root: str | Path = "AttCT/datasets/sycophancy_bct",
) -> List[str]:
    """
    Loads clean (control) sycophancy BCT prompts from local dataset dumps.
    Only `style` varies: CoT vs non-CoT.
    """
    local = Path(local_root)
    if style == "cot":
        local_control = local / "control_cot.jsonl"
    else:
        local_control = local / "control_non_cot.jsonl"

    if local_control.exists():
        return _read_jsonl_user_messages(local_control)

    raise FileNotFoundError(
        "Local sycophancy_bct clean file not found. Expected:\n"
        f"- {local_control}"
    )


def get_prompts(
    source: str = "sycophancy_bct",
    split: str = "train",
    limit: Optional[int] = None,
    *,
    sycophancy_bct_style: Literal["cot", "non_cot"] = "cot",
) -> List[str]:
    """Load clean cot-transparency sycophancy prompts (control prompts only)."""
    prompts = []

    if source == "sycophancy_bct":
        prompts = _load_sycophancy_bct_clean_prompts(
            style=sycophancy_bct_style,
        )
        print(
            f"    Loaded {len(prompts)} prompts from sycophancy_bct "
            f"(clean, style={sycophancy_bct_style})"
        )
    else:
        # Assume it's a file path
        try:
            with open(source, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Warning: Failed to load from file {source}: {e}")

    # Apply limit if specified
    if limit and len(prompts) > limit:
        prompts = prompts[:limit]

    return prompts


class AttCTDataset(Dataset):
    """
    Dataset for Attention Consistency Training.

    Returns pairs of clean and adversarially-wrapped prompts with
    tokenization metadata needed for attention slicing.
    """

    def __init__(
        self,
        prompts: List[str],
        tokenizer,
        wrapper: Optional[AdversarialWrapper] = None,
        add_special_tokens: bool = True,
        use_strong_templates: bool = True
    ):
        """
        Args:
            prompts: List of clean prompts
            tokenizer: HuggingFace tokenizer
            wrapper: AdversarialWrapper instance (default: random sycophancy templates)
            add_special_tokens: Whether to add special tokens to sequences
            use_strong_templates: Kept for API compatibility; if True, uses sycophancy templates
        """
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.wrapper = wrapper or AdversarialWrapper(
            templates=SYCOPHANCY_TEMPLATES if use_strong_templates else None,
            strategy="random"
        )
        self.add_special_tokens = add_special_tokens

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary with:
                - clean_input_ids: tokens for clean prompt
                - wrapped_input_ids: tokens for wrapped prompt
                - start_index: token index where clean prompt starts in wrapped version
                - clean_len: length of clean prompt in tokens
                - wrapper_mask: bool mask over wrapped tokens (True on wrapper tokens)
        """
        clean_text = str(self.prompts[idx])

        # Get wrapped version
        wrapped_text, prefix_len_chars, clean_len_chars = self.wrapper.wrap(clean_text)

        # Tokenize clean prompt FIRST (to avoid special token issues)
        clean_ids = self.tokenizer(
            clean_text,
            add_special_tokens=False,  # Clean prompt without special tokens
            return_tensors=None
        )['input_ids']

        # Tokenize wrapped prompt parts to find boundaries
        # This is necessary because tokenization isn't always character-aligned
        prefix_text = wrapped_text[:prefix_len_chars]
        suffix_text = wrapped_text[prefix_len_chars + clean_len_chars:]

        # Prefix gets special tokens
        prefix_ids = self.tokenizer(
            prefix_text,
            add_special_tokens=self.add_special_tokens,
            return_tensors=None
        )['input_ids']

        # Suffix without special tokens
        suffix_ids = self.tokenizer(
            suffix_text,
            add_special_tokens=False,
            return_tensors=None
        )['input_ids']

        # Construct wrapped sequence: prefix + clean + suffix
        wrapped_ids = prefix_ids + clean_ids + suffix_ids
        start_index = len(prefix_ids)
        clean_len = len(clean_ids)
        wrapper_mask = [True] * len(wrapped_ids)
        for token_idx in range(start_index, start_index + clean_len):
            wrapper_mask[token_idx] = False

        return {
            'clean_input_ids': torch.tensor(clean_ids, dtype=torch.long),
            'wrapped_input_ids': torch.tensor(wrapped_ids, dtype=torch.long),
            'start_index': start_index,
            'clean_len': clean_len,
            'wrapper_mask': torch.tensor(wrapper_mask, dtype=torch.bool),
        }


def collate_fn_batch1(batch):
    """
    Collate function for batch size 1 (current requirement).
    Adds attention masks on the fly.
    """
    item = batch[0]
    return {
        'clean_input_ids': item['clean_input_ids'].unsqueeze(0),
        'clean_attention_mask': torch.ones(1, len(item['clean_input_ids']), dtype=torch.long),
        'wrapped_input_ids': item['wrapped_input_ids'].unsqueeze(0),
        'wrapped_attention_mask': torch.ones(1, len(item['wrapped_input_ids']), dtype=torch.long),
        'start_index': torch.tensor([item['start_index']], dtype=torch.long),
        'clean_len': torch.tensor([item['clean_len']], dtype=torch.long),
        'wrapper_mask': item['wrapper_mask'].unsqueeze(0),
    }
