import random
from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset as _hf_load_dataset
except ImportError:
    _hf_load_dataset = None


class UltraChatKLDataset(Dataset):
    """
    Dataset of tokenised user prompts from UltraChat for KL regularization.

    Each item returns:
        input_ids      : LongTensor [seq_len]
        attention_mask : LongTensor [seq_len]

    No clean/wrapped pairs, no start_index metadata — just a single
    tokenised prompt per sample.

    Args:
        tokenizer:   HuggingFace tokenizer (must support apply_chat_template).
        n_samples:   Number of prompts to sample from UltraChat.
        max_length:  Maximum token length (truncates longer prompts).
        split:       HuggingFace dataset split.  UltraChat-200k has
                     "train_sft" and "test_sft".
        seed:        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        tokenizer,
        n_samples: int = 500,
        max_length: int = 512,
        split: str = "train_sft",
        seed: int = 42,
    ):
        if _hf_load_dataset is None:
            raise ImportError(
                "UltraChatKLDataset requires the `datasets` library. "
                "Install with: pip install datasets"
            )

        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"--> Loading UltraChat-200k ({split}, sampling {n_samples} prompts)...")
        ds = _hf_load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split=split,
            streaming=True,
        )

        # Collect first-turn user messages.
        self.prompts: list[str] = []
        for item in ds:
            if len(self.prompts) >= n_samples:
                break
            messages = item.get("messages", [])
            if not messages:
                continue
            first_msg = messages[0]
            if (
                isinstance(first_msg, dict)
                and first_msg.get("role") == "user"
                and first_msg.get("content", "").strip()
            ):
                self.prompts.append(first_msg["content"].strip())

        # Shuffle so repeated epochs see prompts in different order.
        rng = random.Random(seed)
        rng.shuffle(self.prompts)
        print(f"    Loaded {len(self.prompts)} UltraChat prompts for KL regularization")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        text = self.prompts[idx]

        # Format with the model's chat template.
        # add_generation_prompt=True appends the assistant header so logits
        # are computed at the generation boundary.
        try:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (ValueError, AttributeError):
            formatted = text

        encoding = self.tokenizer(
            formatted,
            add_special_tokens=False,  # chat template already includes BOS
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,  # return plain lists
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        }


def kl_collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids_out: list[torch.Tensor] = []
    attention_mask_out: list[torch.Tensor] = []

    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        pad_len = max_len - ids.shape[0]
        input_ids_out.append(
            torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        attention_mask_out.append(
            torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_out),
        "attention_mask": torch.stack(attention_mask_out),
    }


def get_kl_dataloader(
    config: dict,
    tokenizer,
    n_samples: int = 500,
    split: str = "train_sft",
) -> DataLoader:
    """
    Build a DataLoader for KL regularization using UltraChat prompts.

    Args:
        config:    Full config dict (uses data.max_length if available).
        tokenizer: HuggingFace tokenizer.
        n_samples: Number of UltraChat prompts to load.
        split:     UltraChat-200k split name.

    Returns:
        DataLoader yielding batches with 'input_ids' and 'attention_mask'.
    """
    max_length = config.get("data", {}).get("max_length", 512)

    dataset = UltraChatKLDataset(
        tokenizer=tokenizer,
        n_samples=n_samples,
        max_length=max_length,
        split=split,
    )

    collate = partial(kl_collate_fn, pad_token_id=tokenizer.pad_token_id or 0)

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate,
    )