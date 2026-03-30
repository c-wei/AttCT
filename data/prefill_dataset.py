from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import load_dataset
import torch


PREFILL_VARIANTS = [
    "Sure! ",
    "Of course, here's how:",
    "Absolutely. Step 1:",
    "Here is the information you requested:",
]


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

        start_index= 0 # where prefill begins in wrapped seq
        clean_start_index = 0 # where assistant turn starts in clean seq
        clean_len = clean_len


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