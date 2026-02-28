"""Tests for unified attct_datasets.py (jailbreak + sycophancy)."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wrappers import AdversarialWrapper, STRONG_JAILBREAK_TEMPLATES, SYCOPHANCY_TEMPLATES
from attct_datasets import (
    get_prompts,
    AttCTDataset,
    collate_fn_batch1,
    _read_jsonl_user_messages,
    _load_sycophancy_bct_clean_prompts,
)


# =============================================
# Mock tokenizer fixture
# =============================================

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()

    def tokenize_side_effect(text, add_special_tokens=True, return_tensors=None):
        tokens = list(range(len(text)))
        if add_special_tokens:
            tokens = [999] + tokens
        return {"input_ids": tokens}

    tokenizer.side_effect = tokenize_side_effect
    tokenizer.__call__ = tokenize_side_effect
    return tokenizer


@pytest.fixture
def simple_jailbreak_wrapper():
    return AdversarialWrapper(templates=["PREFIX {prompt} SUFFIX"], strategy="sequential", mode="jailbreak")


@pytest.fixture
def simple_sycophancy_wrapper():
    return AdversarialWrapper(templates=["PREFIX {prompt} SUFFIX"], strategy="sequential", mode="sycophancy")


# =============================================
# get_prompts Tests (jailbreak / ClearHarm)
# =============================================

class TestGetPromptsJailbreak:
    def test_hardcoded_returns_prompts(self):
        prompts = get_prompts(source="hardcoded")
        assert len(prompts) > 0

    def test_hardcoded_returns_100_prompts(self):
        prompts = get_prompts(source="hardcoded")
        assert len(prompts) == 100

    def test_hardcoded_prompts_are_strings(self):
        prompts = get_prompts(source="hardcoded")
        for p in prompts:
            assert isinstance(p, str)

    def test_limit_parameter(self):
        prompts = get_prompts(source="hardcoded", limit=5)
        assert len(prompts) == 5

    def test_limit_larger_than_dataset(self):
        prompts = get_prompts(source="hardcoded", limit=10000)
        assert len(prompts) == 100

    def test_invalid_file_falls_back_to_hardcoded(self):
        prompts = get_prompts(source="/nonexistent/path/file.txt")
        assert len(prompts) >= 10

    def test_prompts_are_nonempty_strings(self):
        prompts = get_prompts(source="hardcoded")
        for p in prompts:
            assert len(p) > 0

    @patch("attct_datasets._hf_load_dataset")
    def test_clear_harm_loads_and_deduplicates(self, mock_load):
        mock_items = [
            {"content": "harmful prompt 1"},
            {"content": "harmful prompt 2"},
            {"content": "harmful prompt 1"},  # duplicate
            {"content": "harmful prompt 3"},
            {"content": "harmful prompt 4"},
            {"content": "harmful prompt 5"},
            {"content": "harmful prompt 6"},
            {"content": "harmful prompt 7"},
            {"content": "harmful prompt 8"},
            {"content": "harmful prompt 9"},
            {"content": "harmful prompt 10"},
        ]
        mock_load.return_value = mock_items
        prompts = get_prompts(source="clear-harm", split="train")
        assert len(prompts) == 10

    @patch("attct_datasets._hf_load_dataset")
    def test_clear_harm_filters_short_prompts(self, mock_load):
        mock_items = [
            {"content": "short"},  # <= 15 chars, should be filtered
            {"content": "this is a sufficiently long prompt 1"},
            {"content": "this is a sufficiently long prompt 2"},
            {"content": "this is a sufficiently long prompt 3"},
            {"content": "this is a sufficiently long prompt 4"},
            {"content": "this is a sufficiently long prompt 5"},
            {"content": "this is a sufficiently long prompt 6"},
            {"content": "this is a sufficiently long prompt 7"},
            {"content": "this is a sufficiently long prompt 8"},
            {"content": "this is a sufficiently long prompt 9"},
            {"content": "this is a sufficiently long prompt 10"},
        ]
        mock_load.return_value = mock_items
        prompts = get_prompts(source="clear-harm", split="train")
        assert "short" not in prompts

    @patch("attct_datasets._hf_load_dataset", side_effect=Exception("Network error"))
    def test_clear_harm_fallback_on_error(self, mock_load):
        prompts = get_prompts(source="clear-harm")
        assert len(prompts) >= 10

    def test_file_source_with_valid_file(self, tmp_path):
        prompt_file = tmp_path / "prompts.txt"
        lines = [f"harmful prompt number {i}" for i in range(20)]
        prompt_file.write_text("\n".join(lines))
        prompts = get_prompts(source=str(prompt_file))
        assert len(prompts) == 20

    def test_file_source_skips_empty_lines(self, tmp_path):
        prompt_file = tmp_path / "prompts.txt"
        prompt_file.write_text("prompt1\n\nprompt2\n\n\nprompt3\n")
        prompts = get_prompts(source=str(prompt_file))
        # Only 3 non-empty lines, but < 10 so falls back to hardcoded
        assert len(prompts) >= 10


# =============================================
# get_prompts Tests (sycophancy)
# =============================================

class TestGetPromptsSycophancy:
    def test_read_jsonl_user_messages(self, tmp_path):
        path = tmp_path / "test.jsonl"
        rows = [
            {"messages": [{"role": "user", "content": "prompt one"}]},
            {"messages": [{"role": "user", "content": "prompt two"}]},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows))
        prompts = _read_jsonl_user_messages(path)
        assert prompts == ["prompt one", "prompt two"]

    def test_load_sycophancy_bct_clean_prompts_from_local_root(self, tmp_path):
        root = tmp_path / "sycophancy_bct"
        root.mkdir()
        cot_path = root / "control_cot.jsonl"
        cot_path.write_text(
            json.dumps({"messages": [{"role": "user", "content": "cot prompt"}]}) + "\n"
        )
        prompts = _load_sycophancy_bct_clean_prompts(style="cot", local_root=root)
        assert prompts == ["cot prompt"]

    def test_get_prompts_file_source(self, tmp_path):
        path = tmp_path / "prompts.txt"
        path.write_text("a\n\nb\n")
        # file source with < 10 lines falls back to hardcoded for non-sycophancy sources
        # but for sycophancy_bct it doesn't apply. Let's test with sycophancy_bct source
        # Actually, file source for any source will use the file path branch
        # For < 10 prompts it falls back unless source is sycophancy_bct
        # Test raw file source:
        prompts = get_prompts(source=str(path))
        # Only 2 prompts, < 10 → falls back to hardcoded
        assert len(prompts) >= 10

    def test_get_prompts_limit(self, tmp_path):
        path = tmp_path / "prompts.txt"
        lines = [f"prompt number {i} is long enough" for i in range(20)]
        path.write_text("\n".join(lines))
        prompts = get_prompts(source=str(path), limit=2)
        assert len(prompts) == 2


# =============================================
# AttCTDataset Tests (jailbreak mode)
# =============================================

class TestAttCTDatasetJailbreak:
    def test_len(self, mock_tokenizer, simple_jailbreak_wrapper):
        prompts = ["prompt1", "prompt2", "prompt3"]
        dataset = AttCTDataset(prompts, mock_tokenizer, wrapper=simple_jailbreak_wrapper, mode="jailbreak")
        assert len(dataset) == 3

    def test_getitem_returns_dict(self, mock_tokenizer, simple_jailbreak_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_jailbreak_wrapper, mode="jailbreak")
        item = dataset[0]
        assert isinstance(item, dict)

    def test_getitem_has_required_keys(self, mock_tokenizer, simple_jailbreak_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_jailbreak_wrapper, mode="jailbreak")
        item = dataset[0]
        required_keys = {"clean_input_ids", "adv_input_ids", "start_index", "clean_len"}
        assert required_keys == set(item.keys())

    def test_getitem_clean_ids_are_tensor(self, mock_tokenizer, simple_jailbreak_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_jailbreak_wrapper, mode="jailbreak")
        item = dataset[0]
        assert isinstance(item["clean_input_ids"], torch.Tensor)
        assert item["clean_input_ids"].dtype == torch.long

    def test_getitem_adv_ids_are_tensor(self, mock_tokenizer, simple_jailbreak_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_jailbreak_wrapper, mode="jailbreak")
        item = dataset[0]
        assert isinstance(item["adv_input_ids"], torch.Tensor)
        assert item["adv_input_ids"].dtype == torch.long

    def test_adv_longer_than_clean(self, mock_tokenizer, simple_jailbreak_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_jailbreak_wrapper, mode="jailbreak")
        item = dataset[0]
        assert len(item["adv_input_ids"]) > len(item["clean_input_ids"])

    def test_start_index_is_int(self, mock_tokenizer, simple_jailbreak_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_jailbreak_wrapper, mode="jailbreak")
        item = dataset[0]
        assert isinstance(item["start_index"], int)

    def test_clean_len_matches_clean_ids(self, mock_tokenizer, simple_jailbreak_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_jailbreak_wrapper, mode="jailbreak")
        item = dataset[0]
        assert item["clean_len"] == len(item["clean_input_ids"])

    def test_adv_ids_contain_clean_ids_at_offset(self, mock_tokenizer, simple_jailbreak_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_jailbreak_wrapper, mode="jailbreak")
        item = dataset[0]
        start = item["start_index"]
        clean_len = item["clean_len"]
        adv_slice = item["adv_input_ids"][start:start + clean_len]
        assert torch.equal(adv_slice, item["clean_input_ids"])

    def test_default_wrapper_is_adversarial(self, mock_tokenizer):
        dataset = AttCTDataset(["test"], mock_tokenizer, mode="jailbreak")
        assert isinstance(dataset.wrapper, AdversarialWrapper)

    def test_multiple_prompts_different_items(self, mock_tokenizer, simple_jailbreak_wrapper):
        prompts = ["short", "a longer prompt here"]
        dataset = AttCTDataset(prompts, mock_tokenizer, wrapper=simple_jailbreak_wrapper, mode="jailbreak")
        item0 = dataset[0]
        item1 = dataset[1]
        assert item0["clean_len"] != item1["clean_len"]

    def test_handles_numeric_prompt_via_str_cast(self, mock_tokenizer, simple_jailbreak_wrapper):
        dataset = AttCTDataset([12345], mock_tokenizer, wrapper=simple_jailbreak_wrapper, mode="jailbreak")
        item = dataset[0]
        assert "clean_input_ids" in item


# =============================================
# AttCTDataset Tests (sycophancy mode)
# =============================================

class TestAttCTDatasetSycophancy:
    def test_getitem_keys_and_types(self, mock_tokenizer, simple_sycophancy_wrapper):
        ds = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_sycophancy_wrapper, mode="sycophancy")
        item = ds[0]
        assert set(item.keys()) == {
            "clean_input_ids", "wrapped_input_ids", "start_index", "clean_len", "wrapper_mask"
        }
        assert item["clean_input_ids"].dtype == torch.long
        assert item["wrapped_input_ids"].dtype == torch.long
        assert item["wrapper_mask"].dtype == torch.bool
        assert isinstance(item["start_index"], int)
        assert isinstance(item["clean_len"], int)

    def test_wrapped_contains_clean_at_offset(self, mock_tokenizer, simple_sycophancy_wrapper):
        ds = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_sycophancy_wrapper, mode="sycophancy")
        item = ds[0]
        s = item["start_index"]
        n = item["clean_len"]
        assert torch.equal(item["wrapped_input_ids"][s:s+n], item["clean_input_ids"])

    def test_wrapper_mask_matches_clean_span(self, mock_tokenizer, simple_sycophancy_wrapper):
        ds = AttCTDataset(["abc"], mock_tokenizer, wrapper=simple_sycophancy_wrapper, mode="sycophancy")
        item = ds[0]
        s = item["start_index"]
        n = item["clean_len"]
        mask = item["wrapper_mask"]
        assert mask.shape[0] == item["wrapped_input_ids"].shape[0]
        assert torch.all(mask[:s])
        assert torch.all(~mask[s:s+n])
        assert torch.all(mask[s+n:])

    def test_default_wrapper_is_sycophancy(self, mock_tokenizer):
        prompt = "Q\n(A) x\n(B) y\n(C) z"
        ds = AttCTDataset([prompt], mock_tokenizer, mode="sycophancy")
        assert isinstance(ds.wrapper, AdversarialWrapper)
        assert ds.wrapper.mode == "sycophancy"


# =============================================
# collate_fn_batch1 Tests (jailbreak mode)
# =============================================

class TestCollateFnBatch1Jailbreak:
    def test_collate_returns_dict(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch, mode="jailbreak")
        assert isinstance(result, dict)

    def test_collate_has_required_keys(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch, mode="jailbreak")
        expected_keys = {
            "clean_input_ids", "clean_attention_mask",
            "wrapped_input_ids", "wrapped_attention_mask",
            "start_index", "clean_len"
        }
        assert expected_keys == set(result.keys())

    def test_collate_adds_batch_dimension(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch, mode="jailbreak")
        assert result["clean_input_ids"].shape == (1, 3)
        assert result["wrapped_input_ids"].shape == (1, 5)

    def test_collate_attention_masks_all_ones(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch, mode="jailbreak")
        assert torch.all(result["clean_attention_mask"] == 1)
        assert torch.all(result["wrapped_attention_mask"] == 1)

    def test_collate_attention_mask_shapes_match_inputs(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2]),
            "adv_input_ids": torch.tensor([10, 1, 2, 20, 30]),
            "start_index": 1,
            "clean_len": 2,
        }]
        result = collate_fn_batch1(batch, mode="jailbreak")
        assert result["clean_attention_mask"].shape == result["clean_input_ids"].shape
        assert result["wrapped_attention_mask"].shape == result["wrapped_input_ids"].shape

    def test_collate_start_index_is_tensor(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch, mode="jailbreak")
        assert isinstance(result["start_index"], torch.Tensor)
        assert result["start_index"].item() == 1

    def test_collate_clean_len_is_tensor(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch, mode="jailbreak")
        assert isinstance(result["clean_len"], torch.Tensor)
        assert result["clean_len"].item() == 3

    def test_collate_uses_only_first_item(self):
        batch = [
            {
                "clean_input_ids": torch.tensor([1, 2]),
                "adv_input_ids": torch.tensor([10, 1, 2]),
                "start_index": 1,
                "clean_len": 2,
            },
            {
                "clean_input_ids": torch.tensor([3, 4, 5]),
                "adv_input_ids": torch.tensor([20, 3, 4, 5]),
                "start_index": 1,
                "clean_len": 3,
            },
        ]
        result = collate_fn_batch1(batch, mode="jailbreak")
        assert result["clean_input_ids"].shape == (1, 2)  # first item's shape

    def test_collate_dtype_long(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch, mode="jailbreak")
        assert result["clean_input_ids"].dtype == torch.long
        assert result["clean_attention_mask"].dtype == torch.long
        assert result["wrapped_input_ids"].dtype == torch.long
        assert result["wrapped_attention_mask"].dtype == torch.long
        assert result["start_index"].dtype == torch.long
        assert result["clean_len"].dtype == torch.long


# =============================================
# collate_fn_batch1 Tests (sycophancy mode)
# =============================================

class TestCollateFnBatch1Sycophancy:
    def test_collate_returns_training_loop_keys(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "wrapped_input_ids": torch.tensor([10, 1, 2, 3, 20], dtype=torch.long),
            "start_index": 1,
            "clean_len": 3,
            "wrapper_mask": torch.tensor([True, False, False, False, True], dtype=torch.bool),
        }]
        out = collate_fn_batch1(batch, mode="sycophancy")
        assert set(out.keys()) == {
            "clean_input_ids",
            "clean_attention_mask",
            "wrapped_input_ids",
            "wrapped_attention_mask",
            "start_index",
            "clean_len",
            "wrapper_mask",
        }

    def test_collate_shapes_and_dtypes(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "wrapped_input_ids": torch.tensor([10, 1, 2, 3, 20], dtype=torch.long),
            "start_index": 1,
            "clean_len": 3,
            "wrapper_mask": torch.tensor([True, False, False, False, True], dtype=torch.bool),
        }]
        out = collate_fn_batch1(batch, mode="sycophancy")
        assert out["clean_input_ids"].shape == (1, 3)
        assert out["clean_attention_mask"].shape == (1, 3)
        assert out["wrapped_input_ids"].shape == (1, 5)
        assert out["wrapped_attention_mask"].shape == (1, 5)
        assert out["wrapper_mask"].shape == (1, 5)
        assert out["clean_input_ids"].dtype == torch.long
        assert out["clean_attention_mask"].dtype == torch.long
        assert out["wrapped_input_ids"].dtype == torch.long
        assert out["wrapped_attention_mask"].dtype == torch.long
        assert out["start_index"].dtype == torch.long
        assert out["clean_len"].dtype == torch.long
        assert out["wrapper_mask"].dtype == torch.bool


# =============================================
# Integration Tests
# =============================================

class TestIntegration:
    def test_jailbreak_end_to_end(self, mock_tokenizer):
        """Full pipeline: load hardcoded prompts -> dataset -> collate."""
        prompts = get_prompts(source="hardcoded", limit=3)
        wrapper = AdversarialWrapper(templates=["Jailbreak: {prompt} End."], strategy="sequential")
        dataset = AttCTDataset(prompts, mock_tokenizer, wrapper=wrapper, mode="jailbreak")
        assert len(dataset) == 3

        for i in range(3):
            item = dataset[i]
            batch = collate_fn_batch1([item], mode="jailbreak")
            assert batch["clean_input_ids"].ndim == 2
            assert batch["wrapped_input_ids"].ndim == 2
            assert batch["clean_attention_mask"].shape == batch["clean_input_ids"].shape

    def test_sycophancy_end_to_end(self, mock_tokenizer):
        """Full pipeline for sycophancy mode."""
        prompts = ["Q\n(A) x\n(B) y\n(C) z"] * 3
        wrapper = AdversarialWrapper(templates=["PREFIX {prompt} SUFFIX"], strategy="sequential", mode="sycophancy")
        dataset = AttCTDataset(prompts, mock_tokenizer, wrapper=wrapper, mode="sycophancy")
        assert len(dataset) == 3

        for i in range(3):
            item = dataset[i]
            batch = collate_fn_batch1([item], mode="sycophancy")
            assert batch["clean_input_ids"].ndim == 2
            assert batch["wrapped_input_ids"].ndim == 2
            assert "wrapper_mask" in batch
