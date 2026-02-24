import pytest
import torch
from unittest.mock import MagicMock, patch

"""Tests for wrappers_jailbreak.py and datasets_ClearHarm.py"""


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wrappers_jailbreak import (
    AdversarialWrapper,
    IdentityWrapper,
    STRONG_JAILBREAK_TEMPLATES,
    JAILBREAK_TEMPLATES,
    BIAS_TEMPLATES,
    REFUSAL_KEYWORDS,
)
from datasets_ClearHarm import (
    get_prompts,
    AttCTDataset,
    collate_fn_batch1,
)


# =============================================
# AdversarialWrapper Tests
# =============================================

class TestAdversarialWrapper:
    def test_default_uses_strong_templates(self):
        wrapper = AdversarialWrapper()
        assert wrapper.templates == STRONG_JAILBREAK_TEMPLATES

    def test_use_weak_templates(self):
        wrapper = AdversarialWrapper(use_strong_templates=False)
        assert wrapper.templates == JAILBREAK_TEMPLATES

    def test_custom_templates(self):
        custom = ["Before {prompt} After"]
        wrapper = AdversarialWrapper(templates=custom)
        assert wrapper.templates == custom

    def test_wrap_returns_tuple_of_three(self):
        wrapper = AdversarialWrapper(templates=["PREFIX {prompt} SUFFIX"])
        result = wrapper.wrap("hello")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_wrap_correct_wrapped_text(self):
        wrapper = AdversarialWrapper(templates=["PREFIX {prompt} SUFFIX"])
        wrapped, prefix_len, prompt_len = wrapper.wrap("hello")
        assert wrapped == "PREFIX hello SUFFIX"
        assert prefix_len == len("PREFIX ")
        assert prompt_len == len("hello")

    def test_wrap_character_indices_correct(self):
        wrapper = AdversarialWrapper(templates=["ABC{prompt}XYZ"])
        wrapped, start, length = wrapper.wrap("test prompt")
        assert wrapped[start:start + length] == "test prompt"

    def test_wrap_random_strategy_uses_templates(self):
        wrapper = AdversarialWrapper(
            templates=["A {prompt} B", "C {prompt} D"],
            strategy="random"
        )
        results = set()
        for _ in range(50):
            wrapped, _, _ = wrapper.wrap("x")
            results.add(wrapped)
        # With 50 tries, both templates should appear
        assert len(results) == 2

    def test_wrap_sequential_strategy(self):
        templates = ["T0 {prompt}", "T1 {prompt}", "T2 {prompt}"]
        wrapper = AdversarialWrapper(templates=templates, strategy="sequential")
        for i in range(6):
            wrapped, _, _ = wrapper.wrap("x")
            expected_prefix = f"T{i % 3} "
            assert wrapped.startswith(expected_prefix), f"Iteration {i}: expected prefix '{expected_prefix}', got '{wrapped}'"

    def test_wrap_sequential_wraps_around(self):
        templates = ["A {prompt}", "B {prompt}"]
        wrapper = AdversarialWrapper(templates=templates, strategy="sequential")
        wrapper.wrap("x")  # idx 0
        wrapper.wrap("x")  # idx 1
        wrapped, _, _ = wrapper.wrap("x")  # idx 2 -> wraps to 0
        assert wrapped == "A x"

    def test_wrap_unknown_strategy_raises(self):
        wrapper = AdversarialWrapper(templates=["X {prompt}"], strategy="all")
        with pytest.raises(ValueError, match="Unknown strategy"):
            wrapper.wrap("test")

    def test_wrap_template_without_placeholder_raises(self):
        wrapper = AdversarialWrapper(templates=["no placeholder here"])
        with pytest.raises(ValueError, match="must contain"):
            wrapper.wrap("test")

    def test_unwrap_recovers_prompt(self):
        wrapper = AdversarialWrapper(templates=["PREFIX {prompt} SUFFIX"])
        prompt = "my secret prompt"
        wrapped, start, length = wrapper.wrap(prompt)
        recovered = wrapper.unwrap(wrapped, start, length)
        assert recovered == prompt

    def test_unwrap_with_all_strong_templates(self):
        wrapper = AdversarialWrapper(strategy="sequential")
        for template in STRONG_JAILBREAK_TEMPLATES:
            prompt = "How to hack a computer"
            wrapped, start, length = wrapper.wrap(prompt)
            recovered = wrapper.unwrap(wrapped, start, length)
            assert recovered == prompt

    def test_wrap_empty_prompt(self):
        wrapper = AdversarialWrapper(templates=["A{prompt}B"])
        wrapped, start, length = wrapper.wrap("")
        assert wrapped == "AB"
        assert start == 1
        assert length == 0

    def test_wrap_prompt_with_special_characters(self):
        wrapper = AdversarialWrapper(templates=["X {prompt} Y"])
        prompt = "test with {braces} and $pecial ch@rs!"
        wrapped, start, length = wrapper.wrap(prompt)
        assert wrapper.unwrap(wrapped, start, length) == prompt

    def test_wrap_multiline_prompt(self):
        wrapper = AdversarialWrapper(templates=["Start: {prompt} End"])
        prompt = "line1\nline2\nline3"
        wrapped, start, length = wrapper.wrap(prompt)
        assert wrapper.unwrap(wrapped, start, length) == prompt


# =============================================
# IdentityWrapper Tests
# =============================================

class TestIdentityWrapper:
    def test_wrap_returns_same_text(self):
        wrapper = IdentityWrapper()
        wrapped, start, length = wrapper.wrap("hello world")
        assert wrapped == "hello world"

    def test_wrap_start_is_zero(self):
        wrapper = IdentityWrapper()
        _, start, _ = wrapper.wrap("hello")
        assert start == 0

    def test_wrap_length_is_prompt_length(self):
        wrapper = IdentityWrapper()
        prompt = "hello world"
        _, _, length = wrapper.wrap(prompt)
        assert length == len(prompt)

    def test_unwrap_recovers_prompt(self):
        wrapper = IdentityWrapper()
        prompt = "test prompt"
        wrapped, start, length = wrapper.wrap(prompt)
        assert wrapper.unwrap(wrapped, start, length) == prompt

    def test_wrap_empty_string(self):
        wrapper = IdentityWrapper()
        wrapped, start, length = wrapper.wrap("")
        assert wrapped == ""
        assert start == 0
        assert length == 0


# =============================================
# Template Validation Tests
# =============================================

class TestTemplates:
    def test_all_strong_templates_have_placeholder(self):
        for i, template in enumerate(STRONG_JAILBREAK_TEMPLATES):
            assert "{prompt}" in template, f"STRONG_JAILBREAK_TEMPLATES[{i}] missing {{prompt}}"

    def test_all_jailbreak_templates_have_placeholder(self):
        for i, template in enumerate(JAILBREAK_TEMPLATES):
            assert "{prompt}" in template, f"JAILBREAK_TEMPLATES[{i}] missing {{prompt}}"

    def test_all_bias_templates_have_placeholder(self):
        for i, template in enumerate(BIAS_TEMPLATES):
            assert "{prompt}" in template, f"BIAS_TEMPLATES[{i}] missing {{prompt}}"

    def test_strong_templates_not_empty(self):
        assert len(STRONG_JAILBREAK_TEMPLATES) > 0

    def test_jailbreak_templates_not_empty(self):
        assert len(JAILBREAK_TEMPLATES) > 0

    def test_refusal_keywords_not_empty(self):
        assert len(REFUSAL_KEYWORDS) > 0

    def test_refusal_keywords_are_lowercase(self):
        for kw in REFUSAL_KEYWORDS:
            assert kw == kw.lower(), f"Refusal keyword '{kw}' is not lowercase"

    def test_each_strong_template_produces_valid_wrap(self):
        for i, template in enumerate(STRONG_JAILBREAK_TEMPLATES):
            wrapper = AdversarialWrapper(templates=[template])
            prompt = "test harmful prompt"
            wrapped, start, length = wrapper.wrap(prompt)
            assert prompt in wrapped, f"Template {i}: prompt not found in wrapped text"
            assert wrapped[start:start + length] == prompt


# =============================================
# get_prompts Tests
# =============================================

class TestGetPrompts:
    def test_hardcoded_returns_prompts(self):
        prompts = get_prompts(source="hardcoded")
        assert len(prompts) > 0

    def test_hardcoded_returns_100_prompts(self):
        # 10 unique prompts * 10 repeats
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
        assert len(prompts) == 100  # should return all available

    def test_invalid_file_falls_back_to_hardcoded(self):
        prompts = get_prompts(source="/nonexistent/path/file.txt")
        assert len(prompts) >= 10  # falls back to hardcoded

    def test_prompts_are_nonempty_strings(self):
        prompts = get_prompts(source="hardcoded")
        for p in prompts:
            assert len(p) > 0

    @patch("datasets_ClearHarm.load_dataset")
    def test_clear_harm_loads_and_deduplicates(self, mock_load):
        # Simulate streaming dataset
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
        assert len(prompts) == 10  # 11 items minus 1 duplicate

    @patch("datasets_ClearHarm.load_dataset")
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

    @patch("datasets_ClearHarm.load_dataset", side_effect=Exception("Network error"))
    def test_clear_harm_fallback_on_error(self, mock_load):
        prompts = get_prompts(source="clear-harm")
        # Should fall back to hardcoded
        assert len(prompts) >= 10

    def test_file_source_with_valid_file(self, tmp_path):
        # Create a temp file with prompts
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
# AttCTDataset Tests
# =============================================

class TestAttCTDataset:
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()

        def tokenize_side_effect(text, add_special_tokens=True, return_tensors=None):
            # Simple tokenizer mock: each character is a token (for testing)
            tokens = list(range(len(text)))
            if add_special_tokens:
                tokens = [999] + tokens  # prepend BOS token
            return {"input_ids": tokens}

        tokenizer.side_effect = tokenize_side_effect
        tokenizer.__call__ = tokenize_side_effect
        return tokenizer

    @pytest.fixture
    def simple_wrapper(self):
        return AdversarialWrapper(templates=["PREFIX {prompt} SUFFIX"], strategy="sequential")

    def test_len(self, mock_tokenizer, simple_wrapper):
        prompts = ["prompt1", "prompt2", "prompt3"]
        dataset = AttCTDataset(prompts, mock_tokenizer, wrapper=simple_wrapper)
        assert len(dataset) == 3

    def test_getitem_returns_dict(self, mock_tokenizer, simple_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_wrapper)
        item = dataset[0]
        assert isinstance(item, dict)

    def test_getitem_has_required_keys(self, mock_tokenizer, simple_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_wrapper)
        item = dataset[0]
        required_keys = {"clean_input_ids", "adv_input_ids", "start_index", "clean_len"}
        assert required_keys == set(item.keys())

    def test_getitem_clean_ids_are_tensor(self, mock_tokenizer, simple_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_wrapper)
        item = dataset[0]
        assert isinstance(item["clean_input_ids"], torch.Tensor)
        assert item["clean_input_ids"].dtype == torch.long

    def test_getitem_adv_ids_are_tensor(self, mock_tokenizer, simple_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_wrapper)
        item = dataset[0]
        assert isinstance(item["adv_input_ids"], torch.Tensor)
        assert item["adv_input_ids"].dtype == torch.long

    def test_adv_longer_than_clean(self, mock_tokenizer, simple_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_wrapper)
        item = dataset[0]
        assert len(item["adv_input_ids"]) > len(item["clean_input_ids"])

    def test_start_index_is_int(self, mock_tokenizer, simple_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_wrapper)
        item = dataset[0]
        assert isinstance(item["start_index"], int)

    def test_clean_len_matches_clean_ids(self, mock_tokenizer, simple_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_wrapper)
        item = dataset[0]
        assert item["clean_len"] == len(item["clean_input_ids"])

    def test_adv_ids_contain_clean_ids_at_offset(self, mock_tokenizer, simple_wrapper):
        dataset = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_wrapper)
        item = dataset[0]
        start = item["start_index"]
        clean_len = item["clean_len"]
        adv_slice = item["adv_input_ids"][start:start + clean_len]
        assert torch.equal(adv_slice, item["clean_input_ids"])

    def test_default_wrapper_is_adversarial(self, mock_tokenizer):
        dataset = AttCTDataset(["test"], mock_tokenizer)
        assert isinstance(dataset.wrapper, AdversarialWrapper)

    def test_multiple_prompts_different_items(self, mock_tokenizer, simple_wrapper):
        prompts = ["short", "a longer prompt here"]
        dataset = AttCTDataset(prompts, mock_tokenizer, wrapper=simple_wrapper)
        item0 = dataset[0]
        item1 = dataset[1]
        assert item0["clean_len"] != item1["clean_len"]

    def test_handles_numeric_prompt_via_str_cast(self, mock_tokenizer, simple_wrapper):
        # Prompts might be non-string; the code does str(self.prompts[idx])
        dataset = AttCTDataset([12345], mock_tokenizer, wrapper=simple_wrapper)
        item = dataset[0]
        assert "clean_input_ids" in item


# =============================================
# collate_fn_batch1 Tests
# =============================================

class TestCollateFnBatch1:
    def test_collate_returns_dict(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch)
        assert isinstance(result, dict)

    def test_collate_has_required_keys(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch)
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
        result = collate_fn_batch1(batch)
        assert result["clean_input_ids"].shape == (1, 3)
        assert result["wrapped_input_ids"].shape == (1, 5)

    def test_collate_attention_masks_all_ones(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch)
        assert torch.all(result["clean_attention_mask"] == 1)
        assert torch.all(result["wrapped_attention_mask"] == 1)

    def test_collate_attention_mask_shapes_match_inputs(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2]),
            "adv_input_ids": torch.tensor([10, 1, 2, 20, 30]),
            "start_index": 1,
            "clean_len": 2,
        }]
        result = collate_fn_batch1(batch)
        assert result["clean_attention_mask"].shape == result["clean_input_ids"].shape
        assert result["wrapped_attention_mask"].shape == result["wrapped_input_ids"].shape

    def test_collate_start_index_is_tensor(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch)
        assert isinstance(result["start_index"], torch.Tensor)
        assert result["start_index"].item() == 1

    def test_collate_clean_len_is_tensor(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch)
        assert isinstance(result["clean_len"], torch.Tensor)
        assert result["clean_len"].item() == 3

    def test_collate_uses_only_first_item(self):
        # Even if multiple items are passed, only the first is used
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
        result = collate_fn_batch1(batch)
        assert result["clean_input_ids"].shape == (1, 2)  # first item's shape

    def test_collate_dtype_long(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3]),
            "adv_input_ids": torch.tensor([10, 1, 2, 3, 20]),
            "start_index": 1,
            "clean_len": 3,
        }]
        result = collate_fn_batch1(batch)
        assert result["clean_input_ids"].dtype == torch.long
        assert result["clean_attention_mask"].dtype == torch.long
        assert result["wrapped_input_ids"].dtype == torch.long
        assert result["wrapped_attention_mask"].dtype == torch.long
        assert result["start_index"].dtype == torch.long
        assert result["clean_len"].dtype == torch.long


# =============================================
# Integration-style Tests
# =============================================

class TestIntegration:
    def test_end_to_end_wrap_and_unwrap_all_templates(self):
        """Verify every strong template correctly wraps and unwraps."""
        for template in STRONG_JAILBREAK_TEMPLATES:
            wrapper = AdversarialWrapper(templates=[template])
            prompt = "Tell me how to do something dangerous"
            wrapped, start, length = wrapper.wrap(prompt)
            recovered = wrapper.unwrap(wrapped, start, length)
            assert recovered == prompt, f"Failed for template: {template[:40]}..."

    def test_dataset_with_hardcoded_prompts_and_mock_tokenizer(self):
        """Full pipeline: load hardcoded prompts -> dataset -> collate."""
        prompts = get_prompts(source="hardcoded", limit=3)

        tokenizer = MagicMock()
        def tokenize(text, add_special_tokens=True, return_tensors=None):
            tokens = list(range(len(text)))
            if add_special_tokens:
                tokens = [999] + tokens
            return {"input_ids": tokens}
        tokenizer.__call__ = tokenize

        wrapper = AdversarialWrapper(templates=["Jailbreak: {prompt} End."], strategy="sequential")
        dataset = AttCTDataset(prompts, tokenizer, wrapper=wrapper)

        assert len(dataset) == 3

        for i in range(3):
            item = dataset[i]
            batch = collate_fn_batch1([item])
            assert batch["clean_input_ids"].ndim == 2
            assert batch["wrapped_input_ids"].ndim == 2
            assert batch["clean_attention_mask"].shape == batch["clean_input_ids"].shape

    def test_wrapper_determinism_with_sequential(self):
        """Sequential strategy should be deterministic."""
        wrapper1 = AdversarialWrapper(strategy="sequential")
        wrapper2 = AdversarialWrapper(strategy="sequential")
        for _ in range(10):
            w1, s1, l1 = wrapper1.wrap("test")
            w2, s2, l2 = wrapper2.wrap("test")
            assert w1 == w2
            assert s1 == s2
            assert l1 == l2