import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

"""Tests for wrappers_sycophancy.py and datasets_cot-transparency.py"""


DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))


def _load_module(module_name: str, file_name: str):
    path = DATA_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


datasets_ct = _load_module("datasets_cot_transparency_mod", "datasets_cot-transparency.py")

from AttCT.data.legacy_data_files.wrappers_sycophancy import (  # noqa: E402
    AdversarialWrapper,
    IdentityWrapper,
    SYCOPHANCY_TEMPLATES,
)

get_prompts = datasets_ct.get_prompts
AttCTDataset = datasets_ct.AttCTDataset
collate_fn_batch1 = datasets_ct.collate_fn_batch1
_read_jsonl_user_messages = datasets_ct._read_jsonl_user_messages
_load_sycophancy_bct_clean_prompts = datasets_ct._load_sycophancy_bct_clean_prompts


class TestSycophancyWrapper:
    def test_default_uses_sycophancy_templates(self):
        wrapper = AdversarialWrapper()
        assert wrapper.templates == SYCOPHANCY_TEMPLATES

    def test_custom_templates(self):
        custom = ["P {prompt} S"]
        wrapper = AdversarialWrapper(templates=custom)
        assert wrapper.templates == custom

    def test_wrap_returns_tuple_of_three(self):
        wrapper = AdversarialWrapper(templates=["P {prompt} S"])
        result = wrapper.wrap("hello")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_wrap_recover_prompt_with_indices(self):
        wrapper = AdversarialWrapper(templates=["AAA {prompt} ZZZ"])
        wrapped, start, length = wrapper.wrap("hello")
        assert wrapped[start:start + length] == "hello"

    def test_wrap_sequential_strategy(self):
        wrapper = AdversarialWrapper(
            templates=["T0 {prompt}", "T1 {prompt}"],
            strategy="sequential",
        )
        wrapped0, _, _ = wrapper.wrap("x")
        wrapped1, _, _ = wrapper.wrap("x")
        wrapped2, _, _ = wrapper.wrap("x")
        assert wrapped0.startswith("T0 ")
        assert wrapped1.startswith("T1 ")
        assert wrapped2.startswith("T0 ")

    def test_wrap_unknown_strategy_raises(self):
        wrapper = AdversarialWrapper(templates=["X {prompt}"], strategy="all")
        with pytest.raises(ValueError, match="Unknown strategy"):
            wrapper.wrap("test")

    def test_template_without_placeholder_raises(self):
        wrapper = AdversarialWrapper(templates=["no placeholder"])
        with pytest.raises(ValueError, match="must contain"):
            wrapper.wrap("test")

    def test_sycophancy_template_fills_answer_choice(self):
        prompt = (
            "Question text\n\nAnswer choices:\n"
            "(A) alpha\n(B) beta\n(C) gamma\n"
        )
        wrapper = AdversarialWrapper(
            templates=["I think it is {answer_letter} ({answer_text}).\\n\\n{prompt}"],
            strategy="sequential",
        )
        wrapped, start, length = wrapper.wrap(prompt)
        assert wrapped[start:start + length] == prompt
        assert "I think it is " in wrapped
        assert "{answer_letter}" not in wrapped
        assert "{answer_text}" not in wrapped

    def test_sycophancy_template_raises_without_choices(self):
        wrapper = AdversarialWrapper(
            templates=["I think {answer_letter}.\\n\\n{prompt}"],
            strategy="sequential",
        )
        with pytest.raises(ValueError, match="no MCQ answer choices"):
            wrapper.wrap("Freeform prompt with no options")


class TestIdentityWrapper:
    def test_identity_wrap(self):
        wrapper = IdentityWrapper()
        wrapped, start, length = wrapper.wrap("hello")
        assert wrapped == "hello"
        assert start == 0
        assert length == 5


class TestTemplates:
    def test_templates_not_empty(self):
        assert len(SYCOPHANCY_TEMPLATES) > 0

    def test_templates_have_prompt_placeholder(self):
        for i, template in enumerate(SYCOPHANCY_TEMPLATES):
            assert "{prompt}" in template, f"SYCOPHANCY_TEMPLATES[{i}] missing {{prompt}}"


class TestGetPrompts:
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
        prompts = get_prompts(source=str(path))
        assert prompts == ["a", "b"]

    def test_get_prompts_limit(self, tmp_path):
        path = tmp_path / "prompts.txt"
        path.write_text("a\nb\nc\n")
        prompts = get_prompts(source=str(path), limit=2)
        assert prompts == ["a", "b"]


class TestAttCTDataset:
    @pytest.fixture
    def mock_tokenizer(self):
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
    def simple_wrapper(self):
        return AdversarialWrapper(templates=["PREFIX {prompt} SUFFIX"], strategy="sequential")

    def test_getitem_keys_and_types(self, mock_tokenizer, simple_wrapper):
        ds = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_wrapper)
        item = ds[0]
        assert set(item.keys()) == {
            "clean_input_ids", "wrapped_input_ids", "start_index", "clean_len", "wrapper_mask"
        }
        assert item["clean_input_ids"].dtype == torch.long
        assert item["wrapped_input_ids"].dtype == torch.long
        assert item["wrapper_mask"].dtype == torch.bool
        assert isinstance(item["start_index"], int)
        assert isinstance(item["clean_len"], int)

    def test_wrapped_contains_clean_at_offset(self, mock_tokenizer, simple_wrapper):
        ds = AttCTDataset(["test prompt"], mock_tokenizer, wrapper=simple_wrapper)
        item = ds[0]
        s = item["start_index"]
        n = item["clean_len"]
        assert torch.equal(item["wrapped_input_ids"][s:s+n], item["clean_input_ids"])

    def test_wrapper_mask_matches_clean_span(self, mock_tokenizer, simple_wrapper):
        ds = AttCTDataset(["abc"], mock_tokenizer, wrapper=simple_wrapper)
        item = ds[0]
        s = item["start_index"]
        n = item["clean_len"]
        mask = item["wrapper_mask"]
        assert mask.shape[0] == item["wrapped_input_ids"].shape[0]
        assert torch.all(mask[:s])
        assert torch.all(~mask[s:s+n])
        assert torch.all(mask[s+n:])

    def test_default_wrapper_is_sycophancy(self, mock_tokenizer):
        prompt = "Q\\n(A) x\\n(B) y\\n(C) z"
        ds = AttCTDataset([prompt], mock_tokenizer)
        assert isinstance(ds.wrapper, AdversarialWrapper)


class TestCollateFnBatch1:
    def test_collate_returns_training_loop_keys(self):
        batch = [{
            "clean_input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "wrapped_input_ids": torch.tensor([10, 1, 2, 3, 20], dtype=torch.long),
            "start_index": 1,
            "clean_len": 3,
            "wrapper_mask": torch.tensor([True, False, False, False, True], dtype=torch.bool),
        }]
        out = collate_fn_batch1(batch)
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
        out = collate_fn_batch1(batch)
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
