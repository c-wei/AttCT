"""Tests for unified wrappers.py (jailbreak + sycophancy)."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wrappers import (
    AdversarialWrapper,
    IdentityWrapper,
    STRONG_JAILBREAK_TEMPLATES,
    JAILBREAK_TEMPLATES,
    BIAS_TEMPLATES,
    REFUSAL_KEYWORDS,
    SYCOPHANCY_TEMPLATES,
    _extract_answer_choices,
    _fill_template_placeholders,
)


# =============================================
# Jailbreak AdversarialWrapper Tests
# =============================================

class TestJailbreakWrapper:
    def test_default_uses_strong_templates(self):
        wrapper = AdversarialWrapper(mode="jailbreak")
        assert wrapper.templates == STRONG_JAILBREAK_TEMPLATES

    def test_use_weak_templates(self):
        wrapper = AdversarialWrapper(use_strong_templates=False, mode="jailbreak")
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
        wrapper = AdversarialWrapper(strategy="sequential", mode="jailbreak")
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
# Sycophancy AdversarialWrapper Tests
# =============================================

class TestSycophancyWrapper:
    def test_default_uses_sycophancy_templates(self):
        wrapper = AdversarialWrapper(mode="sycophancy")
        assert wrapper.templates == SYCOPHANCY_TEMPLATES

    def test_custom_templates(self):
        custom = ["P {prompt} S"]
        wrapper = AdversarialWrapper(templates=custom, mode="sycophancy")
        assert wrapper.templates == custom

    def test_wrap_returns_tuple_of_three(self):
        wrapper = AdversarialWrapper(templates=["P {prompt} S"], mode="sycophancy")
        result = wrapper.wrap("hello")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_wrap_recover_prompt_with_indices(self):
        wrapper = AdversarialWrapper(templates=["AAA {prompt} ZZZ"], mode="sycophancy")
        wrapped, start, length = wrapper.wrap("hello")
        assert wrapped[start:start + length] == "hello"

    def test_wrap_sequential_strategy(self):
        wrapper = AdversarialWrapper(
            templates=["T0 {prompt}", "T1 {prompt}"],
            strategy="sequential",
            mode="sycophancy",
        )
        wrapped0, _, _ = wrapper.wrap("x")
        wrapped1, _, _ = wrapper.wrap("x")
        wrapped2, _, _ = wrapper.wrap("x")
        assert wrapped0.startswith("T0 ")
        assert wrapped1.startswith("T1 ")
        assert wrapped2.startswith("T0 ")

    def test_wrap_unknown_strategy_raises(self):
        wrapper = AdversarialWrapper(templates=["X {prompt}"], strategy="all", mode="sycophancy")
        with pytest.raises(ValueError, match="Unknown strategy"):
            wrapper.wrap("test")

    def test_template_without_placeholder_raises(self):
        wrapper = AdversarialWrapper(templates=["no placeholder"], mode="sycophancy")
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
            mode="sycophancy",
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
            mode="sycophancy",
        )
        with pytest.raises(ValueError, match="no MCQ answer choices"):
            wrapper.wrap("Freeform prompt with no options")


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

    def test_all_sycophancy_templates_have_placeholder(self):
        for i, template in enumerate(SYCOPHANCY_TEMPLATES):
            assert "{prompt}" in template, f"SYCOPHANCY_TEMPLATES[{i}] missing {{prompt}}"

    def test_strong_templates_not_empty(self):
        assert len(STRONG_JAILBREAK_TEMPLATES) > 0

    def test_jailbreak_templates_not_empty(self):
        assert len(JAILBREAK_TEMPLATES) > 0

    def test_sycophancy_templates_not_empty(self):
        assert len(SYCOPHANCY_TEMPLATES) > 0

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
# Integration Tests
# =============================================

class TestIntegration:
    def test_end_to_end_wrap_and_unwrap_all_jailbreak_templates(self):
        for template in STRONG_JAILBREAK_TEMPLATES:
            wrapper = AdversarialWrapper(templates=[template])
            prompt = "Tell me how to do something dangerous"
            wrapped, start, length = wrapper.wrap(prompt)
            recovered = wrapper.unwrap(wrapped, start, length)
            assert recovered == prompt, f"Failed for template: {template[:40]}..."

    def test_wrapper_determinism_with_sequential(self):
        wrapper1 = AdversarialWrapper(strategy="sequential", mode="jailbreak")
        wrapper2 = AdversarialWrapper(strategy="sequential", mode="jailbreak")
        for _ in range(10):
            w1, s1, l1 = wrapper1.wrap("test")
            w2, s2, l2 = wrapper2.wrap("test")
            assert w1 == w2
            assert s1 == s2
            assert l1 == l2

    def test_mode_default_is_jailbreak(self):
        wrapper = AdversarialWrapper()
        assert wrapper.mode == "jailbreak"
        assert wrapper.templates == STRONG_JAILBREAK_TEMPLATES
