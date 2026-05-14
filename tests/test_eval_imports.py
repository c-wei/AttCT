"""
Import-level smoke tests for the eval-side scripts.

These catch stale imports (e.g. `from eval_mmlu import run_mmlu` when
`run_mmlu` doesn't exist) at pytest time instead of at run time, after
the GPU has already booted vLLM and the user is on minute 25 of pre-eval.

vLLM is mocked via conftest.py so these run without a GPU.
"""
import importlib

import pytest


@pytest.mark.parametrize("module", [
    "eval_sycophancy_behavioral",
    "eval_clearharm_behavioral",
    "eval_persona_behavioral",
    "eval_mtbench",
    "eval_mmlu",
    "eval_rollout",
    "evaluate_bct",
    "evaluate_sycophancy",
    "run_evals",
])
def test_module_imports_cleanly(module):
    """Every script reachable from run_evals.py / run_act.sh must import."""
    importlib.import_module(module)


def test_evaluate_jailbreak_imports():
    """evaluate_jailbreak.py needs json-repair (declared in pyproject.toml).

    Skipped if json_repair isn't installed (e.g. local Mac venv where the
    full sync fails on nvidia deps); on RunPod it's always present after
    `uv sync`.
    """
    pytest.importorskip("json_repair")
    importlib.import_module("evaluate_jailbreak")


def test_run_evals_call_targets_exist():
    """The functions run_evals.py imports and calls must exist with usable signatures."""
    import inspect

    from shared.eval_mmlu import run_mmlu
    from experiments.frustration.eval_rollout import run_rollouts, _parse_datasets
    from experiments.sycophancy.evaluate_bct import run_brr_with_llm
    from experiments.sycophancy.eval_sycophancy_behavioral import _load_eval_pairs, _extract_answer_letter
    from experiments.jailbreak.eval_clearharm_behavioral import judge_refusal
    from experiments.persona.eval_persona_behavioral import eval_persona, PERSONAS
    from shared.eval_mtbench import judge_response

    # run_mmlu(llm, checkpoint=..., n_samples=..., metric_prefix=...)
    params = inspect.signature(run_mmlu).parameters
    assert "llm" in params
    assert "checkpoint" in params
    assert "n_samples" in params
    assert "metric_prefix" in params

    # run_brr_with_llm(llm, tokenizer, lora_path, test_root, ...)
    params = inspect.signature(run_brr_with_llm).parameters
    assert "llm" in params
    assert "test_root" in params
    assert "metric_prefix" in params

    # PERSONAS is a non-empty iterable
    assert len(PERSONAS) > 0
