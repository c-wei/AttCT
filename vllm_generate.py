"""
Shared vLLM inference utility for all eval scripts.

Replaces HuggingFace model.generate() calls with vLLM's PagedAttention engine,
giving ~3–5× throughput improvement.

Usage:
    from vllm_generate import load_llm, generate

    llm = load_llm("meta-llama/Llama-3.1-8B-Instruct")
    texts = generate(llm, prompts, max_new_tokens=300)

    # With LoRA adapter:
    llm = load_llm("meta-llama/Llama-3.1-8B-Instruct", lora_path="checkpoints/bct_sft/epoch_1")
    texts = generate(llm, prompts, max_new_tokens=300, lora_path="checkpoints/bct_sft/epoch_1")

    # Multi-GPU (e.g. for Gemma-3-27B on two A40s):
    llm = load_llm("google/gemma-3-27b-it", tensor_parallel_size=2)
"""

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def load_llm(
    model_name: str,
    lora_path: str | None = None,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    quantization: str | None = None,
) -> LLM:
    """Load a vLLM engine. Call once and reuse across generate() calls."""
    return LLM(
        model=model_name,
        enable_lora=lora_path is not None,
        max_lora_rank=64,           # generous ceiling; typical rank is 8–16
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        quantization=quantization,
    )


def generate(
    llm: LLM,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float = 0.0,
    lora_path: str | None = None,
) -> list[str]:
    """
    Generate completions for a list of already-formatted prompt strings.

    Prompts should already have the chat template applied (e.g. via
    tokenizer.apply_chat_template(..., add_generation_prompt=True)).

    Returns a list of completion strings (no prompt prefix).
    """
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    lora_request = LoRARequest("adapter", 1, lora_path) if lora_path else None
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]
