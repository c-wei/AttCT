"""
Generate fresh BCT training data by querying the current model via OpenRouter.

For each question in control_cot.jsonl / control_non_cot.jsonl (clean prompts),
queries OpenRouter for a fresh response from the specified model, then pairs it
with the row-aligned biased prompt from bct_cot.jsonl / bct_non_cot.jsonl.

Usage:
    uv run --no-project python generate_fresh_bct_data.py \\
        --model google/gemma-2-2b-it \\
        --bct-root datasets/sycophancy_bct \\
        --output-dir /tmp/fresh_bct_gemma2 \\
        --limit 20   # optional: cap for quick testing

Requires: OPENROUTER_API_KEY env var (or in .env).
"""

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path

import httpx

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# ── vLLM backend ──────────────────────────────────────────────────────────────

def _generate_vllm(
    user_contents: list[str | None],
    model_name: str,
    max_new_tokens: int,
    temperature: float,
) -> list[str | None]:
    """Generate responses using local vLLM. Frees GPU memory before returning."""
    import torch
    import vllm_generate
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Apply chat template so vLLM gets properly formatted prompts
    prompts = []
    for content in user_contents:
        if content is None:
            prompts.append(None)
            continue
        if getattr(tokenizer, "chat_template", None):
            p = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            p = f"User: {content}\nAssistant:"
        prompts.append(p)

    valid_prompts = [p for p in prompts if p is not None]
    valid_indices = [i for i, p in enumerate(prompts) if p is not None]

    llm = vllm_generate.load_llm(model_name)
    responses_raw = vllm_generate.generate(
        llm, valid_prompts, max_new_tokens=max_new_tokens, temperature=temperature
    )

    # Free GPU memory before caller loads training model
    del llm
    torch.cuda.empty_cache()

    results = [None] * len(user_contents)
    for idx, text in zip(valid_indices, responses_raw):
        results[idx] = text
    return results


def _read_user_contents(path: Path) -> list[str]:
    """Read user-turn content from a JSONL file (one per line)."""
    contents = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            user_content = next(
                (m["content"] for m in messages if m["role"] == "user"), None
            )
            contents.append(user_content)
    return contents


async def _fetch_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    idx: int,
) -> tuple[int, str | None]:
    """Send one request to OpenRouter; returns (idx, response_text | None)."""
    async with sem:
        try:
            resp = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return idx, text
        except Exception as e:
            print(f"  [warn] request {idx} failed: {e}")
            return idx, None


async def _generate_all(
    user_contents: list[str | None],
    model: str,
    max_new_tokens: int,
    temperature: float,
    concurrency: int,
    api_key: str,
) -> list[str | None]:
    """Generate responses for all user_contents via OpenRouter chat API.
    Returns a list aligned with the input (None where input was None or request failed).
    """
    results = [None] * len(user_contents)
    sem = asyncio.Semaphore(concurrency)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [
            _fetch_one(client, sem, model, content, max_new_tokens, temperature, i)
            for i, content in enumerate(user_contents)
            if content is not None
        ]
        done = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            idx, text = await coro
            results[idx] = text
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  {done}/{total} responses received")
    return results


def _write_fresh_jsonl(
    output_path: Path,
    biased_user_contents: list[str | None],
    responses: list[str | None],
) -> tuple[int, int]:
    """
    Write (biased_prompt, fresh_response) pairs to output_path.
    Skips rows where either side is None.
    Returns (written, skipped).
    """
    written = skipped = 0
    with open(output_path, "w") as f:
        for biased, response in zip(biased_user_contents, responses):
            if biased is None or response is None:
                skipped += 1
                continue
            record = {
                "messages": [
                    {"role": "user", "content": biased},
                    {"role": "assistant", "content": response},
                ]
            }
            f.write(json.dumps(record) + "\n")
            written += 1
    return written, skipped


def run_generation(
    model_name: str,
    bct_root: str | Path = "datasets/sycophancy_bct",
    output_dir: str | Path = "/tmp/fresh_bct",
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    limit: int | None = None,
    concurrency: int = 32,
    api_key: str | None = None,
    use_vllm: bool = False,
) -> Path:
    """
    Generate fresh BCT data and write to output_dir.
    Returns the output directory path.
    Callable from run.py as well as from the CLI.
    """
    bct_root = Path(bct_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not use_vllm:
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

    pairs = [
        ("control_cot.jsonl",     "bct_cot.jsonl"),
        ("control_non_cot.jsonl", "bct_non_cot.jsonl"),
    ]

    for control_fname, bct_fname in pairs:
        control_path = bct_root / control_fname
        bct_path     = bct_root / bct_fname
        out_path     = output_dir / bct_fname

        if not control_path.exists():
            print(f"  [skip] {control_path} not found")
            continue
        if not bct_path.exists():
            print(f"  [skip] {bct_path} not found")
            continue

        print(f"\n  Processing {bct_fname}...")
        clean_contents  = _read_user_contents(control_path)
        biased_contents = _read_user_contents(bct_path)

        if limit:
            clean_contents  = clean_contents[:limit]
            biased_contents = biased_contents[:limit]

        n_valid = sum(c is not None for c in clean_contents)
        if use_vllm:
            print(f"  Generating {n_valid} responses via vLLM ({model_name})...")
            responses = _generate_vllm(clean_contents, model_name, max_new_tokens, temperature)
        else:
            print(f"  Generating {n_valid} responses via OpenRouter ({model_name})...")
            responses = asyncio.run(
                _generate_all(clean_contents, model_name, max_new_tokens, temperature, concurrency, api_key)
            )

        written, skipped = _write_fresh_jsonl(out_path, biased_contents, responses)
        print(f"  Written: {written}  Skipped: {skipped}  → {out_path}")

    # Copy instruct_samples.jsonl unchanged (Alpaca/GPT-4T — not model-specific)
    instruct_src = bct_root / "instruct_samples.jsonl"
    instruct_dst = output_dir / "instruct_samples.jsonl"
    if instruct_src.exists():
        shutil.copy2(instruct_src, instruct_dst)
        print(f"\n  Copied instruct_samples.jsonl → {instruct_dst}")

    print(f"\n  Fresh BCT data written to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate fresh BCT training data via OpenRouter")
    parser.add_argument("--model",          default="google/gemma-2-2b-it",
                        help="OpenRouter model ID (default: google/gemma-2-2b-it)")
    parser.add_argument("--bct-root",       default="datasets/sycophancy_bct",
                        help="Directory containing control_*.jsonl and bct_*.jsonl")
    parser.add_argument("--output-dir",     required=True,
                        help="Directory to write fresh JSONL files")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--limit",          type=int, default=None,
                        help="Cap number of pairs per file (for quick testing)")
    parser.add_argument("--concurrency",    type=int, default=32,
                        help="Max parallel OpenRouter requests (default: 32)")
    parser.add_argument("--use-vllm",       action="store_true",
                        help="Use local vLLM instead of OpenRouter (for models not available on OpenRouter)")
    args = parser.parse_args()

    run_generation(
        model_name=args.model,
        bct_root=args.bct_root,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        limit=args.limit,
        concurrency=args.concurrency,
        use_vllm=args.use_vllm,
    )


if __name__ == "__main__":
    main()
