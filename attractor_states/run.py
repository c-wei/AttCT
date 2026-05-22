#!/usr/bin/env python3
"""
Attractor states experiment.

Two instances of a model (or two different models) hold an open-ended
conversation seeded with a freeform prompt. We log the full transcript
to wandb (one run per seed prompt) and to disk.

Follows the message-passing setup from ajobi-uhc/attractor-states.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import wandb
import yaml
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    retries: int = 3,
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=180,
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]

            if response.status_code == 402:
                raise RuntimeError(f"Insufficient credits: {response.text}")
            if response.status_code == 429:
                wait = (attempt + 1) * 10
                print(f"    Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue

            last_error = f"API error {response.status_code}: {response.text}"
            print(f"    {last_error}, retrying...", flush=True)
            time.sleep(2)

        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            print("    Timeout, retrying...", flush=True)
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            print(f"    Request error: {e}, retrying...", flush=True)
            time.sleep(2)

    raise RuntimeError(f"Failed after {retries} attempts: {last_error}")


def run_conversation(
    model_1: str,
    model_2: str,
    seed_prompt: str,
    turns: int,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    wandb_run,
) -> list[dict]:
    """
    Run a conversation between model_1 and model_2.

    `turns` is the number of messages PER MODEL. Total messages = 2 * turns.
    Model 1 receives the seed prompt as its first user message.
    Model 2 sees Model 1's first response as its first user message.
    Each model keeps its own message history (its own outputs as
    'assistant', the other model's outputs as 'user').
    """
    history_1: list[dict] = []
    history_2: list[dict] = []
    transcript: list[dict] = []

    table = wandb.Table(columns=["turn_idx", "speaker", "model", "content"])

    # --- Model 1 opens with the seed prompt ---
    history_1.append({"role": "user", "content": seed_prompt})
    messages_1 = [{"role": "system", "content": system_prompt}] + history_1
    response = call_openrouter(model_1, messages_1, max_tokens, temperature)
    history_1.append({"role": "assistant", "content": response})

    turn_idx = 1
    entry = {
        "turn_idx": turn_idx,
        "speaker": "model_1",
        "model": model_1,
        "content": response,
    }
    transcript.append(entry)
    table.add_data(turn_idx, "model_1", model_1, response)
    print(f"  Turn {turn_idx} (model_1 / {model_1.split('/')[-1]})", flush=True)

    last_response = response

    # --- Alternate. We need `turns` messages PER model. ---
    # model_1 has already spoken once. Now do (2*turns - 1) more messages.
    total_remaining = 2 * turns - 1
    for _ in range(total_remaining):
        turn_idx += 1
        # Even-indexed turns (2, 4, ...) are model_2; odd > 1 are model_1.
        if turn_idx % 2 == 0:
            history_2.append({"role": "user", "content": last_response})
            messages_2 = [{"role": "system", "content": system_prompt}] + history_2
            response = call_openrouter(model_2, messages_2, max_tokens, temperature)
            history_2.append({"role": "assistant", "content": response})
            speaker, current_model = "model_2", model_2
        else:
            history_1.append({"role": "user", "content": last_response})
            messages_1 = [{"role": "system", "content": system_prompt}] + history_1
            response = call_openrouter(model_1, messages_1, max_tokens, temperature)
            history_1.append({"role": "assistant", "content": response})
            speaker, current_model = "model_1", model_1

        entry = {
            "turn_idx": turn_idx,
            "speaker": speaker,
            "model": current_model,
            "content": response,
        }
        transcript.append(entry)
        table.add_data(turn_idx, speaker, current_model, response)
        print(
            f"  Turn {turn_idx} ({speaker} / {current_model.split('/')[-1]})",
            flush=True,
        )
        last_response = response

    wandb_run.log({"transcript": table})

    return transcript


def slugify(s: str, max_len: int = 40) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in s).strip("_")
    return cleaned[:max_len]


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_models(cfg: dict) -> tuple[str, str]:
    if cfg.get("cross_model"):
        m1, m2 = cfg.get("model_1"), cfg.get("model_2")
        if not m1 or not m2:
            raise ValueError("cross_model=true requires both model_1 and model_2")
        return m1, m2
    m = cfg.get("model")
    if not m:
        raise ValueError("`model` must be set when cross_model=false")
    return m, m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_1, model_2 = resolve_models(cfg)
    turns = int(cfg.get("turns", 30))
    max_tokens = int(cfg.get("max_tokens", 1000))
    temperature = float(cfg.get("temperature", 1.0))
    seed_prompts = cfg["seed_prompts"]
    system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
    wandb_project = cfg.get("wandb_project", "attractor-states")
    wandb_entity = cfg.get("wandb_entity")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_1 == model_2:
        run_label = f"{slugify(model_1.split('/')[-1])}_{timestamp}"
    else:
        run_label = (
            f"{slugify(model_1.split('/')[-1])}_vs_"
            f"{slugify(model_2.split('/')[-1])}_{timestamp}"
        )
    results_dir = Path(__file__).parent / "results" / run_label
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"Model 1: {model_1}", flush=True)
    print(f"Model 2: {model_2}", flush=True)
    print(f"Turns per model: {turns} (total messages: {2 * turns})", flush=True)
    print(f"Max tokens: {max_tokens}  Temperature: {temperature}", flush=True)
    print(f"Seed prompts: {len(seed_prompts)}", flush=True)
    print(f"Results dir: {results_dir}", flush=True)
    print("=" * 60, flush=True)

    for i, seed_prompt in enumerate(seed_prompts):
        print(
            f"\n[{i + 1}/{len(seed_prompts)}] Seed: {seed_prompt}",
            flush=True,
        )
        seed_slug = slugify(seed_prompt, max_len=30)
        run_name = f"{run_label}__{seed_slug}"

        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            group=run_label,
            tags=[
                "attractor-states",
                "cross-model" if model_1 != model_2 else "self",
            ],
            config={
                "model_1": model_1,
                "model_2": model_2,
                "cross_model": model_1 != model_2,
                "turns_per_model": turns,
                "total_messages": 2 * turns,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "seed_prompt": seed_prompt,
                "seed_index": i,
            },
            reinit=True,
        )

        try:
            transcript = run_conversation(
                model_1=model_1,
                model_2=model_2,
                seed_prompt=seed_prompt,
                turns=turns,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                wandb_run=wandb_run,
            )

            transcript_payload = {
                "model_1": model_1,
                "model_2": model_2,
                "seed_prompt": seed_prompt,
                "turns_per_model": turns,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "generated_at": datetime.now().isoformat(),
                "transcript": transcript,
            }
            transcript_path = results_dir / f"seed_{i:02d}_{seed_slug}.json"
            with open(transcript_path, "w") as f:
                json.dump(transcript_payload, f, indent=2)

            artifact = wandb.Artifact(
                name=f"transcript_seed_{i:02d}",
                type="conversation",
            )
            artifact.add_file(str(transcript_path))
            wandb_run.log_artifact(artifact)

            print(f"  ✓ Saved transcript: {transcript_path}", flush=True)
        except Exception as e:
            print(f"  ✗ Conversation failed: {e}", flush=True)
            wandb_run.finish(exit_code=1)
            raise
        finally:
            if wandb_run is not None and wandb_run is wandb.run:
                wandb_run.finish()

    print(f"\nDone. Results: {results_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        sys.exit(1)