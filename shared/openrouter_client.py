import os
import time
import requests
import json
from json_repair import repair_json

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "meta-llama/llama-3.3-70b-instruct:free"


def chat(messages: list[dict], **kwargs) -> str:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        },
        data=json.dumps({
            "model": MODEL,
            "messages": messages,
            **kwargs,
        }),
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def chat_structured(messages: list[dict], schema: dict, model: str, retries: int = 5, **kwargs) -> dict:
    """Like chat(), but enforces a JSON schema via response_format and returns parsed JSON."""
    for attempt in range(retries):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            },
            data=json.dumps({
                "model": model,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema, "strict": True},
                },
                **kwargs,
            }),
        )
        if response.status_code in (429, 502, 503, 504):
            wait = 2 ** attempt
            print(f"    [OpenRouter] {response.status_code}, retrying in {wait}s (attempt {attempt+1}/{retries})...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        if choice.get("error", {}).get("code") == 429:
            wait = 2 ** attempt
            print(f"    [OpenRouter] 429 in response body, retrying in {wait}s (attempt {attempt+1}/{retries})...")
            time.sleep(wait)
            continue
        content = choice["message"].get("content")
        if not content:
            raise ValueError(f"Empty content in response: {body}")
        # Extract only the outermost { ... } or [ ... ] block
        open_char, close_char = ('{', '}') if '{' in content else ('[', ']')
        start = content.find(open_char)
        end = content.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            content = content[start:end + 1]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            repaired = repair_json(content, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
            return None
    raise RuntimeError(f"chat_structured failed after {retries} retries (persistent 429)")


if __name__ == "__main__":
    result = chat([{"role": "user", "content": "What is the meaning of life?"}])
    print(result)
