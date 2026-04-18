import os
import time
import requests
import json

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
        content = body["choices"][0]["message"].get("content")
        if not content:
            raise ValueError(f"Empty content in response: {body}")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response content {content!r}: {e}") from e
    raise RuntimeError(f"chat_structured failed after {retries} retries (persistent 429)")


if __name__ == "__main__":
    result = chat([{"role": "user", "content": "What is the meaning of life?"}])
    print(result)
