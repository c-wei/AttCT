import os
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


if __name__ == "__main__":
    result = chat([{"role": "user", "content": "What is the meaning of life?"}])
    print(result)
