import httpx
from config import settings

DEFAULT_TIMEOUT = 120

async def call_openrouter(messages, model=None, max_tokens=2048, reasoning=None, temperature=0.7):
    model = model or settings.OPENROUTER_MODEL
    url = f"{settings.OPENROUTER_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": temperature,
        "extra_body": {"reasoning": reasoning or {"effort": "medium", "exclude": True}},
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data or not data["choices"]:
            return {"text": "", "raw": data}

        choice = data["choices"][0]
        # cover all possible fields
        text = (
            choice.get("message", {}).get("content")
            or choice.get("text")
            or choice.get("content")
            or ""
        )

        return {"text": text.strip(), "raw": data}
