import json
import logging
import httpx
from config import settings
from status import record_call_result
DEFAULT_TIMEOUT = 120
logger = logging.getLogger(__name__)
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
        # This is a raw HTTP request. ``extra_body`` is an OpenAI SDK keyword,
        # not part of OpenRouter's wire format; reasoning belongs at top level.
        "reasoning": reasoning or {"effort": "medium", "exclude": True},
    }
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data or not data["choices"]:
                record_call_result(False)
                return {"text": "", "raw": data}
            choice = data["choices"][0]
            text = (
                choice.get("message", {}).get("content")
                or choice.get("text")
                or choice.get("content")
                or ""
            )
            record_call_result(True)
            return {"text": text.strip(), "raw": data}
    except httpx.HTTPStatusError as exc:
        # OpenRouter errors do not contain the submitted prompt. Log only a
        # bounded error envelope and never headers/API credentials.
        try:
            detail = json.dumps(exc.response.json(), ensure_ascii=True)[:1000]
        except Exception:
            detail = exc.response.text[:1000]
        logger.error("OpenRouter rejected request: status=%s model=%s body=%s",
                     exc.response.status_code, model, detail)
        record_call_result(False, f"HTTP {exc.response.status_code}: {detail}")
        raise
    except Exception as exc:
        logger.error("OpenRouter request failed: model=%s error=%s",
                     model, str(exc)[:500])
        record_call_result(False, str(exc))
        raise
