"""
Gemini AI yordamida obyektivkadan ma'lumotlarni ajratib olish xizmati (ixtiyoriy zaxira).
Asosiy yo'l — ollama_parser.py (OLLAMA_MODEL, qwen3-vl:8b).
google-genai SDK (yangi versiya) ishlatiladi.
"""
import asyncio
import re
from google import genai
from google.genai import types

from ..core.config import settings
from ..schemas.obyektivka import ObyektivkaData
from .extraction import build_obyektivka_data, build_prompt, extract_json_object

_RETRY_DELAYS = [30, 35, 40]


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(code in msg for code in ("429", "resource_exhausted", "too many requests", "503", "unavailable"))


def _retry_delay(exc: Exception, default: float) -> float:
    match = re.search(r"retry[_ ](?:in|delay)['\"]?\s*[:\s]*['\"]?(\d+(?:\.\d+)?)", str(exc), re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 2, 60)
    return default


def parse_gemini_response(response_text: str) -> dict:
    """Gemini javobidan JSON ni ajratib oladi."""
    return extract_json_object(response_text)


async def parse_obyektivka_with_gemini(text: str) -> ObyektivkaData:
    """
    Gemini yordamida obyektivka matnini tahlil qiladi.
    """
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY sozlanmagan. .env faylini tekshiring.")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    prompt = build_prompt(text)

    last_exc: Exception | None = None
    next_sleep = 0.0
    for attempt in range(len(_RETRY_DELAYS) + 1):
        if next_sleep:
            await asyncio.sleep(next_sleep)
            next_sleep = 0.0
        try:
            response = await client.aio.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            break
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == len(_RETRY_DELAYS):
                raise
            next_sleep = _retry_delay(exc, _RETRY_DELAYS[attempt])
    else:
        raise RuntimeError(f"Gemini so'rovida xato: {last_exc}") from last_exc

    # Faqat asosiy matn qismlarini olamiz (thinking qismlarini o'tkazib yuboramiz)
    response_text = ""
    try:
        for part in response.candidates[0].content.parts:
            if not getattr(part, "thought", False) and part.text:
                response_text += part.text
    except (AttributeError, IndexError):
        pass
    if not response_text:
        response_text = response.text or ""

    return build_obyektivka_data(parse_gemini_response(response_text))
