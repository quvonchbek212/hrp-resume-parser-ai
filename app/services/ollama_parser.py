"""
Obyektivka (ma'lumotnoma) matnini vLLM modeli yordamida JSON ga o'giradi.
Server, kalit va model `.env` dagi VLLM_BASE_URL / VLLM_API_KEY / VLLM_MODEL
(yoki OBYEKTIVKA_MODEL) dan olinadi. Tashqi SDK talab qilmaydi — OpenAI-compatible
`/v1/chat/completions` REST endpointi ishlatiladi.

Tezlik haqida (avvalgi Ollama serverida o'lchangan):
  - butun JSON bitta so'rovda -> 1789 token, 115 s
  - prompt eval               -> atigi 3 s, ya'ni sabab prompt emas, javob hajmi
  - shu 1789 tokendan 1039 tasi (~58%) — qarindoshlar jadvali

Asosiy xulosa: yagona haqiqiy o'lchov — yozilgan tokenlar soni. Shuning uchun
tokenlar soni kamaytirilgan:
  - qarindoshlar jadvali umuman modelga yuborilmaydi. U `file_extractor` dan
    ustunlarga bo'lingan holda keladi va kod bilan o'qiladi (-1039 token).
  - profil va mehnat faoliyati promptlaridan o'sha jadval olib tashlanadi
    (prompt ~2 barobar qisqaradi).
Natija: ~1789 -> ~667 token.

Bo'laklar ketma-ket yuboriladi. Parallel yuborish Ollama serverida sinab
ko'rilgan va rad etilgan (o'lchov: ketma-ket 17 s -> parallel 145 s).
"""
import asyncio
import json
import urllib.error
import urllib.request

from ..core.config import settings
from ..schemas.obyektivka import ObyektivkaData
from .extraction import (
    SYSTEM_PROMPT,
    build_obyektivka_data,
    build_profile_prompt,
    build_relatives_prompt,
    build_work_history_prompt,
    clean_document_text,
    extract_json_object,
    parse_relatives_table,
    strip_relatives_table,
)

# Ulanish/vaqtinchalik xatolarda qayta urinish oraliqlari (soniya).
_RETRY_DELAYS = [2.0, 5.0]

# Zaxira himoya: model butun max_tokens budjetini o'ylashga yoki uzun javobga
# sarflab, finish_reason="length" bilan JSONsiz javob qaytarsa, budjetni
# ikkilantirib bir marta qayta uramiz.
_TRUNCATED_REASON = "length"
_MAX_NUM_PREDICT = 32768


class OllamaParserError(RuntimeError):
    """Model so'rovi bajarilmaganda ko'tariladi."""


def _chat_endpoint() -> str:
    """VLLM_BASE_URL (yoki OLLAMA_HOST) dan `/v1/chat/completions` manzilini yasaydi."""
    base = (settings.VLLM_BASE_URL or settings.OLLAMA_HOST or "").strip().rstrip("/")
    if not base:
        raise ValueError("VLLM_BASE_URL sozlanmagan. .env faylini tekshiring.")
    if base.endswith("/chat/completions"):
        return base
    for suffix in ("/api/chat", "/api/generate"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if not base.endswith("/v1"):
        base = f"{base.rstrip('/')}/v1"
    return f"{base}/chat/completions"


def _post_chat(payload: dict) -> dict:
    endpoint = _chat_endpoint()
    headers = {"Content-Type": "application/json"}
    if settings.VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.VLLM_API_KEY}"
    if "ngrok" in endpoint.lower():
        headers["ngrok-skip-browser-warning"] = "true"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.OBYEKTIVKA_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise OllamaParserError(f"Model server HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise OllamaParserError(f"Model serverga ulanib bo'lmadi ({endpoint}): {type(exc).__name__}: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OllamaParserError(f"Model server JSON bo'lmagan javob qaytardi: {body[:300]}") from exc
    if not isinstance(parsed, dict):
        raise OllamaParserError("Model server javobi JSON obyekt emas.")
    return parsed


def _first_choice(parsed: dict) -> dict:
    choices = parsed.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _json_from_response(parsed: dict) -> dict | None:
    """
    Javobdan JSON obyektni ajratadi.

    Avval odatdagi `message.content`, so'ng reasoning maydonlari tekshiriladi —
    model ba'zan natijani o'ylash bloki ichiga yozib yuboradi. Hech qayeridan
    JSON chiqmasa None qaytadi.
    """
    message = _first_choice(parsed).get("message")
    if not isinstance(message, dict):
        message = {}

    for candidate in (message.get("content"), message.get("reasoning_content"), message.get("reasoning")):
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        try:
            return extract_json_object(candidate)
        except ValueError:
            continue
    return None


def _is_truncated(parsed: dict) -> bool:
    """Javob token budjeti tugagani uchun uzilib qolganmi?"""
    return _first_choice(parsed).get("finish_reason") == _TRUNCATED_REASON


def _empty_response_error(parsed: dict, num_predict: int) -> OllamaParserError:
    """Bo'sh/JSONsiz javob uchun sababni ko'rsatuvchi xato yasaydi."""
    usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
    completion_tokens = usage.get("completion_tokens")
    if _is_truncated(parsed):
        return OllamaParserError(
            f"Model ({settings.OBYEKTIVKA_MODEL}) javobi tugallanmadi: {completion_tokens} token "
            f"yozilgach max_tokens={num_predict} budjeti tugadi. \"Thinking\" modellari "
            "budjetni o'ylashga sarflab qo'yadi — OBYEKTIVKA_MODEL ni o'ylamaydigan "
            "instruct modelga almashtiring yoki OBYEKTIVKA_NUM_PREDICT ni oshiring."
        )
    return OllamaParserError(
        f"Model javobida JSON topilmadi (finish_reason={_first_choice(parsed).get('finish_reason')}, "
        f"completion_tokens={completion_tokens})."
    )


def _build_payload(prompt: str, num_predict: int, strict_json: bool = False) -> dict:
    """
    Chat so'rovini yig'adi.

    JSON rejimi (`response_format`) ataylab standart holatda YUBORILMAYDI:
    Ollama'da grammatika cheklovi bir xil hajmdagi javobga 24 sekund qo'shgan,
    model esa tizim promptiga ko'ra allaqachon toza JSON qaytaradi.
    Javobdan JSON chiqmaganda `strict_json=True` bilan qayta uriladi.
    Kontekst hajmi (num_ctx) vLLM serverida qat'iy belgilangan.
    """
    payload = {
        "model": settings.OBYEKTIVKA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": num_predict,
        "temperature": settings.OBYEKTIVKA_TEMPERATURE,
        "top_p": 1,
    }
    if strict_json:
        payload["response_format"] = {"type": "json_object"}
    return payload


async def _post_chat_with_retries(payload: dict) -> dict:
    """Ulanish/tunnel xatolarida qayta urinib so'rov yuboradi."""
    last_exc: OllamaParserError | None = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        if attempt:
            await asyncio.sleep(_RETRY_DELAYS[attempt - 1])
        try:
            return await asyncio.to_thread(_post_chat, payload)
        except OllamaParserError as exc:
            last_exc = exc
    raise OllamaParserError(f"Model so'rovi muvaffaqiyatsiz: {last_exc}") from last_exc


async def _request_json(prompt: str) -> dict:
    """Bitta bo'lakni so'rab, javobidan JSON obyektni qaytaradi."""
    num_predict = settings.OBYEKTIVKA_NUM_PREDICT
    parsed = await _post_chat_with_retries(_build_payload(prompt, num_predict))
    raw_json = _json_from_response(parsed)

    # Budjet tugagani uchun uzilgan bo'lsa — kattaroq budjet bilan bir marta qayta uramiz.
    if raw_json is None and _is_truncated(parsed) and num_predict < _MAX_NUM_PREDICT:
        num_predict = min(num_predict * 2, _MAX_NUM_PREDICT)
        parsed = await _post_chat_with_retries(_build_payload(prompt, num_predict))
        raw_json = _json_from_response(parsed)

    # Model JSON o'rniga erkin matn yozgan bo'lsa — JSON rejimi bilan oxirgi marta uramiz.
    if raw_json is None:
        parsed = await _post_chat_with_retries(_build_payload(prompt, num_predict, strict_json=True))
        raw_json = _json_from_response(parsed)

    if raw_json is None:
        raise _empty_response_error(parsed, num_predict)
    return raw_json


async def parse_obyektivka_with_ollama(text: str) -> ObyektivkaData:
    """
    Obyektivka matnini OBYEKTIVKA_MODEL (standart: VLLM_MODEL, gemma4-31b) bilan tahlil qiladi.

    Nom tarixiy sabab bilan saqlangan (router shu nomni import qiladi).
    Qarindoshlar jadvali kod bilan o'qilgan bo'lsa modelga faqat ikkita so'rov ketadi.
    """
    cleaned = clean_document_text(text)

    relatives = parse_relatives_table(cleaned)
    # Profil va mehnat faoliyatiga qarindoshlar jadvali kerak emas — uni olib
    # tashlash promptni ~2 barobar qisqartiradi (2076 -> 1354 token).
    without_table = strip_relatives_table(cleaned)
    prompts = [build_profile_prompt(without_table), build_work_history_prompt(without_table)]
    if relatives is None:
        # Jadval tanilmadi (boshqa shakl yoki notanish qarindoshlik so'zi) —
        # bu bo'limni model o'qiydi, unga to'liq matn kerak.
        prompts.append(build_relatives_prompt(cleaned))

    # So'rovlar KETMA-KET yuboriladi: avvalgi serverda bir vaqtda ikkita so'rov
    # ikkinchi slot ajratishni juda qimmatlashtirgan (17 s -> 145 s).
    merged: dict = {}
    for prompt in prompts:
        merged.update(await _request_json(prompt))
    if relatives is not None:
        merged["relatives"] = relatives

    return build_obyektivka_data(merged)
