"""
Obyektivka (Ma'lumotnoma) fayl yuklash va tahlil qilish router.
"""
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException

from ..core.config import settings
from ..schemas.obyektivka import ObyektivkaResponse
from ..services.file_extractor import extract_text

router = APIRouter(
    prefix="/obyektivka",
    tags=["Obyektivka Parser"],
)

ALLOWED_EXTENSIONS = {".docx", ".pdf"}
MAX_FILE_SIZE = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # MB → bytes


@router.post(
    "/parse",
    response_model=ObyektivkaResponse,
    summary="Obyektivka faylini tahlil qilish",
    responses={
        200: {"description": "Muvaffaqiyatli tahlil qilindi"},
        400: {"description": "Noto'g'ri fayl turi yoki hajmi"},
        422: {"description": "Fayl o'qib bo'lmadi"},
        500: {"description": "Model (Ollama) xatosi"},
    },
)
async def parse_obyektivka(
    file: UploadFile = File(..., description="DOCX yoki PDF formatidagi obyektivka fayli"),
):
    from ..services.ollama_parser import parse_obyektivka_with_ollama

    # 1. Kengaytmani tekshirish
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Noto'g'ri fayl turi: '{suffix}'. Faqat {', '.join(ALLOWED_EXTENSIONS)} qabul qilinadi.",
        )

    # 2. Faylni o'qish
    file_bytes = await file.read()

    # 3. Hajmni tekshirish
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Fayl juda katta: {len(file_bytes) / 1024 / 1024:.1f} MB. Maksimal: {settings.MAX_FILE_SIZE_MB} MB.",
        )

    # 4. Matnni ajratib olish
    try:
        extracted_text, file_type = extract_text(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not extracted_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Fayldan matn ajratib olib bo'lmadi. Fayl bo'sh yoki himoyalangan bo'lishi mumkin.",
        )

    # 5. vLLM modeli (OBYEKTIVKA_MODEL, standart: VLLM_MODEL=gemma4-31b) bilan tahlil qilish
    try:
        parsed_data = await parse_obyektivka_with_ollama(extracted_text)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model ({settings.OBYEKTIVKA_MODEL}) xatosi: {str(e)}",
        )

    return ObyektivkaResponse(
        success=True,
        filename=file.filename,
        file_type=file_type,
        data=parsed_data,
        raw_text_preview=extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
    )

