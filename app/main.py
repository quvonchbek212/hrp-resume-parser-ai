"""
Obyektivka Parser API
Uzbekiston 3.5-shakl ma'lumotnomasini DOCX/PDF dan JSON ga o'girish.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import obyektivka

app = FastAPI(
    title="Obyektivka Parser API",
    description="""
## 📄 Obyektivka Parser

Uzbekiston davlat hujjati **Obyektivka (3.5-shakl Ma'lumotnoma)** ni avtomatik tahlil qilish API.

### Imkoniyatlar
- ✅ **DOCX va PDF** formatlarini qo'llab-quvvatlaydi
- 🤖 **Ollama / qwen2.5:14b-instruct** modeli yordamida ma'lumotlarni ajratib oladi
- 📊 Barcha ma'lumotlarni **JSON** formatida qaytaradi
- 👨‍👩‍👧‍👦 Yaqin **qarindoshlar jadvali** ni to'liq parse qiladi
- 📋 **Mehnat faoliyati** tarixini ajratib oladi

### Foydalanish
1. `/obyektivka/parse` endpoint ga DOCX yoki PDF faylni yuboring
2. API avtomatik matnni o'qib, vLLM serverdagi `OBYEKTIVKA_MODEL` (standart: `VLLM_MODEL` = `gemma4-31b`) ga yuboradi
3. Strukturalangan JSON natija qaytariladi
    """,
    version="1.0.0",
    contact={
        "name": "Obyektivka Parser",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    },
)

# CORS sozlamalari (kerak bo'lsa frontend bilan ishlash uchun)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routerlarni ulash
app.include_router(obyektivka.router)


@app.get("/", tags=["Root"], summary="API haqida ma'lumot")
async def root():
    return {
        "service": "Obyektivka Parser API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "parse": "POST /obyektivka/parse",
            "health": "GET /obyektivka/health",
        },
    }
