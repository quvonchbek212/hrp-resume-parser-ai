import os
from pathlib import Path

_PARSER_ROOT = Path(__file__).parent.parent.parent
# config.py -> core -> app -> obyektivka_parser -> loyiha ildizi (.env shu yerda)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=str(_PROJECT_ROOT / ".env"),
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # Asosiy parser: vLLM (OpenAI-compatible) server
        VLLM_BASE_URL: str = ""
        VLLM_API_KEY: str = ""
        VLLM_MODEL: str = "gemma4-31b"
        # Eski Ollama manzili — VLLM_BASE_URL berilmaganda zaxira sifatida.
        OLLAMA_HOST: str = "http://127.0.0.1:11434"
        # Bo'sh bo'lsa VLLM_MODEL ishlatiladi.
        OBYEKTIVKA_MODEL: str = ""
        OBYEKTIVKA_TIMEOUT_SECONDS: float = 300.0
        # Bitta bo'lak (profil / mehnat faoliyati / qarindoshlar) uchun budjet (max_tokens).
        # Butun javob emas — shuning uchun 16384 kerak emas.
        OBYEKTIVKA_NUM_PREDICT: int = 1536
        OBYEKTIVKA_TEMPERATURE: float = 0.0

        # Ixtiyoriy zaxira parser (gemini_parser.py)
        GEMINI_API_KEY: str = ""
        GEMINI_MODEL: str = "gemini-2.5-flash"
        UPLOAD_DIR: Path = _PARSER_ROOT / "uploads"
        MAX_FILE_SIZE_MB: int = 20
        ALLOWED_EXTENSIONS: list[str] = [".docx", ".pdf"]
except ModuleNotFoundError:
    class Settings:
        VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "")
        VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
        VLLM_MODEL = os.getenv("VLLM_MODEL", "gemma4-31b")
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        OBYEKTIVKA_MODEL = os.getenv("OBYEKTIVKA_MODEL", "")
        OBYEKTIVKA_TIMEOUT_SECONDS = float(os.getenv("OBYEKTIVKA_TIMEOUT_SECONDS", "300"))
        OBYEKTIVKA_NUM_PREDICT = int(os.getenv("OBYEKTIVKA_NUM_PREDICT", "1536"))
        OBYEKTIVKA_TEMPERATURE = float(os.getenv("OBYEKTIVKA_TEMPERATURE", "0"))
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        UPLOAD_DIR = _PARSER_ROOT / "uploads"
        MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
        ALLOWED_EXTENSIONS = [".docx", ".pdf"]


settings = Settings()
if not settings.OBYEKTIVKA_MODEL:
    settings.OBYEKTIVKA_MODEL = settings.VLLM_MODEL
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
