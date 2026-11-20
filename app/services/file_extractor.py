"""
Fayl matnini ajratib olish xizmati.
DOCX va PDF formatlarini qo'llab-quvvatlaydi.
"""
import io
from pathlib import Path


def _flatten_cell(text: str) -> str:
    """Katak matnidagi qator uzilishlarini probelga aylantiradi.

    Word kataklarida matn ko'pincha bir necha qatorga o'raladi ("Абдуллаев
Баҳодир").
    Ular tozalanmasa bitta jadval qatori matnda bir necha qatorga bo'linib ketadi
    va "har qator = bitta yozuv" qoidasi buziladi — qarindoshlar jadvalining
    ustunlari siljib, ism tug'ilgan yili katagiga tushib qoladi.
    """
    return " ".join(text.split())


def extract_text_from_docx(file_bytes: bytes) -> str:
    """DOCX fayldan matnni o'qib oladi."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        parts = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        for table in doc.tables:
            for row in table.rows:
                # Bo'sh kataklar ham saqlanadi: ularni tashlab yuborilsa ustunlar
                # siljib, model qo'shni ustun qiymatini (masalan "vafot etgan ...")
                # manzil deb o'qib qoladi.
                row_data = [_flatten_cell(cell.text) for cell in row.cells]
                if any(row_data):
                    parts.append(" | ".join(row_data))

        return "\n".join(parts)
    except Exception as e:
        raise ValueError(f"DOCX o'qishda xato: {e}")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """PDF fayldan matnni o'qib oladi."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        clean_row = [_flatten_cell(cell or "") for cell in row]
                        text_parts.append(" | ".join(clean_row))

        return "\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"PDF o'qishda xato: {e}")


def extract_text(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Fayl turini aniqlab, matnni ajratib oladi.
    Returns: (extracted_text, file_type)
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".docx":
        return extract_text_from_docx(file_bytes), "docx"
    elif suffix == ".pdf":
        return extract_text_from_pdf(file_bytes), "pdf"
    else:
        raise ValueError(f"Qo'llab-quvvatlanmaydigan fayl turi: {suffix}. Faqat .docx va .pdf qabul qilinadi.")
