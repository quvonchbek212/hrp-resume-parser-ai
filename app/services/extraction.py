"""
Obyektivka matnidan ma'lumot ajratish uchun umumiy qism:
prompt, model javobidan JSON ajratish va uni sxemaga o'girish.
Ollama va Gemini parserlari shu moduldan foydalanadi.
"""
import json
import re

from ..schemas.obyektivka import ObyektivkaData, RelationshipType, RelativeInfo, WorkHistory

SYSTEM_PROMPT = (
    "Siz Uzbekiston davlat hujjatlarini tahlil qiluvchi ekspertsiz. "
    "Har doim faqat bitta JSON obyekt qaytarasiz — izoh, markdown yoki qo'shimcha matnsiz."
)

EXTRACTION_PROMPT = """
Siz Uzbekiston davlat hujjati — "Obyektivka" (Ma'lumotnoma, 3.5-shakl) ni tahlil qiluvchi ekspertsiz.

Quyida obyektivka matni berilgan. Ushbu matndan barcha ma'lumotlarni aniq va to'liq ajratib,
faqat JSON formatida qaytaring. Boshqa hech qanday matn, izoh yoki markdown yozmang.

OBYEKTIVKA MATNI:
{text}

Quyidagi JSON strukturasini to'ldiring:
{{
  "surname": "Familiyasi (masalan: Абдуллаев)",
  "name": "Ismi (masalan: Ботир)",
  "patronymic": "Otasining ismi / sharifi (masalan: Баҳодирович)",
  "current_position": "Hozirgi lavozimi va tashkilot nomi",
  "position_since": "Lavozimga qachondan, KK.OO.YYYY formatida (masalan: 06.09.2010). Kun noma'lum bo'lsa 01 qo'ying.",
  "birth_date": "Tug'ilgan sanasi (masalan: 27.02.1972)",
  "birth_place": "Tug'ilgan joyi (viloyat, tuman)",
  "nationality": "Millati",
  "party_membership": "Partiyaviylik yoki 'yo'q'",
  "education_level": "Ma'lumoti (oliy, o'rta-maxsus va h.k.)",
  "graduated_from": "Qaysi ta'lim muassasasi, qachon, qaysi daraja (bakalavriat/magistratura)",
  "specialty": "Mutaxassisligi",
  "academic_degree": "Ilmiy darajasi va yili (masalan: falsafa fanlari nomzodi (2003))",
  "academic_title": "Ilmiy unvoni va yili (masalan: dotsent (2005))",
  "foreign_languages": "Biladigan chet tillari",
  "state_awards": "Davlat mukofotlari yoki 'yo'q'",
  "deputy_status": "Deputatlik holati yoki 'yo'q'",
  "work_history": [
    {{
      "period": "1993-1997 yy.",
      "description": "Toshkent davlat universiteti talabasi"
    }}
  ],
  "relatives": [
    {{
      "relation": "Otasi",
      "full_name": "To'liq ismi sharifi",
      "birth_year_place": "1941 yil, Samarqand shahri",
      "workplace_position": "Ish joyi va lavozimi yoki 'Pensiyada (oldingi lavozimi)'",
      "address": "To'liq manzil"
    }}
  ]
}}

Muhim qoidalar:
1. Faqat JSON qaytaring, boshqa hech narsa yozmang
2. Agar ma'lumot topilmasa, null qo'ying
3. Mehnat faoliyatini barcha yozuvlar bilan to'ldiring
4. Qarindoshlar jadvalidagi BARCHA kishilarni relatives ro'yxatiga qo'shing
5. Vafot etgan qarindoshlar uchun workplace_position ga "YILI yilda vafot etgan (oldingi lavozimi)" yozing
6. "relation" ga jadvaldagi qarindoshlik so'zini AYNAN ko'chiring ("Отаси", "Турмуш ўртоғи",
   "Қайнонаси" kabi) — tarjima qilmang, kodga o'girmang
"""


# ---------------------------------------------------------------------------
# Bo'lingan promptlar
#
# Bitta so'rovda butun JSON so'ralganda model ~1789 token yozadi va bu ~115
# sekund oladi — hammasi generatsiyaga ketadi, prompt eval atigi 3 s. Server
# so'rovlarni ketma-ket bajaradi, ya'ni yagona o'lchov — yozilgan tokenlar soni.
# Shuning uchun javob mustaqil bo'laklarga bo'lingan: har bo'lak faqat o'z
# maydonlarini yozadi va keraksiz matn promptdan olib tashlanadi.
# ---------------------------------------------------------------------------

_PROMPT_HEADER = """
Siz Uzbekiston davlat hujjati — "Obyektivka" (Ma'lumotnoma, 3.5-shakl) ni tahlil qiluvchi ekspertsiz.
Quyidagi matndan so'ralgan ma'lumotni ajratib, faqat JSON qaytaring.
Boshqa hech qanday matn, izoh yoki markdown yozmang. Ma'lumot topilmasa null qo'ying.

OBYEKTIVKA MATNI:
{text}
"""

PROFILE_PROMPT = _PROMPT_HEADER + """
Faqat quyidagi JSON ni to'ldiring (mehnat faoliyati va qarindoshlar SO'RALMAYDI):
{{
  "surname": "Familiyasi (masalan: Абдуллаев)",
  "name": "Ismi (masalan: Ботир)",
  "patronymic": "Otasining ismi / sharifi (masalan: Баҳодирович)",
  "current_position": "Hozirgi lavozimi va tashkilot nomi",
  "position_since": "Lavozimga qachondan, KK.OO.YYYY formatida. Kun noma'lum bo'lsa 01 qo'ying.",
  "birth_date": "Tug'ilgan sanasi (masalan: 27.02.1972)",
  "birth_place": "Tug'ilgan joyi (viloyat, tuman)",
  "nationality": "Millati",
  "party_membership": "Partiyaviylik yoki 'yo'q'",
  "education_level": "Ma'lumoti (oliy, o'rta-maxsus va h.k.)",
  "graduated_from": "Qaysi ta'lim muassasasi, qachon, qaysi daraja",
  "specialty": "Mutaxassisligi",
  "academic_degree": "Ilmiy darajasi va yili",
  "academic_title": "Ilmiy unvoni va yili",
  "foreign_languages": "Biladigan chet tillari",
  "state_awards": "Davlat mukofotlari yoki 'yo'q'",
  "deputy_status": "Deputatlik holati yoki 'yo'q'"
}}
"""

WORK_HISTORY_PROMPT = _PROMPT_HEADER + """
Faqat mehnat faoliyatini qaytaring, boshqa maydonlarni YOZMANG:
{{
  "work_history": [
    {{"period": "1993-1997 yy.", "description": "Toshkent davlat universiteti talabasi"}}
  ]
}}

Qoida: mehnat faoliyati bo'limidagi BARCHA yozuvlarni tartibi bilan qaytaring.
"""

RELATIVES_PROMPT = _PROMPT_HEADER + """
Faqat qarindoshlar ro'yxatini qaytaring, boshqa maydonlarni YOZMANG:
{{
  "relatives": [
    {{
      "relation": "Otasi",
      "full_name": "To'liq ismi sharifi",
      "birth_year_place": "1941 yil, Samarqand shahri",
      "workplace_position": "Ish joyi va lavozimi yoki 'Pensiyada (oldingi lavozimi)'",
      "address": "To'liq manzil"
    }}
  ]
}}

Qoidalar:
1. Qarindoshlar jadvalidagi BARCHA kishilarni qo'shing
2. Vafot etgan qarindoshlar uchun workplace_position ga "YILI yilda vafot etgan (oldingi lavozimi)" yozing
3. "relation" ga jadvaldagi qarindoshlik so'zini AYNAN ko'chiring ("Отаси", "Турмуш ўртоғи",
   "Қайнонаси" kabi) — tarjima qilmang, kodga o'girmang
"""


# Qarindoshlik so'zi -> RelationshipType. Model faqat hujjatdagi so'zni ko'chiradi,
# kodga o'girishni shu jadval bajaradi: promptdan 30 qatorlik enum ro'yxati olib
# tashlangani model "o'ylash"ini qisqartiradi va natijani barqaror qiladi.
# Tartib muhim — aniqrog'i oldin turadi ("қайин ака" > "ака", "амакивачча" > "амаки").
_RELATION_KEYWORDS: list[tuple[tuple[str, ...], RelationshipType]] = [
    (("қайин ака", "қайн ака", "qayin aka", "qayn aka"), RelationshipType.OLDER_BROTHER_IN_LAW),
    (("қайин ука", "қайн ука", "qayin uka", "qayn uka"), RelationshipType.YOUNGER_BROTHER_IN_LAW),
    (("қайин опа", "қайн опа", "qayin opa", "qayn opa"), RelationshipType.OLDER_SISTER_IN_LAW),
    (("қайин синг", "қайн синг", "qayin sing", "qayn sing"), RelationshipType.YOUNGER_SISTER_IN_LAW),
    (("қайнота", "қайната", "qaynota", "qaynata"), RelationshipType.FATHER_IN_LAW),
    (("қайнона", "қайнана", "qaynona", "qaynana"), RelationshipType.MOTHER_IN_LAW),
    (("куёв", "kuyov", "son-in-law", "son in law"), RelationshipType.SON_IN_LAW),
    (("келин", "kelin", "daughter-in-law", "daughter in law"), RelationshipType.DAUGHTER_IN_LAW),
    (("амакивачча", "amakivachcha"), RelationshipType.PATERNAL_UNCLE_CHILD),
    (("аммавачча", "ammavachcha"), RelationshipType.PATERNAL_AUNT_CHILD),
    (("тоғавачча", "тогавачча", "tog'avachcha", "togavachcha"), RelationshipType.MATERNAL_UNCLE_CHILD),
    (("холавачча", "xolavachcha", "holavachcha"), RelationshipType.MATERNAL_AUNT_CHILD),
    (("она томондан бобо", "ona tomondan bobo"), RelationshipType.MATERNAL_GRANDFATHER),
    (("она томондан був", "ona tomondan buv"), RelationshipType.MATERNAL_GRANDMOTHER),
    # Tomoni ko'rsatilmagan "бобоси/бувиси" — shakllarda odatda ota tomoni nazarda tutiladi.
    (("бобо", "bobo"), RelationshipType.PATERNAL_GRANDFATHER),
    (("бувиси", "бува", "buvi", "buva"), RelationshipType.PATERNAL_GRANDMOTHER),
    (("набира", "nabira"), RelationshipType.GRANDCHILD),
    (("жиян", "jiyan"), RelationshipType.NIBLING),
    (("амаки", "amaki"), RelationshipType.PATERNAL_UNCLE),
    (("амма", "amma"), RelationshipType.PATERNAL_AUNT),
    (("тоға", "тога", "tog'a", "toga"), RelationshipType.MATERNAL_UNCLE),
    (("хола", "xola", "hola"), RelationshipType.MATERNAL_AUNT),
    (("васийлик", "vasiylik"), RelationshipType.WARD),
    (("васий", "vasiy"), RelationshipType.GUARDIAN),
    (("турмуш ўрто", "турмуш урто", "turmush o'rto", "turmush urto", "рафиқа", "рафика",
      "rafiq", "эри", "хотин", "xotin", "аёли", "wife", "husband"), RelationshipType.SPOUSE),
    (("ўғли", "угли", "o'g'li", "ogli", "қизи", "кизи", "qizi", "фарзанд", "farzand",
      "son", "daughter", "child"), RelationshipType.CHILD),
    (("ака", "aka", "ука", "uka", "brother"), RelationshipType.BROTHER),
    (("опа", "opa", "сингил", "сингл", "singil", "singl", "sister"), RelationshipType.SISTER),
    (("ота", "ota", "падар"), RelationshipType.FATHER),
    (("она", "ona", "модар"), RelationshipType.MOTHER),
]

_APOSTROPHES = {"ʻ": "'", "ʼ": "'", "‘": "'", "’": "'", "`": "'", "´": "'"}


def _normalize_relation(value: str) -> str:
    """Qarindoshlik so'zini taqqoslashga tayyorlaydi (registr, apostrof, bo'shliq)."""
    text = value.strip().lower()
    for char, replacement in _APOSTROPHES.items():
        text = text.replace(char, replacement)
    return re.sub(r"\s+", " ", text)


def resolve_relation(value: object) -> RelationshipType | None:
    """
    Model qaytargan qarindoshlik qiymatini RelationshipType ga o'giradi.

    Hujjatdagi so'z ("Отаси", "Қайнонаси") ham, tayyor enum nomi ("FATHER") ham
    qabul qilinadi. Tanib bo'lmasa OTHER qaytadi — validatsiya xatosi emas.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = _normalize_relation(value)
    try:
        return RelationshipType(normalized.upper())
    except ValueError:
        pass

    for keywords, relation in _RELATION_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return relation
    return RelationshipType.OTHER


# 3.5-shakl NAMUNA nusxalarida hujjat oxirida "Изоҳ:" bandlari va undan keyin
# "Зарур / Тўғри / Нотўғри" yo'riqnoma jadvali keladi. Bular ma'lumot emas —
# to'ldirish qoidalari, lekin model ularni ham tahlil qilishga urinib, javobni
# bir necha barobar sekinlashtiradi. Shuning uchun promptga qo'shilmaydi.
_FOOTER_MARKERS = (
    "изоҳ:", "изох:", "изоҳ", "izoh:", "izoh",
    # NAMUNA nusxasida qoidalar alohida jadval bo'lib keladi va uning birinchi
    # qatori shu. Kesilmasa ichidagi "Тўғри/Нотўғри" namunalari ("Ўғли |
    # Рўзиев Ахрор Холматович | ...") haqiqiy qarindosh deb o'qib yuboriladi.
    "зарур:", "зарур ", "zarur:", "zarur ",
)
# Shakl chetidagi joylashuv izohlari: "Шрифт 14", "6 пт", "3х4 см ...".
_NOISE_LINE = re.compile(
    r"^\s*(?:(?:шрифт|shrift)\s*\d+"
    r"|\d+\s*(?:пт|pt)"
    r"|намуна"
    r"|\d+\s*[хx]\s*\d+\s*см.*)\s*$",
    re.IGNORECASE,
)


def _starts_with_footer_marker(line: str) -> bool:
    normalized = line.strip().lower()
    return any(normalized.startswith(marker) for marker in _FOOTER_MARKERS)


def clean_document_text(text: str) -> str:
    """Hujjat matnidan shakl yo'riqnomalari va joylashuv izohlarini olib tashlaydi."""
    lines = text.splitlines()

    # Qarindoshlar jadvali qayerda ekanini avval aniqlaymiz: 3.5-shaklda "Изоҳ:"
    # bandlari maʼlumotnomaning old tomoni oxirida, yaʼni jadvaldan OLDIN
    # kelishi mumkin. Oddiy "markerdan oxirigacha kes" qoidasi bunday hujjatda
    # butun jadvalni yo'q qilardi.
    relative_rows = [index for index, line in enumerate(lines) if _is_relative_data_row(line)]
    first_relative = relative_rows[0] if relative_rows else -1
    last_relative = relative_rows[-1] if relative_rows else -1

    # 1) Jadvaldan keyingi yo'riqnoma — oxirigacha kesiladi.
    cutoff = len(lines)
    for index, line in enumerate(lines):
        if index < len(lines) // 2 or index <= last_relative:
            continue
        if _starts_with_footer_marker(line):
            cutoff = index
            break

    # 2) Jadvaldan oldingi yo'riqnoma bloki — faqat o'sha blok tashlanadi,
    #    jadvalning o'zi saqlanib qoladi.
    dropped: set[int] = set()
    if first_relative > 0:
        for index in range(first_relative):
            if _starts_with_footer_marker(lines[index]):
                dropped.update(range(index, first_relative))
                break

    kept = [
        line
        for index, line in enumerate(lines[:cutoff])
        if index not in dropped and line.strip() and not _NOISE_LINE.match(line)
    ]
    cleaned = "\n".join(kept).strip()
    # Tozalash hujjatni buzib yuborgan bo'lsa (kutilmagan shakl), xomini qaytaramiz.
    return cleaned if cleaned else text.strip()


def build_prompt(text: str) -> str:
    """Obyektivka matnini tozalab, promptga joylaydi."""
    return EXTRACTION_PROMPT.format(text=clean_document_text(text))


def build_profile_prompt(cleaned_text: str) -> str:
    """Faqat shaxsiy maydonlar uchun prompt (matn allaqachon tozalangan)."""
    return PROFILE_PROMPT.format(text=cleaned_text)


def build_work_history_prompt(cleaned_text: str) -> str:
    """Faqat mehnat faoliyati uchun prompt."""
    return WORK_HISTORY_PROMPT.format(text=cleaned_text)


def build_relatives_prompt(cleaned_text: str) -> str:
    """Faqat qarindoshlar ro'yxati uchun prompt (jadvalni kod o'qiy olmaganda)."""
    return RELATIVES_PROMPT.format(text=cleaned_text)


# --- Qarindoshlar jadvalini kod bilan o'qish ------------------------------
#
# `file_extractor` docx/pdf jadval qatorlarini " | " bilan ajratib beradi, ya'ni
# qarindoshlar jadvali modelga yetib borgunicha ALLAQACHON ustunlarga bo'lingan.
# Uni modelga qayta ko'chirtirish javobning ~60% tokenini yeydi (o'lchov: 1789
# tokendan ~1080 tasi), holbuki bu sof nusxa ko'chirish. Shuning uchun jadval
# shakli tanilsa kod o'zi o'qiydi; shubha bo'lsa — modelga qaytariladi.

# Kamida shuncha ustun bo'lmasa, bu qarindoshlar jadvali emas.
_MIN_RELATIVE_COLUMNS = 4
# 3.5-shakl jadvali: qarindoshligi | F.I.Sh. | tug'ilgan yili va joyi |
# ish joyi va lavozimi | turar joyi.
_MAX_RELATIVE_COLUMNS = 5

# Haqiqiy qarindoshlik so'zi qisqa ("Турмуш ўртоғи", "Қайнонаси"). Yo'riqnoma
# jadvalidagi katak esa butun boshli gap bo'ladi va ichida "отасининг" kabi
# so'zlar uchraydi — uzunlik shuning uchun tekshiriladi.
_MAX_RELATION_WORDS = 4
_MAX_RELATION_CHARS = 40

# Jadval sarlavhasi ("Қариндошлиги | Ф.И.Ш. | Туғилган йили ..."). Ikkitadan
# ko'p belgi topilsa — bu sarlavha qatori, ma'lumot emas.
_TABLE_HEADER_HINTS = (
    "қариндош", "qarindosh", "ф.и.ш", "f.i.sh", "фиш",
    "иш жойи", "ish joyi", "лавозими", "lavozimi",
    "яшаш", "yashash", "манзил", "manzil",
    "туғилган", "tug'ilgan", "tugilgan",
)

# "1." / "2)" kabi tartib raqami ustuni.
_ROW_NUMBER = re.compile(r"^\d{1,3}[.)]?$")


def _split_row(line: str) -> list[str]:
    """Jadval qatorini ustunlarga bo'ladi va birlashtirilgan kataklarni yig'adi.

    python-docx birlashtirilgan katakni har bir ustun uchun takrorlab beradi —
    ketma-ket bir xil qiymatlar shuning uchun bittaga tushiriladi.
    """
    cells = [raw.strip() for raw in line.split("|")]
    if len(cells) <= _MAX_RELATIVE_COLUMNS:
        # Shakl 5 ustunli. Ikki katak bir xil bo'lishi mumkin (masalan ish joyi
        # va manzil ustuniga bir xil "vafot etgan" yozuvi) — ularni birlashtirib
        # yubormaymiz, aks holda manzil yo'qoladi.
        return cells

    # Ustun ko'p bo'lsa — python-docx birlashtirilgan katakni har ustun uchun
    # takrorlagan. Ketma-ket takrorlarni bittaga tushiramiz.
    merged: list[str] = []
    for value in cells:
        if merged and value and value == merged[-1]:
            continue
        merged.append(value)
    return merged


def _looks_like_relation_cell(value: str) -> bool:
    """Katak qarindoshlik so'ziga o'xshaydimi (gapga emas)?"""
    text = value.strip()
    return bool(text) and len(text) <= _MAX_RELATION_CHARS and len(text.split()) <= _MAX_RELATION_WORDS


def _is_header_row(cells: list[str]) -> bool:
    joined = " ".join(cells).lower()
    return sum(hint in joined for hint in _TABLE_HEADER_HINTS) >= 2


def strip_relatives_table(cleaned_text: str) -> str:
    """Qarindoshlar jadvali qatorlarini matndan olib tashlaydi.

    Profil va mehnat faoliyati promptlariga jadval kerak emas — u prompt
    tokenlarining katta qismini egallaydi (o'lchov: 2076 -> 1354 token).
    """
    kept = [line for line in cleaned_text.splitlines() if not _is_relative_row(line)]
    stripped = "\n".join(kept).strip()
    return stripped or cleaned_text


def _relative_row_cells(line: str) -> list[str] | None:
    """Qatorni qarindoshlar jadvali ustunlariga bo'ladi, mos kelmasa None."""
    if line.count("|") < _MIN_RELATIVE_COLUMNS - 1:
        return None
    cells = _split_row(line)
    if cells and _ROW_NUMBER.match(cells[0]):
        cells = cells[1:]
    return cells if len(cells) >= _MIN_RELATIVE_COLUMNS else None


def _is_relative_data_row(line: str) -> bool:
    """Qator haqiqiy qarindosh yozuvimi (sarlavha va yo'riqnoma emas)?"""
    cells = _relative_row_cells(line)
    if cells is None or _is_header_row(cells) or not _looks_like_relation_cell(cells[0]):
        return False
    relation = resolve_relation(cells[0])
    return relation is not None and relation is not RelationshipType.OTHER


def _is_relative_row(line: str) -> bool:
    """Qator qarindoshlar jadvaliga tegishlimi (sarlavha ham hisoblanadi)?"""
    cells = _relative_row_cells(line)
    if cells is None:
        return False
    return _is_header_row(cells) or _is_relative_data_row(line)


def parse_relatives_table(cleaned_text: str) -> list[dict] | None:
    """Qarindoshlar jadvalini matndan to'g'ridan-to'g'ri o'qiydi.

    Har bir nomzod qator tanilgandagina ro'yxat qaytadi. Bitta qator ham
    tushunilmasa None qaytadi va chaqiruvchi butun bo'limni modelga topshiradi —
    yarim o'qilgan jadval noto'g'ri javobdan ko'ra yomonroq.
    """
    rows: list[dict] = []
    candidates = 0

    for line in cleaned_text.splitlines():
        if line.count("|") < _MIN_RELATIVE_COLUMNS - 1:
            continue
        cells = _split_row(line)
        if cells and _ROW_NUMBER.match(cells[0]):
            cells = cells[1:]
        if len(cells) < _MIN_RELATIVE_COLUMNS:
            continue
        if _is_header_row(cells):
            continue

        if not _looks_like_relation_cell(cells[0]):
            # Butun boshli gap — bu ma'lumot qatori emas (yo'riqnoma yoki izoh).
            continue

        candidates += 1
        relation = resolve_relation(cells[0])
        if relation is None or relation is RelationshipType.OTHER:
            # Qisqa, lekin tanilmagan so'z — haqiqiy qarindoshlik atamasi
            # bo'lishi mumkin. Jadvalni yarim o'qigandan ko'ra modelga beramiz.
            return None

        # 5 tadan ortiq ustun bo'lsa oxirgilari manzilning davomi.
        address = ", ".join(part for part in cells[4:] if part) if len(cells) > 4 else ""
        rows.append({
            "relation": cells[0],
            "full_name": cells[1],
            "birth_year_place": cells[2],
            "workplace_position": cells[3],
            "address": address,
        })

    if not rows or len(rows) != candidates:
        return None
    return rows


def extract_json_object(response_text: str) -> dict:
    """Model javobidan JSON obyektni ajratib oladi."""
    text = response_text.strip()
    # Reasoning modellari <think>...</think> blokini qo'shishi mumkin.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Model javobi JSON formatida emas. Javob: {text[:300]}")


def as_text(value: object) -> str | None:
    """
    Model qaytargan qiymatni matnga keltiradi.

    Sxemada matn kutilgan joyga modellar ba'zan ro'yxat yoki obyekt qaytaradi
    (masalan academic_degree: [{"degree": "...", "year": "2010"}]). Ularni
    o'qishli satrga yig'amiz — aks holda butun so'rov validatsiyada yiqiladi.
    """
    if value is None or isinstance(value, bool):
        return None if value is None else str(value)
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        parts = [as_text(item) for item in value.values()]
        return ", ".join(part for part in parts if part) or None
    if isinstance(value, (list, tuple)):
        parts = [as_text(item) for item in value]
        return "; ".join(part for part in parts if part) or None
    return str(value).strip() or None


def build_obyektivka_data(raw_json: dict) -> ObyektivkaData:
    """Model qaytargan xom JSON ni ObyektivkaData sxemasiga o'giradi."""
    work_history = []
    for item in raw_json.get("work_history") or []:
        if isinstance(item, dict):
            work_history.append(WorkHistory(
                period=as_text(item.get("period")),
                description=as_text(item.get("description")),
            ))

    relatives = []
    for item in raw_json.get("relatives") or []:
        if isinstance(item, dict):
            relatives.append(RelativeInfo(
                relation=resolve_relation(item.get("relation")),
                full_name=as_text(item.get("full_name")),
                birth_year_place=as_text(item.get("birth_year_place")),
                workplace_position=as_text(item.get("workplace_position")),
                address=as_text(item.get("address")),
            ))

    text_fields = (
        "surname", "name", "patronymic", "current_position", "position_since",
        "birth_date", "birth_place", "nationality", "party_membership",
        "education_level", "graduated_from", "specialty", "academic_degree",
        "academic_title", "foreign_languages", "state_awards", "deputy_status",
    )
    return ObyektivkaData(
        **{field: as_text(raw_json.get(field)) for field in text_fields},
        work_history=work_history,
        relatives=relatives,
    )
