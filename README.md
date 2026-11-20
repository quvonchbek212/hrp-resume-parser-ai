# 📄 HRP Obyektivka (CV / Resume) Parser API

An intelligent document extraction service built for the **HRP (Human Resource Planning) Employee Management ERP System** to parse official Uzbekistan **Form 3.5 (Ma'lumotnoma / Obyektivka)** documents into structured JSON format.

---

## 📑 Table of Contents
- [Overview & Business Context](#-overview--business-context)
- [Key Features](#-key-features)
- [Architecture & Extraction Pipeline](#-architecture--extraction-pipeline)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Output Schema](#-output-schema)
- [License](#-license)

---

## 🎯 Overview & Business Context

In Uzbekistan's public and enterprise sectors, employee onboarding relies heavily on the standardized **Form 3.5 (Obyektivka / Ma'lumotnoma)**. These documents contain intricate tabular and textual records covering:
- Personal credentials and demographics
- Educational background and academic degrees
- Detailed chronological employment history
- Comprehensive biographical records of close relatives

Manually transcribing these documents into an HR database is time-consuming and error-prone. The **HRP Obyektivka Parser** automates this onboarding pipeline by accepting `.docx` or `.pdf` files, extracting tables and raw text, and passing them to an LLM (*Qwen 2.5 / Gemma*) to produce normalized, schema-validated JSON.

---

## ✨ Key Features

- **Multi-Format Ingestion**: Supports both Microsoft Word (`.docx`) and Adobe PDF (`.pdf`) documents.
- **Biographical & Demographic Extraction**: Automatically extracts full name, birth date, birthplace, nationality, party affiliation, education level, university, graduation year, specialization, academic degree, state awards, and military rank.
- **Employment History Chronology**: Parses every position held, organization name, start and end years.
- **Relational Family Tree Extraction**: Fully extracts the close relatives table (*Yaqin qarindoshlar*), including kinship, full name, birth year, birthplace, current occupation/position, and living address.
- **Pydantic Validation**: Guarantees typed, predictable JSON structure for direct database ingestion.

---

## 🏗 Architecture & Extraction Pipeline

```
Uploaded Document (DOCX / PDF)
               │
               ▼
┌──────────────────────────────────────────────┐
│  File Extractor (python-docx / pdfplumber)   │
│  - Table cell preservation                   │
│  - Text layout normalization                 │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  LLM Extraction Engine (Ollama / vLLM)       │
│  - Model: qwen2.5:14b-instruct or gemma4     │
│  - Strict JSON mode schema enforcement       │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Data Sanitization & Schema Validation       │
│  - Pydantic response models                  │
│  - Field type normalization                  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
Final Structured JSON (Profile + Work History + Relatives)
```

---

## 🛠 Technology Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)
- **Server**: [Uvicorn](https://www.uvicorn.org/)
- **Document Processing**: [python-docx](https://python-docx.readthedocs.io/) & [pdfplumber](https://github.com/jsvine/pdfplumber)
- **Validation**: [Pydantic v2](https://docs.pydantic.dev/) & [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- **LLM Engine**: [Ollama](https://ollama.ai/) / OpenAI-compatible vLLM endpoint (`qwen2.5:14b-instruct` / `gemma4-31b`)
- **Fallback Parser**: Optional integration with Google GenAI SDK (`gemini-2.5-flash`)

---

## 📂 Project Structure

```
hrp-obyektivka-parser/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py          # Settings & environment configuration
│   ├── routers/
│   │   ├── __init__.py
│   │   └── obyektivka.py      # Upload & parse endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── obyektivka.py      # Pydantic response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── extraction.py      # Prompt construction & extraction rules
│   │   ├── file_extractor.py  # DOCX/PDF text & table extraction
│   │   ├── gemini_parser.py   # Optional Gemini fallback client
│   │   └── ollama_parser.py   # Ollama / OpenAI-compatible parser client
│   ├── __init__.py
│   └── main.py                # FastAPI application entry point
├── .env.example               # Template environment configuration
├── .gitignore                 # Git ignore file (prevents secret leaks)
├── README.md                  # Comprehensive documentation
└── requirements.txt           # Project dependencies
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10+
- Ollama or an OpenAI-compatible LLM endpoint running `qwen2.5:14b-instruct`

### 2. Clone the Repository
```bash
git clone https://github.com/quvonchbek212/hrp-obyektivka-parser.git
cd hrp-obyektivka-parser
```

### 3. Create a Virtual Environment
```bash
# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env`:
```env
OLLAMA_HOST=http://127.0.0.1:11434
OBYEKTIVKA_MODEL=qwen2.5:14b-instruct
OBYEKTIVKA_TIMEOUT_SECONDS=300
MAX_FILE_SIZE_MB=20
```

### 6. Run the Application
```bash
uvicorn app.main:app --reload --port 8000
```

---

## 📖 API Reference

Interactive Swagger documentation is available at:
👉 **`http://localhost:8000/docs`**

### Parse Resume Document
`POST /obyektivka/parse`

**Request:** `multipart/form-data`
- `file`: The `.docx` or `.pdf` file to parse (Max: 20MB)

**Sample cURL Request:**
```bash
curl -X POST "http://localhost:8000/obyektivka/parse" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_obyektivka.docx"
```

---

## 📊 Output Schema

```json
{
  "success": true,
  "data": {
    "personal_info": {
      "full_name": "Karimov Alisher Rustamovich",
      "birth_date": "1990-05-14",
      "birth_place": "Toshkent shahri",
      "nationality": "O'zbek",
      "party_membership": "Yo'q",
      "education": "Oliy",
      "graduated": "2012 y. Toshkent Axborot Texnologiyalari Universiteti",
      "specialization": "Dasturiy injiniring",
      "academic_degree": "Yo'q",
      "languages": ["O'zbek", "Rus", "Ingliz"],
      "state_awards": "Yo'q"
    },
    "work_history": [
      {
        "period": "2012-2016",
        "position": "Kichik dasturchi",
        "organization": "AloqaBank AT"
      },
      {
        "period": "2016-hozirgi vaqtgacha",
        "position": "Bosh mutaxassis",
        "organization": "Raqamli texnologiyalar vazirligi"
      }
    ],
    "relatives": [
      {
        "relationship": "Otasi",
        "full_name": "Karimov Rustam Aliyevich",
        "birth_year": "1963",
        "birth_place": "Samarqand viloyati",
        "workplace_and_position": "Nafaqada",
        "address": "Toshkent sh., Chilonzor tumani, 5-mavze"
      }
    ]
  }
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
