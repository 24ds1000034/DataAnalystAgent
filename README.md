# FastAPI LLM Orchestrator (Railway Deploy)

This service accepts a `question.txt` and optional files/URLs, asks Gemini to generate scraping/analysis code, executes that code in a sandboxed job folder, and returns the final `result.json` / `result.txt`.

No ngrok or shell scripts required. Deploy directly to Railway or run locally.

---

## 🚀 Features

- **FastAPI** API with permissive CORS
- **Per-request sandbox**: data lives under `uploads/<request_id>/`
- **Async-safe execution engine** (`task_engine.py`) with:
  - Per-job working dir: `uploads/<request_id>/job_<8hex>/`
  - On-demand library installs (with mapping like `beautifulsoup4 → bs4`)
  - Timeouts and structured logs (`execution_result.txt`)
- **Gemini integration** (`gemini.py`) that uses **API key from environment** (no hardcoded keys)
- **Robust retries** for JSON-only LLM responses
- **Health checks** at `/`, `/healthz`, and `HEAD /`

---

## 📦 Repo Structure

```
.
├─ main.py                   # FastAPI app & orchestration loop
├─ task_engine.py            # Async code runner + pip installer
├─ gemini.py                 # Gemini client & parse_question_with_llm()
├─ requirements.txt          # Deps installed by Railway / local
├─ Procfile                  # (Option A) Process declaration for Railway
├─ railway.toml              # (Option B) Railway config (use one of the two)
├─ .env.example              # Template for environment variables
├─ .gitignore
├─ uploads/
│  └─ .gitkeep               # Keep folder in git; runtime data is ephemeral
└─ README.md
```

Everything routes through `gemini.py`.

---

## 🔧 Requirements

- **Python** 3.12.x or 3.13.x  
  - On Railway, set `PYTHON_VERSION=3.13.5` (if any wheel fails to build, try 3.12.x).
- `requirements.txt` (provided)

---

## 🔑 Environment Variables

Create a `.env` locally (do **not** commit it) or set these in Railway → **Variables**:

- `GENAI_API_KEY` — your Google Gemini API key **(required)**  
- `PYTHON_VERSION` — e.g. `3.13.5` (recommended on Railway)

> **Security:** `gemini.py` reads the key from the environment. There are **no hardcoded keys** in prompts or code.

Example `.env.example`:

```bash
GENAI_API_KEY=your_gemini_api_key_here
PYTHON_VERSION=3.13.5
```

---

## ▶️ Run Locally

```bash
# 1) create & activate venv (Windows PowerShell shown; adjust for macOS/Linux)
python -m venv .venv
. .venv/Scripts/activate    # Windows
# source .venv/bin/activate # macOS/Linux

# 2) install deps
pip install -r requirements.txt

# 3) set env vars (or use a .env file)
set GENAI_API_KEY=your_key_here     # Windows
# export GENAI_API_KEY=your_key_here  # macOS/Linux

# 4) start server
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Health check**:

```
GET http://localhost:8000/   →  { "ok": true, ... }
GET http://localhost:8000/healthz
HEAD /
```

---

## ☁️ Deploy to Railway

### Option A: **Procfile** (simple)

**Procfile**
```
web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Push to GitHub → Create a new Railway project → Deploy from repo.

Set Variables:
- `GENAI_API_KEY`
- `PYTHON_VERSION=3.13.5` (or 3.12.x)

### Option B: **railway.toml** (alternative)

```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

> Use **either** `Procfile` **or** `railway.toml`, not both.

---

## 📡 API

### `POST /api`
Accepts `multipart/form-data`.

- Recommended field: `question.txt=@/path/to/question.txt`
- You may attach additional files: `-F "file1=@/path/to/file.csv"`
- You can also include plain text fields: `-F "note=anything"`

**cURL (Linux/macOS):**
```bash
curl -X POST "http://localhost:8000/api" \
  -H "Accept: application/json" \
  -F "question.txt=@tests/question1.txt" \
  -F "file1=@tests/sample.csv"
```

**cURL (Windows PowerShell):**
```powershell
curl.exe -X POST "http://localhost:8000/api" `
  -H "Accept: application/json" `
  -F "question.txt=@tests\question1.txt" `
  -F "file1=@tests\sample.csv"
```

**Response**
- **Success**: the JSON content of `uploads/<request_id>/result.json`
- **Failure**: `{"message": "...error..."}`; see logs at `uploads/<request_id>/execution_result.txt`

---

## 🗂️ Runtime Files & Logs

Per request (UUID):

```
uploads/<request_id>/
├─ app.log
├─ execution_result.txt        # combined engine logs (install/run output)
├─ metadata.txt                # intermediate info saved by generated code
├─ result.json / result.txt    # final answer
└─ job_<8hex>/
   └─ script.py                # executed code
```

> Railway storage is **ephemeral**. Do not rely on `uploads/` for persistence.

---

## 🧠 How It Works (High Level)

1. `main.py` receives `question.txt` (+ optional files).  
2. `gemini.py::parse_question_with_llm()` prompts Gemini to **return JSON only** containing:
   - `code` (Python), `libraries` (pip names), `run_this` (0/1)
3. `task_engine.py::run_python_code()`
   - Ensures required libraries are available (maps names like `beautifulsoup4 → bs4` for import checks)
   - Writes `script.py` in a per-job folder and executes it (async-safe, with timeouts)
   - Captures stdout/stderr to `execution_result.txt`
4. The code writes `metadata.txt` (intermediate) and `result.json` or `result.txt` (final).  
5. If needed, `main.py` asks Gemini to **validate** the output and optionally **re-generate** code.  
6. The API returns `result.json` (or `result.txt` fallback).

---

## 🧪 Health Checks

Keep tolerant routes for platform probes:

- `GET /`
- `GET /healthz`
- `HEAD /`

---

## 📝 .gitignore (key entries)

```gitignore
# Virtualenvs
.venv/
venv*/
__pycache__/

# Local env & secrets
.env

# Runtime data
uploads/
!uploads/.gitkeep
logs/
*.log

# OS/IDE artifacts
.DS_Store
.vscode/
.idea/
```

---

## 📄 requirements.txt (preinstalled to avoid slow first runs)

```txt
# Web stack
fastapi>=0.115
uvicorn[standard]>=0.30
aiofiles>=24.1
python-multipart>=0.0.9
httpx>=0.27

# LLM + env
google-generativeai>=0.8
python-dotenv>=1.0

# Engine utilities
black>=24.8

# Common data/scraping
requests>=2.32
beautifulsoup4>=4.12
pandas>=2.2
lxml>=5.2

# Plotting (often emitted by LLM code)
numpy>=2.0
matplotlib>=3.9
seaborn>=0.13
```

---

## 🛠️ Troubleshooting

- **`TypeError: can only concatenate str (not "NoneType") to str`**  
  Ensure `question.txt` was received and read properly before concatenating into the prompt. Your current `main.py` reads it from form-data or falls back to the first uploaded file.

- **`FileNotFoundError: .../script.py`**  
  `task_engine.py` writes `script.py` before executing; verify the job folder exists and has write permissions.

- **Library import checks keep reinstalling**  
  The engine maps common pip names to import names (`beautifulsoup4 → bs4`, `scikit-learn → sklearn`, `opencv-python → cv2`, etc.). Ensure your package list uses pip names, not import names.

- **Pip hangs or slow builds**  
  Railway Nixpacks + Python 3.13.5 generally works. If wheels fail to build, try `PYTHON_VERSION=3.12.x`.

- **Using multiple venvs**  
  If you round-robin interpreters, make sure the paths you pick actually exist on Railway. You can also default to `sys.executable`.

---

## ⚠️ Security Notes

- **Never** commit `.env` or API keys.
- Keep keys **out of prompts and logs**.
- This service can fetch URLs supplied via prompts; **validate inputs** if exposing publicly.
- Consider rate-limiting and auth if you deploy the endpoint on a public URL.

---

## License

MIT (or your preferred license).
