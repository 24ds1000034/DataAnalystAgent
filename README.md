FastAPI LLM Orchestrator (Railway Deploy)

This service accepts a question.txt and optional files/URLs, generates scraping/analysis code via LLM, executes it in a sandboxed job folder, and returns the final result.json/result.txt.

No ngrok or shell scripts required. Deploy directly to Railway or run locally.

🚀 Features

FastAPI API with permissive CORS

Per-request sandbox folder under uploads/<request_id>/job_*

Auto-install of common Python libs requested by the LLM (with safety timeouts)

Logs per request (uploads/<request_id>/execution_result.txt)

Tolerant health check route at /

Works great with cURL

📦 What’s in this repo
.
├─ main.py
├─ task_engine.py
├─ gemini.py
├─ llm_parser.py           # optional; not called by main.py
├─ requirements.txt        # Railway installs from this
├─ Procfile                # or railway.toml (choose one)
├─ .env.example
├─ .gitignore
├─ uploads/
│  └─ .gitkeep             # keep folder in git; runtime data is ephemeral
└─ README.md

🔧 Requirements

Python 3.13.5 (set on Railway via variable PYTHON_VERSION=3.13.5; if any package fails, switch to 3.12.x)

requirements.txt (provided)

🔑 Environment Variables

Create a .env locally (don’t commit it) or set these in Railway → Variables:

GENAI_API_KEY=your_gemini_api_key    # required (used by gemini.py)
AIPIPE_TOKEN=your_openrouter_token   # optional (only for llm_parser.py)
PYTHON_VERSION=3.13.5                # recommended on Railway


Security note: Remove any hardcoded keys from gemini.py system prompts.

A sample .env.example is included.

▶️ Run Locally
# 1) create & activate venv (Windows PowerShell shown; adjust for macOS/Linux)
python -m venv .venv
.venv\Scripts\activate

# 2) install deps
pip install -r requirements.txt

# 3) set env vars (or use a .env file)
set GENAI_API_KEY=your_key_here

# 4) start server
uvicorn main:app --host 0.0.0.0 --port 8000


Health check:

GET http://localhost:8000/   → { "ok": true, ... }

☁️ Deploy to Railway
Option A: Procfile (simple)

Procfile

web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}


Push to GitHub → Create a new Railway project → Deploy from repo.

Set Variables:

GENAI_API_KEY

(optional) AIPIPE_TOKEN

PYTHON_VERSION=3.13.5

Option B: railway.toml (alternative)

If you prefer, use railway.toml instead of a Procfile:

[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3


Make sure you use either Procfile or railway.toml, not both.

📡 API
POST /api

Accepts multipart/form-data.

Recommended field: question.txt=@/path/to/question.txt

You may attach additional files: -F "data=@/path/to/file.csv"

You can also include text fields: -F "note=anything"

cURL example (Linux/macOS):

curl -X POST "http://localhost:8000/api" \
  -H "Accept: application/json" \
  -F "question.txt=@tests/question1.txt" \
  -F "file1=@tests/sample.csv"


cURL example (Windows PowerShell):

curl.exe -X POST "http://localhost:8000/api" `
  -H "Accept: application/json" `
  -F "question.txt=@tests\question1.txt" `
  -F "file1=@tests\sample.csv"


Response

On success: contents of uploads/<request_id>/result.json (JSON)

On failure: {"message": "...error..."} plus logs in uploads/<request_id>/execution_result.txt

🗂️ Runtime Files & Logs

Per request (UUID):

uploads/<request_id>/
├─ app.log
├─ execution_result.txt        # combined engine logs
├─ metadata.txt                # intermediate info saved by generated code
├─ result.json / result.txt    # final answer
└─ job_<8hex>/
   └─ script.py                # executed code


Railway storage is ephemeral. Do not rely on uploads/ for persistence.

🧪 Health Checks

Keep the tolerant root route (JSON 200) so Railway health checks pass:

GET /
GET /healthz  (optional)
HEAD /

📝 .gitignore (key entries)
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

📄 requirements.txt (preinstalled to avoid slow first runs)
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

🛠️ Troubleshooting

FileNotFoundError: .../script.py
Fixed in task_engine.py (we write script.py before executing).

Writes to uploads/<uuid>/... fail
We execute script.py with cwd set to project root so those relative paths resolve.

Repeated beautifulsoup4 installs
Import check maps beautifulsoup4 → bs4, scikit-learn → sklearn, etc.

Pip hangs
We apply a 300s timeout and disable version prompts. If a package consistently fails, pin Python to 3.12.x.

⚠️ Security Notes

Never commit .env or API keys.

Remove any hardcoded keys from prompts/system instructions.

This service can fetch URLs supplied via prompts; validate inputs if exposing publicly.

That’s it—commit, push, set Railway variables, and deploy.