# Data Analyst Agent – Project Blueprint
*Last updated: 2025-08-12 22:26 UTC*

## 1) Purpose
A **generic, prompt-first, parallel-processing** API agent that can **source, prepare, analyze, and visualize** data from uploaded files, web pages, or S3 parquet—returning **only the requested JSON** within **3 minutes**.

- **No hardcoding** for datasets.
- **Uploaded data is primary**; scrape/duckdb run in **parallel** if links exist.
- **Deterministic-first**; LLMs act as planners/coders/adjudicators.

## 2) API Contract
**Endpoint**: `POST /api/` (multipart form-data)  
**Required**: `questions.txt` (always present)  
**Optional**: attachments (`.csv`, `.xlsx`, `.json`, images, etc.)

**Response**:  
- Default: **JSON array** with 1 element per question.  
- If the question explicitly requests an object schema, return that exact **object** shape.

**Example**
```bash
curl "http://127.0.0.1:8000/api/" \
  -F "questions.txt=@tests/question1.txt" \
  -F "data.csv=@tests/edges.csv"
```

## 3) Output Rules
- Exact **shape** decided by **Contract** stage (array by default).  
- **No envelopes** (no `ok/meta`), return **only** the payload.  
- Plots as `data:image/png;base64,...` **< 100kB**.  
- Type-checked per qtype: `count(int)`, `earliest(str)`, `corr(float -1..1)`, `scatter(png b64)`, `graph_*` (see pipeline).

## 4) Pipeline (high level)
1. **Contract** → decide output shape (array/object), normalized questions, and inferred qtypes.  
2. **Inputs & Catalog** → load uploaded CSV/Excel/JSON; build data catalog.  
3. **Link Planner** → extract HTTP and **S3** links; optionally ask LLM to pick **up to 4** canonical URLs; persist to `link_plan_*/links.txt`.  
4. **Parallel Forks** (no early stop):
   - **Scraper** → fetch HTML, parse tables to CSVs (`scrape_run_*/scrape/`).  
   - **DuckDB** → if `s3://` present, materialize parquet **LIMITed** samples to CSV (`duckdb_run_*/`).  
   - **Deterministic (pass 1)** → compute answers from uploaded + any existing tables.  
   - **LLM-on-snippet (opt)** → tiny CSV snippets for fast numeric answers.  
   - **Codegen (opt)** → safe Python for plots/corr/graph; prints one JSON array.  
5. **Deterministic (pass 2)** → re-run quickly after scraper/duckdb to consume fresh tables.  
6. **Adjudication** → type-validate & merge per qtype (priority: det_p2 > det_p1 > codegen > llm_snippet).  
7. **Emit** → return final payload only.

**Time budget ≤ 175s** overall (global SLA 180s).

## 5) Qtypes (examples)
- **count**: integer ≥ 0 (supports currency thresholds & date filters).  
- **earliest**: string title/name (by year or metric threshold).  
- **corr**: Pearson correlation between two numeric columns.  
- **scatter**: PNG with dotted **red** regression line (<100kB).  
- **graph_***: `edge_count`, `highest_degree`, `average_degree`, `density`, `shortest_path`, `plot`, `degree_histogram` (requires edges CSV).

## 6) Intermediates (always written)
- `link_plan_*/links.txt` and `link_plan_*/link_plan.json`  
- `scrape_run_*/scrape/table_*.csv|.md`  
- `duckdb_run_*/table_*.csv`  
- `deterministic_run_*/answers.json`  
- `adjudicator_run_*/candidates.json`, `final.json`

## 7) Env & Secrets
- `OPENAI_API_KEY` (optional; enables LLM-assisted link selection & codegen/snippet tracks)  
- `OPENAI_MODEL` (default `gpt-4o-mini`)  
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (optional for protected S3)  
- `AWS_S3_ENDPOINT` (optional for S3-compatible stores)

## 8) Dependencies
`fastapi uvicorn httpx beautifulsoup4 pandas numpy matplotlib pillow duckdb networkx`

## 9) Runbook
```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

## 10) Design Tenets
- **Generic over bespoke** (no dataset-specific code).  
- **Prompt-first** (decisions in prompts; code is thin, safe).  
- **Parallel all the time** (no early stop).  
- **Deterministic-first**, LLMs as planners/coders/adjudicators.  
- **Reproducible** (persist intermediates).

---
