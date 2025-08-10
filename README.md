
# Data Analyst Agent (FastAPI)

**Endpoint**: `POST /api/`  
**Default output**: JSON array (unless a schema is explicitly specified).  
**SLA**: ≤ 3 minutes.

## Quickstart (VS Code/local)

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8000
```

### Test (multipart)
```bash
curl "http://localhost:8000/api/" -F "questions.txt=@tests/sample questions.txt"
```

### Strict evaluator mode (return raw structure only)
`/api/?strict=1`


#### Folder Structure ####

DataAnalystAgent/                                   — project root
├─ app/                                             — backend service + analysis pipelines
│  ├─ api.py                                       — FastAPI app; /healthz and /api/  endpoints
│  ├─ orchestrator.py                              — runs parallel tracks + final adjudication
│  ├─ contract.py                                  — decides output format + extracts questions/qtypes
│  ├─ planner.py                                   — LLM planner: uploaded_file/html_tables/duckdb_parquet
│  ├─ duckdb_track.py                              — safe DuckDB runner; materializes S3 parquet → CSV
│  ├─ fetcher.py                                   — normalizes incoming files/json; light metadata
│  ├─ loaders.py                                   — robust CSV/Excel/JSON/ZIP/image/audio/video loaders
│  ├─ deterministic.py                             — HTML table scrape + numeric ops (count/corr/plot)
│  ├─ llm_direct.py                                — quick answers from small CSV snippets (no plotting)
│  ├─ llm_on_data.py                               — LLM on sampled data (fallback insights)
│  ├─ gen_code.py                                  — LLM codegen (pandas/polars/duckdb/networkx) executor
│  ├─ image_fallback.py                            — picks latest plot PNG; returns base64 data URI
│  ├─ llm_adjudicator.py                           — merges candidates; enforces types; picks final array
│  └─ utils.py                                     — path helpers, timers, schema finalizer, request dirs
│
├─ prompts/                                        — prompt templates (brains live here)
│  ├─ planner_prompt.txt                           — choose strategy + (safe) SQL if S3/DuckDB present
│  ├─ codegen_prompt.txt                           — generate executable analysis code + base64 plots
│  ├─ on_data_prompt.txt                           — answer from a sampled dataset only
│  ├─ finalize_prompt.txt                          — (optional) adjudication guidance if you externalize
│  ├─ schema_map_prompt.txt                        — (optional) map semantic→actual columns in tables
│  ├─ units_map_prompt.txt                         — (optional) normalize billion/million/k to numeric
│  └─ selector_prompt.txt                          — (optional) pick best table/file among candidates
│
├─ tests/                                          — sample inputs for local runs
│  ├─ question1.txt                                — wiki-like table task (corr + plot)
│  ├─ question2.txt                                — S3/DuckDB parquet task with example SQL
│  ├─ question3.txt                                — graph metrics/plots task (edges.csv expected)
│  ├─ edges.csv                                    — toy undirected edge list for graph tasks
│  └─ sample questions.txt                         — mixed examples; shape/format checks
│
├─ output/                                         — per-request artifacts (auto-created)
│  ├─ .gitkeep                                     — keep folder in git
│  └─ <request-id>/                                — run folder (timestamped subfolders inside)
│      ├─ contract_YYYYmmdd_HHMMSS/                — decided format/questions (contract.json)
│      ├─ planner_YYYYmmdd_HHMMSS/                 — plan.json (targets/strategies)
│      ├─ duckdb_run_YYYYmmdd_HHMMSS/              — table_*.csv + notes from DuckDB
│      ├─ deterministic_run_YYYYmmdd_HHMMSS/       — scraped_data/*.csv|md + answers.json
│      ├─ direct_llm_run_YYYYmmdd_HHMMSS/          — direct_answers.json (+ notes)
│      ├─ on_data_run_YYYYmmdd_HHMMSS/             — sampled data + answers.json
│      ├─ codegen_run_YYYYmmdd_HHMMSS/             — code.py, stdout.txt, plots, answers.json
│      ├─ image_fallback_run_YYYYmmdd_HHMMSS/      — image_fallback.json (selected plot)
│      └─ adjudicator_run_YYYYmmdd_HHMMSS/         — candidates.json + final.json
│
├─ .env.example                                    — put OPENAI_API_KEY/GOOGLE_API_KEY here (copy → .env)
├─ requirements.txt                                — pinned deps (fastapi, uvicorn, pandas, duckdb, etc.)
├─ README.md                                       — how to run, curl examples, design notes
├─ LICENSE                                         — MIT license
└─ .gitignore                                      — ignore venv, caches, output/, local artifacts
