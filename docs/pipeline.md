# Data Analyst Agent – Parallel Pipeline
*Last updated: 2025-08-12 22:26 UTC*

## Stage 0 — Init
- Create `workdir/run_<ts>/` and subfolders: `contract/`, `link_plan/`, `scrape/`, `duckdb/`, `snippets/`, `candidates/`, `final/`.
- SLA guard: **hard cap 175s**.

## Stage 1 — Contract (format + qtypes)
- Prompt: `prompts/format_decider.txt`
- Output JSON envelope (illustrative):
```
{
  "output": {"type": "array", "keys": []},
  "questions": ["Q1", "Q2", "..."],
  "qtypes": ["count", "scatter", "..."]
}
```

## Stage 2 — Inputs & Catalog
- Load uploaded CSV/Excel/JSON → profile columns (numeric/text hints).  
- Build `inputs/catalog.json`.

## Stage 3 — Link Planner
- Extract HTTP + **S3** from `questions.txt` (regex).  
- If `OPENAI_API_KEY`, ask LLM to pick **up to 4** canonical links.  
- Persist **authoritative** list to `link_plan_*/links.txt`.

## Stage 4 — Forks (all parallel)
- **Scraper** (HTTP): concurrent fetch → HTML → tables → CSV (`scrape_run_*/scrape/`).  
- **DuckDB** (S3): `read_parquet('s3://...') LIMIT 50k` → CSV (`duckdb_run_*/`).  
- **Deterministic Pass 1**: compute from uploaded + any existing tables.  
- **LLM-on-snippet (opt)**: tiny CSV + prompt → quick numeric answers.  
- **Codegen (opt)**: safe Python for corr/plots/graph; prints one JSON array.

## Stage 5 — Deterministic Pass 2
- If scraper/duckdb produced tables OR p1 had nulls → run fast pass-2.

## Stage 6 — Adjudication
- Priority per answer position: **det_p2 > det_p1 > codegen > llm_snippet**.  
- **Type validation** by qtype (reject invalid types).  
- Output conforms exactly to contract shape (array | object).

## Budgets (example, ms)
- Contract: 7000  
- Inputs: 8000  
- Link planning: 10000  
- Scraper: 50000  
- DuckDB: 35000  
- Deterministic P1: 55000  
- LLM Snippet: 25000  
- Codegen: 45000  
- Deterministic P2: 25000  
- Adjudication: 20000  
- **Total ≤ 175000 ms** (overlapping).

## Plot Constraints
- PNG base64 data URI `< 100kB`  
- Visible axes; **dotted red** regression line for scatter.  

## Graph Suite
- Inputs: edges CSV with two columns (`source/target` auto-detected; fallback first two columns).  
- Outputs: `edge_count`, `highest_degree`, `average_degree`, `density`, `shortest_path`, `plot`, `degree_histogram`.

---
