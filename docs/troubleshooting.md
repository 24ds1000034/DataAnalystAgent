# Troubleshooting & Tips
*Last updated: 2025-08-12 22:26 UTC*

## Windows build errors (pandas/meson/vswhere)
- Prefer official wheels: `pip install --upgrade pip` then `pip install pandas` (or pin to a wheel version).  
- Our requirements avoid `lxml`—`beautifulsoup4` with `html.parser` is used.

## PowerShell curl
- Use `curl.exe` for multipart:  
  `curl.exe "http://127.0.0.1:8000/api/" -F "questions.txt=@tests/question1.txt"`

## Null answers
- Check `link_plan_*/links.txt` (were links planned?)  
- Check `scrape_run_*/scrape/*.csv` and `duckdb_run_*/table_*.csv`  
- Ensure **Deterministic Pass 2** ran (look for `deterministic_run_*/answers.json`).

## PNG too large
- Deterministic compresses to `<100kB`. If still large, reduce point size or DPI.

## S3 access
- Public buckets: only region is required (`s3_region=...` hint in question).
- Private: set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`.
- Non-AWS S3: set `AWS_S3_ENDPOINT`.

## Exact output shape
- Contract decides shape; orchestrator enforces **no extra elements/keys**.
- If you see envelopes, ensure `api.py` returns `final_payload` directly.

---
