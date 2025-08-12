# app/duckdb_track.py
from __future__ import annotations
"""
Safe DuckDB materializer:
- Detects an explicit ```sql ...``` fence containing a SELECT with read_parquet('s3://...')
  OR any s3:// path in the question text.
- Executes the SELECT safely with httpfs+parquet loaded, adds LIMIT if missing.
- Writes 1-2 small CSVs for downstream tracks and returns their paths.
"""

import os
import re
import time
import json
from typing import List, Dict, Any, Optional

SQL_FENCE = re.compile(r"```sql\s+([\s\S]*?)```", re.I)
HAS_SELECT = re.compile(r"^\s*select\b", re.I)
HAS_READ_PARQUET = re.compile(r"\bread_parquet\s*\(", re.I)
S3_URL = re.compile(r"s3://[^\s'\"`]+", re.I)
S3_REGION_KV = re.compile(r"s3_region=([a-z0-9-]+)", re.I)

def _outdir(workdir: str) -> str:
    d = os.path.join(workdir, f"duckdb_run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(d, exist_ok=True)
    return d

def _safe_sql_from_question(question_text: str) -> Optional[str]:
    # Prefer an explicit SQL fence if it looks like a SELECT + read_parquet
    for m in SQL_FENCE.finditer(question_text or ""):
        sql = m.group(1).strip().rstrip(";")
        if HAS_SELECT.search(sql) and HAS_READ_PARQUET.search(sql):
            return sql
    # Fallback: synthesize a SELECT * if we spot an s3:// parquet path
    m = S3_URL.search(question_text or "")
    if not m:
        return None
    url = m.group(0)
    # Very conservative default
    return f"SELECT * FROM read_parquet('{url}') LIMIT 20000"

def _extract_region(question_text: str) -> Optional[str]:
    m = S3_REGION_KV.search(question_text or "")
    return m.group(1) if m else None

async def materialize_duckdb_samples(question_text: str, workdir: str, budget_ms: int) -> Dict[str, Any]:
    out_dir = _outdir(workdir)
    notes: List[str] = []
    sql = _safe_sql_from_question(question_text)
    if not sql:
        with open(os.path.join(out_dir, "notes.json"), "w", encoding="utf-8") as f:
            json.dump({"notes": ["no duckdb sql or s3 path detected"]}, f, indent=2)
        return {"csv_paths": [], "notes": ["no duckdb sql or s3 path detected"]}

    # Ensure a LIMIT for safety
    if not re.search(r"\blimit\b", sql, re.I):
        sql = sql.rstrip(";") + " LIMIT 20000"

    region = _extract_region(question_text)
    try:
        import duckdb
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL parquet; LOAD parquet;")
        # Optionally set region if present
        if region:
            con.execute(f"SET s3_region='{region}';")
        # Run query
        res = con.execute(sql).df()
        if res is None or res.empty:
            with open(os.path.join(out_dir, "notes.json"), "w", encoding="utf-8") as f:
                json.dump({"notes": ["duckdb query returned no rows"]}, f, indent=2)
            return {"csv_paths": [], "notes": ["duckdb query returned no rows"]}
        # Save one small CSV
        csv1 = os.path.join(out_dir, "table_1.csv")
        res.to_csv(csv1, index=False)
        with open(os.path.join(out_dir, "sql_used.sql"), "w", encoding="utf-8") as f:
            f.write(sql + ";\n")
        with open(os.path.join(out_dir, "notes.json"), "w", encoding="utf-8") as f:
            json.dump({"notes": notes}, f, indent=2)
        return {"csv_paths": [csv1], "notes": notes}
    except Exception as e:
        with open(os.path.join(out_dir, "notes.json"), "w", encoding="utf-8") as f:
            json.dump({"notes": [f"duckdb error: {e}"]}, f, indent=2)
        return {"csv_paths": [], "notes": [f"duckdb error: {e}"]}
