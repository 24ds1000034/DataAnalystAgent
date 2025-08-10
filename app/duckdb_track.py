from __future__ import annotations
import os, time, json
from typing import Dict, Any, List
import duckdb, pandas as pd

def _outdir(base: str) -> str:
    d = os.path.join(base, f"duckdb_run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(d, exist_ok=True); return d

SAFE_PREFIXES = ("select",)

def _is_safe_sql(sql: str) -> bool:
    s = (sql or "").strip().lower()
    return any(s.startswith(p) for p in SAFE_PREFIXES) and "read_parquet(" in s

async def run_duckdb_targets(plan: Dict[str, Any], workdir: str, budget_ms: int) -> Dict[str, Any]:
    out_dir = _outdir(workdir)
    targets = (plan or {}).get("targets") or []
    written = 0
    notes: List[str] = []

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL parquet; LOAD parquet;")
    except Exception as e:
        notes.append(f"duckdb: extensions error: {e}")
        with open(os.path.join(out_dir,"notes.json"),"w") as f: json.dump({"notes":notes},f,indent=2)
        return {"array":[]}

    for t in targets:
        if t.get("strategy") != "duckdb_parquet" or written >= 2:  # cap to 2 files
            continue
        sql = t.get("sql") or ""
        if not _is_safe_sql(sql):
            notes.append("duckdb: unsafe or missing SQL; skipped")
            continue
        try:
            df = con.execute(sql).df()
            if isinstance(df, pd.DataFrame) and not df.empty:
                written += 1
                df.to_csv(os.path.join(out_dir, f"table_{written}.csv"), index=False)
        except Exception as e:
            notes.append(f"duckdb: query failed: {e}")

    with open(os.path.join(out_dir,"notes.json"),"w") as f:
        json.dump({"notes":notes,"written":written},f,indent=2)
    return {"array":[]}
