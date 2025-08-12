# app/deterministic.py
from __future__ import annotations
"""
Deterministic track (generic + minimal):
- Prefer uploaded CSV/Excel/JSON and DuckDB outputs; scrape the web only if no dataframes exist.
- Robust header cleaning & fuzzy column matching.
- Supports: count (with threshold & optional year filters), earliest (threshold),
  corr (Pearson), scatter (base64 PNG <100kB) reusing the same columns as corr.
- Writes clean intermediates under deterministic_run_*/ (questions.txt, sources.json,
  scraped_data/table_*.csv/.md, answers.json, metadata.json).
"""

import io
import os
import re
import math
import json
import time
import glob
import base64
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from PIL import Image

import requests
from bs4 import BeautifulSoup

from .loaders import try_load_known


# ---------- small helpers ----------

def _b64_png(b: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")


def _opt_png(b: bytes, max_bytes: int = 100_000) -> bytes:
    """Palette-compress and, if needed, downscale to fit under max_bytes."""
    im = Image.open(io.BytesIO(b)).convert("P", palette=Image.ADAPTIVE, colors=128)
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    data = out.getvalue()
    # iterative shrink if still too big
    while len(data) > max_bytes and min(im.size) > 320:
        im = im.resize((int(im.size[0] * 0.9), int(im.size[1] * 0.9)))
        out = io.BytesIO()
        im.save(out, format="PNG", optimize=True)
        data = out.getvalue()
    return data


def scatter_with_regression_b64(x, y, xlabel: str, ylabel: str, max_bytes: int = 100_000) -> Optional[str]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    if x.size < 2 or y.size < 2:
        return None
    # cap to 5000 points for file size
    if x.size > 5000:
        idx = np.linspace(0, x.size - 1, 5000).astype(int)
        x, y = x[idx], y[idx]
    slope, inter = np.polyfit(x, y, 1)
    fig = plt.figure(figsize=(6.4, 4.8), dpi=115)
    ax = plt.gca()
    ax.scatter(x, y, s=8, alpha=0.7)
    ax.plot([x.min(), x.max()],
            [slope * x.min() + inter, slope * x.max() + inter],
            linestyle="--", color="red")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return _b64_png(_opt_png(buf.getvalue(), max_bytes))


_URL_RE = re.compile(r"https?://[^\s)]+", re.I)
def extract_urls(text: str) -> List[str]:
    return _URL_RE.findall(text or "")


def _flatten_cols(cols) -> List[str]:
    if isinstance(cols, pd.MultiIndex):
        return [" ".join([str(x) for x in tup if str(x) not in ("", "nan")]).strip()
                for tup in cols]
    return [str(c) for c in cols]


_FOOTNOTE_RX = re.compile(r"\[[^\]]*\]")
def _clean_header_names(names: List[str]) -> List[str]:
    out = []
    for n in names:
        s = _FOOTNOTE_RX.sub("", str(n))
        s = re.sub(r"\s+", " ", s).strip()
        out.append(s)
    return out


def _read_html_tables_lxml(html: str) -> List[pd.DataFrame]:
    try:
        tables = pd.read_html(html, flavor="lxml")
    except Exception:
        return []
    dfs: List[pd.DataFrame] = []
    for df in tables:
        if df is None or df.empty:
            continue
        df = df.copy()
        df.columns = _clean_header_names(_flatten_cols(df.columns))
        # drop empty columns
        df = df.loc[:, ~(df.astype(str).apply(lambda s: s.str.strip()).eq("").all())]
        # drop duplicate header rows
        lower_cols = [c.lower() for c in df.columns]
        row_header = pd.Series(lower_cols, index=df.columns)
        norm = df.astype(str).apply(lambda s: s.str.strip().str.lower())
        mask_dup = norm.eq(row_header).all(axis=1)
        if mask_dup.any():
            df = df.loc[~mask_dup]
        dfs.append(df)
    return dfs


def _text(el) -> str:
    return el.get_text(strip=True) if hasattr(el, "get_text") else str(el).strip()


def read_all_html_tables_bs(html: str) -> List[pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    dfs: List[pd.DataFrame] = []
    for tbl in soup.find_all("table"):
        headers: List[str] = []
        thead = tbl.find("thead")
        if thead:
            headers = [_text(th) for th in thead.find_all("th") if _text(th)]
        if not headers:
            tr0 = tbl.find("tr")
            if tr0:
                headers = [_text(th) for th in tr0.find_all(["th", "td"]) if _text(th)]
        rows: List[List[str]] = []
        trs = tbl.find_all("tr")
        start = 1 if headers and trs else 0
        for i, tr in enumerate(trs):
            if i < start:
                continue
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue
            row = [_text(td) for td in cells]
            if any(row):
                rows.append(row)
        if rows:
            w = max(len(r) for r in rows)
            rows = [r + [""] * (w - len(r)) for r in rows]
            if headers and len(headers) == w:
                df = pd.DataFrame(rows, columns=_clean_header_names([str(h) for h in headers]))
            else:
                df = pd.DataFrame(rows)
                if headers and len(headers) <= df.shape[1]:
                    for i, h in enumerate(_clean_header_names(headers)):
                        if h:
                            df.rename(columns={i: h}, inplace=True)
            df.columns = [str(c) for c in df.columns]
            dfs.append(df)
    return dfs


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())


def _grams(s: str) -> set:
    s2 = _norm(s)
    return set(s2[i:i + 3] for i in range(max(1, len(s2) - 2)))


def _sim(a: str, b: str) -> float:
    A, B = _grams(a), _grams(b)
    return (len(A & B) / len(A | B)) if (A and B) else 0.0


_ARTICLE_RX = re.compile(r"^\s*(the|a|an)\s+", re.I)
def _strip_articles(s: str) -> str:
    return _ARTICLE_RX.sub("", s or "").strip()


def find_best_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Exact case-insensitive match, else fuzzy ≥ 0.30."""
    name = _strip_articles(str(name or ""))
    cols = [str(c) for c in df.columns]
    for c in cols:
        if c.strip().lower() == name.strip().lower():
            return c
    if not cols:
        return None
    best = max(cols, key=lambda c: _sim(c, name))
    return best if _sim(best, name) >= 0.30 else None


def parse_col_pair(q: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"(?:between|of)\s+([A-Za-z0-9 _\-./]+?)\s+(?:and|&)\s+([A-Za-z0-9 _\-./]+)", q, re.I)
    if not m:
        return None
    return (_strip_articles(m.group(1).strip()), _strip_articles(m.group(2).strip()))


def parse_threshold(q: str) -> Optional[float]:
    m = re.search(r"\$?\s*([0-9][0-9,\.]*)\s*(b|bn|m|million|k|thousand)?", q, re.I)
    if not m:
        return None
    x = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit in ("b", "bn"):
        x *= 1_000_000_000
    elif unit in ("m", "million"):
        x *= 1_000_000
    elif unit in ("k", "thousand"):
        x *= 1_000
    return x


def parse_year(q: str, which: str = "before") -> Optional[int]:
    m = re.search(
        r"(?:before|prior to)\s*(\d{4})" if which == "before" else r"(?:after|from)\s*(\d{4})",
        q, re.I
    )
    return int(m.group(1)) if m else None


def wants_correlation(q: str) -> bool:
    return bool(re.search(r"\bcorrelat", q, re.I))


def wants_scatter(q: str) -> bool:
    return bool(re.search(r"\bscatter\s*plot|\bscatterplot", q, re.I))


def wants_count(q: str) -> bool:
    return bool(re.search(r"\bcount|how many\b", q, re.I))


def wants_earliest(q: str) -> bool:
    return bool(re.search(r"\bearliest|first\b", q, re.I))


_CRX = re.compile(r"[^\d\.\-]+")
def to_num(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")
    v = s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    v = v.str.replace(_CRX, "", regex=True)
    return pd.to_numeric(v, errors="coerce")


TEXT_COL_HINTS = ("title", "film", "movie", "name")
def normalize_df_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    for c in out.columns:
        cn = c.strip().lower()
        if any(h in cn for h in TEXT_COL_HINTS):
            out[c] = out[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return out


GENERIC_TERMS = [
    "rank", "peak", "year", "date", "title", "name", "gross", "worldwide gross",
    "revenue", "amount", "total", "value", "price", "score"
]
def _terms(questions: List[str]) -> List[str]:
    t: List[str] = []
    seen: set = set()
    for q in questions:
        q = str(q or "")
        p = parse_col_pair(q)
        if p:
            t.extend([p[0], p[1]])
        if wants_count(q) or wants_earliest(q):
            t.extend(["year", "date", "title", "name", "gross", "worldwide gross",
                      "revenue", "amount", "total", "value", "price"])
        if wants_correlation(q) or wants_scatter(q):
            t.extend(["rank", "peak", "score", "value"])
    t.extend(GENERIC_TERMS)
    out: List[str] = []
    for w in t:
        lw = w.lower()
        if lw not in seen:
            out.append(w)
            seen.add(lw)
    return out


def _score(df: pd.DataFrame, terms: List[str]) -> float:
    cols = [str(c) for c in df.columns]
    if not cols or not terms:
        return 0.0
    sim = sum(max((_sim(c, t) for c in cols), default=0.0) for t in terms) / max(1, len(terms))
    r, c = df.shape
    bonus = min(0.3, math.log10(max(2, r)) / 20.0) + min(0.2, math.log10(max(2, c)) / 20.0)
    # Penalize numeric-only headers (often junk tables)
    num_head = sum(1 for c in cols if re.fullmatch(r"\d+", c.strip()))
    if num_head >= max(3, int(0.5 * len(cols))):
        bonus -= 0.25
    return sim + bonus


def _best_overall(dfs: List[pd.DataFrame], questions: List[str]) -> int:
    terms = _terms(questions)
    best_i, best_s = 0, -1.0
    for i, df in enumerate(dfs):
        s = _score(df, terms)
        if s > best_s:
            best_s, best_i = s, i
    return best_i


def _best_for(q: str, dfs: List[pd.DataFrame]) -> List[int]:
    ids = sorted(range(len(dfs)), key=lambda i: _score(dfs[i], _terms([q])), reverse=True)
    return ids[:3]


def _pick_metric(df: pd.DataFrame) -> Optional[str]:
    """Choose a likely currency/amount column, avoid rank-like columns."""
    best = None
    best_sc = -1.0
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        nonna = s.dropna()
        if nonna.empty:
            continue
        name = str(c).lower()
        name_bonus = 0.8 if any(k in name for k in
                                ["gross", "worldwide gross", "revenue", "amount", "total", "value", "price"]) else 0.0
        if any(k in name for k in ["rank", "peak", "position", "no."]):
            name_bonus -= 0.6
        mag = float(np.nanpercentile(np.abs(nonna.values), 90))
        mag_bonus = 0.5 if mag >= 1e6 else 0.0
        sc = name_bonus + mag_bonus
        if sc > best_sc:
            best, best_sc = c, sc
    return best


def _pair_key(a: str, b: str) -> str:
    a2, b2 = _strip_articles(a).lower().strip(), _strip_articles(b).lower().strip()
    return a2 + "|" + b2


# ---------- main entry ----------

async def try_answer_deterministic(
    questions: List[str],
    sources: Dict[str, Any],
    workdir: str,
    budget_ms: int,
    context_text: str = "",
) -> Dict[str, Any]:
    """
    Returns {"array":[...]} when at least one deterministic answer/plot is produced;
    otherwise {} so other tracks can take over.
    """
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(workdir, f"deterministic_run_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    used_dir = os.path.join(out_dir, "scraped_data")
    os.makedirs(used_dir, exist_ok=True)

    qs = [str(q).strip() for q in questions if str(q).strip()]
    with open(os.path.join(out_dir, "questions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(qs))

    # summarize sources (without storing raw bytes)
    red = {"files": {}, "notes": sources.get("notes", [])}
    for name, meta in (sources.get("files") or {}).items():
        rm = {k: v for k, v in meta.items() if k != "bytes"}
        rm["bytes"] = len(meta.get("bytes", b"") or b"")
        red["files"][name] = rm
    with open(os.path.join(out_dir, "sources.json"), "w", encoding="utf-8") as f:
        json.dump({"files": red, "notes": sources.get("notes", [])}, f, indent=2)

    # ---- gather dataframes ----
    dfs: List[pd.DataFrame] = []

    # 1) Prefer uploaded files
    file_dfs = try_load_known(sources.get("files", {}))
    for df in file_dfs.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            dfs.append(df)

    # 2) Also ingest DuckDB materialized CSVs, if any
    for p in sorted(glob.glob(os.path.join(workdir, "duckdb_run_*", "table_*.csv"))):
        try:
            df = pd.read_csv(p)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass

    # 3) Scrape only if nothing was found
    if not dfs:
        urls: List[str] = []
        for q in qs:
            urls += extract_urls(q)
        urls += extract_urls(context_text or "")
        seen = set()
        urls = [u for u in urls if not (u in seen or seen.add(u))]
        for u in urls:
            try:
                r = requests.get(u, timeout=20, headers={"User-Agent": "DataAnalystAgent/0.6"})
                r.raise_for_status()
                page_html = r.text
                page_dfs = _read_html_tables_lxml(page_html) or read_all_html_tables_bs(page_html)
                for df in page_dfs:
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        dfs.append(df)
            except Exception:
                continue

    if not dfs:
        with open(os.path.join(out_dir, "answers.json"), "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"run_id": run_id, "num_questions": len(qs), "num_dataframes": 0, "produced_any": False},
                f, indent=2
            )
        return {}

    # normalize & clean for storage/ops
    dfs = [normalize_df_for_storage(df) for df in dfs]

    answers: List[Any] = []
    produced = False
    saved_indices: set[int] = set()
    pair_cache: Dict[str, Tuple[int, str, str]] = {}  # cache (df_idx, colA, colB) for corr→scatter reuse

    for q in qs:
        ans = None
        cand_ids = _best_for(q, dfs)

        # correlation
        if wants_correlation(q):
            pair = parse_col_pair(q)
            if pair:
                key = _pair_key(*pair)
                # reuse if known
                if key in pair_cache:
                    i, a, b = pair_cache[key]
                    df = dfs[i]
                    x = to_num(df[a]); y = to_num(df[b])
                    if x.notna().sum() >= 2 and y.notna().sum() >= 2:
                        ans = round(float(x.corr(y)), 6)
                # else search
                if ans is None:
                    search_ids = cand_ids + [k for k in range(len(dfs)) if k not in cand_ids]
                    for i in search_ids:
                        df = dfs[i]
                        a = find_best_column(df, pair[0]); b = find_best_column(df, pair[1])
                        if a and b:
                            x = to_num(df[a]); y = to_num(df[b])
                            if x.notna().sum() >= 2 and y.notna().sum() >= 2:
                                ans = round(float(x.corr(y)), 6)
                                pair_cache[key] = (i, a, b)
                                saved_indices.add(i)
                                break
            answers.append(ans); produced |= (ans is not None); continue

        # scatterplot (base64)
        if wants_scatter(q):
            pair = parse_col_pair(q)
            if pair:
                key = _pair_key(*pair)
                done = False
                if key in pair_cache:
                    i, a, b = pair_cache[key]
                    df = dfs[i]
                    x = to_num(df[a]); y = to_num(df[b]); m = x.notna() & y.notna()
                    if m.sum() >= 2:
                        ans = scatter_with_regression_b64(x[m].values, y[m].values, a, b, 100_000)
                        saved_indices.add(i); done = True
                if not done:
                    search_ids = cand_ids + [k for k in range(len(dfs)) if k not in cand_ids]
                    for i in search_ids:
                        df = dfs[i]
                        a = find_best_column(df, pair[0]); b = find_best_column(df, pair[1])
                        if a and b:
                            x = to_num(df[a]); y = to_num(df[b]); m = x.notna() & y.notna()
                            if m.sum() >= 2:
                                ans = scatter_with_regression_b64(x[m].values, y[m].values, a, b, 100_000)
                                pair_cache[key] = (i, a, b)
                                saved_indices.add(i)
                                break
            answers.append(ans); produced |= (ans is not None); continue

        # count with threshold (+ optional before/after year)
        if wants_count(q):
            thr = parse_threshold(q)
            before = parse_year(q, "before")
            after = parse_year(q, "after")
            if thr is not None:
                best = None
                best_i = None
                search_ids = cand_ids + [k for k in range(len(dfs)) if k not in cand_ids]
                for i in search_ids:
                    df = dfs[i]
                    metric = _pick_metric(df)
                    if not metric:
                        continue
                    s = to_num(df[metric])
                    mask = s >= thr
                    ycol = find_best_column(df, "year") or find_best_column(df, "date")
                    if ycol and ycol in df.columns:
                        y = to_num(df[ycol])
                        if before is not None:
                            mask = mask & (y < before)
                        if after is not None:
                            mask = mask & (y >= after)
                    cnt = int(mask.sum())
                    if best is None or cnt > best:
                        best, best_i = cnt, i
                if best is not None:
                    ans = best
                if best_i is not None:
                    saved_indices.add(best_i)
            answers.append(ans); produced |= (ans is not None); continue

        # earliest title/name reaching threshold
        if wants_earliest(q):
            thr = parse_threshold(q)
            if thr is not None:
                best_title = None
                best_year = 10**9
                best_i = None
                search_ids = cand_ids + [k for k in range(len(dfs)) if k not in cand_ids]
                for i in search_ids:
                    df = dfs[i]
                    metric = _pick_metric(df)
                    if not metric:
                        continue
                    title = (find_best_column(df, "title") or find_best_column(df, "name")
                             or find_best_column(df, "film") or find_best_column(df, "movie"))
                    if not title:
                        continue
                    ycol = find_best_column(df, "year") or find_best_column(df, "date")
                    s = to_num(df[metric])
                    mask = s >= thr
                    if ycol and ycol in df.columns:
                        y = to_num(df[ycol])
                        tmp = df[mask].copy()
                        tmp["_y"] = y[mask]
                        tmp = tmp.dropna(subset=["_y"])
                        if tmp.empty:
                            continue
                        row = tmp.sort_values("_y").iloc[0]
                        cand_title = str(row[title]).strip()
                        cand_year = int(row["_y"])
                        if cand_title and cand_year < best_year:
                            best_year = cand_year
                            best_title = cand_title
                            best_i = i
                    else:
                        tmp = df[mask]
                        if tmp.empty:
                            continue
                        cand_title = str(tmp.iloc[0][title]).strip()
                        if cand_title:
                            best_title = cand_title
                            best_i = i
                            break
                ans = best_title
                if best_i is not None:
                    saved_indices.add(best_i)
            answers.append(ans); produced |= (ans is not None); continue

        # unknown question → let other tracks try
        answers.append(None)

    # persist sampled tables for transparency
    if saved_indices:
        for n, i in enumerate(sorted(saved_indices), start=1):
            df = dfs[i]
            df.to_csv(os.path.join(used_dir, f"table_{n}.csv"), index=False)
            _write_md(df, os.path.join(used_dir, f"table_{n}.md"))
    else:
        j = _best_overall(dfs, qs)
        df = dfs[j]
        df.to_csv(os.path.join(used_dir, "table_1.csv"), index=False)
        _write_md(df, os.path.join(used_dir, "table_1.md"))

    res = {"array": answers} if produced else {}
    with open(os.path.join(out_dir, "answers.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "timestamp": time.time(),
                "num_questions": len(qs),
                "produced_any": produced,
                "saved_tables": max(1, len(saved_indices)) if dfs else 0,
            },
            f, indent=2
        )
    return res


def _write_md(df: pd.DataFrame, path: str) -> None:
    cols = list(map(str, df.columns))
    with open(path, "w", encoding="utf-8") as mf:
        mf.write("| " + " | ".join(cols) + " |\n")
        mf.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for _, row in df.head(2000).iterrows():
            vals = [str(x) if x is not None else "" for x in row.tolist()]
            mf.write("| " + " | ".join(vals) + " |\n")
