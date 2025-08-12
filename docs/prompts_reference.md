# Prompts Reference
*Last updated: 2025-08-12 22:26 UTC*

## format_decider.txt
- Decide: output shape (array/object), normalized questions, and qtypes.
- Default to array if not specified; no commentary; strict JSON.

## link_selector.txt
- Given question + candidate URLs (+ uploaded preview), pick up to 4 authoritative links.
- Prefer canonical sources; may add links not present in candidates.
- Return strict JSON: `{"links": [], "notes": ""}`.

## schema_map_prompt.txt (optional)
- Map semantic names (`rank, peak, year, title, gross, source, target`) → actual column headers.
- Strict JSON: `{"column_map": {"rank": "Rank", ...}}` (null if not found).

## direct_on_snippet_prompt.txt (optional)
- Input: tiny CSV snippet (≤500 rows, selected columns).  
- Answer strictly **from the snippet**; return JSON array; no plots.

## codegen_prompt.txt (optional)
- Generate **safe Python** using whitelisted libs: pandas, numpy, matplotlib, pillow, networkx.  
- No network or file deletion; print exactly **one JSON array**; plots as data URI <100kB.
- Handles qtypes: `count, earliest, corr, scatter, graph_*`.

## finalize_prompt.txt (optional)
- As a tie-breaker if numeric candidates disagree; must return strict JSON array of same length, or pass-through.

---
