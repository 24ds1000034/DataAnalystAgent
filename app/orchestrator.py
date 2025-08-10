from __future__ import annotations
import asyncio
from typing import Dict, Any, List, Optional

from .contract import build_contract
from .planner import build_plan
from .fetcher import prepare_sources
from .deterministic import try_answer_deterministic
from .llm_direct import answer_with_llm_direct
from .llm_on_data import small_sample_insights
from .gen_code import run_codegen
from .image_fallback import image_base64_track
from .duckdb_track import run_duckdb_targets
from .llm_adjudicator import final_llm_adjudicate
from .utils import validate_and_finalize, now_ms

async def run_request(
    question_text: str,
    uploaded_files: Dict[str, bytes],
    json_attachments: List[Dict[str, Any]],
    schema: Optional[Dict[str, Any]],
    request_id: str,
    workdir: str,
    return_envelope: bool=False,
):
    t0 = now_ms()

    # 1) Decide contract (format + questions)
    contract = await build_contract(question_text, workdir)
    questions: List[str] = contract.get("questions") or []
    if not questions:
        questions = [question_text.strip()] if question_text and question_text.strip() else []
    questions_len = len(questions)

    final_schema = schema or contract.get("output") or {"type":"array"}
    if isinstance(final_schema, dict) and "type" not in final_schema:
        final_schema["type"] = "array"

    # 2) Planner (LLM) for strategies
    plan = await build_plan(question_text, uploaded_files, workdir)

    # 3) Prepare uploaded/json sources
    sources = await asyncio.to_thread(prepare_sources, uploaded_files, json_attachments, workdir, 35_000)

    # 4) Conditional ladder + parallel forks
    budgets = dict(deterministic=45_000, direct=14_000, on_data=18_000, codegen=28_000, image=4_000, duckdb=35_000, adjudicate=8_000)

    # Kick off always-useful forks
    det_task    = asyncio.create_task(try_answer_deterministic(questions, sources, workdir, budgets["deterministic"], context_text=question_text))
    direct_task = asyncio.create_task(answer_with_llm_direct(questions, workdir, budgets["direct"], question_text=question_text))
    ondata_task = asyncio.create_task(small_sample_insights(questions, sources, workdir, budgets["on_data"]))
    code_task   = asyncio.create_task(run_codegen(questions, workdir, budgets["codegen"]))
    image_task  = asyncio.create_task(image_base64_track(questions, workdir, budgets["image"]))

    # If DuckDB is needed, materialize CSVs early
    duckdb_needed = any(t.get("strategy") == "duckdb_parquet" for t in (plan.get("targets") or []))
    if duckdb_needed:
        duck_task = asyncio.create_task(run_duckdb_targets(plan, workdir, budgets["duckdb"]))
        await duck_task
        # re-trigger direct LLM quickly to pick up fresh CSVs
        direct_task = asyncio.create_task(answer_with_llm_direct(questions, workdir, 6_000, question_text=question_text))

    direct_res, ondata_res, det_res, code_res, img_res = await asyncio.gather(direct_task, ondata_task, det_task, code_task, image_task)

    def to_array(res: Dict[str, Any]) -> List[Any]:
        if isinstance(res, dict):
            for k in ("array","data","result"):
                v = res.get(k)
                if isinstance(v, list):
                    return (v + [None]*questions_len)[:questions_len]
        return [None] * questions_len

    track_arrays = {
        "direct": to_array(direct_res),
        "on_data": to_array(ondata_res),
        "deterministic": to_array(det_res),
        "codegen": to_array(code_res),
        "image": to_array(img_res),
    }

    final_array = await final_llm_adjudicate(questions, track_arrays, workdir, budgets["adjudicate"])

    result = validate_and_finalize(
        final_array,
        final_schema,
        elapsed_ms=now_ms() - t0,
        envelope=return_envelope,
        questions_len=questions_len,
        object_keys=contract.get("object_keys"),
    )
    return result
