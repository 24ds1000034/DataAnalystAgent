from __future__ import annotations
import uuid
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile

from .orchestrator import run_request
from .utils import ensure_request_dir

app = FastAPI(title="Data Analyst Agent", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def healthz():
    return {"ok": True, "version": app.version}

@app.post("/api/")
async def api_endpoint(request: Request):
    """
    Default behavior: decide output schema from the question (contract-first).
    Optional query ?wrap=1 to return an envelope {ok,data,meta} for debugging.
    """
    content_type = (request.headers.get("content-type") or "")
    rid = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    wrap = request.query_params.get("wrap", "0") == "1" or request.query_params.get("strict", "0") == "1"

    workdir = ensure_request_dir(rid)

    question_text: Optional[str] = None
    files: Dict[str, bytes] = {}
    json_attachments: List[Dict[str, Any]] = []
    response_schema: Optional[Dict[str, Any]] = None

    if "multipart/form-data" in content_type.lower():
        form = await request.form()
        qf: Optional[StarletteUploadFile] = form.get("questions.txt")  # type: ignore
        if not isinstance(qf, StarletteUploadFile):
            raise HTTPException(status_code=422, detail="questions.txt is required as a file field")
        question_text = (await qf.read()).decode("utf-8", errors="replace")
        for key, val in form.multi_items():
            if key == "questions.txt":
                continue
            if isinstance(val, StarletteUploadFile):
                data = await val.read()
                files[val.filename or key] = data

    elif "application/json" in content_type.lower():
        body = await request.json()
        question_text = body.get("question_text")
        if not question_text:
            raise HTTPException(status_code=422, detail="question_text is required in JSON body")
        json_attachments = body.get("attachments", []) or []
        response_schema = body.get("response_schema")
    else:
        raise HTTPException(status_code=415, detail="Unsupported Content-Type")

    try:
        result = await run_request(
            question_text=question_text,
            uploaded_files=files,
            json_attachments=json_attachments,
            schema=response_schema,          # may be None → decided by contract
            request_id=rid,
            workdir=workdir,
            return_envelope=wrap,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": {"code": "INTERNAL_ERROR", "stage": "api", "message": str(e)}})

    if wrap and isinstance(result, dict) and "ok" in result:
        return JSONResponse(status_code=200, content=result)

    # raw return (promptfoo expects raw JSON, no extras)
    if isinstance(result, dict) and "ok" in result and "data" in result:
        return JSONResponse(status_code=200, content=result["data"])
    return JSONResponse(status_code=200, content=result)
