from __future__ import annotations

import uuid
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile

from .orchestrator import run_request
from .utils import ensure_request_dir

app = FastAPI(title="Data Analyst Agent", version="0.2.2")

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
    Accepts multipart/form-data with questions.txt required
    OR application/json with 'question_text'. Optional attachments in both.

    On success: returns ONLY the raw JSON structure requested by the prompt
    (array by default, or object if explicitly requested by the question).
    """
    content_type = (request.headers.get("content-type") or "").lower()
    rid = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    workdir = ensure_request_dir(rid)

    question_text: Optional[str] = None
    files: Dict[str, bytes] = {}
    json_attachments: List[Dict[str, Any]] = []
    response_schema: Optional[Dict[str, Any]] = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        qf: Optional[StarletteUploadFile] = form.get("questions.txt")  # type: ignore
        if not isinstance(qf, StarletteUploadFile):
            raise HTTPException(status_code=422, detail="questions.txt is required as a file field")
        question_text = (await qf.read()).decode("utf-8", errors="replace")

        # collect attachments
        for key, val in form.multi_items():
            if key == "questions.txt":
                continue
            if isinstance(val, StarletteUploadFile):
                data = await val.read()
                if data:
                    files[val.filename or key] = data

    elif "application/json" in content_type:
        body = await request.json()
        question_text = body.get("question_text")
        if not question_text:
            raise HTTPException(status_code=422, detail="question_text is required in JSON body")
        json_attachments = body.get("attachments", []) or []
        response_schema = body.get("response_schema")
    else:
        raise HTTPException(status_code=415, detail="Unsupported Content-Type")

    # Default schema: array of answers (if the prompt wants object, contract will switch it)
    schema = response_schema or {"type": "array"}

    try:
        payload = await run_request(
            question_text=question_text,
            uploaded_files=files,
            json_attachments=json_attachments,
            schema=schema,
            request_id=rid,
            workdir=workdir,
        )
    except Exception as e:
        # On error we return an error envelope so you can debug; success is always raw JSON.
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": {"code": "INTERNAL_ERROR", "stage": "api", "message": str(e)}},
        )

    # SUCCESS → return ONLY raw JSON (array or object), nothing else.
    return JSONResponse(status_code=200, content=payload)
