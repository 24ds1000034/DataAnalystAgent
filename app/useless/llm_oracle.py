from __future__ import annotations
import os, json, time
from typing import Dict, Any, List, Optional


def _outdir(base: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(base, f"oracle_run_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def _load_keys():
    # optional .env support
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY"), os.getenv("GOOGLE_API_KEY")


def _openai_answer(api_key: str, questions: List[str], context_text: str, timeout: int = 10) -> Optional[List[Any]]:
    """
    Use OpenAI to get a JSON array answer.
    Return None on any failure or if the library is missing.
    """
    try:
        import openai  # pip install openai
        client = openai.OpenAI(api_key=api_key)
        system = (
            "You answer analytics questions. Return ONLY a JSON array. "
            "The array length must equal the number of questions (fill with null when unknown). No extra text."
        )
        user = f"Context:\n{context_text}\n\nQuestions:\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            timeout=timeout,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        obj = json.loads(content)
        # Accept common shapes
        for key in ("answers", "data", "array", "result"):
            if key in obj and isinstance(obj[key], list):
                arr = obj[key]
                break
        else:
            if isinstance(obj, list):
                arr = obj
            else:
                return None
        # enforce length
        if len(arr) != len(questions):
            arr = (arr + [None] * len(questions))[: len(questions)]
        return arr
    except Exception:
        return None


def _gemini_answer(api_key: str, questions: List[str], context_text: str, timeout: int = 10) -> Optional[List[Any]]:
    """
    Use Gemini to get a JSON array answer.
    """
    try:
        import google.generativeai as genai  # pip install google-generativeai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        system = (
            "Answer the questions below; return ONLY a JSON array (no preface). "
            "Array length must equal the number of questions; use null when unknown."
        )
        prompt = system + "\n\nContext:\n" + (context_text or "") + "\n\nQuestions:\n" + "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(questions)
        )
        resp = model.generate_content(prompt, request_options={"timeout": timeout})
        txt = (resp.text or "").strip()
        import re
        m = re.search(r"\[[\s\S]*\]", txt)
        if not m:
            return None
        arr = json.loads(m.group(0))
        if not isinstance(arr, list):
            return None
        if len(arr) != len(questions):
            arr = (arr + [None] * len(questions))[: len(questions)]
        return arr
    except Exception:
        return None


async def quick_answer_oracle(
    questions: List[str],
    context_text: str,
    workdir: str,
    budget_ms: int,
) -> Dict[str, Any]:
    """
    Parallel LLM path. If no keys present or model calls fail → {}.
    Always writes artifacts under workdir/oracle_run_*.
    """
    if not questions:
        return {}

    out_dir = _outdir(workdir)
    with open(os.path.join(out_dir, "questions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, questions)))
    with open(os.path.join(out_dir, "context.txt"), "w", encoding="utf-8") as f:
        f.write(context_text or "")

    oai_key, gem_key = _load_keys()
    answers: Optional[List[Any]] = None
    provider = "none"

    if oai_key:
        provider = "openai"
        answers = _openai_answer(oai_key, questions, context_text)
    elif gem_key:
        provider = "gemini"
        answers = _gemini_answer(gem_key, questions, context_text)

    if answers is not None:
        with open(os.path.join(out_dir, "oracle_answers.json"), "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)
        return {"array": answers}

    with open(os.path.join(out_dir, "oracle_answers.json"), "w", encoding="utf-8") as f:
        json.dump({"provider": provider, "status": "no-answer"}, f, indent=2)
    return {}
