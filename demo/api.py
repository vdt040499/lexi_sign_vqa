"""
LexiSign VQA — Web Demo API (Sub Task 1 + Sub Task 2)
Run: uvicorn demo.api:app --reload --port 8000
"""

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from subtask1.extract_signs import crop_signs_from_image
from subtask1.filter_signs import filter_signs, build_client
from subtask1.query_signs import query_signs, build_embedder, build_qdrant_client
from subtask2.answer_signs import get_answer

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="LexiSign VQA — Demo")

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent / "uploads"
LAWDB_PATH = Path("data/lawdb/lawdb_parsed.json")
PUBLIC_TEST_JSON = Path("data/public_test/vlsp_2025_public_test.json")
PUBLIC_TEST_IMAGE_DIR = Path("data/public_test/public_test_images")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Global singletons ─────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_processor    = None
_embed_model  = None
_qdrant_client = None
_llm_client   = None
_executor     = ThreadPoolExecutor(max_workers=2)
_lawdb_index: dict[tuple, dict] = {}   # (law_id, article_id) → {title, text}

# Streaming queues
_streams_task1: dict[str, asyncio.Queue] = {}
_streams_task2: dict[str, asyncio.Queue] = {}

# Persist Subtask 1 results for later Subtask 2 calls
# session_id -> {"session_dir", "sign_dir", "sample", "vlsp_meta": {question_type, question, choices}}
_session_state: dict[str, dict] = {}

_vlsp_public_test: list[dict] = []


def _format_vlsp_choices(choices) -> str:
    """Turn JSON {'A': '...', ...} into lines for the LLM (same shape as LexiSignVQA)."""
    if not choices or not isinstance(choices, dict):
        return ""
    lines: list[str] = []
    for key in sorted(choices.keys()):
        lines.append(f"{key}. {choices[key]}")
    return "\n".join(lines)


def _load_vlsp_public_test() -> None:
    global _vlsp_public_test
    if not PUBLIC_TEST_JSON.exists():
        print(f"[WARN] VLSP public test JSON not found: {PUBLIC_TEST_JSON}")
        _vlsp_public_test = []
        return
    with open(PUBLIC_TEST_JSON, encoding="utf-8") as f:
        _vlsp_public_test = json.load(f)
    print(f"[..] Loaded {len(_vlsp_public_test)} VLSP public_test samples.")


def _vlsp_by_id(sample_id: str) -> dict | None:
    key = (sample_id or "").strip()
    for row in _vlsp_public_test:
        if (row.get("id") or "").strip() == key:
            return row
    return None


def _safe_image_stem(raw: str) -> str | None:
    """image_id must be a single path segment; file on disk is {stem}.jpg"""
    s = (raw or "").strip()
    if not s or "/" in s or "\\" in s:
        return None
    return s


def _build_lawdb_index() -> dict:
    """Build (law_id, article_id) → {title, text} lookup from lawdb_parsed.json."""
    index = {}
    if not LAWDB_PATH.exists():
        print(f"[WARN] lawdb not found at {LAWDB_PATH}")
        return index
    with open(LAWDB_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for law in data:
        law_id = law.get("id", "")
        for article in law.get("articles", []):
            art_id = str(article.get("id", ""))
            index[(law_id, art_id)] = {
                "title": article.get("title", ""),
                "text":  article.get("text", ""),
            }
    print(f"[..] Loaded {len(index)} articles into index.")
    return index


@app.on_event("startup")
async def startup():
    global _processor, _embed_model, _qdrant_client, _llm_client, _lawdb_index
    print(f"[..] Loading models on {DEVICE}...")
    _processor, _embed_model = build_embedder(device=DEVICE)
    _qdrant_client = build_qdrant_client()
    _llm_client    = build_client()
    _lawdb_index   = _build_lawdb_index()
    _load_vlsp_public_test()
    print("[OK] Models ready.")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/vlsp/samples")
async def vlsp_samples():
    """List items from vlsp_2025_public_test.json (no gold answers)."""
    out = []
    for row in _vlsp_public_test:
        iid = _safe_image_stem(str(row.get("image_id", "")))
        if not iid:
            continue
        out.append({
            "id": (row.get("id") or "").strip(),
            "image_id": iid,
            "image_file": f"{iid}.jpg",
            "question": row.get("question", ""),
            "question_type": row.get("question_type", ""),
            "choices_text": _format_vlsp_choices(row.get("choices")),
        })
    return JSONResponse(out)


@app.get("/api/vlsp/image/{image_id}")
async def vlsp_image(image_id: str):
    stem = _safe_image_stem(image_id)
    if not stem:
        return JSONResponse({"error": "Invalid image_id"}, status_code=400)
    path = PUBLIC_TEST_IMAGE_DIR / f"{stem}.jpg"
    if not path.exists():
        return JSONResponse({"error": "Image not found"}, status_code=404)
    return FileResponse(str(path))


@app.post("/api/subtask1/analyze")
async def analyze_subtask1(
    image: UploadFile = File(...),
    question: str = Form(default=""),
):
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # Save uploaded image
    image_path = session_dir / f"input{Path(image.filename).suffix or '.jpg'}"
    image_path.write_bytes(await image.read())

    # Create SSE queue for this session
    queue: asyncio.Queue = asyncio.Queue()
    _streams_task1[session_id] = queue

    # Run pipeline in thread pool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_subtask1,
        session_id,
        image_path,
        question,
        None,
        loop,
        queue,
    )

    return JSONResponse({"session_id": session_id})


@app.post("/api/subtask1/analyze_vlsp")
async def analyze_subtask1_vlsp(sample_id: str = Form(...)):
    row = _vlsp_by_id(sample_id)
    if not row:
        return JSONResponse({"error": f"Unknown sample_id: {sample_id}"}, status_code=404)

    image_id = _safe_image_stem(str(row.get("image_id", "")))
    if not image_id:
        return JSONResponse({"error": "Invalid or missing image_id in JSON row"}, status_code=400)
    src = PUBLIC_TEST_IMAGE_DIR / f"{image_id}.jpg"
    if not src.exists():
        return JSONResponse({"error": f"Image not found: {src.name}"}, status_code=404)

    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    import shutil

    image_path = session_dir / "input.jpg"
    shutil.copy2(src, image_path)

    question = row.get("question", "") or ""
    vlsp_meta = {
        "question_type": row.get("question_type", "Yes/No"),
        "question": question,
        "choices": _format_vlsp_choices(row.get("choices")),
    }

    queue: asyncio.Queue = asyncio.Queue()
    _streams_task1[session_id] = queue

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_subtask1,
        session_id,
        image_path,
        question,
        vlsp_meta,
        loop,
        queue,
    )
    return JSONResponse({"session_id": session_id})


@app.get("/api/subtask1/stream/{session_id}")
async def stream_subtask1(session_id: str):
    if session_id not in _streams_task1:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    queue = _streams_task1[session_id]

    async def event_generator():
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=120)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("event") in ("done", "error"):
                    break
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'event': 'error', 'message': 'Timeout'})}\n\n"
        finally:
            _streams_task1.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/subtask2/answer")
async def answer_subtask2(session_id: str = Form(...)):
    if session_id not in _session_state:
        return JSONResponse({"error": "Session not found. Run Subtask 1 first."}, status_code=404)

    queue: asyncio.Queue = asyncio.Queue()
    _streams_task2[session_id] = queue

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_subtask2,
        session_id,
        loop,
        queue,
    )
    return JSONResponse({"session_id": session_id})


@app.get("/api/subtask2/stream/{session_id}")
async def stream_subtask2(session_id: str):
    if session_id not in _streams_task2:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    queue = _streams_task2[session_id]

    async def event_generator():
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=120)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("event") in ("done", "error"):
                    break
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'event': 'error', 'message': 'Timeout'})}\n\n"
        finally:
            _streams_task2.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/image/{session_id}/{file_path:path}")
async def get_image(session_id: str, file_path: str):
    path = UPLOAD_DIR / session_id / file_path
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(path))


# ── Pipeline runner (runs in thread) ─────────────────────────────────────────

def _push(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, event: dict):
    """Thread-safe push to async queue."""
    asyncio.run_coroutine_threadsafe(queue.put(event), loop)


def _run_subtask1(
    session_id: str,
    image_path: Path,
    question: str,
    vlsp_meta: dict | None,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
):
    session_dir = UPLOAD_DIR / session_id
    sign_dir = session_dir / "signs"
    sign_dir.mkdir(exist_ok=True)

    try:
        # ── Step 1: Extract signs ─────────────────────────────────────────
        _push(loop, queue, {"event": "step1_start"})

        crops = crop_signs_from_image(image_path, device=DEVICE)

        sample = {
            "image_id": session_id,
            "question": question,
            "detected_signs": [],
        }

        for i, crop in enumerate(crops):
            crop_name = f"crop{i}.jpg"
            crop.save(sign_dir / crop_name, format="JPEG")
            sample["detected_signs"].append({"image_name": crop_name, "is_chosen": False})

        _push(loop, queue, {
            "event": "step1_done",
            "data": {
                "total": len(crops),
                "crops": [s["image_name"] for s in sample["detected_signs"]],
            },
        })

        # ── Step 2: Filter signs ──────────────────────────────────────────
        _push(loop, queue, {"event": "step2_start"})

        # Pass image_dir as session_dir (image saved as input.jpg there)
        # filter_signs expects image_dir / f"{image_id}.jpg"
        # We'll use a trick: copy input image as {session_id}.jpg
        import shutil
        shutil.copy(image_path, session_dir / f"{session_id}.jpg")

        sample = filter_signs(
            sample=sample,
            image_dir=session_dir,
            sign_dir=sign_dir,
            client=_llm_client,
            batch_size=10,
        )

        _push(loop, queue, {
            "event": "step2_done",
            "data": {
                "signs": [
                    {
                        "image_name": s["image_name"],
                        "is_chosen": s.get("is_chosen", False),
                    }
                    for s in sample["detected_signs"]
                ],
            },
        })

        # ── Step 3: Query Qdrant ──────────────────────────────────────────
        _push(loop, queue, {"event": "step3_start"})

        sample = query_signs(
            sample=sample,
            sign_dir=sign_dir,
            processor=_processor,
            model=_embed_model,
            qdrant_client=_qdrant_client,
            device=DEVICE,
        )

        enriched_articles = []
        for art in sample.get("predicted_articles", []):
            law_id = art.get("law_id", "")
            art_id = str(art.get("article_id", ""))
            meta   = _lawdb_index.get((law_id, art_id), {})
            enriched_articles.append({
                "law_id":     law_id,
                "article_id": art_id,
                "title":      meta.get("title", ""),
                "text":       meta.get("text", ""),
            })

        _push(loop, queue, {
            "event": "step3_done",
            "data": {
                "articles": enriched_articles,
                "signs": [
                    {
                        "image_name": s["image_name"],
                        "is_chosen":  s.get("is_chosen", False),
                        "sign_name":  s.get("sign_name", ""),
                    }
                    for s in sample["detected_signs"]
                ],
            },
        })

        # Persist state for later Subtask 2 (same question as Subtask 1 + VLSP choices if any)
        if vlsp_meta is not None:
            meta = {
                "question_type": vlsp_meta.get("question_type", "Yes/No"),
                "question": vlsp_meta.get("question", question or ""),
                "choices": vlsp_meta.get("choices", "") or "",
            }
        else:
            meta = {
                "question_type": "Yes/No",
                "question": question or "",
                "choices": "",
            }
        _session_state[session_id] = {
            "session_dir": session_dir,
            "sign_dir": sign_dir,
            "sample": sample,
            "vlsp_meta": meta,
        }

        _push(loop, queue, {"event": "done"})

    except Exception as e:
        import traceback
        _push(loop, queue, {"event": "error", "message": str(e), "trace": traceback.format_exc()})


def _run_subtask2(
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
):
    try:
        state = _session_state.get(session_id)
        if not state:
            _push(loop, queue, {"event": "error", "message": "Session not found. Run Subtask 1 first."})
            return

        session_dir: Path = state["session_dir"]
        sign_dir: Path = state["sign_dir"]
        sample: dict = state["sample"]
        meta: dict = state.get("vlsp_meta") or {}
        question_type = meta.get("question_type", "Yes/No")
        question = (meta.get("question") or sample.get("question") or "").strip()
        choices = meta.get("choices", "") or ""

        if not question:
            _push(loop, queue, {"event": "error", "message": "Missing question for Subtask 2 (run Subtask 1 with a question)."})
            return

        _push(loop, queue, {"event": "step4_start"})
        answer_sample = {
            **sample,
            "question_type": question_type,
            "question": question,
            "choices": choices,
        }

        result = get_answer(
            sample=answer_sample,
            image_dir=session_dir,
            sign_dir=sign_dir,
            client=_llm_client,
            processor=_processor,
            embed_model=_embed_model,
            qdrant_client=_qdrant_client,
            device=DEVICE,
        )

        _push(loop, queue, {
            "event": "step4_done",
            "data": {
                "predict": result.get("predict", ""),
                "explanation": result.get("answer_explanation", ""),
            },
        })
        _push(loop, queue, {"event": "done"})
    except Exception as e:
        import traceback
        _push(loop, queue, {"event": "error", "message": str(e), "trace": traceback.format_exc()})
