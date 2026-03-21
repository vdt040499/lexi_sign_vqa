"""
Sub Task 1 — Web Demo API
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

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="LexiSign VQA — Sub Task 1 Demo")

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent / "uploads"
LAWDB_PATH = Path("data/lawdb/lawdb_parsed.json")
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

# session_id → asyncio.Queue
_sessions: dict[str, asyncio.Queue] = {}


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
    print("[OK] Models ready.")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def analyze(
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
    _sessions[session_id] = queue

    # Run pipeline in thread pool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_pipeline,
        session_id,
        image_path,
        question,
        loop,
        queue,
    )

    return JSONResponse({"session_id": session_id})


@app.get("/api/stream/{session_id}")
async def stream(session_id: str):
    if session_id not in _sessions:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    queue = _sessions[session_id]

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
            _sessions.pop(session_id, None)

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


def _run_pipeline(
    session_id: str,
    image_path: Path,
    question: str,
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

        _push(loop, queue, {"event": "done"})

    except Exception as e:
        import traceback
        _push(loop, queue, {"event": "error", "message": str(e), "trace": traceback.format_exc()})
