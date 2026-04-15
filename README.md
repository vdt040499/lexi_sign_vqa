# LexiSign VQA

A **traffic-sign** visual question answering system grounded in legal text (QCVN / a law corpus): detect and crop signs from images, retrieve relevant articles (vector search in Qdrant), then generate answers with an LLM. The code targets a VLSP-style setup (Sub Task 1: article retrieval, Sub Task 2: answer generation) and includes a **web demo** (FastAPI).

## Models (current stack)

Default configuration lives in `subtask1/config.py` and `preprocessing/config.py` (you can change model names, endpoints, and the Qdrant collection there).

| Component | Model / tool | Role |
|------------|-----------------|--------|
| **Sign detection & cropping (public test)** | **YOLOE** (`ultralytics`), weights `models/yoloe-v8l-seg.pt` | Segment/detect signs using English class prompts (`YOLOE_CLASS_NAMES` in config). Crops are saved under `data/public_test/signs_extracted/`. If no boxes are found, the pipeline uses the **full image** as a single “crop”. |
| **Question-conditioned sign filtering** | **LLM** via an OpenAI-compatible API (`OLLAMA_BASE_URL`, default: Ollama), model `LLM_MODEL` (e.g. `gemma3:12b`) | Takes the original image + crops + question; returns a boolean list selecting which crops are relevant (`filter_signs`). You can point to Gemini by changing base URL/model in config. |
| **Image embeddings (retrieval & task 2)** | **CLIP** on Hugging Face: `zer0int/CLIP-GmP-ViT-L-14` (`EMBED_MODEL_ID`) | `CLIPModel` + `CLIPProcessor`: vectorize each crop, L2-normalize, and use it as the query vector to Qdrant. Ingest vector size: **768** (`VECTOR_SIZE` in preprocessing). |
| **Legal corpus (offline)** | **OpenCV** (preprocessing step 2) | Crop sign regions from the law DB images via thresholds/contours (YOLOE is not used for the law DB pipeline). |
| **Sign descriptions from the law DB** | **LLM** (`PARSE_MODEL_ID` in `preprocessing/config.py`, same endpoint type as Ollama) | `step3_parse.py`: generates `name` + `description` per crop, used as Qdrant payload and as context in Sub Task 2 prompts. |
| **Vector DB** | **Qdrant** (`COLLECTION` default `traffic_signs_clip`, cosine distance) | Stores sign embeddings; Sub Task 1 queries top `QDRANT_TOP_K` hits (default 1) to assign `law_id` / `article_id`. |

**Post-retrieval rules (Sub Task 1):** if `article_id` matches a special prefix (B./C./D./E.), the system adds supplemental articles from `DEFAULT_ARTICLES`; otherwise it adds `DEFAULT_ARTICLE`. If retrieval returns nothing, it also falls back to `DEFAULT_ARTICLE` (see `query_signs.py`).

## Detailed pipeline

### End-to-end flow (public test)

1. **Sub Task 1** (`subtask1/run_subtask1.py`): reads `data/public_test/vlsp_2025_public_test.json` → for each image:
   - **Extract:** YOLOE crops signs → `detected_signs` metadata + JPEGs under `signs_extracted/`.
   - **Filter:** LLM marks `is_chosen` for each crop relevant to the question (batched by `batch_size`).
   - **Query:** CLIP-embed each chosen crop → Qdrant → aggregate `predicted_articles` (with Rules 1–3 applied as above).
   - Writes **incrementally** to `data/public_test/results.json`; samples that already have `predicted_articles` are skipped on resume.

2. **Sub Task 2** (`subtask2/run_subtask2.py`): reads `results.json` → for each sample:
   - Re-embeds chosen crops + queries Qdrant to fetch `sign_name` / descriptions (to provide context for the LLM).
   - LLM answers based on question type (multiple-choice / true-false) and normalizes outputs (e.g. Yes/No → Đúng/Sai when needed).
   - Writes `results_task2.json`; `--submission` also creates a submission-format file per `subtask2/config.py`.

3. **Web demo** (`uvicorn demo.api:app`): calls the same logic from `subtask1` / `subtask2` (run from the **repo root** so imports resolve correctly).

### Building the law DB + Qdrant index (preprocessing)

The order is fixed; paths are defined in `preprocessing/config.py`.

| Step | Script | Input → output | Notes |
|------|--------|------------------|---------|
| 1 | `step1_preprocess.py` | `vlsp_2025_law.json` → `lawdb_preprocessed.json` | Normalize HTML/tables and mark image placeholders inside text. |
| 2 | `step2_extract.py` | Preprocessed JSON + law image folder → `lawdb_extracted.json` + crops in `signs_extracted/` | Crop signs using OpenCV (fixed thresholds/heuristics in code). |
| 3 | `step3_parse.py` | Extracted JSON + crops | VLM/LLM assigns a `name` + `description` per crop → `lawdb_parsed.json`; supports resume and `--article-ids`. |
| 4 | `step4_ingest.py` | `lawdb_parsed.json` + crops | CLIP (`INGEST_MODEL_ID`, same family as `EMBED_MODEL_ID`) → upsert to Qdrant; uses `ingest_state_clip.json` to avoid duplicates when resuming. |

**Embedding consistency requirement:** the model used to ingest the law DB and the model used to query during testing must be the **same** checkpoint (same `EMBED_MODEL_ID` / `INGEST_MODEL_ID`) and the same Qdrant collection; otherwise vector similarity is not meaningful.

## Requirements

- Python 3.10+ (recommended)
- Optional GPU (CUDA/MPS) — CPU works but is slower
- [Ollama](https://ollama.com/) (or another OpenAI-compatible endpoint) for sign filtering / answer generation
- A **Qdrant** account or collection (cloud or self-hosted) for retrieval embeddings

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download YOLOE weights (default path in `subtask1/config.py`):

- `models/yoloe-v8l-seg.pt`

Hugging Face models (CLIP/SigLIP, etc.) download on first run.

## Configuration

Edit endpoints and keys in:

| Purpose | File |
|--------|------|
| Sub Task 1 & 2 (public test images, Qdrant, Ollama, embeddings) | `subtask1/config.py` |
| Sub Task 2 (input/output JSON paths only) | `subtask2/config.py` |
| Law corpus preprocessing + Qdrant ingest | `preprocessing/config.py` |

Do not commit real API keys; use environment variables or an untracked local file if you extend configuration.

## Running from the repository root

Always `cd` to the directory that contains `subtask1/`, `data/`, and so on, so `import subtask1` and relative `data/...` paths resolve correctly.

### Sub Task 1 — Article retrieval

```bash
python subtask1/run_subtask1.py
python subtask1/run_subtask1.py --begin_idx 50 --device cuda --batch_size 10
```

- Reads `data/public_test/vlsp_2025_public_test.json`, writes `data/public_test/results.json`.
- If `results.json` already exists, the pipeline **resumes** (skips samples that already have `predicted_articles`).

### Sub Task 2 — Answer generation

```bash
python subtask2/run_subtask2.py
python subtask2/run_subtask2.py --begin_idx 0 --device cuda --submission
```

- Default input: Sub Task 1 output (`data/public_test/results.json`).
- Output: `data/public_test/results_task2.json`; add `--submission` to also write the competition-style file (see `subtask2/config.py`).

### Web demo (FastAPI)

```bash
uvicorn demo.api:app --reload --port 8000
```

Open `http://127.0.0.1:8000` in a browser.  
**Do not** run `python demo/api.py` from inside `demo/` — the `subtask1` package will not be on `PYTHONPATH`.

### Law corpus preprocessing (internal pipeline)

See **Building the law DB + Qdrant index (preprocessing)** above for step-by-step details and model choices. Run from the repository root:

```bash
python preprocessing/step1_preprocess.py
python preprocessing/step2_extract.py
python preprocessing/step3_parse.py
python preprocessing/step4_ingest.py
```

`step3_parse.py` resumes if a parsed file already exists; restrict to specific articles with `--article-ids 46 47`.

## Repository layout (abbreviated)

```
lexisignVQA_research/
├── data/
│   ├── lawdb/              # Law corpus + images (preprocess / ingest)
│   └── public_test/        # VLSP public test set + images
├── demo/                   # API + static (demo UI)
├── preprocessing/          # Steps 1–4: lawdb processing & Qdrant
├── subtask1/               # Sign detection, filtering, Qdrant query
├── subtask2/               # Answer generation
├── models/                 # YOLOE weights (place manually)
├── requirements.txt
└── test.py                 # Quick smoke script (imports from subtask1)
```
