# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_JSON   = "data/public_test/vlsp_2025_public_test.json"
IMAGE_DIR   = "data/public_test/public_test_images"
SIGN_DIR    = "data/public_test/signs_extracted"
OUTPUT_JSON = "data/public_test/results.json"
YOLOE_MODEL = "models/yoloe-v8l-seg.pt"

# ── YoloE ─────────────────────────────────────────────────────────────────────
YOLOE_CLASS_NAMES = [
    "blue rectangle traffic sign",
    "red circular traffic sign",
    "blue circular traffic sign",
    "red triangle traffic sign",
    "green rectangle traffic sign",
    "white rectangle text traffic sign",
]
YOLOE_SCORE_THRESHOLD = 0.2

# ── Embedding (SigLIP) ────────────────────────────────────────────────────────
EMBED_MODEL_ID = "zer0int/CLIP-GmP-ViT-L-14"

# ── Qdrant Cloud ──────────────────────────────────────────────────────────────
QDRANT_URL     = "https://36f9214c-740a-49d2-8bd2-5ed5d78a319b.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.68I0_4iGajB0bTeMaTN9HVJ5eNjzwqovjuQjnR3xxME"
QDRANT_TIMEOUT = 120
COLLECTION     = "traffic_signs_clip"
QDRANT_TOP_K   = 1

# ── LLM — Ollama local ────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"
LLM_MODEL       = "gemma3:12b"
# # ── LLM — Google Gemini API ───────────────────────────────────────────────────
# OLLAMA_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# OLLAMA_API_KEY  = "YOUR_GOOGLE_API_KEY"
# LLM_MODEL       = "gemma-3-12b-it"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 5000

# ── Business rules ────────────────────────────────────────────────────────────
DEFAULT_ARTICLE = {"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"}

DEFAULT_ARTICLES = {
    "B.": {"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"},
    "C.": {"law_id": "QCVN 41:2024/BGTVT", "article_id": "28"},
    "D.": {"law_id": "QCVN 41:2024/BGTVT", "article_id": "32"},
    "E.": {"law_id": "QCVN 41:2024/BGTVT", "article_id": "36"},
}
