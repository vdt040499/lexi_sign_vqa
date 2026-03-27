# Re-export shared settings from subtask1
from subtask1.config import (
    IMAGE_DIR,
    SIGN_DIR,
    EMBED_MODEL_ID,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_TIMEOUT,
    COLLECTION,
    QDRANT_TOP_K,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Input = output from Sub Task 1
INPUT_JSON      = "data/public_test/results.json"
OUTPUT_JSON     = "data/public_test/results_task2.json"
SUBMISSION_JSON = "data/public_test/submission_task2.json"

# ── Question types ─────────────────────────────────────────────────────────────
QUESTION_TYPE_MULTIPLE_CHOICE = "Multiple choice"
QUESTION_TYPE_YES_NO          = "Yes/No"

# ── Answer normalisation ───────────────────────────────────────────────────────
ANSWER_MAPPING = {
    "Yes":   "Đúng",
    "No":    "Sai",
    "Không": "Sai",
    "False": "Sai",
    "Fail":  "Sai",
    "True":  "Đúng",
}
