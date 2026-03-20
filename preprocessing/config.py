# Step 1: Preprocess the law database
INPUT_FILE = "data/lawdb/vlsp_2025_law.json"
OUTPUT_FILE = "data/lawdb/lawdb_preprocessed.json"
DEBUG_DIR = "debug"

# Step 2: Extract the signs from the images
INPUT_EXTRACT_JSON = "data/lawdb/lawdb_preprocessed.json"
OUTPUT_EXTRACT_JSON = "data/lawdb/lawdb_extracted.json"
LAWDB_IMAGE_PATH = "data/lawdb/images.fld"
EXTRACTED_SIGN_PATH = "data/lawdb/signs_extracted"
IGNORED_ARTICLE_ID = "ignore_prefix_"

# Step 3: Parse the signs
INPUT_PARSE_JSON = "data/lawdb/lawdb_extracted.json"
OUTPUT_PARSE_JSON = "data/lawdb/lawdb_parsed.json"
EXTRACTED_SIGN_PATH = "data/lawdb/signs_extracted"

OLLAMA_API_KEY = "ollama"
BASE_URL = "http://localhost:11434/v1"
MODEL_ID = "gemma3:12b"

# Step 4: Ingest the data into Qdrant
QDRANT_URL = "https://36f9214c-740a-49d2-8bd2-5ed5d78a319b.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.68I0_4iGajB0bTeMaTN9HVJ5eNjzwqovjuQjnR3xxME"
QDRANT_TIMEOUT = 120

INPUT_JSON = "data/lawdb/lawdb_parsed.json"
SIGN_PATH = "data/lawdb/signs_extracted"
INGEST_STATE_JSON = "data/lawdb/ingest_state_clip.json"
COLLECTION = "traffic_signs_clip"
MODEL_ID = "zer0int/CLIP-GmP-ViT-L-14"
VECTOR_SIZE = 768
BATCH_SIZE = 20