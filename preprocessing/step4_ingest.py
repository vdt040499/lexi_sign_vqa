import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import CLIPModel, CLIPProcessor
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_TIMEOUT, INPUT_JSON, SIGN_PATH, INGEST_STATE_JSON, COLLECTION, MODEL_ID, VECTOR_SIZE, BATCH_SIZE

SIGN_DIR = Path(SIGN_PATH)

def load_ingest_state() -> set:
    """Return the set of (law_id, article_id) that have been ingested (as tuple str)."""
    if not os.path.exists(INGEST_STATE_JSON):
        return set()
    with open(INGEST_STATE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = set()
    for law_id, articles in data.items():
        for art_id in articles:
            out.add((str(law_id), str(art_id)))
    return out


def save_ingest_state(ingested: set):
    """Save the state from set (law_id, article_id) to file in the format { law_id: [article_id, ...] }."""
    data = {}
    for law_id, art_id in sorted(ingested):
        data.setdefault(law_id, []).append(art_id)
    for k in data:
        data[k] = sorted(data[k])
    with open(INGEST_STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_collection(client: QdrantClient):
    """Create the collection if it doesn't exist; don't delete old data when resuming."""
    try:
        client.get_collection(COLLECTION)
        return
    except Exception:
        pass
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def next_point_id(client: QdrantClient) -> int:
    """Get the next ID to use for the new point (to avoid duplicates when appending)."""
    try:
        info = client.get_collection(COLLECTION)
        return info.points_count
    except Exception:
        return 0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=QDRANT_TIMEOUT,
    )
    ensure_collection(client)
    p_id = next_point_id(client)

    ingested = load_ingest_state()
    print(f"[INFO] Already ingested {len(ingested)} (law, article), will skip.")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        lawdb = json.load(f)

    points = []
    newly_ingested = set()
    skipped, processed = 0, 0

    for law in lawdb:
        law_id = str(law.get("id", ""))
        for art in tqdm(law.get("articles", []), desc=f"Law {law_id}"):
            if not art.get("__is_sucessfully_parsing_sign"):
                continue
            art_id = str(art.get("id", ""))
            key = (law_id, art_id)
            if key in ingested or key in newly_ingested:
                skipped += 1
                continue

            for detail in art.get("detailed_signs", []):
                path = SIGN_DIR / detail["image"]
                if not path.exists():
                    continue

                img = Image.open(path).convert("RGB")
                inputs = proc(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                    vec = F.normalize(features, p=2, dim=1).cpu().numpy()[0]

                points.append(
                    PointStruct(
                        id=p_id,
                        vector=vec.tolist(),
                        payload={
                            "law_id": law_id,
                            "article_id": art_id,
                            "sign_name": detail.get("name"),
                            "sign_description": detail.get("description"),
                        },
                    )
                )
                p_id += 1

                if len(points) >= BATCH_SIZE:
                    client.upsert(collection_name=COLLECTION, points=points)
                    points = []

            if key not in newly_ingested:
                newly_ingested.add(key)
                ingested.add(key)
                save_ingest_state(ingested)
            processed += 1

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    print(f"[SUCCESS] Ingest completed. Processed {processed} new articles, skipped {skipped} already ingested. Total points in collection: {p_id}.")


if __name__ == "__main__":
    main()
