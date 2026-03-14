import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoProcessor, AutoModel

# --- Cấu hình (Qdrant Cloud) ---
QDRANT_URL = "https://36f9214c-740a-49d2-8bd2-5ed5d78a319b.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.68I0_4iGajB0bTeMaTN9HVJ5eNjzwqovjuQjnR3xxME"
# Timeout (giây) cho HTTP khi gửi lên cloud, tránh WriteTimeout
QDRANT_TIMEOUT = 120

INPUT_JSON = "data/lawdb/lawdb_parsed.json"
SIGN_DIR = Path("data/lawdb/signs_extracted")
INGEST_STATE_JSON = "data/lawdb/ingest_state.json"
COLLECTION = "traffic_signs"
MODEL_ID = "google/siglip-base-patch16-224"
VECTOR_SIZE = 768
# Batch nhỏ hơn để tránh timeout khi upload lên Qdrant Cloud
BATCH_SIZE = 20


def load_ingest_state() -> set:
    """Trả về set các (law_id, article_id) đã ingest (dạng tuple str)."""
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
    """Lưu state từ set (law_id, article_id) sang file dạng { law_id: [article_id, ...] }."""
    data = {}
    for law_id, art_id in sorted(ingested):
        data.setdefault(law_id, []).append(art_id)
    for k in data:
        data[k] = sorted(data[k])
    with open(INGEST_STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_collection(client: QdrantClient):
    """Tạo collection nếu chưa tồn tại; không xóa dữ liệu cũ khi resume."""
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
    """Lấy ID tiếp theo để dùng cho point mới (tránh trùng khi append)."""
    try:
        info = client.get_collection(COLLECTION)
        return info.points_count
    except Exception:
        return 0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=QDRANT_TIMEOUT,
    )
    ensure_collection(client)
    p_id = next_point_id(client)

    ingested = load_ingest_state()
    print(f"[INFO] Đã có {len(ingested)} (law, article) đã ingest, sẽ bỏ qua.")

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
                    vec = model.get_image_features(**inputs).cpu().numpy()[0]

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

    print(f"[SUCCESS] Ingest xong. Đã xử lý {processed} article mới, bỏ qua {skipped} đã có. Tổng points trong collection: {p_id}.")


if __name__ == "__main__":
    main()
