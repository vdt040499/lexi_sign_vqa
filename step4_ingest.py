qdrant_client = QdrantClient(
    url="https://36f9214c-740a-49d2-8bd2-5ed5d78a319b.eu-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.68I0_4iGajB0bTeMaTN9HVJ5eNjzwqovjuQjnR3xxME",
)

import os, json, torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoProcessor, AutoModel

# --- Cấu hình ---
INPUT_JSON = "data/lawdb/lawdb_parsed.json"
SIGN_DIR = Path("data/lawdb/signs_extracted")
QDRANT_URI = "http://localhost:6333" 
COLLECTION = "traffic_signs"
MODEL_ID = "google/siglip-base-patch16-224"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)

    client = QdrantClient(url=QDRANT_URI)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

    with open(INPUT_JSON, "r", encoding="utf-8") as f: lawdb = json.load(f)

    points, p_id = [], 0
    for law in lawdb:
        for art in tqdm(law.get("articles", []), desc=f"Ingesting {law['id']}"):
            for detail in art.get("detailed_signs", []):
                path = SIGN_DIR / detail["image"]
                if not path.exists(): continue

                inputs = proc(images=Image.open(path).convert("RGB"), return_tensors="pt").to(device)
                with torch.no_grad():
                    vec = model.get_image_features(**inputs).cpu().numpy()[0]

                points.append(PointStruct(
                    id=p_id, 
                    vector=vec.tolist(), 
                    payload={
                        "law_id": law["id"],
                        "article_id": art.get("id"),
                        "sign_name": detail.get("name"),
                        "sign_description": detail.get("description")
                    }
                ))
                p_id += 1
                if len(points) >= 50:
                    client.upsert(collection_name=COLLECTION, points=points)
                    points = []

    if points: client.upsert(collection_name=COLLECTION, points=points)
    print(f"Success! Ingested {p_id} points.")

if __name__ == "__main__":
    main()