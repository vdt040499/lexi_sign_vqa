from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient

from subtask1.config import (
    EMBED_MODEL_ID,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_TIMEOUT,
    COLLECTION,
    QDRANT_TOP_K,
    DEFAULT_ARTICLE,
    DEFAULT_ARTICLES,
)


def build_embedder(device: str = "cpu"):
    processor = AutoProcessor.from_pretrained(EMBED_MODEL_ID)
    model = AutoModel.from_pretrained(EMBED_MODEL_ID).to(device)
    model.eval()
    return processor, model


def build_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)


def _embed_image(image: Image.Image, processor, model, device: str) -> list[float]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vec = model.get_image_features(**inputs).cpu().numpy()[0]
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()


def query_signs(
    sample: dict,
    sign_dir: Path,
    processor,
    model,
    qdrant_client: QdrantClient,
    device: str = "cpu",
) -> dict:
    """
    For each chosen sign, embed with SigLIP, query Qdrant, then apply
    Rule 1 / Rule 2 / Rule 3 to build the final predicted_articles list.
    """
    predicted_articles: list[dict] = []

    for sign_info in sample.get("detected_signs", []):
        if not sign_info.get("is_chosen"):
            continue

        image = Image.open(sign_dir / sign_info["image_name"]).convert("RGB")
        vec = _embed_image(image, processor, model, device)

        hits = qdrant_client.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=QDRANT_TOP_K,
        )

        if hits:
            top = hits[0].payload
            sign_info["sign_name"] = top.get("sign_name", "")

        for hit in hits:
            article = {
                "law_id":     hit.payload["law_id"],
                "article_id": hit.payload["article_id"],
            }
            if article not in predicted_articles:
                predicted_articles.append(article)

    # Rule 2 — add supplementary article for special sign types (B./C./D./E.)
    final_articles: list[dict] = []
    has_special = False
    for article in predicted_articles:
        final_articles.append(article)
        for prefix, supplement in DEFAULT_ARTICLES.items():
            if article["article_id"].startswith(prefix):
                if supplement not in final_articles:
                    final_articles.append(supplement)
                    has_special = True
                break

    # Rule 3 — if no special signs found, add the default article
    if not has_special:
        if DEFAULT_ARTICLE not in final_articles:
            final_articles.append(DEFAULT_ARTICLE)

    # Rule 1 — fallback if nothing retrieved at all
    if not final_articles:
        final_articles = [DEFAULT_ARTICLE]

    sample["predicted_articles"] = final_articles
    return sample
