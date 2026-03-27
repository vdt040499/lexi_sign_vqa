"""
Sub Task 2 — Answer Generation

Given a sample that already has detected_signs with is_chosen flags (from Sub Task 1),
embed each chosen sign, retrieve its name/description from Qdrant, and ask an LLM
to answer the question about traffic law.
"""

import base64
import re
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from openai import OpenAI
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient

from subtask1.config import (
    EMBED_MODEL_ID,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_TIMEOUT,
    COLLECTION,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    QDRANT_TOP_K,
)
from subtask2.config import (
    ANSWER_MAPPING,
    QUESTION_TYPE_MULTIPLE_CHOICE,
    QUESTION_TYPE_YES_NO,
)


SYSTEM_PROMPT_MULTIPLE_CHOICE = """
You are a Legal QA assistant specializing in Vietnamese traffic regulations.
You will be given a question, an original traffic image, multiple choices, and
optionally the detected traffic signs with their names and descriptions.
Choose exactly 1 answer from the given choices.
Explain your reasoning first, then return the final selection (A, B, C or D)
inside <answer> and </answer>.
"""

SYSTEM_PROMPT_YES_NO = """
You are a Legal QA assistant specializing in Vietnamese traffic regulations.
You will be given a yes/no question, an original traffic image, and optionally
the detected traffic signs with their names and descriptions.
Explain your reasoning first, then return the final selection (Yes or No)
inside <answer> and </answer>.
"""


def build_client() -> OpenAI:
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


def build_embedder(device: str = "cpu"):
    processor = CLIPProcessor.from_pretrained(EMBED_MODEL_ID, use_fast=True)
    model = CLIPModel.from_pretrained(EMBED_MODEL_ID).to(device).eval()
    return processor, model


def build_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)


def _pil_to_base64(image: Image.Image) -> tuple[str, str]:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"


def _embed_image(
    image: Image.Image,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
) -> list[float]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        vec = F.normalize(features, p=2, dim=1).cpu().numpy()[0]
    return vec.tolist()


def get_sign_parts(
    sample: dict,
    sign_dir: Path,
    processor: CLIPProcessor,
    embed_model: CLIPModel,
    qdrant_client: QdrantClient,
    device: str = "cpu",
    top_k: int = QDRANT_TOP_K,
) -> list[dict]:
    """Build multimodal message parts for each chosen sign."""
    parts: list[dict] = []
    sign_index = 0

    for sign_info in sample.get("detected_signs", []):
        if not sign_info.get("is_chosen"):
            continue

        sign_image = Image.open(Path(sign_dir) / sign_info["image_name"]).convert("RGB")
        vec = _embed_image(sign_image, processor, embed_model, device)

        hits = qdrant_client.query_points(
            collection_name=COLLECTION,
            query=vec,
            limit=top_k,
        ).points

        for hit in hits:
            payload  = hit.payload
            sign_b64, sign_mime = _pil_to_base64(sign_image)
            sign_name = payload.get("sign_name", "")
            sign_desc = payload.get("sign_description", sign_name)
            parts.extend([
                {
                    "type": "text",
                    "text": f"<<<SIGN {sign_index}>>>: {sign_name} - {sign_desc}",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{sign_mime};base64,{sign_b64}"},
                },
                {"type": "text", "text": f"<<<END SIGN {sign_index}>>>"},
            ])

        sign_index += 1

    return parts


def get_final_answer(response: str) -> str:
    """Extract text inside <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return match.group(1).strip() if match else ""


def get_answer(
    sample: dict,
    image_dir: Path,
    sign_dir: Path,
    client: OpenAI,
    processor: CLIPProcessor,
    embed_model: CLIPModel,
    qdrant_client: QdrantClient,
    device: str = "cpu",
    top_k: int = QDRANT_TOP_K,
) -> dict:
    """
    Generate an answer for a sample given its detected signs.

    Returns the sample dict updated with:
      - predict: normalised answer string
      - answer_explanation: raw LLM response
    """
    question_type = sample.get("question_type", "")
    question      = sample.get("question", "")
    choices       = sample.get("choices", "")

    if question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
        system_prompt = SYSTEM_PROMPT_MULTIPLE_CHOICE
        user_text     = f"{question}\n\n{choices}" if choices else question
    elif question_type == QUESTION_TYPE_YES_NO:
        system_prompt = SYSTEM_PROMPT_YES_NO
        user_text     = question
    else:
        system_prompt = SYSTEM_PROMPT_YES_NO
        user_text     = question

    image_path = Path(image_dir) / f"{sample['image_id']}.jpg"
    image = Image.open(image_path).convert("RGB")
    org_b64, org_mime = _pil_to_base64(image)

    parts: list[dict] = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": f"{user_text}\n"},
        {"type": "text", "text": "Input Image:"},
        {"type": "image_url", "image_url": {"url": f"data:{org_mime};base64,{org_b64}"}},
    ]

    sign_parts = get_sign_parts(
        sample, sign_dir, processor, embed_model, qdrant_client, device, top_k
    )
    if sign_parts:
        parts.append({"type": "text", "text": "Detected Signs (keep this order):"})
        parts.extend(sign_parts)

    try:
        response_text: str = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": parts}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        ).choices[0].message.content

        raw_answer = get_final_answer(response_text)
        predict    = ANSWER_MAPPING.get(raw_answer, raw_answer)

    except Exception as e:
        response_text = f"Error: {e}"
        predict = "A" if question_type == QUESTION_TYPE_MULTIPLE_CHOICE else "Đúng"

    return {
        **sample,
        "predict":            predict,
        "answer_explanation": response_text,
    }
