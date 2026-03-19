"""
Sub Task 1 — Article Retrieval Pipeline

Usage:
    python subtask1/run_subtask1.py
    python subtask1/run_subtask1.py --begin_idx 50
    python subtask1/run_subtask1.py --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from openai import OpenAI
from tqdm import tqdm

from subtask1.config import (
    TEST_JSON,
    IMAGE_DIR,
    SIGN_DIR,
    OUTPUT_JSON,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    EMBED_MODEL_ID,
)
from subtask1.extract_signs import crop_signs_from_image
from subtask1.filter_signs import filter_signs, build_client
from subtask1.query_signs import query_signs, build_embedder, build_qdrant_client


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_sample(
    sample: dict,
    image_dir: Path,
    sign_dir: Path,
    llm_client: OpenAI,
    processor,
    embed_model,
    qdrant_client,
    device: str,
    batch_size: int,
) -> dict:
    image_id = sample["image_id"]
    image_path = image_dir / f"{image_id}.jpg"

    # Step 1 — Extract signs
    try:
        crops = crop_signs_from_image(image_path, device=device)
    except Exception as e:
        print(f"[WARN] extract_signs failed for {image_id}: {e}")
        crops = []

    sample["detected_signs"] = []
    for i, crop in enumerate(crops):
        crop_name = f"{image_id}_crop{i}.jpg"
        crop.save(sign_dir / crop_name, format="JPEG")
        sample["detected_signs"].append({"image_name": crop_name, "is_chosen": False})

    # Step 2 — Filter signs
    sample = filter_signs(
        sample=sample,
        image_dir=image_dir,
        sign_dir=sign_dir,
        client=llm_client,
        batch_size=batch_size,
    )

    # Step 3 — Query Qdrant
    sample = query_signs(
        sample=sample,
        sign_dir=sign_dir,
        processor=processor,
        model=embed_model,
        qdrant_client=qdrant_client,
        device=device,
    )

    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sub Task 1: Article Retrieval")
    parser.add_argument("--begin_idx",  "-b", type=int, default=0,
                        help="Start index (for resume)")
    parser.add_argument("--batch_size", "-s", type=int, default=10,
                        help="Number of signs per LLM filter call")
    parser.add_argument("--device",     "-d", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cpu / cuda / mps")
    args = parser.parse_args()

    image_dir = Path(IMAGE_DIR)
    sign_dir  = Path(SIGN_DIR)
    sign_dir.mkdir(parents=True, exist_ok=True)

    # Load data — resume from OUTPUT_JSON if it already exists
    output_path = Path(OUTPUT_JSON)
    if output_path.exists():
        samples = load_json(OUTPUT_JSON)
        print(f"[..] Resuming from existing output: {OUTPUT_JSON}")
    else:
        samples = load_json(TEST_JSON)

    print(f"[..] Device        : {args.device}")
    print(f"[..] Total samples : {len(samples)}")
    print(f"[..] Begin index   : {args.begin_idx}")
    print(f"[..] Batch size    : {args.batch_size}")
    print(f"[..] Output        : {OUTPUT_JSON}")

    # Build clients / models
    llm_client   = build_client()
    qdrant_client = build_qdrant_client()
    processor, embed_model = build_embedder(device=args.device)

    for sample in tqdm(samples[args.begin_idx:], total=len(samples) - args.begin_idx):
        # Skip already processed samples
        if "predicted_articles" in sample:
            continue

        try:
            sample = process_sample(
                sample=sample,
                image_dir=image_dir,
                sign_dir=sign_dir,
                llm_client=llm_client,
                processor=processor,
                embed_model=embed_model,
                qdrant_client=qdrant_client,
                device=args.device,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"[ERROR] sample {sample.get('id')}: {e}")

        save_json(samples, OUTPUT_JSON)

    print(f"\n[OK] Done. Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
