"""
Sub Task 2 — Answer Generation Pipeline

Reads the JSON output from Sub Task 1 (which has detected_signs + predicted_articles),
generates an answer for each sample, and saves the results.

Usage:
    python subtask2/run_subtask2.py
    python subtask2/run_subtask2.py --begin_idx 50
    python subtask2/run_subtask2.py --device cuda
    python subtask2/run_subtask2.py --submission   # also write submission_task2.json
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from subtask1.config import IMAGE_DIR, SIGN_DIR
from subtask2.config import INPUT_JSON, OUTPUT_JSON, SUBMISSION_JSON
from subtask2.answer_signs import (
    build_client,
    build_embedder,
    build_qdrant_client,
    get_answer,
)


def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sub Task 2: Answer Generation")
    parser.add_argument(
        "--begin_idx", "-b", type=int, default=0,
        help="Start index (for resume)",
    )
    parser.add_argument(
        "--device", "-d", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cpu / cuda / mps",
    )
    parser.add_argument(
        "--submission", action="store_true",
        help="Also write submission_task2.json stripped of debug fields",
    )
    args = parser.parse_args()

    image_dir   = Path(IMAGE_DIR)
    sign_dir    = Path(SIGN_DIR)
    output_path = Path(OUTPUT_JSON)

    # Resume from OUTPUT_JSON if it already exists
    if output_path.exists():
        samples = load_json(OUTPUT_JSON)
        print(f"[..] Resuming from existing output: {OUTPUT_JSON}")
    else:
        samples = load_json(INPUT_JSON)

    print(f"[..] Device        : {args.device}")
    print(f"[..] Total samples : {len(samples)}")
    print(f"[..] Begin index   : {args.begin_idx}")
    print(f"[..] Output        : {OUTPUT_JSON}")

    client        = build_client()
    qdrant_client = build_qdrant_client()
    processor, embed_model = build_embedder(device=args.device)

    for sample in tqdm(samples[args.begin_idx:], total=len(samples) - args.begin_idx):
        # Skip already processed samples
        if "predict" in sample:
            continue

        try:
            result = get_answer(
                sample=sample,
                image_dir=image_dir,
                sign_dir=sign_dir,
                client=client,
                processor=processor,
                embed_model=embed_model,
                qdrant_client=qdrant_client,
                device=args.device,
            )
            sample["predict"]            = result["predict"]
            sample["answer_explanation"] = result["answer_explanation"]
        except Exception as e:
            print(f"[ERROR] sample {sample.get('id')}: {e}")

        save_json(samples, OUTPUT_JSON)

    print(f"\n[OK] Done. Results saved to: {OUTPUT_JSON}")

    if args.submission:
        submission = [
            {
                "id":       s.get("id"),
                "image_id": s.get("image_id"),
                "answer":   s.get("predict", ""),
            }
            for s in samples
        ]
        save_json(submission, SUBMISSION_JSON)
        print(f"[OK] Submission saved to: {SUBMISSION_JSON}")


if __name__ == "__main__":
    main()
