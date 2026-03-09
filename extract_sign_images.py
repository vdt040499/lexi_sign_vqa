import sys
import json
import torch
from PIL import Image
from pathlib import Path

IGNORED_ARTICLE_ID = ("A", "G", "I", "K", "M", "O", "P")

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)

lawdb_image_dir = Path("./data/law_db/images.fld")
lawdb_preprocessed_json_path = Path("./data/preprocessed/preprocessed_vlsp_2025_law.json")

try:
    lawdb = load_json(lawdb_preprocessed_json_path)
    
    counts = {}

    for law in lawdb:
        law_id = law["id"]
        print(f"[..] Processing LAW {law_id}\n")
        counts[law_id] = 0

        for i, article in enumerate(law["articles"], 1):
            print(f"\t{i}/{len(law['articles'])} - article id {article['id']}")

            if article["id"].startswith(IGNORED_ARTICLE_ID):
                continue

            signs = []
            for image_name in article["images"]:
                image_path = lawdb_image_dir / image_name

                if not image_path.exists():
                    print(f"\t\t[WARNING] Image not found: {image_path}")
                    continue

                try:
                    # signs detection
                    cropped_images = sign_extractor.crop_signs(image_path)

                    for j, cropped_image in enumerate(cropped_images):
                        cropped_name = f"{image_path.stem}_crop{j}.jpg"
                        cropped_image.save(lawdb_extracted_sign_dir / cropped_name)
                        signs.append(cropped_name)

                except Exception as e:
                    print(f"\t\t[ERROR] Failed to process image {image_name}: {e}")
                    continue

            # Add new field
            article["signs"] = signs
            article["num_signs"] = len(signs)
            counts[law_id] += len(signs)
except FileNotFoundError:
    print(f"[ERROR] LawDB JSON file not found: {lawdb_preprocessed_json_path}")
except Exception as e:
    print(f"[ERROR] Failed to load LawDB JSON: {e}")