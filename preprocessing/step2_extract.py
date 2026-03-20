import os
import sys
import json
import cv2
from pathlib import Path
from config import INPUT_JSON, OUTPUT_JSON, LAWDB_IMAGE_PATH, EXTRACTED_SIGN_PATH, IGNORED_ARTICLE_ID

LAWDB_IMAGE_DIR = Path(LAWDB_IMAGE_PATH)
EXTRACTED_SIGN_DIR = Path(EXTRACTED_SIGN_PATH)

def crop_signs_opencv(image_path):
    """Crop signs using OpenCV"""
    image = cv2.imread(str(image_path))
    if image is None: return []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 50: # Filter noise
            cropped_images.append(image[y:y+h, x:x+w])
    # Reverse the order to keep the order from top to bottom (optional)
    return cropped_images[::-1] 

def main():
    EXTRACTED_SIGN_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        lawdb = json.load(f)

    counts = {}
    for law in lawdb:
        law_id = law["id"]
        print(f"[..] Processing LAW {law_id}")
        counts[law_id] = 0

        for i, article in enumerate(law["articles"], 1):
            if article["id"].startswith(IGNORED_ARTICLE_ID):
                continue

            signs = []
            for image_name in article.get("images", []):
                image_path = LAWDB_IMAGE_DIR / image_name

                if not image_path.exists():
                    print(f"\t\t[WARNING] Image not found: {image_path}")
                    continue

                cropped_images = crop_signs_opencv(image_path)

                for j, cropped_image in enumerate(cropped_images):
                    cropped_name = f"{image_path.stem}_crop{j}.jpg"
                    save_path = EXTRACTED_SIGN_DIR / cropped_name
                    cv2.imwrite(str(save_path), cropped_image)
                    signs.append(cropped_name)

            # Update the articles with the new signs
            article["signs"] = signs
            article["num_signs"] = len(signs)
            counts[law_id] += len(signs)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(lawdb, f, ensure_ascii=False, indent=4)
        
    print(f"[..] Processing complete! Total signs extracted: {counts}")

if __name__ == "__main__":
    main()