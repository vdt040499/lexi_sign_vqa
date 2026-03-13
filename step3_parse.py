import os
import re
import json
import time
import base64
from copy import deepcopy
from pathlib import Path
from typing import Sequence, List, Dict, Any, Tuple
from tqdm import tqdm
from openai import OpenAI

INPUT_JSON = "data/lawdb/lawdb_extracted.json"
OUTPUT_JSON = "data/lawdb/lawdb_parsed.json"
EXTRACTED_SIGN_DIR = Path("data/lawdb/signs_extracted")

OLLAMA_API_KEY = "ollama"
BASE_URL = "http://localhost:11434/v1"
MODEL_ID = "moondream"

# Khởi tạo client
client = OpenAI(api_key=OLLAMA_API_KEY, base_url=BASE_URL)

# Prompt template gốc từ dự án
PARSE_SIGNS_PROMPT = """
Bạn là một chuyên gia về luật giao thông. Dưới đây là nội dung Điều luật: <<TITLE>>.
Nội dung: <<CONTENT>>
Hãy phân tích <<NUM_SIGNS>> hình ảnh biển báo tương ứng từ ảnh thứ <<FROM_INDEX>> đến <<TO_INDEX>>.
Trả về một mảng JSON có cấu trúc: [{"name": "Tên biển", "description": "Mô tả chi tiết"}].
Chỉ trả về JSON, không giải thích gì thêm.
"""

def safe_json_from_llm(text: str) -> any:
    """
    Strip common code fences and parse JSON.
    Accepts ```json ... ``` or ``` ... ``` fences.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)

def group_signs_by_image(article: Dict) -> Tuple[List[str], List[List[str]]]:
    """Gom nhóm các biển báo đã cắt theo ID ảnh trang luật gốc"""
    base_image_ids = [p.split(".jpg")[0] for p in article.get("images", [])]
    detailed_images = [[] for _ in base_image_ids]
    for sign in article.get("signs", []):
        image_id = sign.split("_crop")[0]
        if image_id in base_image_ids:
            idx = base_image_ids.index(image_id)
            detailed_images[idx].append(sign)
    return base_image_ids, detailed_images

def absolute_to_local_indices(detailed_images: Sequence[Sequence[str]]) -> List[List[int]]:
    """Tạo danh sách index tương ứng cho từng nhóm ảnh"""
    output_indices = []
    offset = 0
    for group in detailed_images:
        group_len = len(group)
        output_indices.append(list(range(offset, offset + group_len)))
        offset += group_len
    return output_indices

def reindex_placeholders(content: str, local_indices: Sequence[Sequence[int]]) -> str:
    """Thay thế <<IMAGE_i>> bằng chuỗi các index ảnh đã được cắt"""
    updated = content
    for i in reversed(range(len(local_indices))):
        replacement = "".join(f"<<IMAGE_{j}>>" for j in local_indices[i])
        updated = updated.replace(f"<<IMAGE_{i}>>", replacement)
    return updated

def encode_image_to_base64(path: Path) -> str:
    """Chuyển đổi file ảnh sang chuỗi base64"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def process_article(article: Dict, chunk_size: int = 10, sleep_sec: float = 1.0):
    """Xử lý phân tích biển báo theo từng lô (chunk) để tránh quá tải API"""
    article.setdefault("signs", [])
    if not article["signs"]:
        article["__is_sucessfully_parsing_sign"] = True
        article["detailed_signs"] = []
        return

    # Chuẩn bị dữ liệu chỉ số
    _, detailed_images = group_signs_by_image(article)
    absolute_indices_by_image = absolute_to_local_indices(detailed_images)
    num_cropped = len(article["signs"])
    parsed_response_all = []

    # Chia lô để gọi API
    for start in tqdm(range(0, num_cropped, chunk_size), leave=False, desc="Parsing chunks"):
        stop = min(start + chunk_size, num_cropped)
        allowed = list(range(start, stop))

        # Lọc index cục bộ cho lô hiện tại
        local_indices = [[j for j in idxs if j in allowed] for idxs in absolute_indices_by_image]
        content = reindex_placeholders(article.get("text", ""), local_indices)

        # Xây dựng prompt text
        prompt_text = (
            PARSE_SIGNS_PROMPT.replace("<<TITLE>>", article.get("title", "Không có tiêu đề"))
            .replace("<<CONTENT>>", content)
            .replace("<<NUM_SIGNS>>", str(len(allowed)))
            .replace("<<FROM_INDEX>>", str(allowed[0]))
            .replace("<<TO_INDEX>>", str(allowed[-1]))
        )

        # Chuẩn bị tin nhắn đa phương thức (Text + Nhiều ảnh)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        
        for idx in allowed:
            img_name = article["signs"][idx]
            img_path = EXTRACTED_SIGN_DIR / img_name
            if img_path.exists():
                b64_data = encode_image_to_base64(img_path)
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                })

        # Gọi VLM API
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=0.0
            )
            llm_text = response.choices[0].message.content
            parsed = safe_json_from_llm(llm_text)
            if isinstance(parsed, list):
                parsed_response_all.extend(parsed[: len(allowed)])
        except Exception as e:
            print(f"\n[ERROR] Lỗi gọi API tại lô {start}-{stop}: {e}")

        time.sleep(sleep_sec)

    # Tổng hợp và lưu kết quả
    is_ok = len(parsed_response_all) == len(article["signs"])
    detailed_signs = []
    
    if is_ok:
        for i, sign_parsed in enumerate(parsed_response_all):
            sp = deepcopy(sign_parsed)
            sp["image"] = article["signs"][i]
            detailed_signs.append(sp)
    
    article["__is_sucessfully_parsing_sign"] = is_ok
    article["detailed_signs"] = detailed_signs
    if not is_ok:
        article["__error"] = f"Parse mismatch: {len(parsed_response_all)}/{len(article['signs'])}"

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"[ERROR] Không tìm thấy file: {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        lawdb = json.load(f)

    print(f"[INFO] Bắt đầu phân tích {len(lawdb)} bộ luật...")
    
    for law in lawdb:
        law_id = law.get("id", "Unknown")
        articles = law.get("articles", [])
        print(f"\n[..] Đang xử lý bộ luật: {law_id}")

        for article in tqdm(articles, desc="Articles progress"):
            process_article(article)
            
            # Lưu liên tục để tránh mất dữ liệu nếu gặp sự cố
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(lawdb, f, ensure_ascii=False, indent=4)

    print(f"\n[SUCCESS] Hoàn tất! Dữ liệu đã được lưu tại: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()