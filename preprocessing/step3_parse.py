import os
import re
import json
import time
import base64
import argparse
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
MODEL_ID = "gemma3:12b"

# Khởi tạo client
client = OpenAI(api_key=OLLAMA_API_KEY, base_url=BASE_URL)

# Prompt template gốc từ dự án
PARSE_SIGNS_PROMPT = """
Bạn là chuyên gia luật giao thông. Điều luật: <<TITLE>>.
Nội dung: <<CONTENT>>
Nhiệm vụ: phân tích đúng <<NUM_SIGNS>> hình ảnh biển báo (ảnh thứ <<FROM_INDEX>> đến <<TO_INDEX>>) và trả về DUY NHẤT một mảng JSON, mỗi phần tử có "name" và "description".

QUY TẮC BẮT BUỘC:
- Đầu ra phải là ĐÚNG MỘT mảng JSON, bắt đầu bằng [ và kết thúc bằng ].
- Không được thêm bất kỳ câu chữ, giải thích, markdown hay text nào trước hoặc sau mảng JSON.
- Định dạng: [{"name": "Tên biển báo", "description": "Mô tả ngắn gọn"}]

Ví dụ đầu ra hợp lệ:
[{"name": "Biển cấm", "description": "Cấm đi ngược chiều"}, {"name": "Biển chỉ dẫn", "description": "Chỉ hướng đi"}]

Output ONLY the JSON array above, no other text, no explanation.
"""

def safe_json_from_llm(text: str):
    """
    Parse JSON array from LLM output. Strips code fences, then tries to extract
    a [...] array if the response contains prose before/after.
    """
    if not text or not text.strip():
        raise ValueError("Empty response")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    # Thử parse trực tiếp
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Nếu model bọc JSON bằng văn bản: tìm mảng [...] đầu tiên (từ [ đến ] khớp ngoặc)
    start = cleaned.find("[")
    if start == -1:
        raise ValueError("No JSON array found in response")
    depth = 0
    end = -1
    for i in range(start, len(cleaned)):
        if cleaned[i] == "[":
            depth += 1
        elif cleaned[i] == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise ValueError("Unclosed JSON array in response")
    return json.loads(cleaned[start : end + 1])

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

def process_article(article: Dict, chunk_size: int = 2, sleep_sec: float = 1.0):
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
    for start in tqdm(
        range(0, num_cropped, chunk_size),
        leave=False,
        desc=f"Chunks (size={chunk_size}, {num_cropped} signs)",
    ):
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
            print(f"LLM response: {llm_text}")
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
    parser = argparse.ArgumentParser(description="Parse traffic signs descriptions using VLM.")
    parser.add_argument(
        "--law-ids",
        nargs="*",
        help="(Deprecated) Trước đây dùng để lọc theo id, hiện được dùng như alias cho --article-ids.",
    )
    parser.add_argument(
        "--article-ids",
        nargs="*",
        help="Chỉ xử lý các điều luật (article) có id trong danh sách này (ví dụ: --article-ids 46 47). Nếu bỏ trống sẽ xử lý tất cả.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Số lượng biển báo tối đa trong mỗi lần gọi VLM (mặc định: 3). Tăng lên nếu model ổn định, giảm nếu gặp lỗi/timeout với điều luật có nhiều ảnh crop.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=1.0,
        help="Thời gian nghỉ (giây) giữa các lần gọi VLM.",
    )
    args = parser.parse_args()

    # Ưu tiên resume từ OUTPUT_JSON nếu đã tồn tại, nếu không sẽ đọc từ INPUT_JSON gốc
    load_path = OUTPUT_JSON if os.path.exists(OUTPUT_JSON) else INPUT_JSON
    if not os.path.exists(load_path):
        print(f"[ERROR] Không tìm thấy file: {load_path}")
        return

    with open(load_path, "r", encoding="utf-8") as f:
        lawdb = json.load(f)

    print(f"[INFO] Bắt đầu phân tích {len(lawdb)} bộ luật từ: {load_path}")
    # Gộp cả law-ids (cũ) và article-ids (mới) để dùng như danh sách id điều luật cần xử lý
    target_article_ids = set()
    if args.article_ids:
        target_article_ids.update(str(x) for x in args.article_ids)
    if args.law_ids:
        target_article_ids.update(str(x) for x in args.law_ids)
    if not target_article_ids:
        target_article_ids = None
    
    for law_idx, law in enumerate(lawdb, start=1):
        law_id = str(law.get("id", "Unknown"))

        articles = law.get("articles", [])
        print(f"\n[..] Đang xử lý bộ luật {law_id} ({law_idx}/{len(lawdb)}) - số điều luật: {len(articles)}")

        for art_idx, article in enumerate(
            tqdm(articles, desc=f"Articles of law {law_id}"), start=1
        ):
            art_id = article.get("id", f"{art_idx}")
            num_signs = len(article.get("signs", []))
            print(f"[..]   -> Article {art_id} ({art_idx}/{len(articles)}), số biển báo: {num_signs}")

            # Nếu người dùng chỉ định danh sách article-ids thì bỏ qua các điều luật khác
            if target_article_ids and str(art_id) not in target_article_ids:
                continue

            # Nếu đã parse thành công ở lần chạy trước thì bỏ qua
            if article.get("__is_sucessfully_parsing_sign"):
                continue

            num_chunks = (num_signs + args.chunk_size - 1) // args.chunk_size
            print(f"[..] chunk_size={args.chunk_size}, số lô gọi VLM: {num_chunks}")
            process_article(article, chunk_size=args.chunk_size, sleep_sec=args.sleep_sec)
            
            # Lưu liên tục để tránh mất dữ liệu nếu gặp sự cố
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(lawdb, f, ensure_ascii=False, indent=4)

    print(f"\n[SUCCESS] Hoàn tất! Dữ liệu đã được lưu tại: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()