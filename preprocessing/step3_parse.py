import os
import re
import json
import time
import base64
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Sequence, List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from openai import OpenAI
from config import INPUT_PARSE_JSON, OUTPUT_PARSE_JSON, EXTRACTED_SIGN_PATH, OLLAMA_API_KEY, BASE_URL, PARSE_MODEL_ID

INPUT_JSON = INPUT_PARSE_JSON
OUTPUT_JSON = OUTPUT_PARSE_JSON

EXTRACTED_SIGN_DIR = Path(EXTRACTED_SIGN_PATH)

client = OpenAI(api_key=OLLAMA_API_KEY, base_url=BASE_URL)

PARSE_SIGNS_PROMPT = """
Bạn là chuyên gia luật giao thông. Điều luật: <<TITLE>>.
Nội dung: <<CONTENT>>
Nhiệm vụ: phân tích đúng <<NUM_SIGNS>> hình ảnh biển báo (ảnh thứ <<FROM_INDEX>> đến <<TO_INDEX>>) và trả về DUY NHẤT một mảng JSON, mỗi phần tử có "name" và "description".

QUY TẮC BẮT BUỘC:
- Đầu ra phải là ĐÚNG MỘT mảng JSON, bắt đầu bằng [ và kết thúc bằng ].
- Không được thêm bất kỳ câu chữ, giải thích, markdown hay text nào trước hoặc sau mảng JSON.
- Định dạng: [{"name": "Tên biển báo", "description": "Mô tả ngắn gọn"}]
- Luôn trả về đúng số phần tử = <<NUM_SIGNS>> theo đúng thứ tự ảnh đã gửi.
- Nếu không chắc chắn, vẫn phải điền "name" và "description" ngắn gọn dựa trên quan sát, không được trả lời dạng liệt kê/markdown.
- "name" và "description" PHẢI viết bằng tiếng Việt. Nếu bạn nghĩ ra cụm từ tiếng Anh, hãy dịch sang tiếng Việt trước khi trả về JSON.

Ví dụ đầu ra hợp lệ:
[{"name": "Biển cấm", "description": "Cấm đi ngược chiều"}, {"name": "Biển chỉ dẫn", "description": "Chỉ hướng đi"}]

Output ONLY the JSON array above, no other text, no explanation.
"""

def safe_json_from_llm(text: str) -> list:
    """
    Parse JSON array from LLM output. Strips code fences, then tries to extract
    a [...] array if the response contains prose before/after.
    """
    if not text or not text.strip():
        raise ValueError("Empty response")
    raw = text.strip()

    # Prefer fenced JSON if present anywhere in the response.
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    cleaned = m.group(1).strip() if m else raw

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    # If the model wraps JSON with text: find the first [...] array (from [ to ] match parentheses)
    if parsed is None:
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
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, list):
        raise ValueError("Parsed JSON is not an array")
    return parsed

def _build_repair_messages(previous_text: str) -> List[Dict[str, Any]]:
    repair_prompt = (
        "Chuyển câu trả lời trước đó thành DUY NHẤT một mảng JSON hợp lệ.\n"
        "QUY TẮC:\n"
        "- Chỉ in ra JSON array (bắt đầu [ và kết thúc ]), không thêm chữ.\n"
        "- Mỗi phần tử có đúng 2 field: name, description.\n"
        "- name và description phải viết bằng tiếng Việt (dịch từ tiếng Anh nếu cần).\n"
        "Trả về ngay."
    )
    return [
        {"role": "assistant", "content": previous_text},
        {"role": "user", "content": repair_prompt},
    ]

def group_signs_by_image(article: Dict) -> Tuple[List[str], List[List[str]]]:
    """Group the cropped signs by the image ID in the original law article"""
    base_image_ids = [p.split(".jpg")[0] for p in article.get("images", [])]
    detailed_images = [[] for _ in base_image_ids]
    for sign in article.get("signs", []):
        image_id = sign.split("_crop")[0]
        if image_id in base_image_ids:
            idx = base_image_ids.index(image_id)
            detailed_images[idx].append(sign)
    return base_image_ids, detailed_images

def absolute_to_local_indices(detailed_images: Sequence[Sequence[str]]) -> List[List[int]]:
    """Create a list of indices corresponding to each image group"""
    output_indices = []
    offset = 0
    for group in detailed_images:
        group_len = len(group)
        output_indices.append(list(range(offset, offset + group_len)))
        offset += group_len
    return output_indices

def reindex_placeholders(content: str, local_indices: Sequence[Sequence[int]]) -> str:
    """Replace <<IMAGE_i>> with the string of the cropped image indices"""
    updated = content
    for i in reversed(range(len(local_indices))):
        replacement = "".join(f"<<IMAGE_{j}>>" for j in local_indices[i])
        updated = updated.replace(f"<<IMAGE_{i}>>", replacement)
    return updated

def encode_image_to_base64(path: Path) -> str:
    """Convert the image file to a base64 string"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _call_vlm_json(messages: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Gọi VLM và trả về list đã parse, hoặc None nếu không parse được JSON."""
    llm_text = None
    parsed: List | None = None
    response = client.chat.completions.create(
        model=PARSE_MODEL_ID,
        messages=messages,
        temperature=0.0,
    )
    llm_text = response.choices[0].message.content or ""
    try:
        parsed = safe_json_from_llm(llm_text)
    except Exception:
        parsed = None
    if parsed is None and llm_text.strip():
        repair_messages = deepcopy(messages)
        repair_messages.extend(_build_repair_messages(llm_text))
        response2 = client.chat.completions.create(
            model=PARSE_MODEL_ID,
            messages=repair_messages,
            temperature=0.0,
        )
        llm_text2 = response2.choices[0].message.content or ""
        try:
            parsed = safe_json_from_llm(llm_text2)
        except Exception:
            parsed = None
    if parsed is None:
        return None
    out: List[Dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict) and "name" in item and "description" in item:
            out.append({"name": str(item["name"]), "description": str(item["description"])})
    return out

def _build_chunk_messages(article: Dict, allowed: List[int], absolute_indices_by_image: List[List[int]]) -> List[Dict[str, Any]]:
    local_indices = [[j for j in idxs if j in allowed] for idxs in absolute_indices_by_image]
    content = reindex_placeholders(article.get("text", ""), local_indices)
    prompt_text = (
        PARSE_SIGNS_PROMPT.replace("<<TITLE>>", article.get("title", "Không có tiêu đề"))
        .replace("<<CONTENT>>", content)
        .replace("<<NUM_SIGNS>>", str(len(allowed)))
        .replace("<<FROM_INDEX>>", str(allowed[0]))
        .replace("<<TO_INDEX>>", str(allowed[-1]))
    )
    messages: List[Dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    for idx in allowed:
        img_name = article["signs"][idx]
        img_path = EXTRACTED_SIGN_DIR / img_name
        if img_path.exists():
            b64_data = encode_image_to_base64(img_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
            })
    return messages

def _parse_signs_one_by_one(
    article: Dict,
    indices: List[int],
    absolute_indices_by_image: List[List[int]],
    sleep_sec: float,
) -> List[Dict[str, Any]]:
    """Gọi VLM từng ảnh (chậm nhưng ổn định khi batch thiếu/sai số phần tử)."""
    results: List[Dict[str, Any]] = []
    for idx in indices:
        allowed = [idx]
        messages = _build_chunk_messages(article, allowed, absolute_indices_by_image)
        row = None
        try:
            parsed = _call_vlm_json(messages)
            if parsed and len(parsed) >= 1:
                row = parsed[0]
        except Exception as e:
            print(f"\n[ERROR] single-sign idx={idx}: {e}")
        if row is None:
            row = {
                "name": article["signs"][idx],
                "description": "Không phân tích được biển (model lỗi hoặc thiếu ảnh).",
            }
        results.append(row)
        time.sleep(sleep_sec)
    return results

def _parse_chunk_with_fallback(
    article: Dict,
    allowed: List[int],
    absolute_indices_by_image: List[List[int]],
    sleep_sec: float,
    chunk_retries: int,
    single_fallback: bool,
) -> List[Dict[str, Any]]:
    """
    Parse một chunk: retry batch nếu JSON thiếu phần tử; nếu vẫn sai thì (tuỳ chọn) gọi từng ảnh.
    Chỉ chấp nhận batch khi len(parsed) == len(allowed) (tránh lệch thứ tự khi model trả ít hơn).
    """
    messages = _build_chunk_messages(article, allowed, absolute_indices_by_image)
    for attempt in range(max(1, chunk_retries + 1)):
        try:
            parsed = _call_vlm_json(messages)
        except Exception as e:
            print(f"\n[ERROR] chunk {allowed[0]}-{allowed[-1]} attempt {attempt + 1}: {e}")
            parsed = None
            time.sleep(sleep_sec)
            continue
        if parsed is not None and len(parsed) == len(allowed):
            return parsed[: len(allowed)]
        time.sleep(sleep_sec)
    if single_fallback:
        print(
            f"\n[WARN] Chunk {allowed[0]}-{allowed[-1]}: batch không đủ {len(allowed)} phần tử, "
            f"chuyển sang gọi từng ảnh ({len(allowed)} lần)."
        )
        return _parse_signs_one_by_one(article, allowed, absolute_indices_by_image, sleep_sec)
    print(f"\n[WARN] Chunk {allowed[0]}-{allowed[-1]}: thiếu phần tử sau retry, bỏ qua chunk (không dùng single-fallback).")
    return []

def process_article(
    article: Dict,
    chunk_size: int = 2,
    sleep_sec: float = 1.0,
    chunk_retries: int = 2,
    single_fallback: bool = True,
):
    """Process the parsing of signs by chunks to avoid overloading the API"""
    article.setdefault("signs", [])
    if not article["signs"]:
        article["__is_sucessfully_parsing_sign"] = True
        article["detailed_signs"] = []
        return

    # Prepare the data indices
    _, detailed_images = group_signs_by_image(article)
    absolute_indices_by_image = absolute_to_local_indices(detailed_images)
    num_cropped = len(article["signs"])
    parsed_response_all = []

    # Divide into chunks to call the API
    for start in tqdm(
        range(0, num_cropped, chunk_size),
        leave=False,
        desc=f"Chunks (size={chunk_size}, {num_cropped} signs)",
    ):
        stop = min(start + chunk_size, num_cropped)
        allowed = list(range(start, stop))

        try:
            chunk_rows = _parse_chunk_with_fallback(
                article,
                allowed,
                absolute_indices_by_image,
                sleep_sec=sleep_sec,
                chunk_retries=chunk_retries,
                single_fallback=single_fallback,
            )
            parsed_response_all.extend(chunk_rows)
        except Exception as e:
            print(f"\n[ERROR] Error calling API at chunk {start}-{stop}: {e}")

        time.sleep(sleep_sec)

    # Summarize and save the result
    is_ok = len(parsed_response_all) == len(article["signs"])
    detailed_signs = []
    
    if is_ok:
        for i, sign_parsed in enumerate(parsed_response_all):
            sp = deepcopy(sign_parsed)
            sp["image"] = article["signs"][i]
            detailed_signs.append(sp)
    
    article["__is_sucessfully_parsing_sign"] = is_ok
    article["detailed_signs"] = detailed_signs
    if is_ok:
        article.pop("__error", None)
    else:
        article["__error"] = f"Parse mismatch: {len(parsed_response_all)}/{len(article['signs'])}"

def main():
    parser = argparse.ArgumentParser(description="Parse traffic signs descriptions using VLM.")
    parser.add_argument(
        "--law-ids",
        nargs="*",
        help='Chỉ xử lý các văn bản luật có trường "id" khớp (vd: --law-ids "QCVN 41:2024/BGTVT"). Có thể kết hợp với --article-ids.',
    )
    parser.add_argument(
        "--article-ids",
        nargs="*",
        help='Chỉ xử lý các điều có "id" khớp (vd: --article-ids 47). Nếu kèm --law-ids thì phải khớp cả hai.',
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Maximum number of signs in each VLM call (default: 10). Increase if the model is stable, decrease if encountering errors/timeouts with articles having many cropped images.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=1.0,
        help="Time to sleep (seconds) between VLM calls.",
    )
    parser.add_argument(
        "--chunk-retries",
        type=int,
        default=2,
        help="Số lần thử lại mỗi chunk khi JSON thiếu/sai số phần tử (mặc định: 2).",
    )
    parser.add_argument(
        "--no-single-fallback",
        action="store_true",
        help="Tắt fallback gọi VLM từng ảnh khi chunk batch thất bại (nhanh hơn nhưng dễ Parse mismatch).",
    )
    args = parser.parse_args()

    # Ưu tiên resume từ OUTPUT_JSON nếu đã tồn tại, nếu không sẽ đọc từ INPUT_JSON gốc
    load_path = OUTPUT_JSON if os.path.exists(OUTPUT_JSON) else INPUT_JSON
    if not os.path.exists(load_path):
        print(f"[ERROR] File not found: {load_path}")
        return

    with open(load_path, "r", encoding="utf-8") as f:
        lawdb = json.load(f)

    print(f"[INFO] Starting to parse {len(lawdb)} laws from: {load_path}")
    target_law_ids = {str(x) for x in args.law_ids} if args.law_ids else None
    target_article_ids = {str(x) for x in args.article_ids} if args.article_ids else None
    single_fallback = not args.no_single_fallback

    for law_idx, law in enumerate(lawdb, start=1):
        law_id = str(law.get("id", "Unknown"))

        if target_law_ids is not None and law_id not in target_law_ids:
            continue

        articles = law.get("articles", [])
        print(f"\n[..] Processing law {law_id} ({law_idx}/{len(lawdb)}) - number of articles: {len(articles)}")

        for art_idx, article in enumerate(
            tqdm(articles, desc=f"Articles of law {law_id}"), start=1
        ):
            art_id = article.get("id", f"{art_idx}")
            num_signs = len(article.get("signs", []))
            print(f"[..]   -> Article {art_id} ({art_idx}/{len(articles)}), number of signs: {num_signs}")

            if target_article_ids is not None and str(art_id) not in target_article_ids:
                continue

            # If the article has already been parsed successfully in a previous run, skip it
            if article.get("__is_sucessfully_parsing_sign"):
                continue

            num_chunks = (num_signs + args.chunk_size - 1) // args.chunk_size
            print(f"[..] chunk_size={args.chunk_size}, số lô gọi VLM: {num_chunks}")
            process_article(
                article,
                chunk_size=args.chunk_size,
                sleep_sec=args.sleep_sec,
                chunk_retries=args.chunk_retries,
                single_fallback=single_fallback,
            )
            
            # Save incrementally to avoid losing data if there is an error
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(lawdb, f, ensure_ascii=False, indent=4)

    print(f"\n[SUCCESS] Completed! Data saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()