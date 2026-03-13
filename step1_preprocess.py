import json
import os
import re
import html2text

INPUT_FILE = "data/lawdb/vlsp_2025_law.json"
OUTPUT_FILE = "data/lawdb/lawdb_preprocessed.json"
DEBUG_DIR = "debug"

def html_table_to_markdown(html):
    """Chuyển đổi bảng HTML sang Markdown và xử lý link ảnh đặc thù"""
    # Khởi tạo converter
    h = html2text.HTML2Text()
    h.ignore_links = True
    markdown = h.handle(html).replace("**", "")

    # Regex để tìm các ảnh trong file luật QCVN 41:2024
    pattern = re.compile(
        r"!\[\]\(Quy%20chuẩn%20Việt%20Nam-QCVN%2041_2024-BGTVT\.fld/(image\d+\.(?:png|jpg))\)",
        re.DOTALL,
    )

    def replacer(match):
        return f"<<IMAGE: {match.group(1)} /IMAGE>>"

    return pattern.sub(replacer, markdown)

def replace_tables_with_markdown(text):
    """Tìm các khối TABLE trong text và convert nội dung bên trong sang Markdown"""
    # 1. Tìm và xử lý TABLE
    table_pattern = re.compile(r"<<TABLE:\s*(<table.*?</table>)\s*/TABLE>>", re.DOTALL)
    
    matches = table_pattern.findall(text)
    markdowns = [html_table_to_markdown(match) for match in matches]

    def table_replacer(match):
        html_table = match.group(1)
        markdown = html_table_to_markdown(html_table)
        return f"```markdown\n{markdown}\n```"

    new_text = table_pattern.sub(table_replacer, text)

    # 2. Xử lý và đánh dấu thứ tự IMAGE để dùng cho bước Cắt Ảnh sau này
    img_pattern = re.compile(r"<<IMAGE:\s*(image\d{2,5}\.(?:jpg|png))\s*/IMAGE>>", re.DOTALL)
    images = img_pattern.findall(new_text)

    def make_image_replacer():
        index = 0
        def replacer(match):
            nonlocal index
            result = f"<<IMAGE_{index}>>"
            index += 1
            return result
        return replacer

    new_text = img_pattern.sub(make_image_replacer(), new_text)

    return matches, markdowns, images, new_text

def run_preprocess():
    if not os.path.exists(INPUT_FILE):
        print(f"Lỗi: Không tìm thấy file {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Cấu trúc file gốc thường là một list các văn bản luật
    processed_law_data = []
    
    for law_entry in raw_data:
        law_id = law_entry["id"]
        articles = law_entry["articles"]
        
        for idx, article in enumerate(articles):
            raw_text = article["text"]
            
            # Thực hiện trích xuất logic như file gốc
            matches, markdowns, images, processed_text = replace_tables_with_markdown(raw_text)
            
            # Cập nhật thông tin vào article
            article["images"] = images
            article["num_images"] = len(images)
            article["text"] = processed_text
            
            # Lưu debug nếu cần để kiểm tra kết quả convert
            if len(matches) > 0:
                safe_law_id = re.sub(r'[\\/*?:"<>|]', "_", law_id)
                dirpath = os.path.join(DEBUG_DIR, f"{safe_law_id}_{idx}")
                os.makedirs(dirpath, exist_ok=True)
                with open(os.path.join(dirpath, "raw.html"), "w", encoding="utf-8") as f:
                    f.write(matches[0])
                with open(os.path.join(dirpath, "converted.md"), "w", encoding="utf-8") as f:
                    f.write(markdowns[0])

        processed_law_data.append(law_entry)

    # Lưu kết quả cuối cùng
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_law_data, f, ensure_ascii=False, indent=4)
        
    print(f"Xử lý hoàn tất! File đã lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_preprocess()