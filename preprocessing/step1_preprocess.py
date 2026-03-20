import json
import os
import re
import html2text
from config import INPUT_FILE, OUTPUT_FILE, DEBUG_DIR

def html_table_to_markdown(html):
    """Convert HTML table to Markdown and handle image links"""
    h = html2text.HTML2Text()
    h.ignore_links = True
    markdown = h.handle(html).replace("**", "")

    # Use regex to find images in the law file
    pattern = re.compile(
        r"!\[\]\(Quy%20chuẩn%20Việt%20Nam-QCVN%2041_2024-BGTVT\.fld/(image\d+\.(?:png|jpg))\)",
        re.DOTALL,
    )

    def replacer(match):
        return f"<<IMAGE: {match.group(1)} /IMAGE>>"

    return pattern.sub(replacer, markdown)

def replace_tables_with_markdown(text):
    """Find and process TABLE blocks in text and convert the content inside into Markdown"""
    # 1. Find all TABLE blocks in the text
    table_pattern = re.compile(r"<<TABLE:\s*(<table.*?</table>)\s*/TABLE>>", re.DOTALL)
    matches = table_pattern.findall(text)
    markdowns = [html_table_to_markdown(match) for match in matches]

    def table_replacer(match):
        html_table = match.group(1)
        markdown = html_table_to_markdown(html_table)
        return f"```markdown\n{markdown}\n```"

    new_text = table_pattern.sub(table_replacer, text)

    # 2. Process and mark the order of IMAGE to use for the next step of image cropping
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

    processed_law_data = []
    
    for law_entry in raw_data:
        law_id = law_entry["id"]
        articles = law_entry["articles"]
        
        for idx, article in enumerate(articles):
            raw_text = article["text"]
            matches, markdowns, images, processed_text = replace_tables_with_markdown(raw_text)
            
            # Update the article with the new images and text
            article["images"] = images
            article["num_images"] = len(images)
            article["text"] = processed_text
            
            # Save debug if needed to check the conversion result
            if len(matches) > 0:
                safe_law_id = re.sub(r'[\\/*?:"<>|]', "_", law_id)
                dirpath = os.path.join(DEBUG_DIR, f"{safe_law_id}_{idx}")
                os.makedirs(dirpath, exist_ok=True)
                with open(os.path.join(dirpath, "raw.html"), "w", encoding="utf-8") as f:
                    f.write(matches[0])
                with open(os.path.join(dirpath, "converted.md"), "w", encoding="utf-8") as f:
                    f.write(markdowns[0])

        processed_law_data.append(law_entry)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_law_data, f, ensure_ascii=False, indent=4)
        
    print(f"Processing complete! File saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_preprocess()