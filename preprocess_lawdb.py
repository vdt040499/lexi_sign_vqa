import pandas as pd
import json
import random
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

import base64

import os
import time
from io import BytesIO

import html2text
import re
from bs4 import BeautifulSoup

def save_txt(text: str, path: str):
    with open(path, 'w') as f:
        f.write(text)

def save_json(data: dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)
    
def load_law_data(path: str):
    raw_law_data = load_json(path)
    return {e["id"]: e["articles"]  for e in raw_law_data}

def save_law_data(law_data: dict, path: str):
    raw_law_data = [{"id": k, "articles": v} for k, v in law_data.items()]
    save_json(raw_law_data, path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def replace_tables_with_markdown(text):
    pattern = re.compile(r"<<TABLE:\s*(<table.*?</table>)\s*/TABLE>>", re.DOTALL)

    def table_replacer(match):
        html_table = match.group(1)
        markdown = html_table_to_markdown(html_table)
        return f"""```markdown
{markdown}
```
"""

    matches = pattern.findall(text)
    markdowns = [html_table_to_markdown(match) for match in matches]

    new_text = pattern.sub(table_replacer, text)

    pattern = re.compile(r"<<IMAGE:\s*(image\d{2,5}\.jpg)\s*/IMAGE>>", re.DOTALL)
    images = pattern.findall(new_text)

    def make_image_replacer():
        index = 0

        def replacer(match):
            nonlocal index
            result = f"<<IMAGE_{index}>>"
            index += 1
            return result

        return replacer

    image_replacer = make_image_replacer()
    new_text = pattern.sub(image_replacer, new_text)

    return matches, markdowns, images, new_text

def html_table_to_markdown(html):
    markdown = html2text.html2text(html).replace("**", "")

    pattern = re.compile(
        r"!\[\]\(Quy%20chuẩn%20Việt%20Nam-QCVN%2041_2024-BGTVT\.fld/(image\d+\.(?:png|jpg))\)",
        re.DOTALL,
    )

    def replacer(match):
        return f"<<IMAGE: {match.group(1)} /IMAGE>>"

    return pattern.sub(replacer, markdown)

law_data = load_law_data("./data/lawdb/vlsp_2025_law.json")

i = 0

debug_dir = "debug"

for k, v in law_data.items():
    for idx, e in enumerate(v):
        text: str = e["text"]
        matches, markdowns, images, new_text = replace_tables_with_markdown(text)
        count = len(matches)
        for match, markdown in zip(matches, markdowns):
            dirpath = f"{debug_dir}/{i}"
            os.makedirs(dirpath, exist_ok=True)
            save_txt(match, os.path.join(dirpath, f"file.html"))
            save_txt(markdown, os.path.join(dirpath, f"file.markdown"))
            i+=1

        e["images"] = images
        e["num_images"] = len(images)

        e["text"] = new_text

processed_law_data = [
    {"id": law_id, "articles": articles} for law_id, articles in law_data.items()
]
save_json(processed_law_data, './data/preprocessed/preprocessed_vlsp_2025_law.json')
        