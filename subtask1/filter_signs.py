import ast
import base64
import re
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image

from subtask1.config import OLLAMA_BASE_URL, OLLAMA_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


SIGN_FILTER_PROMPT = """
You are an expert in legal question answering, specializing in traffic and road-related regulations.

You will be provided with:
1. A question related to traffic or road regulations.
2. An original input image.
3. <<NUM_SIGNS>> detected sign(s) from the given image.

Your task:
1. Read the question carefully and determine whether it refers to a sign at a specific position. Explicitly extract the referenced position(s) if present (e.g., left/right/top/bottom/center; near/far; overhead/ahead/behind). If none, state "No explicit position".
2. From the question, identify the expected visual characteristics of the referenced sign, including (if mentioned or implied) its shape, color(s), symbol(s), icon(s), or other notable features. If none, state "No explicit visual features". IMPORTANT: If the question mentions multiple descriptive features (e.g., color AND shape AND icon), then ALL of them must be satisfied simultaneously (logical AND), NOT just one (NOT logical OR).
IMPORTANT COLOR RULE:
In Vietnamese, the phrase "màu xanh" MUST be interpreted as "blue".
Example: If the question says "biển màu xanh", a blue-colored sign will satisfy this condition.
3. Give a concise but complete description of the entire original image, noting road layout, vehicles, pedestrians, environment, and the positions of all detected signs (left/right/top/bottom/center).
4. For EACH detected sign (in the given order), decide whether it is related to the question.
5. When deciding, pay close attention to:
    + The position of the sign in the original image relative to the viewpoint and to any position(s) referenced by the question.
    + Whether the question explicitly or implicitly refers to a specific location or direction (e.g., "sign on the right", "overhead sign").
    + The visual appearance and meaning of the sign.
    + If the question is about a prohibitory sign, also consider any supplementary sign(s) immediately below it that is in rectangle shape, contain text or, as they may modify the prohibition's scope.
6. For EACH sign, explain your reasoning clearly and briefly, including position relevance if applicable.
7. Provide the FINAL decision as a Python-style list of boolean values (True/False), where:
    + EACH ELEMENT MUST CORRESPOND EXACTLY to the matching detected sign in the SAME ORDER they were provided.
    + The length of the list MUST equal <<NUM_SIGNS>>.
8. STRICT REQUIREMENT: The final boolean list MUST contain at least one True value. If no detected sign clearly matches the question, then choose the single most prominent/main sign in the image (e.g., the largest or most central sign) and mark it as True.
9. Enclose ONLY the final boolean list between <<ANSWER>> and <</ANSWER>> tags, with nothing else inside.

Output format:
Question-referenced position(s): ...
Question-referenced visual features: ...
Full image description: ...
Explanation for sign 1: ...
Explanation for sign 2: ...
...
<<ANSWER>>[
    <True/False answer for sign 1>,
    ...
    <True/False answer for sign <<NUM_SIGNS>>>
  ]<</ANSWER>>

IMPORTANT NOTES:
- 4-wheeled vehicles may include car, truck, van, bus, jeep, ...
- 3-wheeled vehicles may include tricycle, auto-rickshaw, cycle rickshaw, ...
- The question may ask about multiple signs, not only one sign.
- "màu xanh" = blue in Vietnamese
- "phương tiện"/"loại xe" = all pedestrians, bicycles, cars, trucks, motorbikes, auto-rickshaws, ...
"""


def _encode_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _normalize_bool_tokens(text: str) -> str:
    text = re.sub(r"\btrue\b",  "True",  text, flags=re.IGNORECASE)
    text = re.sub(r"\bfalse\b", "False", text, flags=re.IGNORECASE)
    return text


def _extract_final_answer(response_text: str) -> list[bool] | None:
    m = re.search(r"<<ANSWER>>(.*?)<?</ANSWER>>", response_text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None

    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", m.group(1).strip(), flags=re.IGNORECASE)
    normalized = _normalize_bool_tokens(raw)

    try:
        parsed = ast.literal_eval(normalized if normalized.startswith("[") else f"[{normalized}]")
        if isinstance(parsed, list) and all(isinstance(x, bool) for x in parsed):
            return parsed
    except Exception:
        pass

    parts = [p.strip() for p in normalized.strip("[]").split(",") if p.strip()]
    if parts and all(p in ("True", "False") for p in parts):
        return [p == "True" for p in parts]

    return None


def filter_signs(
    sample: dict,
    image_dir: Path,
    sign_dir: Path,
    client: OpenAI,
    batch_size: int = 10,
) -> dict:
    """
    For each sample, call the LLM to filter which detected signs are relevant
    to the question. Updates each sign's 'is_chosen' field in-place.
    """
    print(f"[DEBUG] Filtering signs for sample: {sample}")
    detected_signs = sample.get("detected_signs", [])

    if not detected_signs:
        return sample

    # Single sign → always chosen
    if len(detected_signs) == 1:
        detected_signs[0]["is_chosen"] = True
        return sample

    image_path = image_dir / f"{sample['image_id']}.jpg"
    original_image = Image.open(image_path).convert("RGB")
    question = sample["question"]

    chosen: list[bool] = []

    for i in range(0, len(detected_signs), batch_size):
        batch = detected_signs[i : i + batch_size]

        prompt_text = SIGN_FILTER_PROMPT.replace("<<NUM_SIGNS>>", str(len(batch)))

        content: list[dict] = [
            {"type": "text", "text": prompt_text + f"\nQuestion: {question}\n"},
            {"type": "text", "text": "Original Image:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode_image(original_image)}"},
            },
            {"type": "text", "text": "Detected Signs (keep this order):"},
        ]

        for j, sign_info in enumerate(batch, start=1):
            crop = Image.open(sign_dir / sign_info["image_name"]).convert("RGB")
            content += [
                {"type": "text", "text": f"<<<SIGN {j}>>>"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image(crop)}"}},
                {"type": "text", "text": f"<<<END SIGN {j}>>>"},
            ]

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            llm_text = response.choices[0].message.content
            print(f"[DEBUG] LLM raw response:\n{llm_text}\n{'─'*60}")
            parsed = _extract_final_answer(llm_text)
            print(f"[DEBUG] Parsed answer: {parsed}")
        except Exception as e:
            print(f"[WARN] LLM filter error: {e}")
            parsed = None

        if isinstance(parsed, list):
            needed = len(batch)
            parsed = (parsed + [False] * needed)[:needed]
            chosen.extend(parsed)
        else:
            chosen.extend([False] * len(batch))

    for idx, sign_info in enumerate(detected_signs):
        sign_info["is_chosen"] = chosen[idx] if idx < len(chosen) else False

    return sample


def build_client() -> OpenAI:
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
