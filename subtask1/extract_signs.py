from pathlib import Path
from PIL import Image

import supervision as sv
from ultralytics import YOLOE

from subtask1.utils_yoloe import remove_tiny_boxes, remove_duplicated_boxes
from subtask1.config import YOLOE_MODEL, YOLOE_CLASS_NAMES, YOLOE_SCORE_THRESHOLD


_model_cache: dict = {}


def _get_model(device: str = "cpu") -> YOLOE:
    if device not in _model_cache:
        model = YOLOE(model=YOLOE_MODEL).to(device)
        model.set_classes(YOLOE_CLASS_NAMES, model.get_text_pe(YOLOE_CLASS_NAMES))
        model.to(device).eval()
        _model_cache[device] = model
    return _model_cache[device]


def crop_signs_from_image(
    image_path: Path,
    device: str = "cpu",
) -> list[Image.Image]:
    """
    Detect and crop traffic signs from a query image using YoloE.
    Falls back to the full image if no signs are detected.
    """
    image = Image.open(str(image_path)).convert("RGB")
    model = _get_model(device)

    results = model(str(image_path), conf=YOLOE_SCORE_THRESHOLD, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    boxes       = detections.xyxy
    confidences = detections.confidence
    class_names = detections.data.get("class_name", [])

    if len(boxes) == 0:
        return [image]

    boxes, confidences, class_names = remove_tiny_boxes(
        boxes, confidences, class_names, image.size, thres=0.05
    )
    boxes, confidences, class_names = remove_duplicated_boxes(
        boxes, confidences, class_names, iou_thres=0.8
    )

    if len(boxes) == 0:
        return [image]

    return [image.crop(box) for box in boxes.tolist()]
