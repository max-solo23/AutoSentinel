from io_tools import load_image_from_bytes
from preprocess import preprocess
from detector import detect_plate_bbox, crop
from ocr import read_text
from custom_types import PlateResult, PlateBBox
import os


def run_pipeline(image_bytes: bytes) -> PlateResult:
    img = load_image_from_bytes(image_bytes)
    print(f"[pipeline] loaded image: shape={getattr(img, 'shape', None)}")

    save_dbg = os.getenv("AUTOSENTINEL_DEBUG", "0") == "1"
    pre = preprocess(img, save_debug=save_dbg)

    bbox, det_conf = detect_plate_bbox(pre if pre is not None else img)
    roi = crop(pre if pre is not None else img, bbox)

    text, ocr_conf = read_text(roi)

    conf = round(float(min(max(det_conf, 0.0), 1.0) * 0.3 + min(max(ocr_conf, 0.0), 1.0) * 0.7), 3)

    return PlateResult(
        status="success" if text else "not_found",
        plate_text=text,
        confidence=round(conf, 3),
        bbox=bbox
    )

