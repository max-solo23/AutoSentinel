from io_tools import load_image_from_bytes
from preprocess import preprocess
from detector import detect_plate_bbox, crop
from ocr import read_text
from custom_types import PlateResult, PlateBBox


def run_pipeline(image_bytes: bytes) -> PlateResult:
    img = load_image_from_bytes(image_bytes)
    print(f"[pipeline] loaded image: shape={getattr(img, 'shape', None)}")
    pre = preprocess(img)
    bbox, det_conf = detect_plate_bbox(pre)
    roi = crop(pre, bbox)
    text, ocr_conf = read_text(roi)

    conf = round(float(min(max(det_conf, 0.0), 1.0) * 0.3 + min(max(ocr_conf, 0.0), 1.0) * 0.7), 3)

    return PlateResult(
        status="success" if text else "not_found",
        plate_text=text,
        confidence=conf,
        bbox=bbox
    )

