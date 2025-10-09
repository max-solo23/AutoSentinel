from io_tools import load_image_from_bytes
from preprocess import preprocess
from detector import detect_plate_bbox, crop
from ocr import read_text
from custom_types import PlateResult
import os


def run_pipeline(image_bytes: bytes) -> PlateResult:
    img = load_image_from_bytes(image_bytes)
    print(f"[pipeline] loaded image: shape={getattr(img, 'shape', None)}")

    save_dbg = os.getenv("AUTOSENTINEL_DEBUG", "0") == "1"
    pre = preprocess(img, save_debug=save_dbg)

    bbox, det_conf = detect_plate_bbox(pre if pre is not None else img)
    roi = crop(pre if pre is not None else img, bbox)

    if roi.size == 0:
        return PlateResult(status="not_found", plate_text="", confidence=0.0, bbox=bbox)

    import cv2, numpy as np
    ph, pw = roi.shape[:2]
    pad = int(0.10 * max(ph, pw))
    roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    h, w = roi.shape[:2]
    if h < 220:
        s = 220.0 / h
        roi = cv2.resize(roi, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    thr = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi

    text, ocr_conf = read_text(thr)
    if not text:
        text, ocr_conf = read_text(roi)

    conf = float(min(max(det_conf, 0.0), 1.0)*0.3 + min(max(ocr_conf, 0.0), 1.0)*0.7)

    return PlateResult(
        status="success" if text else "not_found",
        plate_text=text or "",
        confidence=round(conf, 3),
        bbox=bbox,
    )


