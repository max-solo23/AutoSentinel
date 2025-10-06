import numpy as np
from custom_types import PlateBBox


def detect_plate_bbox(img: np.ndarray) -> tuple[PlateBBox, float]:
    h, w = img.shape[:2]
    x1 = w * 0.25
    y1 = h * 0.40
    x2 = w * 0.75
    y2 = h * 0.60
    bbox = PlateBBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return bbox, 0.50


def crop(img: np.ndarray, bbox: PlateBBox) -> np.ndarray:
    x1, y1, x2, y2 = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
    x1 = max(0, x1); y1 = max(0, y1)
    return img[y1:y2, x1:x2].copy()
