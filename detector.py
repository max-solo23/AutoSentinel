import os
import numpy as np
from custom_types import PlateBBox
import cv2


# Step 3: YOLOv8-based detector with graceful fallback to stub.
USE_STUB = False
MODEL = None
_MODEL_PATH = os.getenv("AUTOSENTINEL_YOLO", os.path.join("models", "yolo_plate.pt"))


try:
    from ultralytics import YOLO
    if os.path.exists(_MODEL_PATH):
        MODEL = YOLO(_MODEL_PATH)
        print(f"[detector] YOLO model loaded: {_MODEL_PATH}")
    else:
        print(f"[detector] Model not found at '{_MODEL_PATH}'. Using stub detector.")
        USE_STUB = True
except Exception as e: # ultralytics not installed or GPU/onnx issues
    print(f"[detector] YOLO load failed: {e}. Using stub detector.")
    USE_STUB = True


def _ensure_3ch(img: np.ndarray) -> np.ndarray:
    # YOLO expects 3 channels. If single-channel, replicate.
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3]

    return np.ascontiguousarray(img)


def _stub_bbox(img: np.ndarray) -> tuple[PlateBBox, float]:
    h, w = img.shape[:2]
    x1 = w * 0.25
    y1 = h * 0.40
    x2 = w * 0.75
    y2 = h * 0.60
    bbox = PlateBBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return bbox, 0.50


def detect_plate_bbox(img: np.ndarray) -> tuple[PlateBBox, float]:
    if USE_STUB or MODEL is None:
        return _stub_bbox(img)

    rgb = _ensure_3ch(img)
    try:
        # conf=0.25 default NMS. imgsz 640 works well for plates.
        res = MODEL.predict(source=rgb, conf=0.25, imgsz=640, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return _stub_bbox(img)
        boxes_xyxy = res[0].boxes.xyxy.cpu().numpy()
        confs = res[0].boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes_xyxy[idx]
        conf = float(confs[idx])
        return PlateBBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)), conf
    except Exception as e:
        print(f"[detector] YOLO inference error: {e}. Falling back to stub.")
        return _stub_bbox(img)


def crop(img: np.ndarray, bbox: PlateBBox) -> np.ndarray:
    x1, y1, x2, y2 = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
    x1 = max(0, x1); y1 = max(0, y1)
    return img[y1:y2, x1:x2].copy()


def draw_bbox(img: np.ndarray, bbox: PlateBBox, color=(0, 255, 0), thickness=2) -> np.ndarray:
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis
