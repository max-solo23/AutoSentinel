from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from custom_types import PlateBBox


@dataclass(slots=True)
class DetectorConfig:
    """Configuration parameters for the YOLO detector."""

    model_path: str = os.path.join("models", "yolo_plate.pt")
    confidence_threshold: float = 0.25
    image_size: int = 640


class PlateDetector:
    """Wrapper around a YOLOv8 model with a deterministic stub fallback."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._model = None
        self._model_available = False

    def detect(self, image: np.ndarray) -> Tuple[PlateBBox, float]:
        """
        Detect a single licence plate bounding box.

        Falls back to a deterministic stub if the model cannot be loaded or yields no detections.
        """
        model = self._ensure_model()
        if model is None:
            return self._stub_bbox(image)

        rgb = _prepare_for_model(image)
        try:
            results = model.predict(
                source=rgb,
                conf=self.config.confidence_threshold,
                imgsz=self.config.image_size,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[detector] YOLO inference error: {exc}. Falling back to stub.")
            return self._stub_bbox(image)

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return self._stub_bbox(image)

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        index = int(np.argmax(confidences))
        x1, y1, x2, y2 = boxes_xyxy[index]
        conf = float(confidences[index])
        return PlateBBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)), conf

    def crop(self, image: np.ndarray, bbox: PlateBBox) -> np.ndarray:
        """Crop the detected bounding box from the source image."""
        x1, y1, x2, y2 = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, *image.shape[2:]), dtype=image.dtype)
        x1 = max(0, x1)
        y1 = max(0, y1)
        return image[y1:y2, x1:x2].copy()

    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: PlateBBox,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Return a copy of ``image`` with the bounding box drawn."""
        vis = image.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        return vis

    # --------------------------------------------------------------------- #
    # Internals                                                             #
    # --------------------------------------------------------------------- #
    def _ensure_model(self):
        if self._model is not None or self._model_available:
            return self._model

        model_path = os.getenv("AUTOSENTINEL_YOLO", self.config.model_path)
        try:
            from ultralytics import YOLO  # imported lazily
        except Exception as exc:  # pragma: no cover - import issues
            print(f"[detector] YOLO import failed: {exc}. Using stub detector.")
            self._model_available = False
            return None

        if not os.path.exists(model_path):
            print(f"[detector] Model not found at '{model_path}'. Using stub detector.")
            self._model_available = False
            return None

        try:
            self._model = YOLO(model_path)
            self._model_available = True
            print(f"[detector] YOLO model loaded: {model_path}")
        except Exception as exc:  # pragma: no cover - runtime issues
            print(f"[detector] YOLO load failed: {exc}. Using stub detector.")
            self._model = None
            self._model_available = False
        return self._model

    @staticmethod
    def _stub_bbox(image: np.ndarray) -> Tuple[PlateBBox, float]:
        height, width = image.shape[:2]
        x1 = width * 0.25
        y1 = height * 0.40
        x2 = width * 0.75
        y2 = height * 0.60
        return PlateBBox(x1=x1, y1=y1, x2=x2, y2=y2), 0.5


_DEFAULT_DETECTOR: Optional[PlateDetector] = None


def get_detector() -> PlateDetector:
    """Return the shared detector instance."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = PlateDetector()
    return _DEFAULT_DETECTOR


def detect_plate_bbox(image: np.ndarray) -> Tuple[PlateBBox, float]:
    """Module-level helper that delegates to the shared detector instance."""
    return get_detector().detect(image)


def crop(image: np.ndarray, bbox: PlateBBox) -> np.ndarray:
    return get_detector().crop(image, bbox)


def draw_bbox(
    image: np.ndarray,
    bbox: PlateBBox,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    return get_detector().draw_bbox(image, bbox, color=color, thickness=thickness)


def _prepare_for_model(image: np.ndarray) -> np.ndarray:
    """
    Ensure the image has three channels as expected by YOLO.

    The ultralytics implementation works with numpy arrays in either RGB or BGR.
    """
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape for detector: {image.shape}")


__all__ = [
    "DetectorConfig",
    "PlateDetector",
    "crop",
    "detect_plate_bbox",
    "draw_bbox",
    "get_detector",
]
