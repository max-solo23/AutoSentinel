from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from custom_types import PlateBBox, PlateResult
from detector import PlateDetector, get_detector
from io_tools import load_image_from_bytes
from ocr import PlateRecognizer, get_recognizer
from preprocess import Preprocessor, resolve_debug_config


@dataclass(slots=True)
class PipelineConfig:
    """Tunables for the high-level pipeline."""

    pad_ratio: float = 0.10
    min_roi_height: int = 220
    blur_sigma: float = 1.2
    sharpen_amount: float = 1.7
    conf_weights: Tuple[float, float] = (0.3, 0.7)  # (detector, ocr)


@dataclass(slots=True)
class ROIArtifacts:
    """Intermediate images produced when preparing the ROI for OCR."""

    raw: np.ndarray
    binarized: np.ndarray


class PlatePipeline:
    """Coordinates image loading, detection, and OCR of licence plates."""

    def __init__(
        self,
        *,
        detector: PlateDetector | None = None,
        recognizer: PlateRecognizer | None = None,
        preprocessor: Preprocessor | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.detector = detector or get_detector()
        self.recognizer = recognizer or get_recognizer()
        self.preprocessor = preprocessor or Preprocessor()
        self.config = config or PipelineConfig()

    def run(self, image_bytes: bytes) -> PlateResult:
        """Run the full recognition pipeline on encoded image bytes."""
        image = load_image_from_bytes(image_bytes)
        print(f"[pipeline] loaded image: shape={getattr(image, 'shape', None)}")

        if _debug_is_enabled():
            debug_cfg = resolve_debug_config(save_debug=True)
            self.preprocessor.run(image, debug=debug_cfg)

        bbox, detector_conf = self.detector.detect(image)
        roi = self.detector.crop(image, bbox)

        if roi.size == 0:
            return PlateResult(status="not_found", plate_text="", confidence=0.0, bbox=bbox)

        artifacts = self._prepare_roi_for_ocr(roi)
        text, ocr_conf = self.recognizer.read(artifacts.binarized)
        if not text:
            text, ocr_conf = self.recognizer.read(artifacts.raw)

        confidence = self._blend_confidence(detector_conf, ocr_conf)
        status = "success" if text else "not_found"
        return PlateResult(
            status=status,
            plate_text=text or "",
            confidence=round(confidence, 3),
            bbox=bbox,
        )

    # ------------------------------------------------------------------ #
    # Helper methods                                                     #
    # ------------------------------------------------------------------ #
    def _prepare_roi_for_ocr(self, roi: np.ndarray) -> ROIArtifacts:
        padded = _pad_roi(roi, pad_ratio=self.config.pad_ratio)
        resized = _ensure_min_height(padded, min_height=self.config.min_roi_height)
        gray = _ensure_grayscale(resized)

        blurred = cv2.GaussianBlur(gray, (0, 0), self.config.blur_sigma)
        sharpened = cv2.addWeighted(gray, self.config.sharpen_amount, blurred, -0.7, 0)
        thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        closed = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        return ROIArtifacts(raw=resized, binarized=closed)

    def _blend_confidence(self, detector_conf: float, ocr_conf: float) -> float:
        det_weight, ocr_weight = self.config.conf_weights
        det = min(max(detector_conf, 0.0), 1.0)
        ocr = min(max(ocr_conf, 0.0), 1.0)
        return det * det_weight + ocr * ocr_weight


_DEFAULT_PIPELINE: Optional[PlatePipeline] = None


def get_pipeline() -> PlatePipeline:
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = PlatePipeline()
    return _DEFAULT_PIPELINE


def run_pipeline(image_bytes: bytes) -> PlateResult:
    """Backwards compatible helper for existing imports."""
    return get_pipeline().run(image_bytes)


# ---------------------------------------------------------------------- #
# Standalone helpers (kept small for readability)                        #
# ---------------------------------------------------------------------- #
def _pad_roi(roi: np.ndarray, *, pad_ratio: float) -> np.ndarray:
    height, width = roi.shape[:2]
    pad = int(pad_ratio * max(height, width))
    if pad <= 0:
        return roi
    return cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def _ensure_min_height(roi: np.ndarray, *, min_height: int) -> np.ndarray:
    height, width = roi.shape[:2]
    if height >= min_height:
        return roi
    scale = min_height / float(height)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(roi, new_size, interpolation=cv2.INTER_CUBIC)


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Unsupported ROI shape: {image.shape}")


def _debug_is_enabled() -> bool:
    return os.getenv("AUTOSENTINEL_DEBUG", "0") == "1"


__all__ = [
    "PipelineConfig",
    "PlatePipeline",
    "ROIArtifacts",
    "get_pipeline",
    "run_pipeline",
]
