from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessDebugConfig:
    """Configuration for saving and visualising intermediate preprocessing steps."""

    save: bool = False
    prefix: str = "test_images/out/pre"
    show: bool = False
    delay_ms: int = 500

    @property
    def enabled(self) -> bool:
        return self.save or self.show


@dataclass(slots=True)
class PreprocessConfig:
    """Tunable parameters for the preprocessing pipeline."""

    min_width: int = 640
    blur_kernel_size: Tuple[int, int] = (3, 3)
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    tophat_kernel_size: Tuple[int, int] = (17, 3)
    resize_interpolation: int = cv2.INTER_CUBIC


class Preprocessor:
    """Runs a deterministic sequence of OpenCV transforms tailored for licence plates."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        self._tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.config.tophat_kernel_size,
        )

    def run(
        self,
        image: np.ndarray,
        *,
        debug: PreprocessDebugConfig | None = None,
    ) -> np.ndarray:
        """
        Apply preprocessing to an image.

        Parameters
        ----------
        image:
            Source image (RGB, RGBA, BGR, or grayscale).
        debug:
            Optional debugging configuration; when provided, intermediate steps are
            persisted to disk or shown interactively.
        """
        dbg = debug or resolve_debug_config()
        frames: List[Tuple[str, np.ndarray]] = []

        gray = _to_grayscale(image)
        _record_frame(dbg, frames, "00_gray", gray)

        blurred = cv2.GaussianBlur(gray, self.config.blur_kernel_size, 0)
        _record_frame(dbg, frames, "01_blur", blurred)

        equalised = self._clahe.apply(blurred)
        _record_frame(dbg, frames, "02_clahe", equalised)

        normalised = cv2.normalize(equalised, None, 0, 255, cv2.NORM_MINMAX)
        _record_frame(dbg, frames, "03_norm", normalised)

        high_pass = cv2.morphologyEx(normalised, cv2.MORPH_TOPHAT, self._tophat_kernel)
        _record_frame(dbg, frames, "04_tophat", high_pass)

        processed = _ensure_min_width(
            high_pass,
            min_width=self.config.min_width,
            interpolation=self.config.resize_interpolation,
            dbg=dbg,
            frames=frames,
        )

        if dbg.show:
            _show_sequence(frames, delay_ms=dbg.delay_ms)

        return processed


def resolve_debug_config(
    *,
    save_debug: bool | None = None,
    prefix: str = "test_images/out/pre",
    show: bool = False,
    delay_ms: int = 500,
) -> PreprocessDebugConfig:
    """Derive the debug configuration, honouring the AUTOSENTINEL_DEBUG env flag."""
    env_debug = os.getenv("AUTOSENTINEL_DEBUG", "0") == "1"
    save = (save_debug if save_debug is not None else False) or env_debug
    return PreprocessDebugConfig(save=save, prefix=prefix, show=show or False, delay_ms=delay_ms)


def preprocess(
    image: np.ndarray,
    *,
    save_debug: bool = False,
    debug_prefix: str = "test_images/out/pre",
    show: bool = False,
    delay_ms: int = 500,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """Convenience functional wrapper around ``Preprocessor`` for backwards compatibility."""
    debug_cfg = PreprocessDebugConfig(
        save=save_debug or os.getenv("AUTOSENTINEL_DEBUG", "0") == "1",
        prefix=debug_prefix,
        show=show,
        delay_ms=delay_ms,
    )
    return Preprocessor(config=config).run(image, debug=debug_cfg)


def _record_frame(
    debug: PreprocessDebugConfig,
    frames: List[Tuple[str, np.ndarray]],
    name: str,
    image: np.ndarray,
) -> None:
    frames.append((name, image))
    if debug.save:
        _save_frame(debug.prefix, name, image)


def _save_frame(prefix: str, stage: str, image: np.ndarray) -> None:
    out_dir = os.path.dirname(prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ext = "png" if image.ndim == 2 else "jpg"
    path = f"{prefix}_{stage}.{ext}"
    to_write = image
    if image.ndim == 3:
        if image.shape[2] == 3:
            to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[2] == 4:
            to_write = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, to_write)


def _ensure_min_width(
    image: np.ndarray,
    *,
    min_width: int,
    interpolation: int,
    dbg: PreprocessDebugConfig,
    frames: List[Tuple[str, np.ndarray]],
) -> np.ndarray:
    height, width = image.shape[:2]
    if width >= min_width:
        _record_frame(dbg, frames, "05_keep", image)
        return image

    scale = min_width / float(width)
    resized = cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=interpolation,
    )
    _record_frame(dbg, frames, "05_resize", resized)
    return resized


def _show_sequence(frames: List[Tuple[str, np.ndarray]], *, delay_ms: int) -> None:
    window_name = "AutoSentinel Preprocess"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    for name, frame in frames:
        vis = frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            vis = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        vis = vis.copy()
        cv2.putText(
            vis,
            name,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            255 if frame.ndim == 2 else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, vis)
        if cv2.waitKey(delay_ms) & 0xFF == 27:
            break
    cv2.destroyWindow(window_name)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert input to single-channel grayscale."""
    if image.ndim == 2:
        return image.copy()
    channels = image.shape[2]
    if channels == 3:
        # Assume RGB input produced by Pillow; fall back to BGR if needed.
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if _is_low_contrast(gray):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    if channels == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        if _is_low_contrast(gray):
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return gray
    raise ValueError(f"Unsupported image shape {image.shape}")


def _is_low_contrast(image: np.ndarray, threshold: float = 15.0) -> bool:
    return float(image.max() - image.min()) < threshold


__all__ = [
    "PreprocessConfig",
    "PreprocessDebugConfig",
    "Preprocessor",
    "preprocess",
    "resolve_debug_config",
]
