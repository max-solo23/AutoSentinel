"""Utility helpers for I/O operations used across the pipeline."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Decode ``data`` into an RGB numpy array.

    Raises
    ------
    UnidentifiedImageError
        If Pillow cannot decode the provided byte buffer.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return np.asarray(img.convert("RGB"))
    except UnidentifiedImageError:
        print(f"[io] Unidentified image. len={len(data)} first16={data[:16]!r}")
        raise


__all__ = ["load_image_from_bytes"]
