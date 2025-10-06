from PIL import Image, UnidentifiedImageError
import io
import numpy as np


def load_image_from_bytes(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return np.array(img.convert("RGB"))
    except UnidentifiedImageError:
        print(f"[io] Unidentified image. len={len(data)} first16={data[:16]!r}")
        raise
