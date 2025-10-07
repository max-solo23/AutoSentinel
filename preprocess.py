import numpy as np
import os
import uuid
import cv2


_DEF_MIN_W = 640


def _maybe_save(stage: str, img: np.ndarray, prefix: str):
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    ext = "png" if img.ndim == 2 else "jpg"
    path = f"{prefix}_{stage}.{ext}"
    cv2.imwrite(path, img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _show_sequence(
        frames: list[tuple[str, np.ndarray]], delay_ms: int = 500, window: str = "AutoSentinel Preprocess"):
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    for name, frame in frames:
        vis = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vis = vis.copy()
        cv2.putText(
            vis, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, 255 if frame.ndim == 2 else (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, vis)
        if cv2.waitKey(delay_ms) & 0xFF == 27:  # ESC to break early
            break
    cv2.destroyWindow(window)


def preprocess(
        img: np.ndarray,
        *,
        save_debug: bool = False,
        debug_prefix: str = "test_images/out/pre",
        show: bool = False,
        delay_ms: int = 500
) -> np.ndarray:
    """OpenCV preprocessing tuned for plates.
    Steps: RGB->GRAY -> denoise -> CLAHE -> normalize -> optional tophat -> resize min width.
    Returns single-channel uint8 image.
    """
    dbg = save_debug or os.getenv("AUTOSENTINEL_DEBUG", "0") == "1"
    uid = uuid.uuid4().hex[:6]
    prefix = f"{debug_prefix}_{uid}"

    frames: list[tuple[str, np.ndarray]] = []

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    frames.append(("00_gray", gray))
    if dbg: _maybe_save("00_gray", gray, prefix)

    den = cv2.GaussianBlur(gray, (3, 3), 0)
    if dbg: _maybe_save("01_blur", den, prefix)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(den)
    frames.append(("02_clahe", eq))
    if dbg: _maybe_save("02_clahe", eq, prefix)

    norm = cv2.normalize(eq, None, 0, 255, cv2.NORM_MINMAX)
    frames.append(("03_norm", norm))
    if dbg: _maybe_save("03_norm", norm, prefix)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    tophat = cv2.morphologyEx(norm, cv2.MORPH_TOPHAT, kernel)
    frames.append(("04_tophat", tophat))
    if dbg: _maybe_save("04_tophat", tophat, prefix)

    h, w = tophat.shape[:2]
    if w < _DEF_MIN_W:
        scale = _DEF_MIN_W / float(w)
        new_size = (int(w * scale), int(h * scale))
        proc = cv2.resize(tophat, new_size, interpolation=cv2.INTER_CUBIC)
        frames.append(("05_resize", proc))
        if dbg: _maybe_save("05_resize", proc, prefix)
    else:
        proc = tophat
        frames.append(("05_keep", proc))

    if show:
        _show_sequence(frames, delay_ms=delay_ms)

    return proc
