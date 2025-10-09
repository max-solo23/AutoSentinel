import sys, os, time
import cv2
from functools import lru_cache


# ensure project root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from io_tools import load_image_from_bytes
from preprocess import preprocess
from detector import detect_plate_bbox, crop
from ocr import read_text


def _save(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("saved:", path)


@lru_cache(maxsize=1)
def _dbg_reader():
    import easyocr, torch
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/test_ocr.py <image_path>")
        sys.exit(1)
    src = sys.argv[1]
    with open(src, "rb") as f:
        b = f.read()
    img = load_image_from_bytes(b)

    pre = preprocess(img, show=False, save_debug=True, debug_prefix="test_images/out/ocr_pre")
    bbox, det_conf = detect_plate_bbox(pre)
    roi = crop(pre, bbox)

    if roi.size == 0:
        print("[error] empty ROI from detector; adjust detector or bbox.")
        sys.exit(2)

    ph, pw = roi.shape[:2]
    pad = int(0.10 * max(ph, pw))
    roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    h, w = roi.shape[:2]
    if h < 220:
        s = 220.0 / h
        roi = cv2.resize(roi, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)

    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]

    ts = int(time.time())

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    thr = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    _save(f"test_images/out/roi_{ts}.png", roi)
    _save(f"test_images/out/roi_thr_{ts}.png", thr)

    t1, c1 = read_text(thr)
    t2, c2 = read_text(roi)
    print(f"ocr_thr='{t1}' c={c1:.3f} | ocr_rgb='{t2}' c={c2:.3f}")

    reader = _dbg_reader()
    res = reader.readtext(roi)  # [(bbox, text, conf), ...]
    print("easyocr_raw:", [(t, round(c, 3)) for _, t, c in res])
    # keep current function too for comparison
    text, conf = read_text(roi)
    print(f"det_conf={det_conf:.3f}  ocr='{text}'  ocr_conf={conf:.3f}")