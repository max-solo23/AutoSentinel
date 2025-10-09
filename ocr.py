import re
from functools import lru_cache
import numpy as np


# def read_text(plate_img: np.ndarray) -> tuple[str, float]:
#     return "DUMMY123", 0.90


@lru_cache(maxsize=1)
def _get_reader():
    import easyocr
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False
    print(f"[ocr] EasyOCR init. gpu={use_gpu}")
    return easyocr.Reader(['en'], gpu=True)


def _pick_best(lines):
    if not lines:
        return "", 0.0
    best = max(lines, key=lambda x: (x[2] or 0) * len(x[1] or ""))
    return best[1] or "", float(best[2] or 0.0)


_EU_IT = re.compile(r"^[A-Z]{2}\d{3}[A-Z]{2}$")
_FALLBACK = re.compile(r"[A-Z0-9]{5,10}")


def _postprocess(text: str) -> str:
    t = text.upper().replace(" ", "").replace("-", "")
    t = t.replace("O", "0").replace("I", "1").replace("Z", "2")
    if _EU_IT.match(t):
        return t
    m = _FALLBACK.search(t)
    return m.group(0) if m else t


def read_text(plate_img: np.ndarray) -> tuple[str, float]:
    reader = _get_reader()
    results = reader.readtext(plate_img)
    # results: List[ (bbox, text, conf) ]
    if not results:
        return "", 0.0
    raw, conf = _pick_best([(b, s, c) for (b, s, c) in results])
    clean = _postprocess(raw)

    if not clean:
        clean = raw.strip().upper()
    return clean, float(conf)
