from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class OCRConfig:
    """Configuration values for OCR and post-processing."""

    languages: Tuple[str, ...] = ("en",)
    min_length: int = 5
    max_length: int = 10
    enforce_eu_it: bool = True


class PlateRecognizer:
    """EasyOCR wrapper with EU-focused post-processing helpers."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self._reader = None

    def read(self, plate_img: np.ndarray) -> Tuple[str, float]:
        reader = self._ensure_reader()
        results = reader.readtext(plate_img)
        if not results:
            return "", 0.0

        text, confidence = _pick_best(results)
        cleaned = _postprocess(
            text,
            min_len=self.config.min_length,
            max_len=self.config.max_length,
            enforce_eu_it=self.config.enforce_eu_it,
        )
        if not cleaned:
            cleaned = text.strip().upper()
        return cleaned, float(confidence)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _ensure_reader(self):
        if self._reader is not None:
            return self._reader

        reader = _create_easyocr_reader(self.config.languages)
        self._reader = reader
        return reader


_DEFAULT_RECOGNIZER: Optional[PlateRecognizer] = None


def get_recognizer() -> PlateRecognizer:
    global _DEFAULT_RECOGNIZER
    if _DEFAULT_RECOGNIZER is None:
        _DEFAULT_RECOGNIZER = PlateRecognizer()
    return _DEFAULT_RECOGNIZER


def read_text(plate_img: np.ndarray) -> Tuple[str, float]:
    return get_recognizer().read(plate_img)


# ---------------------------------------------------------------------- #
# Post-processing helpers                                                #
# ---------------------------------------------------------------------- #
_EU_IT = re.compile(r"^[A-Z]{2}\d{3}[A-Z]{2}$")
_LETTER_FIX = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "B",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "G",
}
_DIGIT_FIX = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}
_PUNCT_REPLACEMENTS = {
    "~": "",
    ":": "",
    ";": "",
    ".": "",
    ",": "",
    "'": "",
    "|": "I",
    "/": "",
    "\\": "",
    "*": "",
    "\u20ac": "E",
    "@": "A",
    "-": "",
    " ": "",
}


def _pick_best(results: Iterable[Tuple[np.ndarray, str, float]]) -> Tuple[str, float]:
    best_text = ""
    best_conf = 0.0
    best_score = -1.0
    for _, text, conf in results:
        text = text or ""
        conf = float(conf or 0.0)
        score = conf * max(len(text), 1)
        if score > best_score:
            best_score = score
            best_text = text
            best_conf = conf
    return best_text, best_conf


def _postprocess(
    text: str,
    *,
    min_len: int,
    max_len: int,
    enforce_eu_it: bool,
) -> str:
    candidate = _coerce_it_plate(text) if enforce_eu_it else ""
    if candidate:
        return candidate
    tokens = _sanitize(text)
    if not tokens:
        return ""
    fallback = re.search(rf"[A-Z0-9]{{{min_len},{max_len}}}", tokens)
    return fallback.group(0) if fallback else tokens


def _sanitize(text: str) -> str:
    allowed: List[str] = []
    for char in text.upper():
        if char in _PUNCT_REPLACEMENTS:
            allowed.append(_PUNCT_REPLACEMENTS[char])
            continue
        if char.isalnum():
            allowed.append(char)
    sanitized = "".join(allowed)
    return _normalize_confusions(sanitized)


def _coerce_it_plate(text: str) -> str:
    clean = _sanitize(text)
    if len(clean) < 7:
        return ""
    for start in range(len(clean) - 6):
        segment = list(clean[start : start + 7])
        if not _coerce_letters(segment, indices=(0, 1, 5, 6)):
            continue
        if not _coerce_digits(segment, indices=(2, 3, 4)):
            continue
        candidate = "".join(segment)
        if _EU_IT.match(candidate):
            return candidate
    return ""


def _coerce_letters(chars: List[str], indices: Tuple[int, ...]) -> bool:
    for idx in indices:
        ch = chars[idx]
        if ch.isalpha():
            continue
        mapped = _LETTER_FIX.get(ch, ch)
        if not mapped.isalpha():
            return False
        chars[idx] = mapped
    return True


def _coerce_digits(chars: List[str], indices: Tuple[int, ...]) -> bool:
    for idx in indices:
        ch = chars[idx]
        if ch.isdigit():
            continue
        mapped = _DIGIT_FIX.get(ch, ch)
        if not mapped.isdigit():
            return False
        chars[idx] = mapped
    return True


@lru_cache(maxsize=1)
def _create_easyocr_reader(languages: Tuple[str, ...]):
    import easyocr

    use_gpu = _detect_gpu()
    print(f"[ocr] EasyOCR init. gpu={use_gpu}")
    return easyocr.Reader(list(languages), gpu=use_gpu)


def _detect_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover - torch not installed
        return False


def _normalize_confusions(text: str) -> str:
    mapping = {
        "O": "0",
        "I": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }
    for source, target in mapping.items():
        text = text.replace(source, target)
    return text


__all__ = [
    "OCRConfig",
    "PlateRecognizer",
    "get_recognizer",
    "read_text",
]
