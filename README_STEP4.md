# AutoSentinel — Step 4: OCR with EasyOCR (EU plates)


## What changed
- Integrated EasyOCR in `ocr.py` with lazy GPU reader and EU-style postprocessing.
- Postprocess rules: uppercase, remove spaces/dashes, fix O→0/I→1/Z→2, regex `[A-Z0-9]{4,10}`.


## Install
```powershell
pip install easyocr==1.7.1
```
If GPU is set up (PyTorch CUDA), EasyOCR will use it automatically.


## Quick test
```powershell
python -m tests.smoke_test test_images/dummy.png
```
Expect the stub bbox but OCR now processed via EasyOCR; result still depends on input.


## Next
- Step 5: Add country-specific plate formats and confidence calibration.
- Optional: fallback to Tesseract if EasyOCR unavailable.