# AutoSentinel — Step 2: OpenCV preprocessing


## What changed
- Added real preprocessing: grayscale → denoise → CLAHE → normalize → light tophat → resize if width < 640.
- Optional debug dumps to `test_images/out/*` when `AUTOSENTINEL_DEBUG=1` or `save_debug=True`.
- Added `opencv-python` to requirements.


## Run a quick check
```bash
set AUTOSENTINEL_DEBUG=1 # PowerShell: $env:AUTOSENTINEL_DEBUG=1
python -m tests.smoke_test test_images/dummy.png

or one line: $env:AUTOSENTINEL_DEBUG=1; python -m tests.smoke_test test_images/dummy.png

```
Check `test_images/out/` for intermediate images: `00_gray`, `01_blur`, `02_clahe`, `03_norm`, `04_tophat`, `05_resize`.


## API run
```bash
uvicorn main:app --reload --port 8000
```


## Next
- Step 3: Replace stub detector with YOLOv8.
- Step 4: Replace stub OCR with EasyOCR.