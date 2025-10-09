# AutoSentinel — Step 3: YOLOv8 plate detector


## What changed
- Implemented YOLOv8 detector in `detector.py` with graceful fallback to stub if model missing.
- Model path configurable via `AUTOSENTINEL_YOLO` env var. Default: `models/yolo_plate.pt`.
- Auto-converts grayscale input to 3-channel for YOLO.


## Install
```bash
pip install ultralytics==8.3.10
```


## Model
Place your trained weights at `models/yolo_plate.pt` or set env var:
```powershell
$env:AUTOSENTINEL_YOLO="A:\models\my_plate.pt"
```


## Quick test
```powershell
# Without model present it falls back to stub and logs a notice
python -m tests.smoke_test test_images/dummy.png
```
Expect a real bbox when model exists; otherwise stub bbox.


## Next
- Step 4: EasyOCR for plate text, with country-specific postprocessing.