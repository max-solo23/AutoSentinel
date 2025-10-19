# AutoSentinel

AutoSentinel is a FastAPI-based automatic number plate recognition (ANPR) pipeline that combines classical image preprocessing, a YOLOv8 detector, and EasyOCR for text recognition. The codebase is structured so each piece can be validated in isolation while still supporting an end-to-end API demo.

## Project Layout
```
AutoSentinel/
    custom_types.py
    io_tools.py
    preprocess.py
    detector.py
    ocr.py
    pipeline.py
    main.py
    requirements.txt
    tests/
        smoke_test.py
    test_images/
        dummy.png
    tools/
        make_dummy_image.py
```

## Installation
- Use Python 3.10 or newer.
- Create and activate a virtual environment (recommended):

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

  macOS/Linux:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- Install dependencies:

```powershell
pip install -r requirements.txt
```

The requirements file includes OpenCV, Ultralytics (YOLOv8), and EasyOCR. If you have a CUDA-capable GPU, install a compatible PyTorch build so EasyOCR can take advantage of it.

## Quick Start
- **Run the smoke test without the API**

  ```powershell
  python -m tests.smoke_test test_images/dummy.png
  ```

  The pipeline falls back to deterministic stubs whenever a detector model is missing, so you always receive a response for debugging.

- **Launch the REST API**

  ```powershell
  uvicorn main:app --reload --port 8000
  ```

  Open http://localhost:8000/docs to exercise the `/recognize_plate` endpoint with sample images.

## Configuration
- `AUTOSENTINEL_DEBUG=1` enables debug image dumps for every preprocessing stage under `test_images/out/`. You can also pass `save_debug=True` when invoking the pipeline programmatically.
- `AUTOSENTINEL_YOLO` points to your YOLOv8 weights. The default path is `models/yolo_plate.pt`. If the file is missing, the detector falls back to the stub and logs a notice.

```powershell
$env:AUTOSENTINEL_DEBUG = 1
$env:AUTOSENTINEL_YOLO = "A:\models\my_plate.pt"
```

## Pipeline Stages
- **Preprocessor (`Preprocessor`)**: deterministic sequence (grayscale → denoise → CLAHE → normalize → tophat → optional resize). Mostly used for debug dumps, but exposed for standalone experiments.
- **Detector (`PlateDetector`)**: lazy-loads YOLOv8 weights, falls back to a deterministic stub when weights are missing, and exposes helpers for cropping/visualisation.
- **Recognizer (`PlateRecognizer`)**: runs EasyOCR, then applies EU-centric cleanup and heuristics to coerce doubtful characters (O↔0, I↔1, Z↔2, etc.).
- **Orchestrator (`PlatePipeline`)**: ties the components together, pads and sharpens the ROI before OCR, and blends detector/OCR confidences into a single score.

## Debugging Tips
- Inspect `tests/smoke_test.py` for an end-to-end example that feeds an image and prints the resulting JSON.
- With `AUTOSENTINEL_DEBUG` enabled, review `test_images/out/` to understand how each preprocessing step transforms the plate.
- Replace the dummy image with your own samples to benchmark model performance and tune thresholds.

## Roadmap
- Extend the recognizer with country-specific plate formats and confidence calibration.
- Provide a Tesseract fallback when EasyOCR cannot load.
- Enrich the FastAPI contract with batch endpoints and improved error reporting.
