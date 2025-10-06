# AutoSentinel — Step 1: Skeleton


## Goals
- Create a runnable API with a deterministic stub pipeline.
- Observe end-to-end JSON without ML deps.

## Layout
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
        dummy.png (generated)
    tools/
        make_dummy_image.py
```


## Run
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Open http://localhost:8000/docs and POST an image to /recognize_plate.


## Quick check without API
```bash
python tests/smoke_test.py test_images/any_car.jpg
```
Expected: `DUMMY123` with ~0.9 confidence and a center bbox.


## Next steps
- Step 2: real preprocessing with OpenCV.
- Step 3: YOLOv8 detector.
- Step 4: EasyOCR recognition.
- Step 5: Validation, formats, EU templates.