import sys, os
from pipeline import run_pipeline


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/smoke_test.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    print("Path:", path)
    print("Exists:", os.path.exists(path))
    print("Size:", os.path.getsize(path) if os.path.exists(path) else "n/a")
    with open(path, "rb") as f:
        b = f.read()
    print("Bytes len:", len(b))
    print("Head 16:", b[:16])
    out = run_pipeline(b)
    print("Status:", out.status)
    print("Plate:", out.plate_text)
    print("Confidence:", out.confidence)
    print("BBox:", out.bbox.model_dump())
