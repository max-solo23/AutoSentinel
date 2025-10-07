import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from io_tools import load_image_from_bytes
from preprocess import preprocess


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/preview_preproc.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, "rb") as f:
        b = f.read()
    img = load_image_from_bytes(b)
    # show=True and delay_ms=500 for 0.5s
    _ = preprocess(img, show=True, delay_ms=1000)
    print("Done.")
