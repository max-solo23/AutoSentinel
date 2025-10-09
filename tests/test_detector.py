import sys
import os
import time
import cv2


# ensure project root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from io_tools import load_image_from_bytes
from preprocess import preprocess
from detector import detect_plate_bbox, draw_bbox


def _save_vis(img, out_dir="test_images/out", stem="detector_vis"):
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    path = os.path.join(out_dir, f"{stem}_{ts}.jpg")
    import cv2
    cv2.imwrite(path, img)
    print("saved:", path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/test_detector.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, "rb") as f:
        b = f.read()
    img = load_image_from_bytes(b)

    # run preprocessing to simulate pipeline
    pre = preprocess(img, show=False)
    bbox, conf = detect_plate_bbox(pre)
    print(f"bbox: {bbox.model_dump()} conf: {conf:.3f}")

    # visualize
    vis = draw_bbox(pre if pre is not None else img, bbox)
    try:
        import cv2

        cv2.imshow("AutoSentinel detector", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        # >>> NEW: fallback when GUI is unavailable
        print("[info] GUI not available, saving visualization instead. reason:", e)
        _save_vis(vis)
