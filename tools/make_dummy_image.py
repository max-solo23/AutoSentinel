from PIL import Image, ImageDraw, ImageFont
import os


os.makedirs("test_images", exist_ok=True)

w, h = 800, 450
img = Image.new("RGB", (w, h), (40, 40, 40))
plate = (int(w*0.25), int(h*0.40), int(w*0.75), int(h*0.60))

draw = ImageDraw.Draw(img)
try:
    font = ImageFont.load_default()
except Exception:
    font = None

draw.text((plate[0]+10, plate[1]+10), "AB123CD", fill=(0, 0, 0), font=font)
img.save("test_images/dummy.png")
print("Saved test_images/dummy.png")
