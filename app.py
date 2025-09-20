from flask import Flask, request, render_template, jsonify
import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
import io, base64
import numpy as np
import rasterio
import os

app = Flask(__name__, template_folder="templates")

# Load model
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights).eval()
transform = weights.transforms()

# Color palette (RGBA) - background now light gray
PALETTE = {
    0: (200, 200, 200, 180), # background -> light gray
    3: (160, 82, 45, 200),   # ground -> brown
    8: (34, 139, 34, 220),   # vegetation
    21: (0, 255, 0, 220),    # trees
    90: (135, 206, 235, 200) # sky
}

def apply_palette(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, color in PALETTE.items():
        color_mask[mask == cls] = color
    return Image.fromarray(color_mask, mode="RGBA")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    bounds = None
    if file.filename.lower().endswith((".tif", ".tiff")):
        with rasterio.open(file) as src:
            img = src.read([1, 2, 3])
            img = np.transpose(img, (1, 2, 0))
            bounds = src.bounds
            img = Image.fromarray(img.astype(np.uint8))
    else:
        img = Image.open(file).convert("RGB")

    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    pred = output.argmax(0).byte().cpu().numpy()

    mask_img = apply_palette(pred)

    # Encode both images
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    img.save(buf1, format="PNG")
    mask_img.save(buf2, format="PNG")
    original_b64 = base64.b64encode(buf1.getvalue()).decode("utf-8")
    mask_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    resp = {"original": original_b64, "mask": mask_b64}
    if bounds:
        resp["bounds"] = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    return jsonify(resp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
