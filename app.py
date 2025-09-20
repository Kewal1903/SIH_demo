from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
import numpy as np
import io, base64, os
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

app = Flask(__name__, template_folder="templates")

# Load SegFormer pretrained on ADE20K (covers vegetation, ground, sky, water, etc.)
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).eval()

# ADE20K palette (simplified subset)
PALETTE = {
    0: (200,200,200,255), # background
    2: (0,128,0,255),     # vegetation/trees
    3: (160,82,45,255),   # ground/soil
    5: (70,130,180,255),  # water
    6: (135,206,235,255)  # sky
}

def decode_segmentation(mask):
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
    img = Image.open(file).convert("RGB")

    inputs = feature_extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # (batch, num_classes, h, w)
    pred = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()

    # Colorize mask
    mask_img = decode_segmentation(pred)

    # Encode original and mask
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    img.save(buf1, format="PNG")
    mask_img.save(buf2, format="PNG")
    original_b64 = base64.b64encode(buf1.getvalue()).decode("utf-8")
    mask_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    return jsonify({"original": original_b64, "mask": mask_b64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
