"""Flask web app that serves the trained CIFAR-10 CNN.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import io
import json
import os
import sys

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

MODEL_PATH = "models/cifar10_cnn.keras"
if not os.path.exists(MODEL_PATH):
    sys.exit("Model not found. Train it first:  python train.py")

import tensorflow as tf  # noqa: E402  (imported after the check for a fast failure)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

model = tf.keras.models.load_model(MODEL_PATH)
with open("models/classes.json") as f:
    CLASSES = json.load(f)
METRICS = {}
if os.path.exists("models/metrics.json"):
    with open("models/metrics.json") as f:
        METRICS = json.load(f)


def preprocess(raw: bytes) -> np.ndarray:
    """Centre-crop to a square, resize to 32x32, return a (1, 32, 32, 3) batch."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((32, 32), Image.LANCZOS)
    return np.asarray(img, dtype="float32")[None, ...]


@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES, metrics=METRICS)


@app.route("/api/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image received. Choose a file and try again."), 400
    try:
        batch = preprocess(file.read())
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file isn't a readable image. Use JPG, PNG or WebP."), 400

    probs = model.predict(batch, verbose=0)[0]
    top = np.argsort(probs)[::-1][:5]
    return jsonify(predictions=[{"label": CLASSES[i], "confidence": float(probs[i])}
                                for i in top])


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is larger than 8 MB. Choose a smaller file."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
