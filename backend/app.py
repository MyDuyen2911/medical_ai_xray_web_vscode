import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "resnet101v2_xray_model.h5")

IMG_SIZE = 400

class_names = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TUBERCULOSIS"
]

print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("File trong backend:", os.listdir(BASE_DIR))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")

print("Đang load model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Load model thành công!")


def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image).astype("float32")
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Backend ResNet101V2 X-ray AI đang chạy",
        "model": "ResNet101V2",
        "classes": class_names
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Không tìm thấy file ảnh"
        }), 400

    image_file = request.files["image"]

    try:
        img_array = preprocess_image(image_file)

        predictions = model.predict(img_array, verbose=0)[0]

        pred_index = int(np.argmax(predictions))
        pred_label = class_names[pred_index]
        confidence = float(predictions[pred_index]) * 100

        probabilities = {
            class_names[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(class_names))
        }

        return jsonify({
            "model": "ResNet101V2",
            "prediction": pred_label,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)