import os
import io
import base64

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "ResNet101"
APP_VERSION = "REAL_XAI_BACKEND_V3"
MODEL_PATH = os.path.join(BASE_DIR, "resnet101_model_fixed.keras")

DEFAULT_IMG_SIZE = 400

CLASS_NAMES = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TUBERCULOSIS"
]

DISPLAY_NAMES = {
    "COVID19": "COVID-19",
    "NORMAL": "Normal",
    "PNEUMONIA": "Pneumonia",
    "TUBERCULOSIS": "Tuberculosis"
}


print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("Files in backend:", os.listdir(BASE_DIR))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")

print("Đang load model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Load model thành công!")


def get_model_input_size():
    try:
        input_shape = model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = input_shape[1]
        width = input_shape[2]

        if height is not None and width is not None:
            return int(height), int(width)

    except Exception:
        pass

    return DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE


IMG_HEIGHT, IMG_WIDTH = get_model_input_size()
print("Model input size:", IMG_HEIGHT, IMG_WIDTH)


def pil_to_data_url(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def normalize_map(arr):
    arr = np.array(arr, dtype=np.float32)
    arr = arr - np.min(arr)
    max_value = np.max(arr)

    if max_value > 1e-8:
        arr = arr / max_value

    return arr


def jet_colormap(gray):
    gray = normalize_map(gray)

    r = np.clip(1.5 - np.abs(4 * gray - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * gray - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * gray - 1), 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(original_pil, heatmap, alpha=0.48):
    heatmap = normalize_map(heatmap)

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(original_pil.size, Image.BILINEAR)

    heatmap_arr = np.array(heatmap_img).astype(np.float32) / 255.0
    color_arr = jet_colormap(heatmap_arr).astype(np.float32)

    original_arr = np.array(original_pil.convert("RGB")).astype(np.float32)

    overlay = original_arr * (1 - alpha) + color_arr * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def preprocess_image(image_file):
    original = Image.open(image_file).convert("RGB")
    resized = original.resize((IMG_WIDTH, IMG_HEIGHT))

    img_array = np.array(resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return original, img_array


def normalize_predictions(predictions):
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]

    predictions = np.array(predictions)

    if predictions.ndim == 2:
        predictions = predictions[0]

    return predictions.astype("float32")


def find_last_conv_layer(keras_model):
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

        if isinstance(layer, tf.keras.Model):
            nested_conv = find_last_conv_layer(layer)
            if nested_conv is not None:
                return nested_conv

    return None


def find_conv_model_and_tail_layers(keras_model):
    """
    Hỗ trợ tốt cho model dạng Sequential:
    [ResNet101/ResNet101V2 backbone] -> pooling -> dense -> softmax.

    Nếu model có backbone nested, hàm sẽ lấy output của Conv2D cuối trong backbone
    và chạy tiếp các layer phía sau để tính prediction.
    """
    for index, layer in enumerate(keras_model.layers):
        if isinstance(layer, tf.keras.Model):
            last_conv = find_last_conv_layer(layer)

            if last_conv is not None:
                conv_model = tf.keras.models.Model(
                    inputs=layer.inputs,
                    outputs=[last_conv.output, layer.output]
                )

                tail_layers = keras_model.layers[index + 1:]
                return conv_model, tail_layers

    last_conv = find_last_conv_layer(keras_model)

    if last_conv is None:
        raise RuntimeError("Không tìm thấy Conv2D layer để tạo Grad-CAM.")

    grad_model = tf.keras.models.Model(
        inputs=keras_model.inputs,
        outputs=[last_conv.output, keras_model.output]
    )

    return grad_model, None


def forward_from_conv_model(img_array, conv_model, tail_layers):
    conv_outputs, x = conv_model(img_array, training=False)

    if tail_layers is None:
        predictions = x
        return conv_outputs, predictions

    for layer in tail_layers:
        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)

    predictions = x
    return conv_outputs, predictions


def generate_gradcam_heatmap(img_array, pred_index):
    conv_model, tail_layers = find_conv_model_and_tail_layers(model)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = forward_from_conv_model(
            img_array,
            conv_model,
            tail_layers
        )

        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError("Không tính được gradient cho Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy()

    return normalize_map(heatmap)


def generate_gradcampp_heatmap(img_array, pred_index):
    conv_model, tail_layers = find_conv_model_and_tail_layers(model)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = forward_from_conv_model(
            img_array,
            conv_model,
            tail_layers
        )

        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError("Không tính được gradient cho Grad-CAM++.")

    conv_outputs = conv_outputs[0]
    grads = grads[0]

    grads_power_2 = tf.square(grads)
    grads_power_3 = grads_power_2 * grads

    sum_activations = tf.reduce_sum(conv_outputs, axis=(0, 1))
    eps = tf.keras.backend.epsilon()

    alpha_num = grads_power_2
    alpha_denom = 2.0 * grads_power_2 + grads_power_3 * sum_activations
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, eps)

    alphas = alpha_num / alpha_denom
    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))

    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy()

    return normalize_map(heatmap)


def generate_integrated_gradients_heatmap(img_array, pred_index, steps=32):
    baseline = np.zeros_like(img_array).astype("float32")
    input_img = img_array.astype("float32")

    alphas = np.linspace(0.0, 1.0, steps + 1).astype("float32")
    total_grads = np.zeros_like(input_img).astype("float32")

    for alpha in alphas:
        interpolated = baseline + alpha * (input_img - baseline)
        interpolated = tf.convert_to_tensor(interpolated)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = model(interpolated, training=False)
            class_score = predictions[:, pred_index]

        grads = tape.gradient(class_score, interpolated)

        if grads is not None:
            total_grads += grads.numpy()

    avg_grads = total_grads / float(len(alphas))
    integrated_grads = (input_img - baseline) * avg_grads

    heatmap = np.mean(np.abs(integrated_grads[0]), axis=-1)
    return normalize_map(heatmap)


def generate_occlusion_heatmap(img_array, pred_index, grid_size=8):
    base_prediction = model.predict(img_array, verbose=0)
    base_prediction = normalize_predictions(base_prediction)
    base_score = float(base_prediction[pred_index])

    _, h, w, _ = img_array.shape
    patch_h = max(1, h // grid_size)
    patch_w = max(1, w // grid_size)

    heatmap = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, patch_h):
        for x in range(0, w, patch_w):
            occluded = img_array.copy()
            y2 = min(y + patch_h, h)
            x2 = min(x + patch_w, w)

            occluded[:, y:y2, x:x2, :] = 0.0

            prediction = model.predict(occluded, verbose=0)
            prediction = normalize_predictions(prediction)
            score = float(prediction[pred_index])

            drop = max(0.0, base_score - score)
            heatmap[y:y2, x:x2] = drop

    return normalize_map(heatmap)


def generate_xai_images(original_pil, img_array, pred_index):
    gradcam_heatmap = generate_gradcam_heatmap(img_array, pred_index)
    gradcampp_heatmap = generate_gradcampp_heatmap(img_array, pred_index)
    ig_heatmap = generate_integrated_gradients_heatmap(img_array, pred_index)
    occlusion_heatmap = generate_occlusion_heatmap(img_array, pred_index)

    gradcam_img = overlay_heatmap(original_pil, gradcam_heatmap, alpha=0.52)
    gradcampp_img = overlay_heatmap(original_pil, gradcampp_heatmap, alpha=0.52)
    ig_img = overlay_heatmap(original_pil, ig_heatmap, alpha=0.50)
    occlusion_img = overlay_heatmap(original_pil, occlusion_heatmap, alpha=0.55)

    comparison_img = create_comparison_strip(
        original_pil,
        gradcam_img,
        gradcampp_img,
        ig_img,
        occlusion_img
    )

    return {
        "gradcam": gradcam_img,
        "gradcampp": gradcampp_img,
        "integrated_gradients": ig_img,
        "occlusion": occlusion_img,
        "comparison": comparison_img
    }


def create_comparison_strip(original, gradcam, gradcampp, ig, occlusion):
    labels = [
        "Original",
        "Grad-CAM",
        "Grad-CAM++",
        "Integrated Gradients",
        "Occlusion"
    ]

    images = [
        original.convert("RGB"),
        gradcam.convert("RGB"),
        gradcampp.convert("RGB"),
        ig.convert("RGB"),
        occlusion.convert("RGB")
    ]

    item_w = 260
    item_h = 260
    title_h = 32

    canvas = Image.new(
        "RGB",
        (item_w * len(images), item_h + title_h),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for idx, (img, label) in enumerate(zip(images, labels)):
        x = idx * item_w
        resized = img.resize((item_w, item_h), Image.BILINEAR)

        draw.text((x + 10, 8), label, fill="black")
        canvas.paste(resized, (x, title_h))

    return canvas


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Backend ResNet101 X-ray AI đang chạy",
        "model": MODEL_NAME,
        "input_size": [IMG_HEIGHT, IMG_WIDTH],
        "classes": CLASS_NAMES
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" in request.files:
        image_file = request.files["image"]
    elif "file" in request.files:
        image_file = request.files["file"]
    else:
        return jsonify({
            "error": "Không tìm thấy file ảnh. Vui lòng gửi ảnh với key 'image' hoặc 'file'."
        }), 400

    try:
        original_pil, img_array = preprocess_image(image_file)

        raw_predictions = model.predict(img_array, verbose=0)
        predictions = normalize_predictions(raw_predictions)

        if len(predictions) != len(CLASS_NAMES):
            return jsonify({
                "error": "Model không trả về đúng 4 lớp.",
                "output_shape": list(predictions.shape),
                "output": predictions.tolist()
            }), 500

        pred_index = int(np.argmax(predictions))
        pred_key = CLASS_NAMES[pred_index]
        pred_label = DISPLAY_NAMES[pred_key]

        confidence = float(predictions[pred_index]) * 100

        probabilities = {}

        for i, class_key in enumerate(CLASS_NAMES):
            display_name = DISPLAY_NAMES[class_key]
            probabilities[display_name] = round(float(predictions[i]) * 100, 2)

        print("Đang tạo XAI images...")
        xai = generate_xai_images(original_pil, img_array, pred_index)
        print("Tạo XAI images thành công!")

        return jsonify({
            "debug_version": APP_VERSION,
            "model": MODEL_NAME,
            "prediction": pred_label,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "original_image": pil_to_data_url(original_pil),
            "xai_comparison": pil_to_data_url(xai["comparison"]),
            "xai_images": {
                "gradcam": pil_to_data_url(xai["gradcam"]),
                "gradcampp": pil_to_data_url(xai["gradcampp"]),
                "integrated_gradients": pil_to_data_url(xai["integrated_gradients"]),
                "occlusion": pil_to_data_url(xai["occlusion"])
                }
        })
    except Exception as error:
        print("Lỗi predict:", error)
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
