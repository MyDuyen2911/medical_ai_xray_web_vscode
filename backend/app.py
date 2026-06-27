import os
import io
import base64
import sqlite3
import secrets
import json
from datetime import datetime
from functools import wraps

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import bcrypt


app = Flask(__name__)
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "mediai.db")
MODEL_NAME = "ResNet101"
MODEL_PATH = os.path.join(BASE_DIR, "resnet101_model_fixed.keras")

DEFAULT_IMG_SIZE = 400

CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

DISPLAY_NAMES = {
    "COVID19":      "COVID-19",
    "NORMAL":       "Normal",
    "PNEUMONIA":    "Pneumonia",
    "TUBERCULOSIS": "Tuberculosis"
}


# ── Database ──────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT    UNIQUE NOT NULL,
        password_hash TEXT    NOT NULL,
        created_at    TEXT    NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS tokens (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL,
        token      TEXT    UNIQUE NOT NULL,
        created_at TEXT    NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS diagnoses (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id             INTEGER NOT NULL,
        filename            TEXT,
        prediction          TEXT    NOT NULL,
        confidence          REAL    NOT NULL,
        probabilities       TEXT    NOT NULL,
        original_image      TEXT,
        gradcam             TEXT,
        gradcampp           TEXT,
        integrated_gradients TEXT,
        occlusion           TEXT,
        xai_comparison      TEXT,
        created_at          TEXT    NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


init_db()
print("Database ready:", DB_PATH)


# ── Auth helpers ──────────────────────────────────────────────────────────────

def get_current_user():
    """Return user dict if token valid, else None."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not token:
        return None
    conn = get_db()
    row = conn.execute(
        "SELECT u.id, u.username FROM tokens t JOIN users u ON t.user_id = u.id WHERE t.token = ?",
        (token,)
    ).fetchone()
    conn.close()
    return {"id": row["id"], "username": row["username"]} if row else None


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Chưa đăng nhập hoặc phiên đã hết hạn."}), 401
        request.current_user = user
        return f(*args, **kwargs)
    return decorated


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/register", methods=["POST"])
def register():
    data     = request.get_json() or {}
    username = str(data.get("username") or "").strip()
    password = str(data.get("password") or "")

    if len(username) < 3:
        return jsonify({"error": "Tên đăng nhập phải có ít nhất 3 ký tự."}), 400
    if len(password) < 6:
        return jsonify({"error": "Mật khẩu phải có ít nhất 6 ký tự."}), 400

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, pw_hash, datetime.now().isoformat())
        )
        conn.commit()
        user_id = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()["id"]
        token   = secrets.token_hex(32)
        conn.execute(
            "INSERT INTO tokens (user_id, token, created_at) VALUES (?, ?, ?)",
            (user_id, token, datetime.now().isoformat())
        )
        conn.commit()
        return jsonify({"token": token, "username": username})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Tên đăng nhập đã tồn tại."}), 400
    finally:
        conn.close()


@app.route("/login", methods=["POST"])
def login():
    data     = request.get_json() or {}
    username = str(data.get("username") or "").strip()
    password = str(data.get("password") or "")

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not user or not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        conn.close()
        return jsonify({"error": "Tên đăng nhập hoặc mật khẩu không đúng."}), 401

    token = secrets.token_hex(32)
    conn.execute(
        "INSERT INTO tokens (user_id, token, created_at) VALUES (?, ?, ?)",
        (user["id"], token, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return jsonify({"token": token, "username": user["username"]})


@app.route("/logout", methods=["POST"])
@require_auth
def logout():
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    conn  = get_db()
    conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Đã đăng xuất."})


@app.route("/me", methods=["GET"])
@require_auth
def me():
    user = request.current_user
    conn = get_db()

    total = conn.execute(
        "SELECT COUNT(*) as cnt FROM diagnoses WHERE user_id = ?", (user["id"],)
    ).fetchone()["cnt"]

    class_rows = conn.execute(
        "SELECT prediction, COUNT(*) as cnt FROM diagnoses WHERE user_id = ? GROUP BY prediction",
        (user["id"],)
    ).fetchall()

    u_row = conn.execute("SELECT created_at FROM users WHERE id = ?", (user["id"],)).fetchone()
    conn.close()

    return jsonify({
        "id":          user["id"],
        "username":    user["username"],
        "created_at":  u_row["created_at"] if u_row else "",
        "total_scans": total,
        "stats":       {r["prediction"]: r["cnt"] for r in class_rows}
    })


@app.route("/change-password", methods=["POST"])
@require_auth
def change_password():
    user = request.current_user
    data = request.get_json() or {}
    old_pw  = str(data.get("old_password") or "")
    new_pw  = str(data.get("new_password") or "")

    if len(new_pw) < 6:
        return jsonify({"error": "Mật khẩu mới phải có ít nhất 6 ký tự."}), 400

    conn     = get_db()
    user_row = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()

    if not bcrypt.checkpw(old_pw.encode(), user_row["password_hash"].encode()):
        conn.close()
        return jsonify({"error": "Mật khẩu cũ không đúng."}), 401

    new_hash = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt()).decode()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user["id"]))
    conn.commit()
    conn.close()
    return jsonify({"message": "Đổi mật khẩu thành công."})


# ── History routes ────────────────────────────────────────────────────────────

@app.route("/history", methods=["GET"])
@require_auth
def get_history():
    user = request.current_user
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, prediction, confidence, probabilities, original_image, created_at "
        "FROM diagnoses WHERE user_id = ? ORDER BY created_at DESC LIMIT 100",
        (user["id"],)
    ).fetchall()
    conn.close()

    return jsonify({"history": [{
        "id":          r["id"],
        "fileName":    r["filename"] or "xray-image",
        "prediction":  r["prediction"],
        "confidence":  r["confidence"],
        "probabilities": json.loads(r["probabilities"] or "{}"),
        "originalImage": r["original_image"] or "",
        "date":        r["created_at"]
    } for r in rows]})


@app.route("/history/<int:diag_id>", methods=["GET"])
@require_auth
def get_history_detail(diag_id):
    user = request.current_user
    conn = get_db()
    row  = conn.execute(
        "SELECT * FROM diagnoses WHERE id = ? AND user_id = ?", (diag_id, user["id"])
    ).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Không tìm thấy bản ghi."}), 404

    return jsonify({
        "id":          row["id"],
        "fileName":    row["filename"] or "xray-image",
        "prediction":  row["prediction"],
        "confidence":  row["confidence"],
        "probabilities": json.loads(row["probabilities"] or "{}"),
        "originalImage": row["original_image"] or "",
        "xai_images": {
            "gradcam":               row["gradcam"] or "",
            "gradcampp":             row["gradcampp"] or "",
            "integrated_gradients":  row["integrated_gradients"] or "",
            "occlusion":             row["occlusion"] or ""
        },
        "xai_comparison": row["xai_comparison"] or "",
        "date": row["created_at"]
    })


@app.route("/history/<int:diag_id>", methods=["DELETE"])
@require_auth
def delete_history(diag_id):
    user   = request.current_user
    conn   = get_db()
    result = conn.execute(
        "DELETE FROM diagnoses WHERE id = ? AND user_id = ?", (diag_id, user["id"])
    )
    conn.commit()
    conn.close()

    if result.rowcount == 0:
        return jsonify({"error": "Không tìm thấy hoặc không có quyền xóa."}), 404
    return jsonify({"message": "Đã xóa bản ghi."})


# ── Model loading ─────────────────────────────────────────────────────────────

print("FILES:", os.listdir(BASE_DIR))
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model không tồn tại: {MODEL_PATH}")

print("Loading model…")
import tf_keras
model = tf_keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded!")


def get_model_input_size():
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        h, w = shape[1], shape[2]
        if h and w:
            return int(h), int(w)
    except Exception:
        pass
    return DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE


IMG_HEIGHT, IMG_WIDTH = get_model_input_size()
print("Model input size:", IMG_HEIGHT, IMG_WIDTH)


# ── Image / XAI helpers ───────────────────────────────────────────────────────

def pil_to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def normalize_map(arr):
    arr = np.array(arr, dtype=np.float32)
    arr -= arr.min()
    mx = arr.max()
    return arr / mx if mx > 1e-8 else arr


def jet_colormap(gray):
    gray = normalize_map(gray)
    r = np.clip(1.5 - np.abs(4 * gray - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * gray - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * gray - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(original_pil, heatmap, alpha=0.48):
    heatmap = normalize_map(heatmap)
    hm_img  = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(original_pil.size, Image.BILINEAR)
    hm_arr  = np.array(hm_img).astype(np.float32) / 255.0
    color   = jet_colormap(hm_arr).astype(np.float32)
    orig    = np.array(original_pil.convert("RGB")).astype(np.float32)
    blended = np.clip(orig * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def preprocess_image(image_file):
    original = Image.open(image_file).convert("RGB")
    resized  = original.resize((IMG_WIDTH, IMG_HEIGHT))
    arr      = np.expand_dims(np.array(resized).astype("float32") / 255.0, axis=0)
    return original, arr


def normalize_predictions(preds):
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = np.array(preds)
    if preds.ndim == 2:
        preds = preds[0]
    return preds.astype("float32")


def find_last_conv_layer(m):
    for layer in reversed(m.layers):
        if "conv2d" in layer.__class__.__name__.lower():
            return layer
        if hasattr(layer, 'layers'):
            nested = find_last_conv_layer(layer)
            if nested:
                return nested
    return None


def build_grad_model():
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, 'layers'):
            last_conv = find_last_conv_layer(layer)
            if last_conv:
                conv_model = tf_keras.models.Model(layer.inputs, [last_conv.output, layer.output])
                return conv_model, model.layers[idx + 1:]
    last_conv = find_last_conv_layer(model)
    if not last_conv:
        raise RuntimeError("Không tìm thấy Conv2D layer.")
    return tf_keras.models.Model(model.inputs, [last_conv.output, model.output]), None


def forward_grad(img_array, conv_model, tail):
    conv_out, x = conv_model(img_array, training=False)
    if tail:
        for layer in tail:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
    return conv_out, x


def generate_gradcam(img_array, pred_index):
    conv_model, tail = build_grad_model()
    with tf.GradientTape() as tape:
        conv_out, preds = forward_grad(img_array, conv_model, tail)
        score = preds[:, pred_index]
    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError("Không tính được gradient.")
    pw = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm = tf.nn.relu(tf.reduce_sum(conv_out[0] * pw, axis=-1))
    return normalize_map(hm.numpy())


def generate_gradcampp(img_array, pred_index):
    conv_model, tail = build_grad_model()
    with tf.GradientTape() as tape:
        conv_out, preds = forward_grad(img_array, conv_model, tail)
        score = preds[:, pred_index]
    grads   = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError("Không tính được gradient.")
    co      = conv_out[0]
    g       = grads[0]
    g2, g3  = tf.square(g), tf.square(g) * g
    denom   = 2.0 * g2 + g3 * tf.reduce_sum(co, axis=(0, 1))
    eps     = tf.keras.backend.epsilon()
    alpha   = g2 / tf.where(denom != 0.0, denom, eps)
    weights = tf.reduce_sum(alpha * tf.nn.relu(g), axis=(0, 1))
    hm      = tf.nn.relu(tf.reduce_sum(weights * co, axis=-1))
    return normalize_map(hm.numpy())


def generate_integrated_gradients(img_array, pred_index, steps=32):
    baseline    = np.zeros_like(img_array, dtype="float32")
    total_grads = np.zeros_like(img_array, dtype="float32")
    for alpha in np.linspace(0.0, 1.0, steps + 1):
        interp = tf.convert_to_tensor(baseline + alpha * (img_array - baseline))
        with tf.GradientTape() as tape:
            tape.watch(interp)
            score = model(interp, training=False)[:, pred_index]
        g = tape.gradient(score, interp)
        if g is not None:
            total_grads += g.numpy()
    ig = (img_array - baseline) * (total_grads / float(steps + 1))
    return normalize_map(np.mean(np.abs(ig[0]), axis=-1))


def generate_occlusion(img_array, pred_index, grid_size=8):
    base_score = float(normalize_predictions(model.predict(img_array, verbose=0))[pred_index])
    _, h, w, _ = img_array.shape
    ph, pw     = max(1, h // grid_size), max(1, w // grid_size)
    heatmap    = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, ph):
        for x in range(0, w, pw):
            occ = img_array.copy()
            occ[:, y:y+ph, x:x+pw, :] = 0.0
            score = float(normalize_predictions(model.predict(occ, verbose=0))[pred_index])
            heatmap[y:y+ph, x:x+pw] = max(0.0, base_score - score)
    return normalize_map(heatmap)


def generate_xai_images(original_pil, img_array, pred_index):
    gc  = overlay_heatmap(original_pil, generate_gradcam(img_array, pred_index),              alpha=0.52)
    gpp = overlay_heatmap(original_pil, generate_gradcampp(img_array, pred_index),            alpha=0.52)
    ig  = overlay_heatmap(original_pil, generate_integrated_gradients(img_array, pred_index), alpha=0.50)
    occ = overlay_heatmap(original_pil, generate_occlusion(img_array, pred_index),            alpha=0.55)
    strip = create_comparison_strip(original_pil, gc, gpp, ig, occ)
    return {"gradcam": gc, "gradcampp": gpp, "integrated_gradients": ig, "occlusion": occ, "comparison": strip}


def create_comparison_strip(orig, gc, gpp, ig, occ):
    images = [i.convert("RGB") for i in [orig, gc, gpp, ig, occ]]
    labels = ["Original", "Grad-CAM", "Grad-CAM++", "Integrated Gradients", "Occlusion"]
    iw, ih, th = 260, 260, 32
    canvas = Image.new("RGB", (iw * 5, ih + th), "white")
    draw   = ImageDraw.Draw(canvas)
    for i, (img, lbl) in enumerate(zip(images, labels)):
        draw.text((i * iw + 10, 8), lbl, fill="black")
        canvas.paste(img.resize((iw, ih), Image.BILINEAR), (i * iw, th))
    return canvas


# ── Main routes ───────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message":    "Backend ResNet101 X-ray AI đang chạy",
        "model":      MODEL_NAME,
        "input_size": [IMG_HEIGHT, IMG_WIDTH],
        "classes":    CLASS_NAMES
    })


@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image") or request.files.get("file")
    if not image_file:
        return jsonify({"error": "Không tìm thấy file ảnh."}), 400

    filename = image_file.filename or "xray-image"

    try:
        original_pil, img_array = preprocess_image(image_file)
        preds     = normalize_predictions(model.predict(img_array, verbose=0))

        if len(preds) != len(CLASS_NAMES):
            return jsonify({"error": "Model output sai số lớp.", "shape": list(preds.shape)}), 500

        pred_index  = int(np.argmax(preds))
        pred_key    = CLASS_NAMES[pred_index]
        pred_label  = DISPLAY_NAMES[pred_key]
        confidence  = float(preds[pred_index]) * 100
        probs       = {DISPLAY_NAMES[CLASS_NAMES[i]]: round(float(preds[i]) * 100, 2) for i in range(4)}

        print("Tạo XAI images…")
        xai = generate_xai_images(original_pil, img_array, pred_index)
        print("XAI done.")

        orig_url  = pil_to_data_url(original_pil)
        xai_urls  = {k: pil_to_data_url(v) for k, v in xai.items() if k != "comparison"}
        comp_url  = pil_to_data_url(xai["comparison"])

        # ── Save to history if user is authenticated ──
        current_user = get_current_user()
        diag_id = None
        if current_user:
            conn = get_db()
            cur  = conn.execute(
                "INSERT INTO diagnoses (user_id, filename, prediction, confidence, probabilities, "
                "original_image, gradcam, gradcampp, integrated_gradients, occlusion, xai_comparison, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    current_user["id"], filename, pred_label, round(confidence, 2),
                    json.dumps(probs), orig_url,
                    xai_urls["gradcam"], xai_urls["gradcampp"],
                    xai_urls["integrated_gradients"], xai_urls["occlusion"],
                    comp_url, datetime.now().isoformat()
                )
            )
            diag_id = cur.lastrowid
            conn.commit()
            conn.close()

        return jsonify({
            "model":          MODEL_NAME,
            "prediction":     pred_label,
            "confidence":     round(confidence, 2),
            "probabilities":  probs,
            "original_image": orig_url,
            "xai_comparison": comp_url,
            "diagnosis_id":   diag_id,
            "xai_images":     xai_urls
        })

    except Exception as err:
        print("Predict error:", err)
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)