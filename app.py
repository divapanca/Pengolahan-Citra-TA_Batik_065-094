"""
Aplikasi Flask: Klasifikasi Motif Batik Nusantara menggunakan
Gray Level Co-occurrence Matrix (GLCM) + SVM.
"""

import os
import io
import base64
import uuid
from datetime import datetime

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glcm_utils import (
    extract_glcm_detail,
    extract_combined_features,
    preprocess_image,
    GLCM_PROPS,
    GLCM_ANGLE_LABELS,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

# ====== Load model ======
MODEL_PATH = os.path.join(MODEL_DIR, "glcm_svm_model.pkl")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.pkl")

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
label_encoder = joblib.load(LE_PATH) if os.path.exists(LE_PATH) else None
meta = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}

PROP_LABELS = {
    "contrast": "Contrast",
    "dissimilarity": "Dissimilarity",
    "homogeneity": "Homogeneity",
    "energy": "Energy",
    "correlation": "Correlation",
    "ASM": "Angular Second Moment (ASM)",
}

# Nama tampilan kelas batik (label asli -> "Daerah - Motif")
def pretty_class_name(cls_name):
    return cls_name.replace("_", " ")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def array_to_base64_png(arr, cmap="gray"):
    """Konversi array numpy 2D menjadi gambar PNG base64 (untuk ditampilkan di HTML)."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(arr, cmap=cmap)
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/")
def index():
    classes = meta.get("classes", [])
    sample_classes = []
    for c in classes:
        sample_path = url_for("static", filename=f"samples/{c}.jpg")
        sample_classes.append({"name": c, "pretty": pretty_class_name(c), "img": sample_path})

    stats = {
        "n_classes": len(classes),
        "n_train": meta.get("n_train", "-"),
        "n_test": meta.get("n_test", "-"),
        "accuracy": round(meta.get("test_accuracy", 0) * 100, 2),
    }
    return render_template("index.html", page="dashboard", samples=sample_classes, stats=stats)


@app.route("/classify")
def classify_page():
    return render_template("classify.html", page="classify")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model belum tersedia. Jalankan train_model.py terlebih dahulu."}), 500

    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang dikirim."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung. Gunakan JPG, JPEG, PNG, atau BMP."}), 400

    # Simpan file upload sementara
    ext = file.filename.rsplit(".", 1)[1].lower()
    fname = f"{uuid.uuid4().hex}.{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    file.save(fpath)

    try:
        # Ekstraksi fitur GLCM (untuk tampilan detail)
        _, detail = extract_glcm_detail(fpath)
        gray_arr, quant_arr = preprocess_image(fpath)

        # Ekstraksi fitur gabungan (GLCM + LBP + Warna) untuk prediksi
        combined_feats = extract_combined_features(fpath)
        feats_2d = combined_feats.reshape(1, -1)
        pred_idx = model.predict(feats_2d)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        proba = model.predict_proba(feats_2d)[0]
        order = np.argsort(proba)[::-1][:5]
        top5 = [
            {
                "label": label_encoder.inverse_transform([idx])[0],
                "pretty": pretty_class_name(label_encoder.inverse_transform([idx])[0]),
                "prob": round(float(proba[idx]) * 100, 2),
            }
            for idx in order
        ]

        # Visualisasi gambar grayscale & kuantisasi untuk ditampilkan
        gray_b64 = array_to_base64_png(gray_arr, cmap="gray")
        quant_b64 = array_to_base64_png(quant_arr, cmap="gray")

        # Susun detail fitur GLCM agar rapi
        glcm_table = []
        for prop in GLCM_PROPS:
            d = detail[prop]
            glcm_table.append({
                "name": PROP_LABELS.get(prop, prop),
                "key": prop,
                "mean": round(d["mean"], 5),
                "per_angle": {a: round(v, 5) for a, v in d["per_angle"].items()},
            })

        result = {
            "success": True,
            "uploaded_image": url_for("static", filename=f"uploads/{fname}"),
            "gray_image": f"data:image/png;base64,{gray_b64}",
            "quant_image": f"data:image/png;base64,{quant_b64}",
            "prediction": {
                "label": pred_label,
                "pretty": pretty_class_name(pred_label),
                "confidence": round(float(np.max(proba)) * 100, 2),
            },
            "top5": top5,
            "glcm_features": glcm_table,
            "angle_labels": GLCM_ANGLE_LABELS,
            "timestamp": datetime.now().strftime("%d %b %Y, %H:%M:%S"),
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan saat memproses gambar: {str(e)}"}), 500


@app.route("/about")
def about():
    return render_template("about.html", page="about", meta=meta)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
