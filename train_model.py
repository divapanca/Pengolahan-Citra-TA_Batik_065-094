"""
Training script: Ekstraksi fitur GLCM + LBP + Warna dari dataset Batik
Nusantara (dengan augmentasi data), lalu melatih model SVM (dengan
grid search) untuk klasifikasi motif batik.
"""

import os
import numpy as np
import joblib
from PIL import Image, ImageEnhance
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

from glcm_utils import (
    extract_combined_features,
    get_combined_feature_names,
    IMG_SIZE,
    GLCM_DISTANCES,
    GLCM_ANGLES,
    GLCM_LEVELS,
    GLCM_PROPS,
)

# ====== KONFIGURASI ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "D:/semester 6/Pengolahan Citra/data/backup batik/Batik Nusantara/raw_batik_v2/raw_batik_v2"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "model")


# ====== AUGMENTASI ======
def augment_image(img):
    """Hasilkan beberapa versi augmentasi dari sebuah PIL Image (RGB)."""
    augmented = [img]  # original

    # Rotasi
    for angle in [90, 180, 270, 45, 135]:
        augmented.append(img.rotate(angle, expand=True, fillcolor=(255, 255, 255)))

    # Flip
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    augmented.append(img.transpose(Image.FLIP_TOP_BOTTOM))

    # Brightness variation
    b_enhancer = ImageEnhance.Brightness(img)
    augmented.append(b_enhancer.enhance(1.3))
    augmented.append(b_enhancer.enhance(0.7))

    # Contrast variation
    c_enhancer = ImageEnhance.Contrast(img)
    augmented.append(c_enhancer.enhance(1.3))

    # Center crop (zoom-in) ~85%
    w, h = img.size
    cw, ch = int(w * 0.85), int(h * 0.85)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch)).resize((w, h))
    augmented.append(cropped)

    return augmented


def load_dataset(folder, augment=False, cache_name=None):
    cache_path = os.path.join(MODEL_DIR, cache_name) if cache_name else None
    if cache_path and os.path.exists(cache_path):
        print(f"  Memuat fitur dari cache: {cache_path}")
        d = joblib.load(cache_path)
        return d["X"], d["y"], d["classes"]

    X, y = [], []
    classes = sorted(os.listdir(folder))
    for cls in classes:
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path):
            continue
        files = os.listdir(cls_path)
        for fname in files:
            fpath = os.path.join(cls_path, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                variants = augment_image(img) if augment else [img]

                for variant in variants:
                    feats = extract_combined_features(variant)
                    X.append(feats)
                    y.append(cls)
            except Exception as e:
                print(f"  [SKIP] {fpath}: {e}")
        print(f"  selesai kelas: {cls} ({len(files)} file)")

    X, y = np.array(X), np.array(y)
    if cache_path:
        joblib.dump({"X": X, "y": y, "classes": classes}, cache_path)
    return X, y, classes


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=== Ekstraksi fitur (GLCM + LBP + Warna) dari data TRAIN (dengan augmentasi) ===")
    X_train, y_train, classes = load_dataset(TRAIN_DIR, augment=True, cache_name="cache_train.pkl")
    print(f"Jumlah sampel train (setelah augmentasi): {X_train.shape[0]}, jumlah fitur: {X_train.shape[1]}")
    print(f"Kelas ({len(classes)}): {classes}")

    print("\n=== Ekstraksi fitur dari data TEST (tanpa augmentasi) ===")
    X_test, y_test, _ = load_dataset(TEST_DIR, augment=False, cache_name="cache_test.pkl")
    print(f"Jumlah sampel test: {X_test.shape[0]}")

    le = LabelEncoder()
    le.fit(classes)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=50, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
    ])

    print("\n=== Training SVM (tanpa grid search) ===")
    best_model = pipeline
    best_model.fit(X_train, y_train_enc)
    grid_best_params = {"svm__C": 50, "svm__gamma": "scale", "svm__class_weight": "balanced"}
    grid_best_score = None

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"\nAkurasi pada data TEST: {acc*100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

    joblib.dump(best_model, os.path.join(MODEL_DIR, "glcm_svm_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    meta = {
        "img_size": IMG_SIZE,
        "distances": GLCM_DISTANCES,
        "angles": GLCM_ANGLES,
        "levels": GLCM_LEVELS,
        "props": GLCM_PROPS,
        "feature_names": get_combined_feature_names(),
        "classes": list(le.classes_),
        "test_accuracy": acc,
        "cv_accuracy": grid_best_score,
        "best_params": grid_best_params,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "augmented": True,
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta.pkl"))

    print(f"\nModel tersimpan di: {MODEL_DIR}")


if __name__ == "__main__":
    main()
