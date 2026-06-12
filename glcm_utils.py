"""
Modul ekstraksi fitur GLCM (Gray Level Co-occurrence Matrix).
Digunakan oleh train_model.py dan app.py agar konsisten.
"""

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

IMG_SIZE = (128, 128)
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_ANGLE_LABELS = ["0°", "45°", "90°", "135°"]
GLCM_LEVELS = 32
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

# LBP params
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
LBP_NBINS = LBP_POINTS + 2  # uniform LBP -> P+2 bins

# Color histogram params (HSV)
COLOR_BINS = 8  # bins per channel


def _open_image(image_path_or_obj):
    if isinstance(image_path_or_obj, Image.Image):
        return image_path_or_obj
    return Image.open(image_path_or_obj)


def preprocess_image(image_path_or_obj):
    """Buka gambar, konversi ke grayscale, resize, dan kuantisasi level keabuan."""
    img = _open_image(image_path_or_obj).convert("L")
    img = img.resize(IMG_SIZE)
    arr = np.array(img)
    arr_q = (arr.astype(np.float64) / 255.0 * (GLCM_LEVELS - 1)).astype(np.uint8)
    return arr, arr_q


def extract_glcm_features(image_path_or_obj):
    """Ekstrak vektor fitur GLCM dari satu gambar."""
    _, arr_q = preprocess_image(image_path_or_obj)

    glcm = graycomatrix(
        arr_q,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    features = []
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)
        features.extend(vals.flatten())

    return np.array(features, dtype=np.float64), glcm


def extract_glcm_detail(image_path_or_obj):
    """
    Ekstrak fitur GLCM beserta detail per-properti (rata-rata tiap arah)
    untuk ditampilkan ke pengguna.
    """
    feats, glcm = extract_glcm_features(image_path_or_obj)

    detail = {}
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)          # shape (n_distances, n_angles)
        detail[prop] = {
            "mean": float(np.mean(vals)),
            "per_angle": {
                GLCM_ANGLE_LABELS[a]: float(np.mean(vals[:, a]))
                for a in range(len(GLCM_ANGLES))
            },
        }

    return feats, detail


def get_feature_names():
    names = []
    for prop in GLCM_PROPS:
        for d in GLCM_DISTANCES:
            for a_idx in range(len(GLCM_ANGLES)):
                names.append(f"{prop}_d{d}_{GLCM_ANGLE_LABELS[a_idx]}")
    return names


def extract_lbp_features(image_path_or_obj):
    """Ekstrak histogram Local Binary Pattern (LBP) dari citra grayscale."""
    img = _open_image(image_path_or_obj).convert("L")
    img = img.resize(IMG_SIZE)
    arr = np.array(img)

    lbp = local_binary_pattern(arr, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_NBINS + 1),
        range=(0, LBP_NBINS),
    )
    hist = hist.astype(np.float64)
    hist /= (hist.sum() + 1e-7)  # normalisasi
    return hist


def get_lbp_feature_names():
    return [f"lbp_bin_{i}" for i in range(LBP_NBINS)]


def extract_color_features(image_path_or_obj):
    """
    Ekstrak fitur warna: histogram HSV (per channel) + color moments
    (mean, std, skewness) per channel HSV.
    """
    img = _open_image(image_path_or_obj).convert("RGB").resize(IMG_SIZE)
    hsv = np.array(img.convert("HSV")).astype(np.float64)

    features = []

    # Histogram per channel HSV
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=COLOR_BINS, range=(0, 256))
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-7)
        features.extend(hist)

    # Color moments: mean, std, skewness per channel
    for ch in range(3):
        channel = hsv[:, :, ch].ravel()
        mean = np.mean(channel)
        std = np.std(channel)
        skew = np.mean(((channel - mean) / (std + 1e-7)) ** 3)
        features.extend([mean, std, skew])

    return np.array(features, dtype=np.float64)


def get_color_feature_names():
    names = []
    for ch_name in ["H", "S", "V"]:
        for b in range(COLOR_BINS):
            names.append(f"color_hist_{ch_name}_bin{b}")
    for ch_name in ["H", "S", "V"]:
        for stat in ["mean", "std", "skew"]:
            names.append(f"color_{ch_name}_{stat}")
    return names


def extract_combined_features(image_path_or_obj):
    """Gabungkan fitur GLCM + LBP + Warna menjadi satu vektor."""
    glcm_feats, _ = extract_glcm_features(image_path_or_obj)
    lbp_feats = extract_lbp_features(image_path_or_obj)
    color_feats = extract_color_features(image_path_or_obj)
    return np.concatenate([glcm_feats, lbp_feats, color_feats])


def get_combined_feature_names():
    return get_feature_names() + get_lbp_feature_names() + get_color_feature_names()
