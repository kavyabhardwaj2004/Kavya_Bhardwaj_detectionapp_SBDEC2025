# pages/5_🌍_Erosion_Prediction.py
import streamlit as st
import cv2, numpy as np, joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.ui_utils import apply_custom_css

apply_custom_css()
st.set_page_config(page_title="Erosion Prediction", layout="wide")
st.title("🌍 Soil Erosion Susceptibility Prediction")

MODEL_PATH = Path(__file__).parent.parent / "erosion_rf_pipeline.pkl"

# ---------- feature & heatmap helpers (same as training) ----------
def norm(arr):
    a = arr.copy().astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx - mn < 1e-6:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def local_variance(arr, win):
    arrf = arr.astype(np.float32)
    m = cv2.blur(arrf, (win,win))
    m2 = cv2.blur(arrf*arrf, (win,win))
    var = m2 - m*m
    var[var < 0] = 0
    return var


def compute_features_array(img_bgr):
    """
    Extract EXACTLY the same 9 features used during training.
    This must match the training script feature order:

    1. sobel_small_mean
    2. sobel_large_mean
    3. lap_mean
    4. edge_density_mean
    5. blackhat_mean
    6. tophat_mean
    7. var_comb_mean
    8. green_ratio_mean
    9. soil_exposure_mean
    """

    # -------------------------
    # Resize for consistency
    # -------------------------
    h, w = img_bgr.shape[:2]
    scale = 512 / max(h, w) if max(h, w) > 512 else 1.0
    if scale != 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    # -------------------------
    # Convert to grayscale
    # -------------------------
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray_u8 = img_gray.astype(np.uint8)  # IMPORTANT for avoiding OpenCV AVX crash

    # -------------------------
    # Sobel small scale
    # -------------------------
    sobel_small = np.sqrt(
        cv2.Sobel(img_gray_u8, cv2.CV_64F, 1, 0, ksize=3) ** 2 +
        cv2.Sobel(img_gray_u8, cv2.CV_64F, 0, 1, ksize=3) ** 2
    )

    # -------------------------
    # Sobel large scale
    # -------------------------
    sobel_large = np.sqrt(
        cv2.Sobel(img_gray_u8, cv2.CV_64F, 1, 0, ksize=7) ** 2 +
        cv2.Sobel(img_gray_u8, cv2.CV_64F, 0, 1, ksize=7) ** 2
    )

    # -------------------------
    # Laplacian
    # -------------------------
    lap = np.abs(cv2.Laplacian(img_gray_u8, cv2.CV_64F))

    # -------------------------
    # Edge density using Canny
    # -------------------------
    edges = cv2.Canny(img_gray_u8, 50, 150)
    edge_density = cv2.blur(edges.astype(np.float32) / 255.0, (25, 25))

    # -------------------------
    # Morphological blackhat / tophat
    # -------------------------
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(img_gray_u8, cv2.MORPH_BLACKHAT, se)
    tophat = cv2.morphologyEx(img_gray_u8, cv2.MORPH_TOPHAT, se)

    # -------------------------
    # Variance (texture roughness)
    # -------------------------
    def local_variance(arr, win):
        arr = arr.astype(np.float32)
        m = cv2.blur(arr, (win, win))
        m2 = cv2.blur(arr * arr, (win, win))
        v = m2 - m * m
        v[v < 0] = 0
        return v

    def norm(a):
        mn, mx = a.min(), a.max()
        if mx - mn < 1e-6:
            return np.zeros_like(a)
        return (a - mn) / (mx - mn)

    var_small = local_variance(img_gray_u8, 9)
    var_large = local_variance(img_gray_u8, 31)
    var_comb = 0.5 * norm(var_small) + 0.5 * norm(var_large)

    # -------------------------
    # Vegetation proxy
    # -------------------------
    green = img_bgr[:, :, 1].astype(np.float32)
    rgb_sum = img_bgr.sum(axis=2).astype(np.float32) + 1
    green_ratio = green / rgb_sum
    soil_exposure = 1 - green_ratio

    # -------------------------
    # FINAL 9 FEATURES (ORDER MATCHES TRAINING MODEL)
    # -------------------------
    features = [
        sobel_small.mean(),
        sobel_large.mean(),
        lap.mean(),
        edge_density.mean(),
        blackhat.mean(),
        tophat.mean(),
        var_comb.mean(),
        green_ratio.mean(),
        soil_exposure.mean(),
    ]

    return np.array(features, dtype=np.float32)


def compute_heatmap_and_score(image, original, debug=False):
    # image: preprocessed grayscale normalized (0..1)
    # original: BGR uint8
    img = (image * 255.0).astype(np.float32)
    sobel_small_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_small_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_small = np.sqrt(sobel_small_x**2 + sobel_small_y**2)
    sobel_large_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=7)
    sobel_large_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=7)
    sobel_large = np.sqrt(sobel_large_x**2 + sobel_large_y**2)
    def n(a):
        mn, mx = a.min(), a.max()
        if mx-mn < 1e-6: return np.zeros_like(a)
        return (a-mn)/(mx-mn)
    sobel_small_n = n(sobel_small); sobel_large_n = n(sobel_large)
    img_u8 = np.clip(img,0,255).astype(np.uint8)
    edges = cv2.Canny(img_u8, 50, 150)
    k = 25 if max(img.shape) > 200 else 15
    kernel = np.ones((k,k), dtype=np.float32)/(k*k)
    edge_density = cv2.filter2D(edges.astype(np.float32)/255.0, -1, kernel)
    edge_density_n = n(edge_density)
    se_size = 21 if max(img.shape)>300 else 11
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size,se_size))
    blackhat = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, se).astype(np.float32)
    tophat = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, se).astype(np.float32)
    blackhat_n = n(blackhat); tophat_n = n(tophat)
    var_small = local_variance(img, 9); var_large = local_variance(img, 31 if max(img.shape)>200 else 15)
    var_comb = 0.5*n(var_small) + 0.5*n(var_large)
    green = original[:,:,1].astype(np.float32); rgb_sum = original.sum(axis=2).astype(np.float32)+1e-6
    green_ratio = green / rgb_sum
    soil_exposure_n = n(1.0 - green_ratio)

    heatmap = (
        0.22 * n(sobel_small) +
        0.28 * n(sobel_large) +
        0.20 * edge_density_n +
        0.15 * blackhat_n +
        0.10 * var_comb +
        0.05 * soil_exposure_n
    )
    heatmap = n(heatmap)
    score = (
        0.22 * sobel_small_n.mean() +
        0.28 * sobel_large_n.mean() +
        0.20 * edge_density_n.mean() +
        0.15 * blackhat_n.mean() +
        0.10 * var_comb.mean() +
        0.05 * soil_exposure_n.mean()
    )
    score = float(np.clip(score + 0.12 * blackhat_n.mean(), 0.0, 1.0))
    if debug:
        debug_dict = {
            "sobel_small": n(sobel_small),
            "sobel_large": n(sobel_large),
            "edges": edge_density_n,
            "valley": blackhat_n,
            "ridge": tophat_n,
            "variance": var_comb,
            "soil_exposure": soil_exposure_n,
            "heatmap": heatmap,
            "score": score
        }
        return heatmap, score, debug_dict
    return heatmap, score

# --------- load model if present ----------
model_obj = None
if MODEL_PATH.exists():
    try:
        model_obj = joblib.load(str(MODEL_PATH))
        pipeline = model_obj['pipeline']
        feature_names = model_obj.get('feature_names', None)
        st.success("Loaded trained erosion model.")
    except Exception as e:
        st.warning("Failed to load model: " + str(e))
        pipeline = None
else:
    pipeline = None

# UI
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
show_debug = st.checkbox("Show intermediate feature maps (debug)", value=False)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if original is None:
        st.error("Failed to read image.")
    else:
        st.image(original, channels="BGR", use_column_width=True)
        # preprocess
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        processed = gray.astype(np.float32) / 255.0

        # compute heatmap + rule-based score
        if show_debug:
            heatmap, score, debug = compute_heatmap_and_score(processed, original, debug=True)
        else:
            heatmap, score = compute_heatmap_and_score(processed, original, debug=False)

        overlay = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

        st.subheader("🌡️ Erosion Heatmap Overlay")
        st.image(overlay, channels="BGR", use_column_width=True)
        st.metric("Rule-based Risk Level (0–1)", f"{score:.3f}")

        # ML prediction if model available
        if pipeline is not None:
            feat_vec = compute_features_array(original).reshape(1, -1)
            pred = pipeline.predict(feat_vec)[0]
            pred_proba = pipeline.predict_proba(feat_vec)[0] if hasattr(pipeline, "predict_proba") else None
            labels = {0: "Low", 1: "High"}
            st.subheader("🧠 ML Prediction (Random Forest)")
            st.write("Predicted class:", labels.get(pred, str(pred)))
            if pred_proba is not None:
                st.write("Probability:", {labels[i]: float(pred_proba[i]) for i in [0,1]})
        else:
            st.info("No trained ML model found. Run `train_erosion_model.py` to create one.")

        if show_debug:
            st.subheader("Intermediate feature maps (debug)")
            cols = st.columns(4)
            maps = ["sobel_small","sobel_large","edges","valley","variance","soil_exposure","heatmap"]
            for i, name in enumerate(maps):
                arr = debug[name]
                # Convert to displayable 0..255 grayscale
                disp = (np.clip(arr,0,1)*255).astype(np.uint8)
                cols[i % 4].image(disp, use_column_width=True)
