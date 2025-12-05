import cv2, numpy as np, joblib
from pathlib import Path
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------
# Correct image paths
# --------------------------
low_files = [
    Path("lessprone.png"),
    Path("lessprone1.jpg"),
    Path("lessprone2.jpg"),
    Path("lessprone3.jpg"),
    Path("lessprone4.jpg"),
]

high_files = [
    Path("moreprone1.jpg"),
    Path("moreprone2.jpg"),
    Path("moreprone3.jpg"),
    Path("moreprone4.jpg"),
    Path("moreprone5.jpg"),
]

def norm(a):
    mn, mx = a.min(), a.max()
    if mx - mn < 1e-6:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def local_variance(arr, win):
    arrf = arr.astype(np.float32)
    m = cv2.blur(arrf, (win,win))
    m2 = cv2.blur(arrf*arrf, (win,win))
    v = m2 - m*m
    v[v < 0] = 0
    return v

def compute_features(img_bgr):
    # Make consistent size
    h, w = img_bgr.shape[:2]
    scale = 512 / max(h, w) if max(h, w) > 512 else 1.0
    if scale != 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))

    # Use uint8 grayscale to avoid OpenCV AVX bug
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray_u8 = img_gray.astype(np.uint8)

    # --------------------------
    # Multi-scale Sobel (SAFE)
    # --------------------------
    sobel_small = np.sqrt(
        cv2.Sobel(img_gray_u8, cv2.CV_64F, 1, 0, 3)**2 +
        cv2.Sobel(img_gray_u8, cv2.CV_64F, 0, 1, 3)**2
    )
    sobel_large = np.sqrt(
        cv2.Sobel(img_gray_u8, cv2.CV_64F, 1, 0, 7)**2 +
        cv2.Sobel(img_gray_u8, cv2.CV_64F, 0, 1, 7)**2
    )

    # --------------------------
    # Laplacian (SAFE)
    # --------------------------
    lap = np.abs(cv2.Laplacian(img_gray_u8, cv2.CV_64F))

    # --------------------------
    # Edge density
    # --------------------------
    edges = cv2.Canny(img_gray_u8, 50, 150)
    edge_density = cv2.blur(edges.astype(np.float32)/255.0, (25,25))

    # --------------------------
    # Morphology (SAFE)
    # --------------------------
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    blackhat = cv2.morphologyEx(img_gray_u8, cv2.MORPH_BLACKHAT, se)
    tophat  = cv2.morphologyEx(img_gray_u8, cv2.MORPH_TOPHAT, se)

    # --------------------------
    # Variance (still float safe)
    # --------------------------
    var_comb = 0.5 * norm(local_variance(img_gray_u8, 9)) + \
               0.5 * norm(local_variance(img_gray_u8, 31))

    # --------------------------
    # Vegetation proxy
    # --------------------------
    green = img_bgr[:,:,1].astype(np.float32)
    rgb_sum = img_bgr.sum(axis=2).astype(np.float32) + 1
    green_ratio = green / rgb_sum

    feats = OrderedDict()
    feats["sobel_small"] = sobel_small.mean()
    feats["sobel_large"] = sobel_large.mean()
    feats["lap"] = lap.mean()
    feats["edge_density"] = edge_density.mean()
    feats["blackhat"] = blackhat.mean()
    feats["tophat"] = tophat.mean()
    feats["var"] = var_comb.mean()
    feats["green"] = green_ratio.mean()
    feats["soil_exposure"] = (1 - green_ratio).mean()

    return list(feats.values()), list(feats.keys())


# --------------------------
# Build dataset
# --------------------------
X, y = [], []

for p in low_files:
    img = cv2.imread(str(p))
    if img is None:
        print("Missing:", p); continue
    fvals, fnames = compute_features(img)
    X.append(fvals); y.append(0)

for p in high_files:
    img = cv2.imread(str(p))
    if img is None:
        print("Missing:", p); continue
    fvals, fnames = compute_features(img)
    X.append(fvals); y.append(1)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("Loaded dataset:", X.shape, y.shape)

# --------------------------
# Train RF model
# --------------------------
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print("CV Accuracy:", scores, "Mean =", scores.mean())

pipe.fit(X, y)

# Save
joblib.dump(
    {"pipeline": pipe, "feature_names": fnames},
    "erosion_rf_pipeline.pkl"
)

print("Saved model: erosion_rf_pipeline.pkl")
