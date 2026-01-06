from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import cv2
import numpy as np


@dataclass
class DetectResult:
    pred_label: str  # "authentic" / "tampered"
    score: float
    artifacts: Dict[str, Any]


# ----------------------------
# Utils
# ----------------------------

def _clamp_rect(rect: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def crop_rect(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clamp_rect(rect, w, h)
    return img[y1:y2, x1:x2].copy()


def _mad_z(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Robust z-score yang stabil:
    - buang NaN/inf
    - guard MAD minimum
    - clip z agar tidak meledak
    """
    x = x.astype(np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))

    # Guard: jika MAD terlalu kecil, anggap tidak ada variasi -> z ~ 0
    if mad < 1e-3:
        return np.zeros_like(x, dtype=np.float32)

    z = (x - med) / (1.4826 * mad + eps)
    z = np.clip(z, -10.0, 10.0)  # clip agar tidak ekstrem
    return z.astype(np.float32)



# ----------------------------
# Baseline (global)
# ----------------------------

def baseline_global(image_bgr: np.ndarray, thr: float = 18.0) -> DetectResult:
    """
    Baseline sederhana: edge density global.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    score = float(edges.mean())

    pred = "tampered" if score > thr else "authentic"
    return DetectResult(pred_label=pred, score=score, artifacts={"edge_mean": score, "thr": thr})


# ----------------------------
# Morphology
# ----------------------------

def _adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    # Foreground putih (digit/stroke) -> pakai INV
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )


def _morph_ops(bin_img: np.ndarray, ksize: int = 3) -> Dict[str, np.ndarray]:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)

    # Gap #2: Morphological Gradient
    gradient = cv2.morphologyEx(closed, cv2.MORPH_GRADIENT, k, iterations=1)

    return {
        "bin": bin_img,
        "opened": opened,
        "closed": closed,
        "gradient": gradient,
    }


# ----------------------------
# Connected Components + Shape Features
# ----------------------------

def _cc_stats(mask: np.ndarray, min_area: int = 10):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        comps.append({
            "id": int(i),
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(area),
            "centroid": (float(cx), float(cy)),
        })
    return comps, labels


def _component_features(mask: np.ndarray, comp_bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
    """
    Hitung fitur bentuk untuk 1 komponen (di dalam ROI mask biner).

    Fitur yang dihitung (sesuai klaim paper):
      - area (berbasis contour area)
      - perimeter (keliling kontur)
      - aspect_ratio
      - extent
      - solidity
      - circularity
      - holes_count (proxy topology/Euler)
      - stroke_thickness (estimasi via distance transform)
      - bbox_w, bbox_h
    """
    x, y, w, h = comp_bbox
    roi = mask[y:y + h, x:x + w]
    if roi.size == 0:
        return {}

    # Pastikan biner 0/255 (uint8)
    roi_bin = (roi > 0).astype(np.uint8) * 255

    # Kontur eksternal untuk perimeter/area/hull
    contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}

    cnt = max(contours, key=cv2.contourArea)

    peri = float(cv2.arcLength(cnt, True))
    cnt_area = float(cv2.contourArea(cnt)) + 1e-6  # guard

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) + 1e-6

    aspect_ratio = float(w) / float(h + 1e-6)
    extent = float(cnt_area) / float(w * h + 1e-6)
    solidity = float(cnt_area) / float(hull_area)

    circularity = 0.0
    if peri > 1e-6:
        circularity = float(4.0 * np.pi * cnt_area / (peri * peri))

    # holes_count (proxy): hitung kontur child dengan RETR_CCOMP
    contours2, hier = cv2.findContours(roi_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hier is not None:
        for i in range(len(contours2)):
            parent = hier[0][i][3]
            if parent != -1:
                holes += 1
    holes_count = float(holes)

    # Stroke thickness (distance transform), stabil + clamp
    dt = cv2.distanceTransform(roi_bin, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    fg = dt[roi_bin > 0]

    if fg.size > 0:
        fg = fg[np.isfinite(fg)]
        if fg.size > 0:
            fg = np.clip(fg, 0.0, 50.0)   # clamp jarak
            med_dt = float(np.median(fg))
            stroke_thickness = float(2.0 * med_dt)
        else:
            stroke_thickness = 0.0
    else:
        stroke_thickness = 0.0

    # ----------------------------
    # Clamp fitur agar stabil
    # ----------------------------
    solidity = float(np.clip(solidity, 0.0, 1.0))
    extent = float(np.clip(extent, 0.0, 1.0))
    circularity = float(np.clip(circularity, 0.0, 1.2))   # circularity normal <= 1
    holes_count = float(np.clip(holes_count, 0.0, 10.0))  # digit biasanya 0-2 holes; clamp aman
    stroke_thickness = float(np.clip(stroke_thickness, 0.0, 30.0))
    peri = float(np.clip(peri, 0.0, 2000.0))
    aspect_ratio = float(np.clip(aspect_ratio, 0.05, 5.0))

    return {
        "area": float(cnt_area),
        "perimeter": float(peri),
        "aspect_ratio": float(aspect_ratio),
        "extent": float(extent),
        "solidity": float(solidity),
        "circularity": float(circularity),
        "holes_count": float(holes_count),
        "stroke_thickness": float(stroke_thickness),
        "bbox_w": float(w),
        "bbox_h": float(h),
    }

def _filter_digit_like(comps: List[Dict[str, Any]], roi_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
    """
    Gap #3: arahkan analisis ke 'digit-like components' dalam ROI total.
    Heuristik digit-like (ringan, defensible):
      - ukuran bbox masuk akal (tidak terlalu kecil/noise, tidak terlalu besar/line)
      - aspect ratio w/h dalam rentang tertentu
      - area relatif terhadap bbox (extent) tidak ekstrem
    """
    H, W = roi_shape
    out = []
    for c in comps:
        x, y, w, h = c["bbox"]
        if w < 4 or h < 6:
            continue
        if w > 0.9 * W or h > 0.9 * H:
            continue

        ar = w / (h + 1e-6)
        if ar < 0.15 or ar > 1.20:
            # digit biasanya tidak terlalu pipih horizontal / tidak terlalu lebar
            continue

        # extent rough: area/(w*h) but use stored feature later if available
        out.append(c)
    return out


def _score_anomaly_interdigit(features: List[Dict[str, float]]) -> Tuple[float, Dict[str, Any]]:
    """
    Skor anomali berbasis konsistensi antar 'digit-like components' di ROI.
    Pakai robust z-score (MAD) pada fitur-fitur tertentu.

    Return:
      final_score, detail
    """
    if len(features) < 3:
        # terlalu sedikit komponen untuk analisis antar-digit
        return 0.0, {"n_digits": len(features), "reason": "too_few_components"}

    # pilih fitur paling informatif dan stabil
    keys = ["stroke_thickness", "solidity", "extent", "circularity", "holes_count", "aspect_ratio"]
    X = {}
    for k in keys:
        X[k] = np.array([f.get(k, 0.0) for f in features], dtype=np.float32)

    z = {}
    for k, v in X.items():
        zz = _mad_z(v)
        # jika _mad_z mengembalikan array kosong (shouldn't), fallback zeros
        if zz.size == 0:
            zz = np.zeros_like(v, dtype=np.float32)
        # pastikan panjang sama
        if zz.shape[0] != v.shape[0]:
            zz = np.zeros_like(v, dtype=np.float32)
        z[k] = np.abs(zz)


    # aggregate per component
    per_comp = np.zeros(len(features), dtype=np.float32)
    for k in keys:
        per_comp += z[k]

    # anomali = jumlah komponen yang melewati ambang z robust
    # ambang 2.5 cukup standar
    flagged = per_comp > 2.5 * len(keys) * 0.15  # skala ringan agar tidak terlalu ketat
    n_flag = int(np.sum(flagged))

    # score utama: proporsi komponen anomali + tingkat z rata-rata
    prop = n_flag / max(1, len(features))
    mean_z = float(np.mean(per_comp))

    score = 6.0 * prop + 0.15 * mean_z  # bobot interpretable

    detail = {
        "n_digits": len(features),
        "n_flagged": n_flag,
        "prop_flagged": float(prop),
        "mean_zsum": mean_z,
        "flagged_indices": np.where(flagged)[0].tolist(),
    }
    return float(score), detail


# ----------------------------
# Morph + CC + Shape Feature detector (ROI)
# ----------------------------

def morphcc_roi(
    image_bgr: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    thr: float = 4.0,
    ksize: int = 3,
    min_area: int = 10,
    include_gradient_in_score: bool = False,
) -> DetectResult:
    """
    Versi yang konsisten dengan paper:
    - Morphology (OPEN, CLOSE, + GRADIENT sebagai artefak)
    - Connected components
    - Shape features per component (perimeter, solidity, holes, thickness, etc.)
    - Skor anomali relatif antar 'digit-like components' (MAD z-score)
    """
    roi = crop_rect(image_bgr, roi_rect)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    bin_img = _adaptive_binarize(gray)
    morph = _morph_ops(bin_img, ksize=ksize)
    mask = morph["closed"]
    gradient = morph["gradient"]

    comps, labels = _cc_stats(mask, min_area=min_area)
    comps = _filter_digit_like(comps, roi_shape=mask.shape)

    # hitung fitur per komponen
    feats = []
    for c in comps:
        f = _component_features(mask, c["bbox"])
        if not f:
            continue

        # buang fitur yang NaN/inf
        ok = True
        for k, v in f.items():
            if not np.isfinite(float(v)):
                ok = False
                break
        if not ok:
            continue

        c["features"] = f
        feats.append(f)

    # skor anomali antar-digit
    score_digits, detail = _score_anomaly_interdigit(feats)

    grad_mean = float(gradient.mean())
    score = float(score_digits)

    # optional: gradient contribution (minor) sebagai “edge/contour emphasis”
    grad_mean = float(gradient.mean())
    # Clip skor agar stabil dan threshold meaningful
    score = float(np.clip(score, 0.0, 50.0))

    if include_gradient_in_score:
        score += 0.01 * grad_mean

    pred = "tampered" if score > float(thr) else "authentic"

    return DetectResult(
        pred_label=pred,
        score=float(score),
        artifacts={
            "roi_rect": roi_rect,
            "mask_closed": mask,
            "mask_gradient": gradient,
            "components": [c["bbox"] + (c["area"],) for c in comps],  # (x,y,w,h,area)
            "features_per_comp": feats,
            "digit_anomaly": detail,
            "thr": float(thr),
            "ksize": int(ksize),
            "min_area": int(min_area),
            "grad_mean": grad_mean,
            "include_gradient_in_score": bool(include_gradient_in_score),
        },
    )
