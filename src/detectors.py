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


def _clamp_rect(rect: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def crop_rect(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clamp_rect(rect, w, h)
    return img[y1:y2, x1:x2].copy()


# ----------------------------
# Baseline detector
# ----------------------------

def baseline_global(image_bgr: np.ndarray, thr: float = 18.0) -> DetectResult:
    """
    Baseline sederhana (global): edge density seluruh citra.
    Ini sengaja lemah untuk membuktikan bahwa pemalsuan lokal tidak tertangkap oleh statistik global.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    score = float(edges.mean())  # 0..255

    pred = "tampered" if score > thr else "authentic"
    return DetectResult(
        pred_label=pred,
        score=score,
        artifacts={
            "edge_mean": score,
            "thr": thr,
        },
    )


# ----------------------------
# Morphology + CC on ROI
# ----------------------------

def _adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    # THRESH_BINARY_INV agar foreground putih (digit/stroke)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )


def _morph_clean(bin_img: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)
    return closed


def _connected_components(bin_img: np.ndarray, min_area: int = 10) -> List[Tuple[int, int, int, int, int]]:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    comps = []
    for i in range(1, n):  # skip background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comps.append((int(x), int(y), int(w), int(h), int(area)))
    return comps


def morphcc_roi(
    image_bgr: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    thr: float = 6.0,
    ksize: int = 3,
    min_area: int = 10,
) -> DetectResult:
    """
    Novelty awal: morfologi + connected components di ROI total.
    Score berbasis fragmentasi komponen + edge density lokal.

    thr bisa:
    - angka tetap (default 6.0)
    - hasil data-driven tuning (median/MAD atau percentile)
    """
    roi = crop_rect(image_bgr, roi_rect)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    bin_img = _adaptive_binarize(gray)
    mask = _morph_clean(bin_img, ksize=ksize)

    comps = _connected_components(mask, min_area=min_area)

    edges = cv2.Canny(gray, 80, 160)
    edge_mean = float(edges.mean())

    # komponen kecil = indikasi fragmentasi/artefak paste
    small = sum(1 for (_, _, w, h, area) in comps if (area < 60) or (w * h < 120))

    # skor sederhana tapi interpretable (bisa Anda ablate nanti)
    score = 0.7 * float(small) + 0.03 * float(edge_mean)
    pred = "tampered" if score > thr else "authentic"

    return DetectResult(
        pred_label=pred,
        score=float(score),
        artifacts={
            "roi_rect": roi_rect,
            "mask": mask,
            "components": comps,
            "edge_mean_roi": edge_mean,
            "small_components": small,
            "thr": thr,
            "ksize": ksize,
            "min_area": min_area,
        },
    )
