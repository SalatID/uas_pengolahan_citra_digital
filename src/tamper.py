from pathlib import Path
import cv2
import numpy as np

def tamper_patch(image_bgr: np.ndarray, rect, seed: int = 1):
    """
    rect: (x1,y1,x2,y2) area yang akan ditamper (TOTAL box)
    Mengembalikan: tampered_image, mask (uint8 0/255)
    """
    rng = np.random.default_rng(seed)
    x1, y1, x2, y2 = rect
    h, w = image_bgr.shape[:2]

    # safety clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid rect for tamper")

    patch = image_bgr[y1:y2, x1:x2].copy()
    ph, pw = patch.shape[:2]

    # transform parameters (ringan, realistis)
    angle = float(rng.uniform(-4.0, 4.0))
    scale = float(rng.uniform(0.95, 1.05))
    tx = int(rng.integers(-4, 5))
    ty = int(rng.integers(-3, 4))

    # rotate+scale
    M = cv2.getRotationMatrix2D((pw/2, ph/2), angle, scale)
    warped = cv2.warpAffine(patch, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # optional blur ringan (kadang copy-paste terlihat blur)
    if rng.random() < 0.5:
        warped = cv2.GaussianBlur(warped, (3, 3), 0)

    # lokasi tempel (sekitar ROI asli)
    dst_x1 = int(np.clip(x1 + tx, 0, w - pw))
    dst_y1 = int(np.clip(y1 + ty, 0, h - ph))
    dst_x2 = dst_x1 + pw
    dst_y2 = dst_y1 + ph

    tampered = image_bgr.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    # tempel patch (hard paste untuk memunculkan tepi)
    tampered[dst_y1:dst_y2, dst_x1:dst_x2] = warped
    mask[dst_y1:dst_y2, dst_x1:dst_x2] = 255

    return tampered, mask, (dst_x1, dst_y1, dst_x2, dst_y2)
