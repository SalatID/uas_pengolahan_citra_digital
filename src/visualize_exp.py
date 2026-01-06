from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

def save_overlay_morphcc(image_bgr: np.ndarray, roi_rect: Tuple[int,int,int,int], comps, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = image_bgr.copy()
    x1,y1,x2,y2 = roi_rect
    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 2)  # ROI total

    # draw CC boxes inside ROI (offset)
    for (x, y, w, h, area) in comps:
        cv2.rectangle(vis, (x1+x, y1+y), (x1+x+w, y1+y+h), (0,200,0), 1)

    cv2.imwrite(str(out_path), vis)

def save_mask(mask: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)
