from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from .io import SROIEBox

def draw_box(img: np.ndarray, box: SROIEBox, color: Tuple[int,int,int], label: Optional[str] = None) -> None:
    x1, y1, x2, y2 = box.rect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def save_overlay(image_bgr: np.ndarray, kw: SROIEBox, val: SROIEBox, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image_bgr.copy()

    # keyword: hijau, value: merah (konvensi evidence)
    draw_box(canvas, kw, (0, 200, 0), "KEY")
    draw_box(canvas, val, (0, 0, 255), "VALUE")

    cv2.imwrite(str(out_path), canvas)
    
def save_overlay_single(image_bgr, box, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image_bgr.copy()
    draw_box(canvas, box, (0, 0, 255), "TOTAL+VALUE")
    cv2.imwrite(str(out_path), canvas)