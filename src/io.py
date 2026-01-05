from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

@dataclass
class SROIEBox:
    poly: List[Tuple[int, int]]          # 4 points
    text: str
    rect: Tuple[int, int, int, int]      # x_min, y_min, x_max, y_max
    center: Tuple[float, float]
    h: int

def parse_sroie_box_file(box_path: Path) -> List[SROIEBox]:
    boxes: List[SROIEBox] = []
    lines = box_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue

        parts = raw.split(",")
        if len(parts) < 9:
            continue

        coords = list(map(int, parts[:8]))
        text = ",".join(parts[8:]).strip()

        poly = [(coords[0], coords[1]), (coords[2], coords[3]),
                (coords[4], coords[5]), (coords[6], coords[7])]

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        h = max(1, y_max - y_min)

        boxes.append(
            SROIEBox(poly=poly, text=text, rect=(x_min, y_min, x_max, y_max), center=(cx, cy), h=h)
        )

    return boxes
