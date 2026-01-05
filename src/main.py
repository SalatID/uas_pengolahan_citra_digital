import json
from pathlib import Path
import cv2

from .io import parse_sroie_box_file
from .roi_extract import debug_total_candidates, extract_total_value, normalize_money_text
from .visualize import save_overlay,save_overlay_single

def run_one(split_dir: Path, file_id: str, out_dir: Path) -> None:
    img_path = split_dir / "img" / f"{file_id}.jpg"
    box_path = split_dir / "box" / f"{file_id}.txt"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not box_path.exists():
        raise FileNotFoundError(f"Box not found: {box_path}")

    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    boxes = parse_sroie_box_file(box_path)
    pair = extract_total_value(boxes)
    if pair is None:
        debug_total_candidates(boxes)
        raise RuntimeError(f"No TOTAL/value pair found for: {file_id}")

    kind, kw, val, norm_value = pair
    raw_value = (kw.text.strip() if kind == "single" else val.text.strip())
    norm_value = normalize_money_text(raw_value)

    run_id = "run1"
    overlay_path = out_dir / run_id / "overlays" / f"{file_id}_overlay.png"
    report_path  = out_dir / run_id / "reports"  / f"{file_id}.json"

    if kind == "single":
        save_overlay_single(image, kw, overlay_path)
    else:
        save_overlay(image, kw, val, overlay_path)


    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
    "image": f"{file_id}.jpg",
    "mode": "roi_extract",
    "method": kind,
    "total_box": {"text": kw.text, "bbox": list(kw.rect)},
    "value": {"raw_text": raw_value, "norm_text": norm_value},
    "notes": ["currency suffix stripped for computation"]
    }
    if kind == "pair":
        report["value"]["bbox"] = list(val.rect)
    else:
        report["value"]["bbox"] = list(kw.rect)

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] overlay: {overlay_path}")
    print(f"[OK] report : {report_path}")
    print(f"[PAIR] {kw.text} -> raw='{raw_value}' norm='{norm_value}'")

if __name__ == "__main__":
    # Sesuaikan file_id dan split (train/test) di sini untuk test cepat
    root = Path(".")
    out_dir = root / "outputs"

    # Ganti ini sesuai lokasi file Anda (train atau test)
    split = root / "train"
    file_id = "X51007846371"

    run_one(split, file_id, out_dir)
