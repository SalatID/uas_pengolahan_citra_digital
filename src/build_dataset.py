import csv
from pathlib import Path
import cv2

from .io import parse_sroie_box_file
from .roi_extract import extract_total_value
from .tamper import tamper_patch

def build(split_dir: Path, out_data_dir: Path, n_limit: int = 50):
    img_dir = split_dir / "img"
    box_dir = split_dir / "box"

    out_img_dir = out_data_dir / "images"
    out_mask_dir = out_data_dir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    rows = [("filename", "label")]

    ids = [p.stem for p in img_dir.glob("*.jpg")]
    ids = ids[:n_limit]

    for i, file_id in enumerate(ids, start=1):
        img_path = img_dir / f"{file_id}.jpg"
        box_path = box_dir / f"{file_id}.txt"
        if not box_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        boxes = parse_sroie_box_file(box_path)
        res = extract_total_value(boxes)
        if res is None:
            continue

        kind, kw, val, norm = res
        rect = kw.rect if kind == "single" else val.rect

        # simpan authentic (copy)
        auth_name = f"{file_id}_auth.jpg"
        cv2.imwrite(str(out_img_dir / auth_name), image)
        rows.append((auth_name, "authentic"))

        # buat tampered
        tampered, mask, tampered_rect = tamper_patch(image, rect, seed=i)
        tamp_name = f"{file_id}_tamp.jpg"
        mask_name = f"{file_id}_mask.png"

        cv2.imwrite(str(out_img_dir / tamp_name), tampered)
        cv2.imwrite(str(out_mask_dir / mask_name), mask)
        rows.append((tamp_name, "tampered"))

        print(f"[{i}/{len(ids)}] OK {file_id}: total_norm={norm}")

    # write labels.csv
    labels_path = out_data_dir / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    print(f"[DONE] labels: {labels_path}")
    print(f"[DONE] images: {out_img_dir}")

if __name__ == "__main__":
    root = Path(".")
    split = root / "train"
    out_data = root / "data"
    build(split, out_data, n_limit=50)
