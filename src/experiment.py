from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

from .io import parse_sroie_box_file
from .roi_extract import extract_total_value
from .detectors import baseline_global, morphcc_roi
from .eval import compute_metrics, save_metrics
from .visualize_exp import save_overlay_morphcc, save_mask


def get_file_id(filename: str) -> str:
    return filename.split("_")[0]


def load_roi_from_sroie(project_root: Path, file_id: str) -> Optional[Tuple[int, int, int, int]]:
    for split in ["train", "test"]:
        box_path = project_root / split / "box" / f"{file_id}.txt"
        if not box_path.exists():
            continue
        boxes = parse_sroie_box_file(box_path)
        res = extract_total_value(boxes)
        if res is None:
            return None
        kind, kw, val, norm = res
        return kw.rect if kind == "single" else val.rect
    return None


def _collect_scores_auth(
    project_root: Path,
    data_dir: Path,
    mode: str,
    baseline_thr: float,
    morph_params: Dict,
) -> List[float]:
    labels_path = data_dir / "labels.csv"
    images_dir = data_dir / "images"
    scores_auth: List[float] = []

    with labels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["filename"]
            true = row["label"]
            if true != "authentic":
                continue
            img_path = images_dir / fn
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            if mode == "baseline_global":
                res = baseline_global(image, thr=baseline_thr)
                scores_auth.append(res.score)

            elif mode == "morphcc_roi":
                file_id = get_file_id(fn)
                roi = load_roi_from_sroie(project_root, file_id)
                if roi is None:
                    continue
                # pakai thr dummy agar hanya collect score
                res = morphcc_roi(image, roi, thr=999999.0, **morph_params)
                scores_auth.append(res.score)
            else:
                raise ValueError("Unknown mode")

    return scores_auth


def choose_threshold(scores_auth: List[float], method: str = "mad15") -> float:
    """
    Default dibuat konsisten dengan paper: MAD-based.
    method:
      - mad15: median + 1.5*MAD (sensitif, cocok UAS)
      - mad30: median + 3.0*MAD (lebih konservatif)
      - p85 / p90 (opsional)
    """
    if not scores_auth:
        return 0.0

    s = np.array(scores_auth, dtype=np.float32)
    s = s[np.isfinite(s)]  # buang NaN/inf

# Winsorize: buang pengaruh outlier ekstrem pada threshold
    if s.size >= 10:
        hi = float(np.percentile(s, 95))
        s = np.clip(s, 0.0, hi)
    if method == "mad10":
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        if mad < 1e-6:
            return float(np.percentile(s, 85))
        return float(med + 1.0 * mad)

    if method == "mad15":
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))

        # guard: jika MAD hampir nol -> fallback percentile
        if mad < 1e-6:
            return float(np.percentile(s, 85))

        return float(med + 1.5 * mad)

    if method == "mad30":
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        return float(med + 3.0 * mad)

    if method == "p85":
        return float(np.percentile(s, 85))

    if method == "p90":
        return float(np.percentile(s, 90))

    raise ValueError(f"Unknown threshold method: {method}")


def run(
    mode: str,
    project_root: Path,
    data_dir: Path,
    out_dir: Path,
    run_id: str = "exp1",
    baseline_thr: float = 18.0,
    threshold_tuning: bool = True,
    threshold_method: str = "mad15",   # <- FIX: default mad15
    morph_params: Optional[Dict] = None,
):
    morph_params = morph_params or {"ksize": 3, "min_area": 6, "include_gradient_in_score": False}

    labels_path = data_dir / "labels.csv"
    images_dir = data_dir / "images"

    run_dir = out_dir / run_id
    overlays_dir = run_dir / "overlays"
    masks_dir = run_dir / "masks"
    grads_dir = run_dir / "gradients"  # simpan morphological gradient mask

    tuned_thr = None
    if threshold_tuning:
        scores_auth = _collect_scores_auth(
            project_root=project_root,
            data_dir=data_dir,
            mode=mode,
            baseline_thr=baseline_thr,
            morph_params=morph_params,
        )
        s = np.array(scores_auth, dtype=np.float32)
        finite = s[np.isfinite(s)]
        print(f"[SCORE-AUTH] mode={mode} n={len(scores_auth)} finite={finite.size} "
            f"min={float(np.min(finite)) if finite.size else 'NA'} "
            f"med={float(np.median(finite)) if finite.size else 'NA'} "
            f"max={float(np.max(finite)) if finite.size else 'NA'}")
        tuned_thr = choose_threshold(scores_auth, method=threshold_method)
        print(f"[THR] mode={mode} method={threshold_method} thr={tuned_thr:.4f} (from {len(scores_auth)} authentic)")

    y_true, y_pred = [], []
    preds_rows = [("filename", "true_label", "pred_label", "score")]

    n_skipped = 0

    with labels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["filename"]
            true = row["label"]

            img_path = images_dir / fn
            image = cv2.imread(str(img_path))
            if image is None:
                n_skipped += 1
                continue

            if mode == "baseline_global":
                thr = float(tuned_thr) if tuned_thr is not None else float(baseline_thr)
                res = baseline_global(image, thr=thr)

            elif mode == "morphcc_roi":
                file_id = get_file_id(fn)
                roi = load_roi_from_sroie(project_root, file_id)
                if roi is None:
                    n_skipped += 1
                    continue

                thr = float(tuned_thr) if tuned_thr is not None else 4.0
                res = morphcc_roi(image, roi, thr=thr, **morph_params)

                # evidence overlay + masks
                # NOTE: visualize_exp.save_overlay_morphcc expects comps format (x,y,w,h,area)
                comps = res.artifacts.get("components", [])
                save_overlay_morphcc(image, roi, comps, overlays_dir / f"{fn}.png")

                # closed mask
                save_mask(res.artifacts["mask_closed"], masks_dir / f"{fn}_mask_closed.png")

                # gradient mask (Gap #2 alignment)
                save_mask(res.artifacts["mask_gradient"], grads_dir / f"{fn}_mask_gradient.png")

            else:
                raise ValueError("Unknown mode")

            y_true.append(true)
            y_pred.append(res.pred_label)
            preds_rows.append((fn, true, res.pred_label, f"{res.score:.6f}"))

    run_dir.mkdir(parents=True, exist_ok=True)

    preds_path = run_dir / f"preds_{mode}.csv"
    with preds_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(preds_rows)

    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "mode": mode,
        "n_samples": len(y_true),
        "n_skipped": n_skipped,
        "threshold_tuning": threshold_tuning,
        "threshold_method": threshold_method if threshold_tuning else None,
        "threshold_value": float(tuned_thr) if tuned_thr is not None else (baseline_thr if mode == "baseline_global" else 4.0),
        "morph_params": morph_params if mode == "morphcc_roi" else None,
    })

    metrics_path = run_dir / f"metrics_{mode}.json"
    save_metrics(metrics, metrics_path)

    print(f"[DONE] mode={mode} samples={len(y_true)} skipped={n_skipped}")
    print(f"[OUT]  {preds_path}")
    print(f"[OUT]  {metrics_path}")


if __name__ == "__main__":
    root = Path(".")
    data_dir = root / "data"
    out_dir = root / "outputs"

    # Baseline
    run(
        mode="baseline_global",
        project_root=root,
        data_dir=data_dir,
        out_dir=out_dir,
        run_id="exp1",
        baseline_thr=18.0,
        threshold_tuning=True,
        threshold_method="mad10",  # dibuat konsisten
    )

    # Morph + CC + Shape Features + (Gradient as artifact)
    run(
        mode="morphcc_roi",
        project_root=root,
        data_dir=data_dir,
        out_dir=out_dir,
        run_id="exp1",
        threshold_tuning=True,
        threshold_method="mad10",
        morph_params={"ksize": 3, "min_area": 10, "include_gradient_in_score": False},
    )
