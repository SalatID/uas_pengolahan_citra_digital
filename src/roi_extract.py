import re
from typing import List, Optional, Tuple

from .io import SROIEBox

# ----------------------------
# Patterns & normalization
# ----------------------------

KEY_PATTERNS = [
    (re.compile(r"\bTOTAL\s+AMOUNT\b", re.I), 110),
    (re.compile(r"\bNETT\s+TOTAL\b", re.I), 105),
    (re.compile(r"\bGRAND\s+TOTAL\b", re.I), 100),
    (re.compile(r"\bTOTAL\b", re.I), 90),
    (re.compile(r"\bAMOUNT\s+DUE\b", re.I), 85),
    (re.compile(r"\bBALANCE\s+DUE\b", re.I), 85),
]

# Tangkap angka utama dengan 2 desimal (boleh ada ribuan)
# Contoh yang valid: "RM8.85", "8.85", "1,293.00 MYR"
NUM_CAPTURE_RE = re.compile(r"(?P<num>[\d,]+\.\d{2})")


def normalize_money_text(text: str) -> str:
    """
    Strip currency/suffix untuk perhitungan:
    'TOTAL AMOUNT: RM8.85' -> '8.85'
    'RM1,293.00 MYR' -> '1293.00'
    """
    t = text.strip()
    m = NUM_CAPTURE_RE.search(t)
    if not m:
        return ""
    return m.group("num").replace(",", "")


def is_money_like(text: str) -> bool:
    return bool(NUM_CAPTURE_RE.search(text.strip()))


def _keyword_priority(text: str) -> Optional[int]:
    # normalize whitespace agar match stabil
    t = " ".join(text.strip().split())
    for rx, prio in KEY_PATTERNS:
        if rx.search(t):
            return prio
    return None


# ----------------------------
# Geometry helpers
# ----------------------------

def _y_overlap_ratio(a: SROIEBox, b: SROIEBox) -> float:
    ax1, ay1, ax2, ay2 = a.rect
    bx1, by1, bx2, by2 = b.rect
    inter = max(0, min(ay2, by2) - max(ay1, by1))
    denom = min(ay2 - ay1, by2 - by1)
    if denom <= 0:
        return 0.0
    return inter / denom


# ----------------------------
# Extractors
# ----------------------------

def extract_total_from_single_box(boxes: List[SROIEBox]) -> Optional[Tuple[SROIEBox, str]]:
    """
    Skenario B: keyword + nilai berada pada 1 box yang sama.
    Contoh: 'TOTAL AMOUNT: RM8.85'
    Return (box_total, norm_value)
    """
    cands = []
    for b in boxes:
        prio = _keyword_priority(b.text)
        if prio is None:
            continue
        if not is_money_like(b.text):
            continue
        norm = normalize_money_text(b.text)
        if not norm:
            continue

        # tie-break: prio tinggi + teks lebih informatif
        # (lebih panjang biasanya mengandung keyword + value)
        cands.append((prio, -len(b.text), b, norm))

    if not cands:
        return None

    cands.sort(key=lambda x: (-x[0], x[1]))  # prio desc, len desc
    _, _, box, norm = cands[0]
    return box, norm


def extract_total_value_pair(boxes: List[SROIEBox]) -> Optional[Tuple[SROIEBox, SROIEBox]]:
    """
    Skenario A: keyword box terpisah dari value box.
    Contoh:
      'TOTAL:'  (keyword box)
      '193.00'  (value box)
    Return (keyword_box, value_box)
    """
    candidates = []

    for kw in boxes:
        prio = _keyword_priority(kw.text)
        if prio is None:
            continue

        best_val = None
        best_score = None

        for bx in boxes:
            if bx is kw:
                continue

            # harus sebaris (overlap vertikal)
            if _y_overlap_ratio(kw, bx) < 0.30:
                continue

            # harus di kanan (pakai center-x agar robust)
            if bx.center[0] <= kw.center[0]:
                continue

            # harus angka
            if not is_money_like(bx.text):
                continue

            # skor: jarak horizontal + penalti panjang teks (prefer tanpa suffix)
            dist = bx.rect[0] - kw.rect[2]  # xmin_value - xmax_kw (bisa negatif kecil)
            raw = bx.text.strip()
            penalty = 0.5 * len(raw)

            score = max(dist, 0) + penalty

            if best_score is None or score < best_score:
                best_score = score
                best_val = bx

        if best_val is not None:
            candidates.append((prio, best_score, kw, best_val))

    if not candidates:
        return None

    # pilih keyword prioritas tertinggi, lalu score terkecil
    candidates.sort(key=lambda x: (-x[0], x[1]))
    _, _, kw, val = candidates[0]
    return kw, val


def extract_total_value(boxes: List[SROIEBox]):
    """
    Wrapper final:
    - Coba single-box extraction dulu (TOTAL + value dalam satu box)
    - Kalau tidak ada, fallback ke pair extraction (keyword box + value box terpisah)

    Return:
      ("single", total_box, None, norm_value)
      ("pair", keyword_box, value_box, norm_value)
      None jika tidak ditemukan
    """
    single = extract_total_from_single_box(boxes)
    if single is not None:
        box, norm = single
        return ("single", box, None, norm)

    pair = extract_total_value_pair(boxes)
    if pair is None:
        return None

    kw, val = pair
    norm = normalize_money_text(val.text)
    return ("pair", kw, val, norm)


# ----------------------------
# Debug helper (optional)
# ----------------------------

def debug_total_candidates(boxes: List[SROIEBox], limit: int = 20) -> None:
    print("=== DEBUG: keyword candidates ===")
    kws = []
    for b in boxes:
        pr = _keyword_priority(b.text)
        if pr is not None:
            kws.append((pr, b.text, b.rect))
    kws.sort(key=lambda x: -x[0])
    for pr, txt, rect in kws[:limit]:
        print(f"[KW prio={pr}] '{txt}' rect={rect}")

    print("\n=== DEBUG: money-like boxes ===")
    shown = 0
    for b in boxes:
        if is_money_like(b.text):
            print(f"[VAL] '{b.text}' rect={b.rect}")
            shown += 1
            if shown >= limit:
                break
