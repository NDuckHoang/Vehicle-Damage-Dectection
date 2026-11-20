"""
Giai doan 3 - Truc quan hoa annotation de kiem tra nhan truoc khi train.

Moi anh ket qua gom 2 nua dat canh nhau:
    TRAI : polygon goc doc tu VIA JSON (ten class tieng Viet)
    PHAI : polygon doc tu file label YOLO, doi nguoc ve pixel (ten class tieng Anh)
Neu hai ben trung nhau -> chuyen doi dung.

Chon anh:
    - Moi class 3 anh (tap train)
    - Anh co nhieu vung hu hong / nhieu loai hu hong
    - Anh co polygon bi clip (vuot bien) va anh co polygon bi bo qua (lay tu conversion_log)
    - Vai anh tap val

Chay:
    python scripts/visualize_annotations.py            (mac dinh)
    python scripts/visualize_annotations.py --per-class 5
"""
import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
OUT_DIR = PROJECT_ROOT / "results" / "annotations"
LOG_FILE = PROJECT_ROOT / "reports" / "conversion_log.txt"
DATASET_ROOT = Path(r"D:\Dataset")
JSON_FILES = {"train": DATASET_ROOT / "0Train_via_annos.json",
              "val": DATASET_ROOT / "0Val_via_annos.json"}

# Mau co dinh cho tung class_id (BGR)
COLORS = [(0, 0, 255), (0, 200, 255), (255, 128, 0), (200, 0, 200),
          (0, 200, 0), (255, 0, 128), (0, 128, 255)]
VI_TO_ID = {"mop_lom": 0, "tray_son": 1, "vo_kinh": 2, "mat_bo_phan": 3,
            "thung": 4, "rach": 5, "be_den": 6}  # giong CLASS_MAP trong convert_via_to_yolo.py


def load_class_names():
    """Doc ten class tu data.yaml."""
    names = {}
    for line in (DATA_DIR / "data.yaml").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s+(\d+):\s*([^#]+)", line)
        if m:
            names[int(m.group(1))] = m.group(2).strip()
    return names


def read_yolo_label(path, w, h):
    """Doc file label YOLO-Seg -> list (class_id, mang diem pixel Nx2)."""
    polys = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        v = line.split()
        pts = np.array(v[1:], dtype=float).reshape(-1, 2) * [w, h]  # bo chuan hoa
        polys.append((int(v[0]), pts))
    return polys


def draw_polygons(img, polys, title):
    """
    Ve polygon: to mau ban trong suot + vien + nhan ten class.
    polys: list (class_id, ten hien thi, mang diem Nx2)
    """
    overlay = img.copy()
    for cid, _, pts in polys:
        cv2.fillPoly(overlay, [pts.astype(np.int32)], COLORS[cid % len(COLORS)])
    out = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)  # tron 35% mau mask

    scale = max(img.shape[:2]) / 1000  # co chu theo kich thuoc anh
    thick = max(1, int(2 * scale))
    for cid, label, pts in polys:
        color = COLORS[cid % len(COLORS)]
        cv2.polylines(out, [pts.astype(np.int32)], True, color, thick)
        # Dat nhan o diem tren cung cua polygon
        x, y = pts[np.argmin(pts[:, 1])].astype(int)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, thick)
        x = min(max(x, 0), out.shape[1] - tw)
        y = max(y, th + 4)
        cv2.rectangle(out, (x, y - th - 4), (x + tw, y), color, -1)
        cv2.putText(out, label, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

    # Tieu de goc trai tren
    cv2.rectangle(out, (0, 0), (out.shape[1], int(40 * scale)), (0, 0, 0), -1)
    cv2.putText(out, title, (10, int(28 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def visualize(split, rec, names, tag):
    name = rec["name"]
    img = cv2.imread(str(DATA_DIR / "images" / split / name))
    h, w = img.shape[:2]

    # Trai: polygon goc tu JSON (chua clip, ve dung nhu annotation)
    orig = [(VI_TO_ID[r["class"]], r["class"],
             np.array(list(zip(r["all_x"], r["all_y"])), dtype=float)) for r in rec["regions"]]
    # Phai: polygon tu file label YOLO
    yolo = [(cid, names[cid], pts) for cid, pts in
            read_yolo_label(DATA_DIR / "labels" / split / (Path(name).stem + ".txt"), w, h)]

    left = draw_polygons(img, orig, f"VIA JSON goc: {len(orig)} vung")
    right = draw_polygons(img, yolo, f"YOLO label: {len(yolo)} vung")
    combo = np.hstack([left, right])
    # Thu nho cho de xem (chieu rong toi da 1600px)
    if combo.shape[1] > 1600:
        s = 1600 / combo.shape[1]
        combo = cv2.resize(combo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    out_path = OUT_DIR / f"{tag}__{split}__{Path(name).stem}.jpg"
    cv2.imwrite(str(out_path), combo, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path, len(orig), len(yolo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=3, help="so anh moi class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = load_class_names()
    data = {s: json.load(open(p, encoding="utf-8")) for s, p in JSON_FILES.items()}
    rng = random.Random(args.seed)

    selected = []  # (split, ten anh, tag)
    # 1. Moi class N anh tu tap train
    for vi, cid in VI_TO_ID.items():
        cands = [k for k, r in data["train"].items() if any(g["class"] == vi for g in r["regions"])]
        for k in rng.sample(cands, min(args.per_class, len(cands))):
            selected.append(("train", k, f"class{cid}_{names[cid].replace(' ', '')}"))
    # 2. Anh co nhieu loai hu hong khac nhau nhat
    multi = sorted(data["train"], key=lambda k: -len({g["class"] for g in data["train"][k]["regions"]}))
    selected += [("train", k, "multi_class") for k in multi[:2]]
    # 3. Anh co polygon bi clip / bi bo qua (doc tu log cua Giai doan 2)
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\[(train|val)\] (\S+\.jpg) region \d+ .*(vuot bien|BO QUA)", line)
            if m:
                tag = "clipped" if m.group(3) == "vuot bien" else "skipped"
                selected.append((m.group(1), m.group(2), tag))
    # 4. Vai anh tap val
    selected += [("val", k, "val_random") for k in rng.sample(list(data["val"]), 3)]

    # Bo trung lap, giu thu tu
    seen, final = set(), []
    for s, k, t in selected:
        if (s, k) not in seen:
            seen.add((s, k))
            final.append((s, k, t))

    print(f"Ve {len(final)} anh -> {OUT_DIR}")
    for s, k, t in final:
        path, n_orig, n_yolo = visualize(s, data[s][k], names, t)
        note = "" if n_orig == n_yolo else f"  (JSON {n_orig} != YOLO {n_yolo}, xem conversion_log)"
        print(f"  {path.name}{note}")


if __name__ == "__main__":
    main()
