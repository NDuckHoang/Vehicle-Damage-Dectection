"""
Giai doan 2 - Chuyen annotation VIA JSON (VehiDE) sang dinh dang YOLOv8-Seg.

Dinh dang 1 dong trong file label YOLO-Seg:
    <class_id> x1 y1 x2 y2 ... xn yn      (toa do da chuan hoa ve [0, 1])

Nguyen tac:
- Ghep anh-nhan bang ten file CHINH XAC (truong "name" trong JSON).
- Class ID lay tu bang CLASS_MAP co dinh ben duoi (khong tu sinh).
- Giu nguyen cach chia train/val goc cua dataset.
- Anh trong project la HARDLINK toi anh goc (khong ton them dung luong, khong sua anh goc).
  Rieng anh JPEG bi cat cut: ghi 1 ban sao ma hoa lai bang OpenCV (anh goc giu nguyen).
- Polygon vuot bien -> clip ve trong anh. Polygon < 3 diem / dien tich ~0 -> bo qua, ghi log.

Chay:
    python scripts/convert_via_to_yolo.py
"""
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path(r"D:\Dataset")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "dataset"
LOG_FILE = PROJECT_ROOT / "reports" / "conversion_log.txt"

SPLITS = {
    "train": (DATASET_ROOT / "0Train_via_annos.json", DATASET_ROOT / "image" / "image"),
    "val": (DATASET_ROOT / "0Val_via_annos.json", DATASET_ROOT / "validation" / "validation"),
}

# Bang anh xa class: ten trong JSON -> (class_id, ten tieng Anh).
# Thu tu co dinh, KHONG doi sau khi da train (neu doi, model cu se sai nhan).
CLASS_MAP = {
    "mop_lom":     (0, "Dent"),
    "tray_son":    (1, "Scratch"),
    "vo_kinh":     (2, "Broken glass"),
    "mat_bo_phan": (3, "Lost part"),
    "thung":       (4, "Punctured"),
    "rach":        (5, "Torn"),
    "be_den":      (6, "Broken light"),
}

MIN_AREA_PX = 1.0       # dien tich toi thieu (pixel^2) sau khi clip
BIG_OVERSHOOT_PX = 50   # canh bao neu polygon vuot bien nhieu hon muc nay


def is_truncated_jpeg(path):
    """JPEG chuan ket thuc bang 2 byte FF D9; thieu -> file bi cat cut."""
    with open(path, "rb") as f:
        f.seek(-2, os.SEEK_END)
        return f.read(2) != b"\xff\xd9"


def prepare_image(src, dst):
    """
    Dua anh vao thu muc dataset cua project.
    Tra ve "link" | "reencode" | "exists".
    """
    if dst.exists():
        return "exists"
    if is_truncated_jpeg(src):
        # Anh bi cat cut: doc bang OpenCV roi ghi file MOI (khong dong cham anh goc)
        img = cv2.imread(str(src))
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return "reencode"
    os.link(src, dst)  # hardlink: cung o dia D:, khong copy du lieu
    return "link"


def convert_polygon(xs, ys, w, h):
    """
    Chuyen 1 polygon sang toa do YOLO chuan hoa.
    Tra ve (list toa do chuan hoa hoac None, ly do loi, so pixel vuot bien).
    """
    if xs is None or ys is None or len(xs) != len(ys):
        return None, "all_x/all_y thieu hoac khac do dai", 0
    pts = np.array(list(zip(xs, ys)), dtype=float)

    # Muc vuot bien (de ghi log), sau do clip ve trong khung anh
    over = max(0, -pts[:, 0].min(), -pts[:, 1].min(),
               pts[:, 0].max() - w, pts[:, 1].max() - h)
    pts[:, 0] = np.clip(pts[:, 0], 0, w)
    pts[:, 1] = np.clip(pts[:, 1], 0, h)

    # Bo cac diem trung lien tiep (co the xuat hien sau khi clip)
    keep = [0] + [i for i in range(1, len(pts)) if not np.array_equal(pts[i], pts[i - 1])]
    pts = pts[keep]
    if len(pts) > 1 and np.array_equal(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return None, f"it hon 3 diem ({len(pts)})", over

    # Dien tich theo cong thuc Shoelace
    area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) -
                     np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))
    if area < MIN_AREA_PX:
        return None, f"dien tich ~0 ({area:.1f})", over

    # Chuan hoa: x / width, y / height
    pts[:, 0] /= w
    pts[:, 1] /= h
    return pts.flatten().tolist(), None, over


def convert_split(split, json_path, img_dir, log):
    img_out = OUT_DIR / "images" / split
    lbl_out = OUT_DIR / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    stats = Counter()
    cls_count = Counter()
    for key, rec in data.items():
        name = rec.get("name", key)
        src = img_dir / name
        if not src.exists():
            log(f"[{split}] KHONG CO ANH: {name}")
            stats["missing_image"] += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            log(f"[{split}] KHONG DOC DUOC ANH: {name}")
            stats["unreadable"] += 1
            continue
        h, w = img.shape[:2]

        lines = []
        for j, r in enumerate(rec.get("regions", []) or []):
            cls_name = r.get("class")
            if cls_name not in CLASS_MAP:
                log(f"[{split}] {name} region {j}: class khong xac dinh '{cls_name}' -> bo qua")
                stats["unknown_class"] += 1
                continue
            coords, err, over = convert_polygon(r.get("all_x"), r.get("all_y"), w, h)
            if over > 0:
                stats["clipped"] += 1
            if over > BIG_OVERSHOOT_PX:
                log(f"[{split}] {name} region {j} ({cls_name}): vuot bien {over:.0f}px -> da clip")
            if coords is None:
                log(f"[{split}] {name} region {j} ({cls_name}): BO QUA - {err}")
                stats["skipped_polygon"] += 1
                continue
            cls_id = CLASS_MAP[cls_name][0]
            lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords))
            cls_count[cls_id] += 1

        # Ten file label = ten anh, doi duoi thanh .txt
        # Anh khong con polygon hop le van giu, file label rong = anh negative
        (lbl_out / (Path(name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        if not lines:
            log(f"[{split}] {name}: khong con polygon hop le -> label rong")
            stats["empty_label"] += 1

        mode = prepare_image(src, img_out / name)
        stats[f"image_{mode}"] += 1
        if mode == "reencode":
            log(f"[{split}] {name}: JPEG bi cat cut -> ghi ban sao ma hoa lai")
        stats["images"] += 1
        stats["polygons"] += len(lines)

    return stats, cls_count, len(data)


def write_data_yaml():
    names = {cid: en for _, (cid, en) in sorted(CLASS_MAP.items(), key=lambda kv: kv[1][0])}
    lines = [
        "# YOLOv8-Seg dataset - VehiDE (tao boi convert_via_to_yolo.py)",
        f"path: {OUT_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(names)}",
        "names:",
    ]
    vi = {cid: vn for vn, (cid, _) in CLASS_MAP.items()}
    for cid, en in names.items():
        lines.append(f"  {cid}: {en}  # {vi[cid]}")
    (OUT_DIR / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        log_lines.append(msg)

    summary = ["TOM TAT CHUYEN DOI VIA -> YOLOv8-Seg", ""]
    for split, (json_path, img_dir) in SPLITS.items():
        print(f"Dang chuyen doi tap {split} ...")
        stats, cls_count, n_json = convert_split(split, json_path, img_dir, log)

        n_img = len(list((OUT_DIR / "images" / split).glob("*")))
        n_lbl = len(list((OUT_DIR / "labels" / split).glob("*.txt")))
        summary.append(f"[{split}] ban ghi JSON: {n_json} | da xu ly: {stats['images']}")
        summary.append(f"[{split}] so anh trong dataset: {n_img} | so file label: {n_lbl} | "
                       f"khop: {'OK' if n_img == n_lbl == n_json else 'KHONG KHOP'}")
        summary.append(f"[{split}] {dict(stats)}")
        id2name = {cid: en for _, (cid, en) in CLASS_MAP.items()}
        summary.append(f"[{split}] polygon theo class: " +
                       ", ".join(f"{cid}:{id2name[cid]}={cls_count[cid]}" for cid in sorted(cls_count)))
        summary.append("")

    write_data_yaml()
    summary.append(f"Da tao: {OUT_DIR / 'data.yaml'}")
    LOG_FILE.write_text("\n".join(summary + ["", "CHI TIET:"] + log_lines), encoding="utf-8")
    print("\n".join(summary))
    print(f"Log chi tiet ({len(log_lines)} dong): {LOG_FILE}")


if __name__ == "__main__":
    main()
