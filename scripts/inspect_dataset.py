"""
Giai doan 1 - Kiem tra dataset VehiDE (VIA JSON).

Chuong trinh chi DOC du lieu goc, khong sua/ghi de bat ky file nao trong D:\\Dataset.
Ket qua:
    reports/dataset_analysis.txt
    reports/dataset_stats.csv          (thong ke tung polygon)
    reports/chart_class_distribution.png
    reports/chart_regions_per_image.png
    reports/chart_area_ratio.png

Chay:
    python scripts/inspect_dataset.py
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # ve bieu do khong can cua so
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Cau hinh duong dan (dataset goc nam ngoai project)
# ----------------------------------------------------------------------------
DATASET_ROOT = Path(r"D:\Dataset")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

SPLITS = {
    "train": {
        "json": DATASET_ROOT / "0Train_via_annos.json",
        "image_dirs": [DATASET_ROOT / "image" / "image", DATASET_ROOT / "image"],
    },
    "val": {
        "json": DATASET_ROOT / "0Val_via_annos.json",
        "image_dirs": [DATASET_ROOT / "validation" / "validation", DATASET_ROOT / "validation"],
    },
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_image_dir(candidates):
    """Chon thu muc dau tien thuc su chua anh (dataset co thu muc long nhau image/image)."""
    for d in candidates:
        if d.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir()):
            return d
    return None


def read_image_size(path):
    """
    Tra ve (w_raw, h_raw, rotated, ok).
    - w_raw, h_raw: kich thuoc pixel goc, BO QUA EXIF orientation.
    - rotated: True neu EXIF orientation lam anh xoay 90/270 do (doi chieu w/h).
    OpenCV mac dinh tu xoay anh theo EXIF, nen ta doc 2 lan de so sanh.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if raw is None:
        return None, None, False, False
    h, w = raw.shape[:2]
    # Doc lan 2 o do phan giai 1/8 (nhanh) co ap dung EXIF de xem co bi xoay khong
    small = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    rotated = False
    if small is not None and w != h:
        sh, sw = small.shape[:2]
        rotated = (sw > sh) != (w > h)
    return w, h, rotated, True


def polygon_area(xs, ys):
    """Dien tich polygon theo cong thuc Shoelace."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def check_polygon(region, w, h):
    """Kiem tra mot region. Tra ve (danh sach loi, dien tich)."""
    errors = []
    xs, ys = region.get("all_x"), region.get("all_y")
    if xs is None or ys is None:
        return ["thieu all_x/all_y"], 0.0
    if not isinstance(xs, list) or not isinstance(ys, list):
        return ["all_x/all_y khong phai list"], 0.0
    if len(xs) != len(ys):
        return [f"so diem x({len(xs)}) != y({len(ys)})"], 0.0
    if any(not isinstance(v, (int, float)) for v in xs + ys):
        return ["toa do khong phai so"], 0.0
    if len(xs) < 3:
        errors.append(f"it hon 3 diem ({len(xs)})")
    area = polygon_area(xs, ys) if len(xs) >= 3 else 0.0
    if len(xs) >= 3 and area < 1.0:
        errors.append("dien tich ~ 0")
    if w is not None:
        if min(xs) < 0 or min(ys) < 0:
            errors.append("toa do am")
        # Cho phep lech 1 pixel o bien
        if max(xs) > w + 1 or max(ys) > h + 1:
            errors.append("vuot kich thuoc anh")
    return errors, area


def overshoot(xs, ys, w, h):
    """So pixel lon nhat ma polygon vuot ra ngoai khung anh (0 neu nam trong anh)."""
    if w is None or not xs or len(xs) != len(ys):
        return 0
    return max(0, -min(xs), -min(ys), max(xs) - w, max(ys) - h)


def is_truncated_jpeg(path):
    """JPEG hop le phai ket thuc bang marker EOI (FF D9). Neu khong -> file bi cat cut."""
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return False
    with open(path, "rb") as f:
        f.seek(-2, os.SEEK_END)
        return f.read(2) != b"\xff\xd9"


def inspect_split(split, cfg, lines, rows):
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log(f"TAP {split.upper()}")
    log("=" * 70)

    # 1. Doc JSON
    with open(cfg["json"], encoding="utf-8") as f:
        data = json.load(f)
    log(f"File annotation: {cfg['json']}")
    log(f"Kieu goc JSON: {type(data).__name__}, so ban ghi: {len(data)}")

    # 2. Liet ke tat ca truong co trong JSON
    record_keys, region_keys, class_types = Counter(), Counter(), Counter()
    for rec in data.values():
        record_keys.update(rec.keys())
        for r in rec.get("regions", []):
            region_keys.update(r.keys())
            class_types[type(r.get("class")).__name__] += 1
    log(f"Cac truong cua ban ghi: {dict(record_keys)}")
    log(f"Cac truong cua region : {dict(region_keys)}")
    log(f"Kieu du lieu truong 'class': {dict(class_types)}")

    # 3. Ghep ten anh
    img_dir = find_image_dir(cfg["image_dirs"])
    log(f"Thu muc anh thuc te: {img_dir}")
    disk_files = {f.name for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS}
    log(f"So anh tren o cung: {len(disk_files)}")

    key_ne_name = [k for k, v in data.items() if v.get("name") != k]
    log(f"So ban ghi co key khac truong 'name': {len(key_ne_name)}")
    names = [v.get("name") for v in data.values()]
    dup_names = [n for n, c in Counter(names).items() if c > 1]
    log(f"So ten anh bi trung trong JSON: {len(dup_names)}")

    names_set = set(names)
    matched = names_set & disk_files
    ann_no_img = sorted(names_set - disk_files)
    img_no_ann = sorted(disk_files - names_set)
    # Kiem tra khop khi bo qua hoa/thuong (chi de bao cao, KHONG dung de ghep)
    lower_disk = {n.lower() for n in disk_files}
    case_only = [n for n in ann_no_img if n.lower() in lower_disk]
    hash_like = [n for n in disk_files if "image" not in n]
    log(f"Ten anh khop CHINH XAC voi file: {len(matched)}")
    log(f"Annotation khong co anh       : {len(ann_no_img)} {ann_no_img[:5]}")
    log(f"Anh khong co annotation       : {len(img_no_ann)} {img_no_ann[:5]}")
    log(f"Chi khop khi bo qua hoa/thuong: {len(case_only)}")
    log(f"So file ten dang hash (khong chua 'image'): {len(hash_like)}, "
        f"trong do co trong JSON: {len(set(hash_like) & names_set)}")

    # 4. Duyet tung anh: kich thuoc + polygon
    n_regions = []
    bad_images, bad_polys, empty_imgs, rotated_imgs, truncated = [], [], [], [], []
    rot_fit_raw, rot_fit_swapped = 0, 0
    for i, (key, rec) in enumerate(data.items()):
        name = rec.get("name", key)
        regions = rec.get("regions", []) or []
        n_regions.append(len(regions))
        if not regions:
            empty_imgs.append(name)
        w = h = None
        rotated = False
        if name in disk_files:
            w, h, rotated, ok = read_image_size(img_dir / name)
            if not ok:
                bad_images.append(name)
            if rotated:
                rotated_imgs.append(name)
            if is_truncated_jpeg(img_dir / name):
                truncated.append(name)
        for j, r in enumerate(regions):
            errs, area = check_polygon(r, w, h)
            if errs:
                bad_polys.append((name, j, r.get("class"), "; ".join(errs)))
            xs, ys = r.get("all_x") or [0], r.get("all_y") or [0]
            # Voi anh bi xoay EXIF: polygon nam vua khung nao (goc hay da xoay)?
            if rotated and w is not None:
                if max(xs) <= w + 1 and max(ys) <= h + 1:
                    rot_fit_raw += 1
                if max(xs) <= h + 1 and max(ys) <= w + 1:
                    rot_fit_swapped += 1
            rows.append({
                "split": split, "image": name, "region_idx": j,
                "class": r.get("class"), "n_points": len(xs),
                "img_w": w, "img_h": h, "exif_rotated": rotated,
                "area_px": round(area, 1),
                "overshoot_px": overshoot(xs, ys, w, h),
                "overshoot_pct": round(100 * overshoot(xs, ys, w, h) / max(w, h), 3) if w else None,
                "area_ratio": round(area / (w * h), 6) if w else None,
                "errors": "; ".join(errs),
            })
        if (i + 1) % 2000 == 0:
            print(f"  ... da kiem tra {i + 1}/{len(data)} anh")

    n_regions = np.array(n_regions)
    log("")
    log(f"Tong so regions (polygon): {n_regions.sum()}")
    log(f"Regions/anh: min={n_regions.min()}, max={n_regions.max()}, "
        f"trung binh={n_regions.mean():.2f}, trung vi={np.median(n_regions):.0f}")
    log(f"Anh co regions rong (khong hu hong/negative): {len(empty_imgs)}")
    log(f"Anh khong doc duoc (loi file): {len(bad_images)} {bad_images[:5]}")
    log(f"Anh JPEG bi cat cut (thieu marker cuoi, van doc duoc): {len(truncated)} {truncated[:5]}")
    log(f"Anh co EXIF xoay 90/270 do: {len(rotated_imgs)} {rotated_imgs[:5]}")
    if rotated_imgs:
        log(f"  Polygon cua anh xoay nam vua khung GOC (bo qua EXIF): {rot_fit_raw}")
        log(f"  Polygon cua anh xoay nam vua khung DA XOAY        : {rot_fit_swapped}")

    # 5. Thong ke class
    cls_counter = Counter(r.get("class") for rec in data.values() for r in rec.get("regions", []))
    img_per_cls = Counter()
    for rec in data.values():
        for c in {r.get("class") for r in rec.get("regions", [])}:
            img_per_cls[c] += 1
    log("")
    log("Phan bo class (so polygon / so anh chua class):")
    for c, n in cls_counter.most_common():
        log(f"  {str(c):15s} {n:7d} polygon   {img_per_cls[c]:6d} anh")

    # 6. Polygon khong hop le
    log("")
    log(f"Polygon khong hop le: {len(bad_polys)}")
    err_types = Counter(e for _, _, _, es in bad_polys for e in es.split("; "))
    for e, n in err_types.most_common():
        log(f"  {e}: {n}")
    for b in bad_polys[:10]:
        log(f"    vi du: {b}")
    # Muc do vuot bien: neu chi vai pixel -> loi click o mep anh, co the clip an toan
    ov = np.array([r["overshoot_px"] for r in rows if r["split"] == split and r["overshoot_px"] > 0])
    if len(ov):
        pct = np.array([r["overshoot_pct"] for r in rows if r["split"] == split and r["overshoot_px"] > 0])
        log(f"  Muc vuot bien (pixel): min={ov.min()}, trung vi={np.median(ov):.0f}, "
            f"p95={np.percentile(ov, 95):.0f}, max={ov.max()}")
        log(f"  Muc vuot bien (% canh dai anh): trung vi={np.median(pct):.2f}%, "
            f"p95={np.percentile(pct, 95):.2f}%, max={pct.max():.2f}%")
        for t in [5, 20, 50]:
            log(f"  So polygon vuot > {t} px: {(ov > t).sum()}")

    return {
        "data": data, "names": names_set, "classes": cls_counter,
        "n_regions": n_regions, "bad_polys": bad_polys, "rotated": rotated_imgs,
        "ann_no_img": ann_no_img, "img_no_ann": img_no_ann, "empty": empty_imgs,
    }


def make_charts(res, df):
    # Bieu do phan bo class
    classes = sorted(set(res["train"]["classes"]) | set(res["val"]["classes"]),
                     key=lambda c: -res["train"]["classes"][c])
    x = np.arange(len(classes))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, [res["train"]["classes"][c] for c in classes], 0.4, label="train")
    plt.bar(x + 0.2, [res["val"]["classes"][c] for c in classes], 0.4, label="val")
    plt.xticks(x, classes, rotation=20)
    plt.ylabel("So polygon")
    plt.title("Phan bo class hu hong - VehiDE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "chart_class_distribution.png", dpi=120)
    plt.close()

    # Bieu do so region moi anh
    plt.figure(figsize=(8, 4))
    for s in ["train", "val"]:
        nr = res[s]["n_regions"]
        plt.hist(nr, bins=range(0, min(nr.max(), 40) + 2), alpha=0.6, label=s)
    plt.xlabel("So vung hu hong / anh")
    plt.ylabel("So anh")
    plt.title("So regions tren moi anh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "chart_regions_per_image.png", dpi=120)
    plt.close()

    # Bieu do ti le dien tich polygon / dien tich anh (theo class)
    plt.figure(figsize=(10, 5))
    data = [df[df["class"] == c]["area_ratio"].dropna().clip(upper=1) * 100 for c in classes]
    plt.boxplot(data, tick_labels=classes, showfliers=False)
    plt.ylabel("Dien tich polygon / anh (%)")
    plt.title("Ti le dien tich vung hu hong theo class")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "chart_area_ratio.png", dpi=120)
    plt.close()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines, rows, res = [], [], {}
    lines.append("BAO CAO PHAN TICH DATASET VEHIDE (Giai doan 1)")
    lines.append(f"Python: {sys.version.split()[0]} | OpenCV: {cv2.__version__}")
    lines.append(f"Dataset goc: {DATASET_ROOT}")
    lines.append("")

    for split, cfg in SPLITS.items():
        res[split] = inspect_split(split, cfg, lines, rows)

    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "dataset_stats.csv", index=False, encoding="utf-8")

    # Kiem tra trung lap giua train va val
    overlap = res["train"]["names"] & res["val"]["names"]
    all_cls = set(res["train"]["classes"]) | set(res["val"]["classes"])
    lines.append("")
    lines.append("=" * 70)
    lines.append("TONG HOP")
    lines.append("=" * 70)
    lines.append(f"Anh trung ten giua train va val: {len(overlap)}")
    lines.append(f"Class co trong train nhung khong co trong val: "
                 f"{set(res['train']['classes']) - set(res['val']['classes'])}")
    lines.append(f"Class co trong val nhung khong co trong train: "
                 f"{set(res['val']['classes']) - set(res['train']['classes'])}")
    lines.append(f"Tat ca class tim thay ({len(all_cls)}): {sorted(map(str, all_cls))}")
    lines.append("Truong severity/muc do: " +
                 ("CO" if any("sever" in k.lower() or "level" in k.lower()
                              for k in df.columns) else "KHONG CO trong annotation"))
    lines.append("")
    lines.append("Thong ke dien tich polygon (% dien tich anh) theo class:")
    summary = (df.assign(pct=df["area_ratio"] * 100)
               .groupby("class")["pct"].describe(percentiles=[0.25, 0.5, 0.75])
               .round(2))
    lines.append(summary.to_string())
    lines.append("")
    lines.append("So diem moi polygon:")
    lines.append(df["n_points"].describe().round(1).to_string())

    # Ket luan va de xuat (dua tren ket qua kiem tra o tren)
    lines += [
        "",
        "=" * 70,
        "KET LUAN VA DE XUAT CHO GIAI DOAN 2",
        "=" * 70,
        "1. Ghep anh-nhan: dung ten file CHINH XAC (truong 'name' == ten file tren dia),",
        "   ke ca anh ten dang hash. Khong can doan/ghep theo thu tu.",
        "2. Class lay tu truong 'class' cua tung region (ten tieng Viet khong dau).",
        "   Anh xa sang ten tieng Anh trong mo ta dataset la DE XUAT, can xac nhan:",
        "     tray_son->Scratch, mop_lom->Dents, rach->Torn, mat_bo_phan->Lost parts,",
        "     be_den->Broken lights, thung->Punctured, vo_kinh->Broken glass.",
        "3. Khong co class 'Non-damaged' va khong co anh regions rong -> khong co anh negative.",
        "4. Khong co nhan severity -> Giai doan 6 phai dung phuong phap rule-based.",
        "5. Polygon vuot bien chu yeu vai pixel -> clip toa do ve [0, w] x [0, h].",
        "   Polygon < 3 diem hoac dien tich ~0 sau khi clip -> bo qua va ghi log.",
        "6. Anh JPEG bi cat cut van doc duoc bang OpenCV -> giu lai, ghi log.",
        "7. Khong co anh xoay EXIF -> toa do polygon dung truc tiep voi kich thuoc anh.",
    ]

    make_charts(res, df)

    out = REPORT_DIR / "dataset_analysis.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-30:]))
    print(f"\nDa ghi bao cao: {out}")


if __name__ == "__main__":
    main()
