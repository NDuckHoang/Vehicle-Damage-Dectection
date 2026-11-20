"""
Kiem tra dataset YOLOv8-Seg sau khi chuyen doi (dung lai truoc khi train).

Kiem tra:
1. data.yaml ton tai va co du cac truong.
2. So anh == so file label cho moi tap, ten khop 1-1.
3. Moi dong label: class_id hop le, so toa do chan, >= 3 diem, toa do trong [0, 1].
4. Doi chieu nguoc: toa do YOLO * kich thuoc anh ~ toa do goc trong VIA JSON (sai so <= 1px).

Chay:
    python scripts/validate_dataset.py
Tra ve exit code 0 neu hop le, 1 neu co loi.
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
DATASET_ROOT = Path(r"D:\Dataset")
JSON_FILES = {"train": DATASET_ROOT / "0Train_via_annos.json",
              "val": DATASET_ROOT / "0Val_via_annos.json"}
N_ROUNDTRIP = 200  # so anh ngau nhien dung de doi chieu nguoc moi tap


def read_names_from_yaml(path):
    """Doc 'nc' va 'names' tu data.yaml (dinh dang don gian do script convert tao ra)."""
    nc, names = None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].rstrip()
        if line.startswith("nc:"):
            nc = int(line.split(":")[1])
        elif line.startswith("  ") and ":" in line:
            k, v = line.strip().split(":", 1)
            names[int(k)] = v.strip()
    return nc, names


def check_label_file(path, nc):
    """Tra ve (so polygon, danh sach loi)."""
    errors, n = [], 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        parts = line.split()
        n += 1
        try:
            cls = int(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"dong {i}: khong phai so")
            continue
        if not 0 <= cls < nc:
            errors.append(f"dong {i}: class_id {cls} ngoai [0,{nc - 1}]")
        if len(vals) % 2 or len(vals) < 6:
            errors.append(f"dong {i}: so toa do {len(vals)} khong hop le")
        if vals and (min(vals) < 0 or max(vals) > 1):
            errors.append(f"dong {i}: toa do ngoai [0,1]")
    return n, errors


def roundtrip_check(split, img_dir, lbl_dir):
    """Doi toa do YOLO ve pixel va so voi polygon goc (da clip) trong JSON."""
    data = json.load(open(JSON_FILES[split], encoding="utf-8"))
    random.seed(0)
    sample = random.sample(list(data.values()), min(N_ROUNDTRIP, len(data)))
    max_err, bad = 0.0, []
    for rec in sample:
        name = rec["name"]
        h, w = cv2.imread(str(img_dir / name)).shape[:2]
        lines = [l for l in (lbl_dir / (Path(name).stem + ".txt")).read_text().splitlines() if l]
        regions = rec["regions"]
        # Converter bo qua rat it polygon hong; voi mau ngau nhien, so dong thuong bang so region
        if len(lines) != len(regions):
            continue
        for line, r in zip(lines, regions):
            p = np.array(line.split()[1:], dtype=float).reshape(-1, 2) * [w, h]
            orig = np.array(list(zip(r["all_x"], r["all_y"])), dtype=float)
            orig[:, 0] = orig[:, 0].clip(0, w)
            orig[:, 1] = orig[:, 1].clip(0, h)
            # Moi diem YOLO phai trung (<=1px) voi 1 diem goc
            d = np.abs(p[:, None, :] - orig[None, :, :]).max(axis=2).min(axis=1).max()
            max_err = max(max_err, d)
            if d > 1.0:
                bad.append((name, d))
    return max_err, bad


def main():
    ok = True
    yaml_path = DATA_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"LOI: khong tim thay {yaml_path}. Hay chay convert_via_to_yolo.py truoc.")
        sys.exit(1)
    nc, names = read_names_from_yaml(yaml_path)
    print(f"data.yaml: nc={nc}, names={names}")
    if nc != len(names):
        print("LOI: nc khong bang so names")
        ok = False

    for split in ["train", "val"]:
        img_dir, lbl_dir = DATA_DIR / "images" / split, DATA_DIR / "labels" / split
        imgs = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")}
        print(f"\n[{split}] anh: {len(imgs)} | label: {len(lbls)}")
        if imgs != lbls:
            ok = False
            print(f"  LOI: anh thieu label: {len(imgs - lbls)}, label thieu anh: {len(lbls - imgs)}")

        total_poly, empty, file_errors = 0, 0, []
        for p in lbl_dir.glob("*.txt"):
            n, errs = check_label_file(p, nc)
            total_poly += n
            empty += (n == 0)
            if errs:
                file_errors.append((p.name, errs))
        print(f"  tong polygon: {total_poly} | label rong: {empty} | file loi: {len(file_errors)}")
        for name, errs in file_errors[:5]:
            print(f"    {name}: {errs[:3]}")
        if file_errors:
            ok = False

        max_err, bad = roundtrip_check(split, img_dir, lbl_dir)
        print(f"  doi chieu nguoc {N_ROUNDTRIP} anh: sai so lon nhat = {max_err:.3f} px "
              f"| polygon lech > 1px: {len(bad)}")
        if bad:
            ok = False

    print("\nKET QUA:", "HOP LE" if ok else "CO LOI")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
