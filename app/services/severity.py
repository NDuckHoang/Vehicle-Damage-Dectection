"""
Giai doan 6 - Uoc luong muc do hu hong (Nhe / Trung binh / Nang) theo phuong phap RULE-BASED.

LUU Y QUAN TRONG:
    Dataset VehiDE KHONG co nhan muc do (severity). Ket qua o day la UOC LUONG theo tieu chi
    thu nghiem, KHONG phai ket luan giam dinh chuyen nghiep.

Phuong phap (3 tang):
1) Tung vung hu hong:
       diem = trong_so_loai + bac_dien_tich
   - trong_so_loai (gia dinh, co the chinh): Scratch=1; Dent/Torn/Punctured=2;
     Broken light/Broken glass/Lost part=3.
   - bac_dien_tich: so ti le dien tich vung (so voi anh) voi phan vi p50, p90 cua CHINH class do
     trong tap train: <= p50 -> 0 (nho), p50..p90 -> 1 (vua), > p90 -> 2 (lon).
   - diem <= 2 -> Nhe, = 3 -> Trung binh, >= 4 -> Nang.
2) Tung anh: muc = muc nang nhat trong cac vung; tang 1 muc neu tong dien tich hu hong
   hoac so vung vuot phan vi p90 cua dataset (hu hong lan rong).
3) Ca truong hop nhieu anh (nguoi dung tu nhom): muc = muc nang nhat trong cac anh.
   KHONG loai bo vung trung lap giua cac anh (khong co phuong phap doi chieu tin cay).

Cac nguong p50/p90 duoc tinh tu nhan that va luu vao severity_config.json:
    python app/services/severity.py --calibrate
"""
import json
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).with_name("severity_config.json")
LEVELS = ["Nhẹ", "Trung bình", "Nặng"]
DISCLAIMER = ("Mức độ chỉ là ước lượng theo tiêu chí thử nghiệm (diện tích, số lượng và loại hư hỏng), "
              "không phải kết luận giám định chuyên nghiệp.")

# Trong so theo loai hu hong (gia dinh chuyen mon, ghi ro trong bao cao)
TYPE_WEIGHT = {
    "Scratch": 1,       # tray son: thuong chi anh huong be mat
    "Dent": 2,          # mop lom: bien dang than vo
    "Torn": 2,          # rach
    "Punctured": 2,     # thung
    "Broken light": 3,  # be den: anh huong an toan, phai thay the
    "Broken glass": 3,  # vo kinh: anh huong an toan, phai thay the
    "Lost part": 3,     # mat bo phan
}
AREA_NAMES = ["nhỏ", "vừa", "lớn"]


# ----------------------------------------------------------------------------
# Hieu chinh nguong tu du lieu
# ----------------------------------------------------------------------------
def polygon_area_norm(pts):
    """Dien tich polygon voi toa do chuan hoa [0,1] = ti le dien tich so voi anh (Shoelace)."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def union_area_ratio(polys, grid=512):
    """Ti le dien tich HOP cac polygon (ve len luoi grid x grid, tranh dem trung vung chong lan)."""
    import cv2
    canvas = np.zeros((grid, grid), np.uint8)
    for pts in polys:
        cv2.fillPoly(canvas, [(pts * grid).astype(np.int32)], 1)
    return float(canvas.mean())


def calibrate(labels_dir, names):
    """Tinh phan vi dien tich tu cac file label YOLO cua tap train."""
    region_areas = {n: [] for n in names.values()}
    image_union, image_count = [], []
    for f in Path(labels_dir).glob("*.txt"):
        polys = []
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            v = line.split()
            pts = np.array(v[1:], dtype=float).reshape(-1, 2)
            region_areas[names[int(v[0])]].append(polygon_area_norm(pts))
            polys.append(pts)
        if polys:
            image_union.append(union_area_ratio(polys))
            image_count.append(len(polys))

    cfg = {"source": str(labels_dir), "classes": {}, "image": {}}
    for cls, a in region_areas.items():
        a = np.array(a)
        cfg["classes"][cls] = {"n": int(len(a)),
                               "p50": round(float(np.percentile(a, 50)), 6),
                               "p90": round(float(np.percentile(a, 90)), 6)}
    cfg["image"] = {"n": len(image_union),
                    "total_area_p90": round(float(np.percentile(image_union, 90)), 6),
                    "region_count_p90": int(np.percentile(image_count, 90))}
    return cfg


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Chua co {CONFIG_PATH}. Chay: python app/services/severity.py --calibrate")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Luat uoc luong muc do
# ----------------------------------------------------------------------------
def score_to_level(score):
    """Doi diem (1..5) ra chi so muc: 0=Nhe, 1=Trung binh, 2=Nang."""
    if score <= 2:
        return 0
    if score == 3:
        return 1
    return 2


def region_severity(class_name, area_ratio, cfg):
    """Muc do cua 1 vung hu hong. area_ratio = dien tich mask / dien tich anh."""
    th = cfg["classes"][class_name]
    if area_ratio <= th["p50"]:
        area_level = 0
    elif area_ratio <= th["p90"]:
        area_level = 1
    else:
        area_level = 2
    weight = TYPE_WEIGHT[class_name]
    score = weight + area_level
    level = score_to_level(score)
    return {
        "level": level, "level_name": LEVELS[level], "score": score,
        "type_weight": weight, "area_level": AREA_NAMES[area_level],
        "reason": f"{class_name} (trọng số {weight}) + diện tích {AREA_NAMES[area_level]} "
                  f"({area_ratio * 100:.2f}% ảnh) = {score} điểm",
    }


def image_severity(regions, total_area_ratio, cfg):
    """
    Muc do cua 1 anh.
    regions: list dict, moi dict co "level" (ket qua region_severity).
    total_area_ratio: ti le dien tich HOP cac mask so voi anh.
    """
    if not regions:
        return {"level": None, "level_name": "Không phát hiện hư hỏng",
                "reasons": ["Model không phát hiện vùng hư hỏng nào (với ngưỡng confidence hiện tại)."]}
    level = max(r["level"] for r in regions)
    reasons = [f"Mức cao nhất trong {len(regions)} vùng: {LEVELS[level]}"]
    im = cfg["image"]
    why = []
    if total_area_ratio > im["total_area_p90"]:
        why.append(f"tổng diện tích {total_area_ratio * 100:.1f}% ảnh > {im['total_area_p90'] * 100:.1f}%")
    if len(regions) > im["region_count_p90"]:
        why.append(f"{len(regions)} vùng > {im['region_count_p90']} vùng")
    if why and level < 2:
        level += 1
        reasons.append("Hư hỏng lan rộng (" + "; ".join(why) + ", vượt p90 của dataset) → tăng 1 mức")
    return {"level": level, "level_name": LEVELS[level], "reasons": reasons}


def case_severity(image_results):
    """
    Tong hop nhieu anh do NGUOI DUNG nhom vao cung 1 truong hop.
    image_results: list dict co "severity" (ket qua image_severity) va "regions" (list co "class").
    """
    levels = [r["severity"]["level"] for r in image_results if r["severity"]["level"] is not None]
    damage_types = {}
    for r in image_results:
        for reg in r["regions"]:
            damage_types[reg["class"]] = damage_types.get(reg["class"], 0) + 1
    if not levels:
        return {"level": None, "level_name": "Không phát hiện hư hỏng", "damage_types": {},
                "note": DISCLAIMER}
    level = max(levels)
    return {
        "level": level, "level_name": LEVELS[level],
        "damage_types": damage_types,  # so vung theo loai, cong don qua cac anh
        "note": ("Số vùng được cộng dồn từ các ảnh; chưa loại bỏ vùng trùng lặp giữa các góc chụp. "
                 + DISCLAIMER),
    }


if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="tinh nguong tu nhan tap train")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.calibrate:
        names = {}
        for line in (root / "dataset" / "data.yaml").read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s+(\d+):\s*([^#]+)", line)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
        cfg = calibrate(root / "dataset" / "labels" / "train", names)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(f"Da luu: {CONFIG_PATH}")
