"""
Phan tich phuong phap uoc luong muc do (Giai doan 6) tren tap validation.

Vi khong co nhan severity that, script KHONG do "do chinh xac" cua muc do. Thay vao do:
1. Phan bo muc do khi ap dung luat len NHAN THAT (polygon goc).
2. Phan bo muc do khi ap dung luat len DU DOAN cua model.
3. Do nhat quan giua 2 cach tren cho tung anh: cho biet sai so cua model
   (bo sot / mask sai dien tich) anh huong the nao den muc do uoc luong.

Chay:
    python scripts/analyze_severity.py
Ket qua: reports/severity_analysis.md, results/severity/*.png
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import cv2  # noqa: E402
from app.services import detection as det  # noqa: E402  (dung CHUNG ham voi web demo)
from app.services import severity as sv  # noqa: E402

MODEL = PROJECT_ROOT / "models" / "trained" / "yolov8n_seg" / "weights" / "best.pt"
VAL_IMG = PROJECT_ROOT / "dataset" / "images" / "val"
VAL_LBL = PROJECT_ROOT / "dataset" / "labels" / "val"
OUT_DIR = PROJECT_ROOT / "results" / "severity"
REPORT = PROJECT_ROOT / "reports" / "severity_analysis.md"
CONF = 0.25
NONE = "Không phát hiện"


def assess(polys_by_class, cfg):
    """polys_by_class: list (class_name, mang diem chuan hoa Nx2) -> (muc anh, list muc vung)."""
    regions = []
    for cls, pts in polys_by_class:
        r = sv.region_severity(cls, sv.polygon_area_norm(pts), cfg)
        r["class"] = cls
        regions.append(r)
    total = sv.union_area_ratio([p for _, p in polys_by_class]) if polys_by_class else 0.0
    return sv.image_severity(regions, total, cfg), regions


def main():
    cfg = sv.load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    names = model.names
    device = 0 if torch.cuda.is_available() else "cpu"

    rows, region_rows = [], []
    imgs = sorted(VAL_IMG.glob("*.jpg"))
    for i, p in enumerate(imgs):
        # --- Nhan that ---
        gt = []
        for line in (VAL_LBL / (p.stem + ".txt")).read_text().splitlines():
            if line.strip():
                v = line.split()
                gt.append((names[int(v[0])], np.array(v[1:], dtype=float).reshape(-1, 2)))
        gt_img, gt_regs = assess(gt, cfg)

        # --- Du doan cua model: dung dung ham analyze_image cua web demo ---
        res = det.analyze_image(model, cv2.imread(str(p)), conf=CONF)
        pr_img = res["severity"]

        rows.append({"image": p.name, "gt_level": gt_img["level_name"],
                     "pred_level": pr_img["level_name"] if pr_img["level"] is not None else NONE,
                     "gt_regions": len(gt), "pred_regions": len(res["regions"])})
        region_rows += [{"source": "Nhãn thật", "class": g["class"], "level": g["level_name"]} for g in gt_regs]
        region_rows += [{"source": "Dự đoán", "class": g["class"], "level": g["severity"]["level_name"]}
                        for g in res["regions"]]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(imgs)}")

    df = pd.DataFrame(rows)
    rdf = pd.DataFrame(region_rows)
    df.to_csv(OUT_DIR / "severity_per_image.csv", index=False, encoding="utf-8")

    order = sv.LEVELS + [NONE]
    gt_dist = df["gt_level"].value_counts().reindex(order, fill_value=0)
    pr_dist = df["pred_level"].value_counts().reindex(order, fill_value=0)

    # Bieu do phan bo muc do anh
    x = np.arange(len(order))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.2, gt_dist.values, 0.4, label="Từ nhãn thật")
    plt.bar(x + 0.2, pr_dist.values, 0.4, label=f"Từ dự đoán (conf≥{CONF})")
    plt.xticks(x, order)
    plt.ylabel("Số ảnh")
    plt.title("Phân bố mức độ ước lượng - tập val (2.324 ảnh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "severity_distribution.png", dpi=120)
    plt.close()

    # Bang cheo nhan that vs du doan
    ct = pd.crosstab(df["gt_level"], df["pred_level"]).reindex(index=sv.LEVELS, columns=order, fill_value=0)
    lvl = {n: i for i, n in enumerate(sv.LEVELS)}
    both = df[df["pred_level"] != NONE]
    exact = (df["gt_level"] == df["pred_level"]).mean()
    within1 = (both["gt_level"].map(lvl) - both["pred_level"].map(lvl)).abs().le(1).sum() / len(df)
    under = (both["pred_level"].map(lvl) < both["gt_level"].map(lvl)).sum()
    over = (both["pred_level"].map(lvl) > both["gt_level"].map(lvl)).sum()
    missed = (df["pred_level"] == NONE).sum()

    # Phan bo muc do vung theo loai (nhan that)
    reg_ct = pd.crosstab(rdf[rdf.source == "Nhãn thật"]["class"],
                         rdf[rdf.source == "Nhãn thật"]["level"]).reindex(columns=sv.LEVELS, fill_value=0)

    def md_table(t):
        cols = list(t.columns)
        out = ["| | " + " | ".join(map(str, cols)) + " |", "|---" * (len(cols) + 1) + "|"]
        for idx, row in t.iterrows():
            out.append(f"| {idx} | " + " | ".join(str(v) for v in row.values) + " |")
        return out

    L = ["# Phân tích ước lượng mức độ hư hỏng (rule-based)", "",
         f"> {sv.DISCLAIMER}", "",
         "Sinh tự động bởi `scripts/analyze_severity.py`, model `yolov8n_seg/best.pt`, tập val 2.324 ảnh.", "",
         "## 0. Phương pháp", "",
         "VehiDE **không có nhãn mức độ**, nên mức độ được ước lượng bằng luật (rule-based), "
         "cài đặt trong `app/services/severity.py`:", "",
         "1. **Từng vùng:** `điểm = trọng số loại + bậc diện tích`.",
         "   - Trọng số loại là **giả định thử nghiệm**: Scratch = 1; Dent, Torn, Punctured = 2; "
         "Broken light, Broken glass, Lost part = 3 (bộ phận phải thay thế hoặc ảnh hưởng an toàn).",
         "   - Bậc diện tích so tỉ lệ diện tích mask/ảnh với phân vị **của chính class đó** trong tập train: "
         "≤ p50 là nhỏ (0), p50–p90 là vừa (1), > p90 là lớn (2).",
         "   - Điểm ≤ 2 là **Nhẹ**, bằng 3 là **Trung bình**, ≥ 4 là **Nặng**.",
         "2. **Từng ảnh:** lấy mức cao nhất trong các vùng; tăng 1 mức nếu tổng diện tích hư hỏng "
         "(hợp các mask) hoặc số vùng vượt p90 của dataset.",
         "3. **Nhiều ảnh cùng trường hợp** (do người dùng nhóm): lấy mức cao nhất trong các ảnh; "
         "số vùng được cộng dồn, **không** loại bỏ trùng lặp giữa các góc chụp.", "",
         "## 1. Ngưỡng diện tích (tính từ nhãn tập train)", "",
         "| Class | Trọng số loại | Số vùng | p50 (% ảnh) | p90 (% ảnh) |", "|---|---|---|---|---|"]
    for c, t in cfg["classes"].items():
        L.append(f"| {c} | {sv.TYPE_WEIGHT[c]} | {t['n']} | {t['p50'] * 100:.2f} | {t['p90'] * 100:.2f} |")
    L += ["", f"Ngưỡng lan rộng cấp ảnh: tổng diện tích > {cfg['image']['total_area_p90'] * 100:.1f}% ảnh "
              f"hoặc số vùng > {cfg['image']['region_count_p90']}.", "",
          "## 2. Phân bố mức độ cấp ảnh", "",
          "| Mức | Từ nhãn thật | Từ dự đoán |", "|---|---|---|"]
    for o in order:
        L.append(f"| {o} | {gt_dist[o]} ({gt_dist[o] / len(df) * 100:.1f}%) | "
                 f"{pr_dist[o]} ({pr_dist[o] / len(df) * 100:.1f}%) |")
    L += ["", "## 3. Mức độ từng vùng theo loại hư hỏng (nhãn thật)", ""] + md_table(reg_ct)
    L += ["", "## 4. Nhất quán giữa mức từ nhãn thật và từ dự đoán", "",
          "Hàng: mức tính từ nhãn thật. Cột: mức tính từ dự đoán của model.", ""] + md_table(ct)
    L += ["",
          f"- Trùng khớp chính xác: **{exact * 100:.1f}%** số ảnh.",
          f"- Lệch không quá 1 mức (không tính ảnh bị bỏ sót): {within1 * 100:.1f}%.",
          f"- Dự đoán thấp hơn: {under} ảnh; cao hơn: {over} ảnh; model không phát hiện gì: {missed} ảnh.",
          "", "Lưu ý: bảng này **không** đo độ chính xác của mức độ (không có nhãn severity thật), "
              "chỉ đo mức ảnh hưởng của sai số model lên kết quả ước lượng.", "",
          "Biểu đồ: `results/severity/severity_distribution.png`. Chi tiết từng ảnh: "
          "`results/severity/severity_per_image.csv`."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-40:]))


if __name__ == "__main__":
    main()
