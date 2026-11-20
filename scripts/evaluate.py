"""
Giai doan 5 - Danh gia model YOLOv8-Seg tren tap validation.

Vi du:
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/trained/yolov8s_seg/weights/best.pt

Ket qua (results/evaluation/<ten_model>/):
    metrics.json                 so lieu tong + tung class (Box va Mask)
    per_class.csv                bang chi so tung class
    training_curves.png          loss va mAP theo epoch (doc tu results.csv)
    confusion_matrix*.png, *_curve.png    do Ultralytics ve
    predictions/                 anh so sanh Nhan that | Du doan
reports/model_evaluation.md      bao cao tong hop, so sanh moi model da danh gia
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
VAL_IMG_DIR = PROJECT_ROOT / "dataset" / "images" / "val"
VAL_LBL_DIR = PROJECT_ROOT / "dataset" / "labels" / "val"
EVAL_DIR = PROJECT_ROOT / "results" / "evaluation"
REPORT = PROJECT_ROOT / "reports" / "model_evaluation.md"

COLORS = [(0, 0, 255), (0, 200, 255), (255, 128, 0), (200, 0, 200),
          (0, 200, 0), (255, 0, 128), (0, 128, 255)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(PROJECT_ROOT / "models/trained/yolov8n_seg/weights/best.pt"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.25, help="nguong confidence cho anh du doan")
    p.add_argument("--n-images", type=int, default=20, help="so anh du doan minh hoa")
    return p.parse_args()


def run_validation(model, args, out_dir):
    """Chay model.val() -> lay chi so Box va Mask (tong + tung class)."""
    device = 0 if torch.cuda.is_available() else "cpu"
    m = model.val(data=str(DATA_YAML), split="val", imgsz=args.imgsz, batch=args.batch,
                  device=device, project=str(out_dir.parent), name=out_dir.name,
                  exist_ok=True, plots=True, workers=2)
    names = m.names
    result = {
        "box": {"precision": m.box.mp, "recall": m.box.mr, "mAP50": m.box.map50, "mAP50-95": m.box.map},
        "mask": {"precision": m.seg.mp, "recall": m.seg.mr, "mAP50": m.seg.map50, "mAP50-95": m.seg.map},
        "speed_ms_per_image": m.speed,
        "per_class": [],
    }
    # Chi so tung class: class_result(i) tra ve (P, R, mAP50, mAP50-95)
    for i, cid in enumerate(m.ap_class_index):
        bp, br, b50, b5095 = m.box.class_result(i)
        sp, sr, s50, s5095 = m.seg.class_result(i)
        result["per_class"].append({
            "class_id": int(cid), "class": names[int(cid)],
            "box_P": bp, "box_R": br, "box_mAP50": b50, "box_mAP50-95": b5095,
            "mask_P": sp, "mask_R": sr, "mask_mAP50": s50, "mask_mAP50-95": s5095,
        })
    # Doi numpy -> float de ghi JSON
    return json.loads(json.dumps(result, default=float))


def plot_training_curves(results_csv, out_path):
    """Ve loss (train/val) va mAP theo epoch tu results.csv cua qua trinh train."""
    d = pd.read_csv(results_csv)
    d.columns = [c.strip() for c in d.columns]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k in ["box_loss", "seg_loss", "cls_loss"]:
        ax[0].plot(d["epoch"], d[f"train/{k}"], label=f"train {k}")
        ax[0].plot(d["epoch"], d[f"val/{k}"], "--", label=f"val {k}")
    ax[0].set_title("Loss")
    ax[0].legend(fontsize=8)
    for k, lb in [("metrics/mAP50(B)", "Box mAP50"), ("metrics/mAP50-95(B)", "Box mAP50-95"),
                  ("metrics/mAP50(M)", "Mask mAP50"), ("metrics/mAP50-95(M)", "Mask mAP50-95")]:
        ax[1].plot(d["epoch"], d[k], label=lb)
    ax[1].set_title("mAP tren tap val")
    ax[1].legend(fontsize=8)
    ax[2].plot(d["epoch"], d["metrics/precision(M)"], label="Mask Precision")
    ax[2].plot(d["epoch"], d["metrics/recall(M)"], label="Mask Recall")
    ax[2].set_title("Precision / Recall (Mask)")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    best = d["metrics/mAP50-95(M)"].idxmax()
    return {"epochs": int(d["epoch"].iloc[-1]), "best_epoch": int(d.loc[best, "epoch"]),
            "train_hours": round(float(d["time"].iloc[-1]) / 3600, 2)}


def draw_gt(img, label_path, names):
    """Ve nhan that (polygon tu file YOLO) len anh."""
    h, w = img.shape[:2]
    overlay = img.copy()
    polys = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            v = line.split()
            pts = (np.array(v[1:], dtype=float).reshape(-1, 2) * [w, h]).astype(np.int32)
            polys.append((int(v[0]), pts))
            cv2.fillPoly(overlay, [pts], COLORS[int(v[0]) % 7])
    out = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    s = max(h, w) / 1000
    for cid, pts in polys:
        cv2.polylines(out, [pts], True, COLORS[cid % 7], max(1, int(2 * s)))
        x, y = pts[np.argmin(pts[:, 1])]
        cv2.putText(out, names[cid], (int(x), max(int(y), 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    return out


def save_predictions(model, names, args, out_dir):
    """Luu anh: trai = nhan that, phai = du doan cua model."""
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(VAL_IMG_DIR.glob("*.jpg"))
    random.Random(0).shuffle(imgs)
    for p in imgs[:args.n_images]:
        img = cv2.imread(str(p))
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        pred = r.plot(boxes=True, labels=True, conf=True)
        gt = draw_gt(img, VAL_LBL_DIR / (p.stem + ".txt"), names)
        panels = []
        s = max(img.shape[:2]) / 1000  # co chu theo kich thuoc anh
        for im, t in [(gt, "Nhan that"), (pred, f"Du doan (conf>={args.conf})")]:
            # Them thanh tieu de PHIA TREN anh (khong ve de len, tranh che nhan)
            im = cv2.copyMakeBorder(im, int(45 * s), 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.putText(im, t, (10, int(32 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255),
                        max(1, int(2 * s)), cv2.LINE_AA)
            panels.append(im)
        combo = np.hstack(panels)
        if combo.shape[1] > 1600:
            combo = cv2.resize(combo, None, fx=1600 / combo.shape[1], fy=1600 / combo.shape[1])
        cv2.imwrite(str(pred_dir / p.name), combo, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return pred_dir


def fmt(x):
    return f"{x:.3f}"


def write_report():
    """Tao reports/model_evaluation.md tu tat ca metrics.json da co (tu dong so sanh model)."""
    runs = sorted(EVAL_DIR.glob("*/metrics.json"))
    all_m = [json.loads(p.read_text(encoding="utf-8")) for p in runs]
    L = ["# Danh gia model YOLOv8-Seg - VehiDE", "",
         "> Bao cao sinh tu dong boi `scripts/evaluate.py`. Moi so lieu lay tu `model.val()` "
         "cua Ultralytics tren tap validation goc (2.324 anh, 6.035 vung hu hong).", ""]

    L += ["## 1. Bang tong hop", "",
          "| Model | Epoch (best) | Thoi gian train | Box P | Box R | Box mAP50 | Box mAP50-95 "
          "| Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | Inference (ms/anh) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in all_m:
        b, s, t = m["box"], m["mask"], m.get("training", {})
        L.append(f"| {m['name']} | {t.get('epochs', '?')} ({t.get('best_epoch', '?')}) | "
                 f"{t.get('train_hours', '?')} h | {fmt(b['precision'])} | {fmt(b['recall'])} | "
                 f"{fmt(b['mAP50'])} | {fmt(b['mAP50-95'])} | {fmt(s['precision'])} | "
                 f"{fmt(s['recall'])} | **{fmt(s['mAP50'])}** | **{fmt(s['mAP50-95'])}** | "
                 f"{m['speed_ms_per_image']['inference']:.1f} |")

    for m in all_m:
        L += ["", f"## 2. Chi so tung class - {m['name']}", "",
              "| Class | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |",
              "|---|---|---|---|---|---|---|---|---|"]
        pc = sorted(m["per_class"], key=lambda c: -c["mask_mAP50"])
        for c in pc:
            L.append(f"| {c['class']} | {fmt(c['box_P'])} | {fmt(c['box_R'])} | {fmt(c['box_mAP50'])} | "
                     f"{fmt(c['box_mAP50-95'])} | {fmt(c['mask_P'])} | {fmt(c['mask_R'])} | "
                     f"{fmt(c['mask_mAP50'])} | {fmt(c['mask_mAP50-95'])} |")
        best, worst = pc[0], pc[-1]
        L += ["", "Nhan xet tu dong:",
              f"- Class tot nhat (Mask mAP50): **{best['class']}** ({fmt(best['mask_mAP50'])}).",
              f"- Class kem nhat (Mask mAP50): **{worst['class']}** ({fmt(worst['mask_mAP50'])}).",
              f"- Mask Precision {fmt(m['mask']['precision'])} vs Recall {fmt(m['mask']['recall'])}: "
              + ("model bo sot nhieu vung hu hong (recall thap hon precision)."
                 if m["mask"]["recall"] < m["mask"]["precision"] else "model co xu huong bao nham nhieu hon bo sot."),
              "", "Hinh anh:",
              f"- Bieu do training: `results/evaluation/{m['name']}/training_curves.png`",
              f"- Confusion matrix: `results/evaluation/{m['name']}/confusion_matrix_normalized.png`",
              f"- Duong cong PR (mask): `results/evaluation/{m['name']}/MaskPR_curve.png`",
              f"- Anh du doan: `results/evaluation/{m['name']}/predictions/`"]
    # Giu lai phan phan tich viet tay (sau dong MARKER) neu bao cao da ton tai
    marker = "<!-- PHAN TICH VIET TAY: noi dung ben duoi duoc giu nguyen khi chay lai -->"
    manual = ""
    if REPORT.exists() and marker in REPORT.read_text(encoding="utf-8"):
        manual = REPORT.read_text(encoding="utf-8").split(marker, 1)[1]
    REPORT.write_text("\n".join(L) + "\n\n" + marker + manual, encoding="utf-8")


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Khong tim thay model: {model_path}")
    # Ten run = ten thu muc train (vd: yolov8n_seg)
    name = model_path.parent.parent.name
    out_dir = EVAL_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    print(f"Danh gia {model_path} tren tap val ...")
    metrics = run_validation(model, args, out_dir)
    metrics["name"] = name
    metrics["model_path"] = str(model_path)

    results_csv = model_path.parent.parent / "results.csv"
    if results_csv.exists():
        metrics["training"] = plot_training_curves(results_csv, out_dir / "training_curves.png")

    print("Ve anh du doan ...")
    save_predictions(model, model.names, args, out_dir)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(metrics["per_class"]).round(4).to_csv(out_dir / "per_class.csv", index=False)
    write_report()

    s, b = metrics["mask"], metrics["box"]
    print(f"\nBox : P={b['precision']:.3f} R={b['recall']:.3f} mAP50={b['mAP50']:.3f} mAP50-95={b['mAP50-95']:.3f}")
    print(f"Mask: P={s['precision']:.3f} R={s['recall']:.3f} mAP50={s['mAP50']:.3f} mAP50-95={s['mAP50-95']:.3f}")
    print(f"Ket qua: {out_dir}\nBao cao: {REPORT}")


if __name__ == "__main__":
    main()
