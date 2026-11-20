"""
Danh gia CHU QUAN - Buoc 1: chon anh va tao phieu cham (1 file HTML) gui cho nguoi danh gia.

- Chon NGAU NHIEN (seed co dinh) anh tu tap validation, chia deu theo muc do he thong du doan:
  3 Nhe, 3 Trung binh, 3 Nang, 1 Khong phat hien (mac dinh 10 anh).
- Chay he thong (dung ham cua web demo) tren tung anh, luu ket qua vao subjective/sample.json.
- Tao subjective/phieu_danh_gia.html: anh nhung san (base64), mo bang trinh duyet, khong can cai dat.

Chay:
    python scripts/subjective_prepare.py
    python scripts/subjective_prepare.py --seed 7 --quota 3 3 3 1
"""
import argparse
import base64
import json
import random
import sys
from pathlib import Path

import cv2
from jinja2 import Environment, FileSystemLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from services import detection as det  # noqa: E402

VAL_IMG = PROJECT_ROOT / "dataset" / "images" / "val"
OUT_DIR = PROJECT_ROOT / "subjective"
TEMPLATE_DIR = PROJECT_ROOT / "app" / "templates"
CONF = 0.25
BUCKETS = ["Nhẹ", "Trung bình", "Nặng", "Không phát hiện hư hỏng"]


def to_b64(img, max_w=1100):
    """Thu nho anh va ma hoa base64 de nhung vao HTML."""
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--quota", type=int, nargs=4, default=[3, 3, 3, 1],
                   help="so anh: Nhe, Trung binh, Nang, Khong phat hien")
    args = p.parse_args()

    if (OUT_DIR / "sample.json").exists():
        print(f"[!] Da ton tai {OUT_DIR / 'sample.json'}. Neu tao lai, phieu cu se KHONG khop voi "
              "ket qua da cham. Hay xoa/doi ten thu muc subjective/ truoc khi tao lai.")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "responses").mkdir(exist_ok=True)

    model = det.load_model()
    quota = dict(zip(BUCKETS, args.quota))
    images = sorted(VAL_IMG.glob("*.jpg"))
    random.Random(args.seed).shuffle(images)

    chosen = {b: [] for b in BUCKETS}
    for path in images:  # duyet anh theo thu tu ngau nhien, dua vao nhom con thieu
        if all(len(chosen[b]) >= quota[b] for b in BUCKETS):
            break
        img = cv2.imread(str(path))
        res = det.analyze_image(model, img, conf=CONF)
        level = res["severity"]["level_name"]
        if len(chosen[level]) < quota[level]:
            chosen[level].append((path, img, res))

    # Tron thu tu de nguoi cham khong doan duoc muc do theo vi tri anh
    items = [x for b in BUCKETS for x in chosen[b]]
    random.Random(args.seed + 1).shuffle(items)

    sample, form_items = [], []
    for i, (path, img, res) in enumerate(items, start=1):
        image_id = f"A{i:02d}"
        regions = [{"class": g["class"], "class_vi": g["class_vi"], "confidence": g["confidence"],
                    "area_pct": round(g["area_ratio"] * 100, 2), "level": g["severity"]["level_name"]}
                   for g in res["regions"]]
        sample.append({"image_id": image_id, "filename": path.name,
                       "system_level": res["severity"]["level_name"],
                       "system_reasons": res["severity"]["reasons"], "regions": regions})
        form_items.append({"image_id": image_id, "original": to_b64(img),
                           "result": to_b64(det.draw_result(img, res)),
                           "system_level": res["severity"]["level_name"], "regions": regions})
        print(f"  {image_id}: {path.name} -> he thong: {res['severity']['level_name']} ({len(regions)} vung)")

    (OUT_DIR / "sample.json").write_text(
        json.dumps({"seed": args.seed, "conf": CONF, "items": sample}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    # Nhung JSON vao the <script>: thay "</" de khong the dong the script som
    items_json = json.dumps(form_items, ensure_ascii=False).replace("</", "<\\/")
    html = env.get_template("subjective_form.html").render(items_json=items_json)
    out = OUT_DIR / "phieu_danh_gia.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nDa tao phieu: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"Ket qua he thong: {OUT_DIR / 'sample.json'}")
    print(f"Nhan file CSV tu nguoi cham -> dat vao: {OUT_DIR / 'responses'}")


if __name__ == "__main__":
    main()
