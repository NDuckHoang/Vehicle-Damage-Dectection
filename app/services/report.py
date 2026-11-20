"""
Xu ly 1 "truong hop" gom nhieu anh (nhieu goc chup) do NGUOI DUNG chu dong nhom lai,
tong hop ket qua va xuat bao cao.

Moi truong hop luu trong app/static/results/<case_id>/ :
    NN_original.jpg, NN_result.jpg   anh goc va anh da ve mask (NN = so thu tu anh)
    report.json                      toan bo ket qua
    regions.csv                      bang cac vung hu hong (mo duoc bang Excel)
    report.html                      bao cao 1 file (anh nhung san) de tai ve / in

Chay thu tu dong lenh:
    python -m app.services.report anh1.jpg anh2.jpg --name "Xe thu nghiem"
"""
import base64
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

import cv2
from jinja2 import Environment, FileSystemLoader

from . import detection as det
from . import severity as sv

APP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = APP_DIR / "static" / "results"
TEMPLATE_DIR = APP_DIR / "templates"


def new_case_id():
    """Ma truong hop duy nhat: thoi gian + 6 ky tu ngau nhien (khong ghi de truong hop cu)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]


def process_case(images, case_name="", conf=0.25, model=None):
    """
    images: list (ten_file, du_lieu) voi du_lieu la bytes hoac duong dan.
    Tra ve dict bao cao (da luu ra dia).
    """
    model = model or det.load_model()
    case_id = new_case_id()
    case_dir = RESULTS_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=False)

    image_results = []
    for idx, (filename, data) in enumerate(images, start=1):
        item = {"index": idx, "filename": filename}
        try:
            img = det.read_image(data)
        except ValueError as e:
            # Anh khong hop le: ghi loi, van xu ly cac anh con lai
            item.update({"error": str(e), "regions": [],
                         "severity": {"level": None, "level_name": "Lỗi ảnh", "reasons": [str(e)]}})
            image_results.append(item)
            continue

        result = det.analyze_image(model, img, conf=conf)
        orig_name, res_name = f"{idx:02d}_original.jpg", f"{idx:02d}_result.jpg"
        cv2.imwrite(str(case_dir / orig_name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(case_dir / res_name), det.draw_result(img, result), [cv2.IMWRITE_JPEG_QUALITY, 92])
        item.update(result)
        item.update({"original": orig_name, "result": res_name})
        image_results.append(item)

    report = {
        "case_id": case_id,
        "case_name": case_name or f"Trường hợp {case_id}",
        "created_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "conf_threshold": conf,
        "images": image_results,
        "summary": summarize(image_results),
        "disclaimer": sv.DISCLAIMER,
    }
    save_report(report, case_dir)
    return report


def summarize(image_results):
    """Tong hop ca truong hop: muc do chung, so vung theo loai, thong ke anh."""
    valid = [r for r in image_results if "error" not in r]
    case = sv.case_severity(valid)
    regions = [g for r in valid for g in r["regions"]]
    by_level = {name: 0 for name in sv.LEVELS}
    for g in regions:
        by_level[g["severity"]["level_name"]] += 1
    worst = max((r for r in valid if r["severity"]["level"] is not None),
                key=lambda r: r["severity"]["level"], default=None)
    return {
        "level": case["level"],
        "level_name": case["level_name"],
        "note": case["note"],
        "n_images": len(image_results),
        "n_valid_images": len(valid),
        "n_error_images": len(image_results) - len(valid),
        "n_images_with_damage": sum(1 for r in valid if r["regions"]),
        "n_regions": len(regions),
        "damage_types": [{"class": c, "class_vi": det.CLASS_VI.get(c, c), "count": n}
                         for c, n in sorted(case["damage_types"].items(), key=lambda kv: -kv[1])],
        "regions_by_level": by_level,
        "worst_image": worst["index"] if worst else None,
    }


def save_report(report, case_dir):
    """Ghi report.json, regions.csv va report.html."""
    (case_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # utf-8-sig: Excel tren Windows doc dung tieng Viet
    with open(case_dir / "regions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["anh", "ten_file", "vung", "loai", "loai_vi", "confidence",
                    "dien_tich_px", "ti_le_dien_tich_%", "muc_do", "ly_do"])
        for r in report["images"]:
            for g in r["regions"]:
                w.writerow([r["index"], r["filename"], g["id"], g["class"], g["class_vi"],
                            g["confidence"], g["area_px"], round(g["area_ratio"] * 100, 3),
                            g["severity"]["level_name"], g["severity"]["reason"]])

    (case_dir / "report.html").write_text(render_html(report, case_dir), encoding="utf-8")


def _img_base64(path, max_w=1000):
    """Thu nho anh va ma hoa base64 de nhung vao file HTML (bao cao chi can 1 file)."""
    img = cv2.imread(str(path))
    if img.shape[1] > max_w:
        img = cv2.resize(img, None, fx=max_w / img.shape[1], fy=max_w / img.shape[1],
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def render_html(report, case_dir):
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    images = {}
    for r in report["images"]:
        if "result" in r:
            images[r["index"]] = {"original": _img_base64(case_dir / r["original"]),
                                  "result": _img_base64(case_dir / r["result"])}
    return env.get_template("report_export.html").render(r=report, images=images)


def load_report(case_id):
    """Doc lai bao cao da luu (dung cho trang ket qua cua web)."""
    path = RESULTS_DIR / case_id / "report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Xu ly 1 truong hop nhieu anh")
    p.add_argument("images", nargs="+")
    p.add_argument("--name", default="")
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()
    rep = process_case([(Path(x).name, x) for x in a.images], a.name, a.conf)
    s = rep["summary"]
    print(f"Ma truong hop: {rep['case_id']}")
    for r in rep["images"]:
        if "error" in r:
            print(f"  Anh {r['index']} ({r['filename']}): LOI - {r['error']}")
            continue
        print(f"  Anh {r['index']} ({r['filename']}): {len(r['regions'])} vung, "
              f"muc do: {r['severity']['level_name']}, canh bao: {r['quality_warnings']}")
        for g in r["regions"]:
            print(f"     #{g['id']} {g['class']:13s} conf={g['confidence']:.2f} "
                  f"dien tich={g['area_ratio'] * 100:.2f}% -> {g['severity']['level_name']}")
    print(f"TONG HOP: {s['level_name']} | {s['n_regions']} vung | loai: "
          f"{[(d['class'], d['count']) for d in s['damage_types']]}")
    print(f"Thu muc: {RESULTS_DIR / rep['case_id']}")
