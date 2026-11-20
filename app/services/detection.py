"""
Phat hien + phan doan vung hu hong tren 1 anh bang YOLOv8-Seg, roi uoc luong muc do.

Ham chinh:
    load_model(path)                 -> nap model 1 lan
    check_image_quality(img)         -> canh bao anh nho / mo / qua toi / qua sang
    analyze_image(model, img, conf)  -> dict ket qua (vung hu hong, dien tich, muc do)
    draw_result(img, result)         -> anh da ve mask + nhan
"""
from pathlib import Path

import cv2
import numpy as np

from . import severity as sv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "trained" / "yolov8n_seg" / "weights" / "best.pt"

# Mau BGR co dinh cho tung class (giong cac script truc quan hoa)
CLASS_COLORS = {
    "Dent": (0, 0, 255), "Scratch": (0, 200, 255), "Broken glass": (255, 128, 0),
    "Lost part": (200, 0, 200), "Punctured": (0, 200, 0), "Torn": (255, 0, 128),
    "Broken light": (0, 128, 255),
}
# Ten tieng Viet de hien thi tren web / bao cao
CLASS_VI = {
    "Dent": "Móp, lõm", "Scratch": "Trầy sơn", "Broken glass": "Vỡ kính",
    "Lost part": "Mất bộ phận", "Punctured": "Thủng", "Torn": "Rách", "Broken light": "Bể đèn",
}

_model_cache = {}


def load_model(path=DEFAULT_MODEL):
    """Nap model YOLO (chi nap 1 lan cho moi duong dan)."""
    from ultralytics import YOLO
    path = str(path)
    if path not in _model_cache:
        if not Path(path).exists():
            raise FileNotFoundError(f"Không tìm thấy model: {path}")
        _model_cache[path] = YOLO(path)
    return _model_cache[path]


def read_image(path_or_bytes):
    """Doc anh tu duong dan hoac bytes. Tra ve anh BGR, hoac raise ValueError neu khong hop le."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes))
    if img is None:
        raise ValueError("Tệp không phải ảnh hợp lệ hoặc ảnh bị hỏng.")
    return img


def check_image_quality(img):
    """
    Canh bao chat luong anh (khong chan xu ly, chi bao cho nguoi dung).
    Nguong mang tinh kinh nghiem.
    """
    warnings = []
    h, w = img.shape[:2]
    if min(h, w) < 320:
        warnings.append(f"Ảnh độ phân giải thấp ({w}x{h}), kết quả có thể kém chính xác.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Do net: phuong sai cua Laplacian (cang nho cang mo)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharp < 30:
        warnings.append(f"Ảnh có vẻ bị mờ (độ nét {sharp:.0f}).")
    mean = gray.mean()
    if mean < 40:
        warnings.append(f"Ảnh quá tối (độ sáng trung bình {mean:.0f}/255).")
    elif mean > 225:
        warnings.append(f"Ảnh quá sáng (độ sáng trung bình {mean:.0f}/255).")
    return warnings


def analyze_image(model, img, conf=0.25, imgsz=640):
    """
    Chay model tren 1 anh BGR. Tra ve:
        width, height, regions (list), total_area_ratio, severity, quality_warnings
    Moi region: id, class, class_vi, confidence, area_px, area_ratio, polygon, severity
    """
    h, w = img.shape[:2]
    cfg = sv.load_config()
    # retina_masks=True: mask tra ve dung kich thuoc anh goc (h, w)
    r = model.predict(img, conf=conf, imgsz=imgsz, retina_masks=True, verbose=False)[0]

    regions = []
    union = np.zeros((h, w), np.uint8)  # hop cac vung hu hong
    if r.masks is not None:
        classes = r.boxes.cls.int().tolist()
        confs = r.boxes.conf.tolist()
        masks = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)  # (n, h, w)
        for cid, c, m in zip(classes, confs, masks):
            # 1 mask co the gom nhieu manh roi -> lay contour ngoai cua TUNG manh
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [k for k in contours if len(k) >= 3]
            if not contours:
                continue
            # To kin cac contour (giong polygon nhan that: khong co lo) roi dem pixel
            filled = np.zeros((h, w), np.uint8)
            cv2.drawContours(filled, contours, -1, 1, thickness=-1)
            area_px = int(filled.sum())
            ratio = area_px / (h * w)  # dien tich mask / dien tich anh
            union |= filled
            name = model.names[cid]
            regions.append({
                "id": len(regions) + 1,
                "class": name,
                "class_vi": CLASS_VI.get(name, name),
                "confidence": round(float(c), 3),
                "area_ratio": round(float(ratio), 6),
                "area_px": area_px,
                # Danh sach polygon (moi manh 1 polygon), rut gon diem de file JSON nho
                "polygons": [cv2.approxPolyDP(k, 1.5, True).reshape(-1, 2).tolist() for k in contours],
                "severity": sv.region_severity(name, ratio, cfg),
            })

    # Dien tich HOP cac mask (khong dem trung phan chong lan)
    total = float(union.mean())
    sev_regions = [{"level": g["severity"]["level"]} for g in regions]
    return {
        "width": w, "height": h,
        "regions": regions,
        "total_area_ratio": round(total, 6),
        "severity": sv.image_severity(sev_regions, total, cfg),
        "quality_warnings": check_image_quality(img),
    }


def draw_result(img, result):
    """Ve mask ban trong suot + vien + nhan '#id Class conf%' len anh."""
    out = img.copy()
    overlay = img.copy()
    for g in result["regions"]:
        polys = [np.array(p, dtype=np.int32) for p in g["polygons"]]
        cv2.fillPoly(overlay, polys, CLASS_COLORS.get(g["class"], (255, 255, 255)))
    out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)

    s = max(img.shape[:2]) / 1000  # co chu theo kich thuoc anh
    thick = max(1, int(2 * s))
    for g in result["regions"]:
        color = CLASS_COLORS.get(g["class"], (255, 255, 255))
        polys = [np.array(p, dtype=np.int32) for p in g["polygons"]]
        cv2.polylines(out, polys, True, color, thick, cv2.LINE_AA)
        label = f"#{g['id']} {g['class']} {g['confidence'] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, thick)
        pts = np.vstack(polys)
        x, y = pts[np.argmin(pts[:, 1])]  # dat nhan o diem cao nhat cua vung
        x = int(min(max(x, 0), out.shape[1] - tw - 4))
        y = int(max(y, th + 6))
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(out, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return out
