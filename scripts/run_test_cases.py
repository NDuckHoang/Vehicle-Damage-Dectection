"""
Giai doan 9 - Chay cac test case va ghi KET QUA THUC TE vao reports/test_cases.md.

Anh test:
- Anh tu tap VALIDATION (model chua tung train tren anh nay), chon NGAU NHIEN (seed co dinh)
  trong nhom anh co nhan that phu hop -> khong chon loc anh "dep".
- Anh khong hu hong: ultralytics/assets/bus.jpg (VehiDE khong co anh khong hu hong).
- Anh chat luong thap / khong hop le: tao tu anh val (lam mo, thu nho, lam toi) hoac tep gia.

Ket qua luu o results/test_cases/ (khong tron vao danh sach truong hop cua web demo).

Chay:
    python scripts/run_test_cases.py
"""
import importlib.util
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_DIR))
from services import detection as det  # noqa: E402
from services import report as rep  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "test_cases"
VAL_IMG = PROJECT_ROOT / "dataset" / "images" / "val"
VAL_LBL = PROJECT_ROOT / "dataset" / "labels" / "val"
BUS = PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "ultralytics" / "assets" / "bus.jpg"
REPORT = PROJECT_ROOT / "reports" / "test_cases.md"
CONF = 0.25

rep.RESULTS_DIR = OUT_DIR / "cases"  # tach ket qua test khoi thu muc cua web demo
NAMES = {0: "Dent", 1: "Scratch", 2: "Broken glass", 3: "Lost part", 4: "Punctured", 5: "Torn", 6: "Broken light"}


def gt_classes(img_path):
    lines = (VAL_LBL / (img_path.stem + ".txt")).read_text().splitlines()
    return [NAMES[int(l.split()[0])] for l in lines if l.strip()]


def pick(condition, seed):
    """Chon ngau nhien 1 anh val co nhan that thoa dieu kien."""
    cands = [p for p in sorted(VAL_IMG.glob("*.jpg")) if condition(gt_classes(p))]
    return random.Random(seed).choice(cands)


def run_single(tc_id, img, model):
    """Chay 1 anh, luu anh ket qua, tra ve dict ket qua."""
    res = det.analyze_image(model, img, conf=CONF)
    cv2.imwrite(str(OUT_DIR / f"{tc_id}.jpg"), np.hstack([img, det.draw_result(img, res)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    return res


def describe(res):
    if not res["regions"]:
        return "0 vùng"
    regs = ", ".join(f"{g['class']} {g['confidence']:.2f} ({g['area_ratio'] * 100:.1f}%, "
                     f"{g['severity']['level_name']})" for g in res["regions"])
    return f"{len(res['regions'])} vùng: {regs}. Mức ảnh: **{res['severity']['level_name']}**"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = det.load_model()
    rows = []  # (ma, ten, dau vao, ky vong, thuc te, ket qua)

    def add(tc, name, inp, expect, actual, ok):
        rows.append((tc, name, inp, expect, actual, "✅ Đạt" if ok else "❌ Không đạt"))
        print(f"{tc} {name}: {'DAT' if ok else 'KHONG DAT'} | {actual}")

    # ---- TC01-03: anh chi co 1 loai hu hong ----
    singles = [("TC01", "Ảnh xe bị móp", "Dent", 1), ("TC02", "Ảnh xe trầy xước", "Scratch", 2),
               ("TC03", "Ảnh vỡ kính", "Broken glass", 3)]
    picked = {}
    for tc, name, cls, seed in singles:
        p = pick(lambda g, c=cls: len(g) > 0 and set(g) == {c}, seed)
        picked[tc] = p
        res = run_single(tc, cv2.imread(str(p)), model)
        found = {g["class"] for g in res["regions"]}
        add(tc, name, f"`{p.name}` (nhãn thật: {len(gt_classes(p))} vùng {cls})",
            f"Phát hiện ít nhất 1 vùng **{cls}**", describe(res), cls in found)

    # ---- TC04: nhieu loai hu hong ----
    p = pick(lambda g: len(set(g)) >= 3, 4)
    res = run_single("TC04", cv2.imread(str(p)), model)
    gt = set(gt_classes(p))
    hit = gt & {g["class"] for g in res["regions"]}
    add("TC04", "Ảnh có nhiều loại hư hỏng", f"`{p.name}` (nhãn thật: {', '.join(sorted(gt))})",
        "Phát hiện ≥ 2 loại có trong nhãn thật",
        describe(res) + f". Trùng nhãn thật: {', '.join(sorted(hit)) or 'không'}", len(hit) >= 2)

    # ---- TC05: khong hu hong ----
    res = run_single("TC05", cv2.imread(str(BUS)), model)
    add("TC05", "Ảnh xe không hư hỏng", "`bus.jpg` (ảnh mẫu Ultralytics, xe buýt nguyên vẹn)",
        "Không phát hiện vùng hư hỏng → \"Không phát hiện hư hỏng\"",
        describe(res) if res["regions"] else "0 vùng → " + res["severity"]["level_name"],
        len(res["regions"]) == 0)

    # ---- TC06: chat luong thap ----
    src = cv2.imread(str(picked["TC01"]))
    low = cv2.GaussianBlur(cv2.resize(src, (240, 180)), (7, 7), 0)
    res = run_single("TC06a", low, model)
    add("TC06a", "Ảnh chất lượng thấp (nhỏ + mờ)", "Ảnh TC01 thu nhỏ 240×180, làm mờ Gaussian 7×7",
        "Vẫn xử lý, có cảnh báo độ phân giải thấp và mờ",
        "Cảnh báo: " + "; ".join(res["quality_warnings"]) + ". " + describe(res),
        len(res["quality_warnings"]) >= 2)
    dark = (src.astype(np.float32) * 0.12).astype(np.uint8)
    res = run_single("TC06b", dark, model)
    add("TC06b", "Ảnh chất lượng thấp (quá tối)", "Ảnh TC01 giảm độ sáng còn 12%",
        "Vẫn xử lý, có cảnh báo ảnh quá tối",
        "Cảnh báo: " + "; ".join(res["quality_warnings"]) + ". " + describe(res),
        any("tối" in w for w in res["quality_warnings"]))

    # ---- TC07: anh khong hop le ----
    fake = OUT_DIR / "fake.jpg"
    fake.write_text("day khong phai anh", encoding="utf-8")
    empty = OUT_DIR / "empty.jpg"
    empty.write_bytes(b"")
    r = rep.process_case([("fake.jpg", fake), ("empty.jpg", empty)], "TC07 ảnh không hợp lệ", CONF, model)
    errs = [im.get("error") for im in r["images"]]
    add("TC07", "Ảnh không hợp lệ", "Tệp văn bản đổi đuôi `.jpg` và tệp rỗng `.jpg`",
        "Không lỗi chương trình; mỗi tệp báo \"không phải ảnh hợp lệ\"; mức chung \"Không phát hiện\"",
        f"Lỗi từng ảnh: {errs}; mức chung: {r['summary']['level_name']}",
        all(errs) and r["summary"]["n_valid_images"] == 0)

    # ---- TC08: nhieu anh cung truong hop ----
    imgs = [(picked[t].name, picked[t]) for t in ["TC01", "TC02", "TC03"]]
    r = rep.process_case(imgs, "TC08 nhiều góc chụp", CONF, model)
    s = r["summary"]
    per_img = [im["severity"]["level"] for im in r["images"] if im["severity"]["level"] is not None]
    total = sum(len(im["regions"]) for im in r["images"])
    ok = (s["n_regions"] == total and (s["level"] == max(per_img) if per_img else s["level"] is None)
          and "trùng lặp" in s["note"])
    add("TC08", "Nhiều ảnh cùng một trường hợp", "3 ảnh của TC01, TC02, TC03 trong 1 lần tải lên",
        "Tổng số vùng = tổng các ảnh; mức chung = mức cao nhất; có ghi chú chưa loại trùng lặp",
        f"Mức từng ảnh: {[im['severity']['level_name'] for im in r['images']]}; mức chung: "
        f"**{s['level_name']}**; {s['n_regions']} vùng; loại: "
        f"{', '.join(d['class'] + '×' + str(d['count']) for d in s['damage_types'])}", ok)

    # ---- TC09: kiem thu web (Flask test client) ----
    spec = importlib.util.spec_from_file_location("webapp", APP_DIR / "app.py")
    web = importlib.util.module_from_spec(spec)
    sys.modules["webapp"] = web  # Flask can dang ky module de tim dung thu muc templates/
    spec.loader.exec_module(web)
    c = web.app.test_client()
    checks = []
    checks.append(("GET /", c.get("/").status_code == 200))
    r0 = c.post("/analyze", data={})
    checks.append(("POST không có ảnh → quay về trang chủ", r0.status_code == 302))
    with open(picked["TC01"], "rb") as f1, open(fake, "rb") as f2:
        r1 = c.post("/analyze", data={"case_name": "TC09 web", "conf": "0.25", "images": [
            (f1, "anh.jpg"), (f2, "loi.jpg"), (open(__file__, "rb"), "ghi_chu.txt")]},
            content_type="multipart/form-data")
    case_id = r1.headers.get("Location", "").rstrip("/").split("/")[-1]
    checks.append(("POST ảnh + tệp hỏng + .txt → trang kết quả", r1.status_code == 302 and "/case/" in r1.location))
    checks.append(("GET trang kết quả", c.get(f"/case/{case_id}").status_code == 200))
    for fmt in ["html", "csv", "json"]:
        checks.append((f"Tải báo cáo {fmt}", c.get(f"/case/{case_id}/download/{fmt}").status_code == 200))
    checks.append(("Mã trường hợp sai → 404", c.get("/case/khong_ton_tai").status_code == 404))
    checks.append(("Định dạng tải sai → 404", c.get(f"/case/{case_id}/download/exe").status_code == 404))
    add("TC09", "Web demo (Flask test client)", "Các request tới các route",
        "Mọi route trả đúng mã HTTP", "; ".join(f"{n}: {'OK' if v else 'LỖI'}" for n, v in checks),
        all(v for _, v in checks))

    # ---- Ghi bao cao ----
    n_ok = sum(1 for r in rows if r[5].startswith("✅"))
    L = ["# Test case – Hệ thống ước lượng hư hỏng phương tiện", "",
         "> Sinh tự động bởi `scripts/run_test_cases.py`. Kết quả là **kết quả chạy thực tế** của model "
         f"`yolov8n_seg/best.pt` (ngưỡng confidence {CONF}). Ảnh hư hỏng được chọn **ngẫu nhiên** "
         "(seed cố định) từ tập validation theo nhãn thật, không chọn lọc.", "",
         f"**Tổng kết: {n_ok}/{len(rows)} test case đạt.**", "",
         "| Mã | Test case | Đầu vào | Kết quả mong đợi | Kết quả thực tế | Đánh giá |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    L += ["", "Ảnh kết quả (trái: ảnh gốc, phải: dự đoán): `results/test_cases/TCxx.jpg`. "
              "Báo cáo của TC07, TC08: `results/test_cases/cases/`.", "",
          "Ghi chú:",
          "- TC01–TC05 đánh giá **chất lượng model**: kết quả \"Không đạt\" phản ánh hạn chế thực tế "
          "của model (recall thấp với vùng nhỏ, báo nhầm), không phải lỗi chương trình.",
          "- TC06–TC09 đánh giá **hành vi hệ thống** (cảnh báo, xử lý lỗi, tổng hợp, web)."]
    # Giu lai phan phan tich viet tay (sau MARKER) khi chay lai
    marker = "<!-- PHAN TICH VIET TAY: noi dung ben duoi duoc giu nguyen khi chay lai -->"
    manual = ""
    if REPORT.exists() and marker in REPORT.read_text(encoding="utf-8"):
        manual = REPORT.read_text(encoding="utf-8").split(marker, 1)[1]
    REPORT.write_text("\n".join(L) + "\n\n" + marker + manual, encoding="utf-8")
    print(f"\n{n_ok}/{len(rows)} dat. Bao cao: {REPORT}")


if __name__ == "__main__":
    main()
