"""
Giai doan 8 - Web demo Flask: uoc luong hu hong phuong tien tu anh hien truong.

Chay (tu thu muc goc project):
    .venv\\Scripts\\python.exe app\\app.py
Mo trinh duyet: http://127.0.0.1:5000

Cac trang:
    GET  /                          trang chu: upload nhieu anh cua 1 truong hop
    POST /analyze                   xu ly anh -> chuyen sang trang ket qua
    GET  /case/<case_id>            trang ket qua + bao cao
    GET  /case/<case_id>/download/<html|csv|json>   tai bao cao
"""
import re
from pathlib import Path

from flask import (Flask, abort, flash, redirect, render_template, request,
                   send_from_directory, url_for)

from services import detection, report

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILES = 10
DOWNLOADS = {"html": "report.html", "csv": "regions.csv", "json": "report.json"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # tong dung luong upload toi da 60MB
app.secret_key = "vehicle-damage-demo"  # dung cho thong bao flash (demo noi bo)


def recent_cases(limit=6):
    """Danh sach cac truong hop da xu ly gan day (doc tu thu muc ket qua)."""
    items = []
    for d in sorted(report.RESULTS_DIR.glob("*/report.json"), reverse=True)[:limit]:
        r = report.load_report(d.parent.name)
        if r:
            items.append(r)
    return items


@app.route("/")
def index():
    return render_template("index.html", cases=recent_cases(), max_files=MAX_FILES,
                           allowed=", ".join(sorted(ALLOWED_EXT)))


@app.route("/analyze", methods=["POST"])
def analyze():
    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        flash("Vui lòng chọn ít nhất 1 ảnh.", "error")
        return redirect(url_for("index"))
    if len(files) > MAX_FILES:
        flash(f"Tối đa {MAX_FILES} ảnh cho một trường hợp (bạn chọn {len(files)}).", "error")
        return redirect(url_for("index"))

    images, rejected = [], []
    for f in files:
        name = Path(f.filename).name
        if Path(name).suffix.lower() not in ALLOWED_EXT:
            rejected.append(name)
            continue
        images.append((name, f.read()))  # doc bytes; anh hong se duoc bao loi trong bao cao
    if rejected:
        flash("Bỏ qua tệp không đúng định dạng ảnh: " + ", ".join(rejected), "warning")
    if not images:
        return redirect(url_for("index"))

    try:
        conf = min(max(float(request.form.get("conf", 0.25)), 0.05), 0.95)
    except ValueError:
        conf = 0.25
    case_name = request.form.get("case_name", "").strip()[:100]

    rep = report.process_case(images, case_name=case_name, conf=conf)
    return redirect(url_for("case", case_id=rep["case_id"]))


def _check_case_id(case_id):
    # Chi chap nhan ma dung dinh dang -> chan duong dan kieu ../../
    if not re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", case_id):
        abort(404)


@app.route("/case/<case_id>")
def case(case_id):
    _check_case_id(case_id)
    rep = report.load_report(case_id)
    if rep is None:
        abort(404)
    return render_template("result.html", r=rep, class_colors=detection.CLASS_COLORS)


@app.route("/case/<case_id>/download/<fmt>")
def download(case_id, fmt):
    _check_case_id(case_id)
    if fmt not in DOWNLOADS:
        abort(404)
    return send_from_directory(report.RESULTS_DIR / case_id, DOWNLOADS[fmt], as_attachment=True,
                               download_name=f"bao_cao_{case_id}.{fmt}")


@app.errorhandler(413)
def too_large(_):
    flash("Tổng dung lượng ảnh vượt quá 60MB. Hãy chọn ít ảnh hơn hoặc giảm kích thước ảnh.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(_):
    return render_template("base.html", not_found=True), 404


@app.template_filter("bgr2hex")
def bgr2hex(bgr):
    """Doi mau BGR (OpenCV) sang ma mau HTML."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":
    print("Đang nạp model ...")
    detection.load_model()  # nap truoc de bao loi som neu thieu model
    app.run(host="127.0.0.1", port=5000, debug=False)
